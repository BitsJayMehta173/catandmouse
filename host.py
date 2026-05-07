import socket
import threading
import time
import cv2
import ctypes
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

# Make all coordinate operations use physical pixels regardless of DPI scaling.
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# Enable 1ms timer resolution for the entire process
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

# ── Performance constants ──────────────────────────────────────────────────────
GAZE_FPS      = 12         # Gaze checks per second (up from 8 → faster switching)
GAZE_INTERVAL = 1.0 / GAZE_FPS
PROCESS_W     = 640        # Frame width fed to MediaPipe
PROCESS_H     = 480        # Frame height fed to MediaPipe
# ──────────────────────────────────────────────────────────────────────────────


class HostController:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.active_client_ip = None
        self.gaze_states = {"host": 0.0}   # ip -> float confidence (0.0-1.0)

        self.mouse_listener = None
        self.vx = 0.5
        self.vy = 0.5
        self.sensitivity = 1.5
        self._gaze_lock = threading.Lock()

        # ── Mouse Throttling ──────────────────────────────────────────────────
        self.last_send_time = 0.0
        self.SEND_INTERVAL  = 1.0 / 120.0  # 120Hz (up from 90Hz)
        # ──────────────────────────────────────────────────────────────────────

        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # TCP Server for Clicks/Commands
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp_server.bind(('0.0.0.0', network_utils.TCP_PORT))
            self.tcp_server.listen(5)
            print(f"[Server] TCP server listening on port {network_utils.TCP_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind TCP server: {e}")
            raise

        # TCP Server for Gaze State
        self.gaze_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.gaze_server.bind(('0.0.0.0', network_utils.GAZE_PORT))
            self.gaze_server.listen(5)
            print(f"[Server] Gaze server listening on port {network_utils.GAZE_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind Gaze server: {e}")
            raise

        self.client_tcp_sockets = {}  # IP -> socket

        self.last_pos = None
        self.mouse_controller = mouse.Controller()

        # Calibration state
        self.is_calibrating = False
        self._trigger_calibration_flag = False

        # Vision shared state (set in run_vision)
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._tracker      = None   # GazeTracker instance

    # ── Win32 mouse filter (CLIENT mode) ──────────────────────────────────────

    def _win32_filter_client_mode(self, msg, data):
        """Suppress all host input and forward movement/clicks to active client."""
        LLMHF_INJECTED = 0x01

        if msg == 0x0200:  # WM_MOUSEMOVE
            if data.flags & LLMHF_INJECTED:
                return False

            user32 = ctypes.windll.user32
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
            cx, cy = sw // 2, sh // 2

            dx = data.pt.x - cx
            dy = data.pt.y - cy

            if (dx != 0 or dy != 0) and self.active_client_ip:
                self.vx = max(0.0, min(1.0, self.vx + (dx / sw) * self.sensitivity))
                self.vy = max(0.0, min(1.0, self.vy + (dy / sh) * self.sensitivity))

                # Throttle sending to ~120Hz
                now = time.perf_counter()  # Higher precision than time.time()
                if now - self.last_send_time >= self.SEND_INTERVAL:
                    try:
                        pkt = network_utils.pack_move(self.vx, self.vy)
                        self.udp_sock.sendto(pkt, (self.active_client_ip, network_utils.UDP_PORT))
                        self.last_send_time = now
                    except Exception:
                        pass

            user32.SetCursorPos(cx, cy)

        elif msg == 0x0201: self.send_manual_click(1, True)
        elif msg == 0x0202: self.send_manual_click(1, False)
        elif msg == 0x0204: self.send_manual_click(2, True)
        elif msg == 0x0205: self.send_manual_click(2, False)
        elif msg == 0x0207: self.send_manual_click(3, True)
        elif msg == 0x0208: self.send_manual_click(3, False)

        elif msg == 0x020A:  # WM_MOUSEWHEEL
            delta = ctypes.c_short(data.mouseData >> 16).value / 120
            self._send_scroll_to_client(0, delta)
        elif msg == 0x020E:  # WM_MOUSEHWHEEL
            delta = ctypes.c_short(data.mouseData >> 16).value / 120
            self._send_scroll_to_client(delta, 0)

        return False  # Suppress ALL host input

    def _win32_filter_host_mode(self, msg, data):
        """HOST mode: pass everything through normally."""
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_scroll_to_client(self, dx, dy):
        if not self.active_client_ip:
            return
        sock = self.client_tcp_sockets.get(self.active_client_ip)
        if sock:
            try:
                sock.sendall(network_utils.pack_scroll(dx, dy))
            except Exception as e:
                print(f"[Scroll] Error: {e}")

    def send_manual_click(self, button_id, pressed):
        if self.active_client_ip:
            data = network_utils.pack_click(button_id, pressed)
            sock = self.client_tcp_sockets.get(self.active_client_ip)
            if sock:
                try:
                    sock.sendall(data)
                except Exception as e:
                    print(f"[Click] Error: {e}")

    def _restart_listener(self, suppress):
        if self.mouse_listener and self.mouse_listener.running:
            self.mouse_listener.stop()

        win32_filter = self._win32_filter_client_mode if suppress else self._win32_filter_host_mode
        self.mouse_listener = mouse.Listener(
            suppress=suppress,
            win32_event_filter=win32_filter
        )
        self.mouse_listener.start()

    # ── Focus switching ───────────────────────────────────────────────────────

    def switch_focus(self, target_ip):
        if self.active_client_ip == target_ip:
            return

        self.active_client_ip = target_ip

        if target_ip:
            print(f"[Focus] → CLIENT {target_ip} (host input suppressed)")
            user32 = ctypes.windll.user32
            cx = user32.GetSystemMetrics(0) // 2
            cy = user32.GetSystemMetrics(1) // 2
            user32.SetCursorPos(cx, cy)
            self.vx, self.vy = 0.5, 0.5
            self._restart_listener(suppress=True)
        else:
            print("[Focus] → HOST (input restored)")
            self._restart_listener(suppress=False)

    def focus_arbiter(self):
        """
        Runs at 15 Hz (up from 10 Hz). Switches focus to whichever device has
        the highest gaze confidence, provided it beats the current owner by MARGIN.
        """
        MARGIN   = 0.06
        INTERVAL = 1.0 / 15.0   # 15 Hz (faster arbitration)
        loop_count = 0

        while True:
            time.sleep(INTERVAL)
            loop_count += 1

            with self._gaze_lock:
                states = dict(self.gaze_states)

            if loop_count % 45 == 0:  # Log every ~3 seconds
                owner = self.active_client_ip or 'host'
                print(f"[Arbiter] {states} | owner={owner}")

            owner_key  = "host" if self.active_client_ip is None else self.active_client_ip
            owner_conf = states.get(owner_key, 0.0)

            best_ip   = self.active_client_ip
            best_conf = owner_conf + MARGIN

            for ip, conf in states.items():
                dev_ip = None if ip == "host" else ip
                if dev_ip == self.active_client_ip:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_ip   = dev_ip

            self.switch_focus(best_ip)

    # ── TCP / Gaze listeners ──────────────────────────────────────────────────

    def accept_tcp_clients(self):
        while True:
            client, addr = self.tcp_server.accept()
            print(f"[TCP] Client connected: {addr[0]}")
            self.client_tcp_sockets[addr[0]] = client

    def accept_gaze_clients(self):
        while True:
            client, addr = self.gaze_server.accept()
            print(f"[Gaze] Client connected: {addr[0]}")
            threading.Thread(
                target=self.handle_gaze_client, args=(client, addr[0]), daemon=True
            ).start()

    def handle_gaze_client(self, client, ip):
        GAZE_SIZE = 4
        try:
            while True:
                data = b''
                while len(data) < GAZE_SIZE:
                    chunk = client.recv(GAZE_SIZE - len(data))
                    if not chunk:
                        raise ConnectionError("Gaze socket closed")
                    data += chunk
                confidence = network_utils.unpack_gaze(data)
                with self._gaze_lock:
                    self.gaze_states[ip] = max(0.0, min(1.0, confidence))
        except Exception as e:
            print(f"[Gaze] Lost connection with {ip}: {e}")
        finally:
            client.close()
            with self._gaze_lock:
                self.gaze_states.pop(ip, None)

    def broadcast_calibration(self):
        data = network_utils.pack_control(1)
        for ip, sock in self.client_tcp_sockets.items():
            try:
                sock.sendall(data)
                print(f"[Control] Sent calibration command to {ip}")
            except Exception as e:
                print(f"[Control] Failed to send to {ip}: {e}")

    # ── Vision (headless, background) ────────────────────────────────────────

    def _listen_calibration_input(self):
        """Background thread: press Enter in terminal to trigger calibration."""
        print("\n[Host] ── Press ENTER at any time to start calibration ──\n")
        while True:
            try:
                input()
                if self._tracker is None:
                    print("[Host] Camera not ready yet — please wait a moment.")
                    continue
                self._trigger_calibration_flag = True
                print("[Host] Calibration starting on next gaze cycle...")
            except (EOFError, KeyboardInterrupt):
                break

    def run_vision(self):
        """
        Opens the camera in a background capture thread.
        Gaze is computed at GAZE_FPS on 640x480 frames.
        No cv2 window is shown — everything runs silently.
        """
        print(f"[Vision] Opening camera {self.camera_source} in background mode...")

        source = self.camera_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Could not open camera {source}")
            return

        print(f"[Vision] Camera ready. Processing at {GAZE_FPS} fps ({PROCESS_W}x{PROCESS_H}).")

        # ── Background frame capture thread ───────────────────────────────────
        def _capture_loop():
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                    with self._frame_lock:
                        self._latest_frame = small

        threading.Thread(target=_capture_loop, daemon=True).start()

        # ── Create tracker (headless = no rendering overhead) ─────────────────
        self._tracker = GazeTracker(headless=True)

        # ── Calibration input listener ────────────────────────────────────────
        threading.Thread(target=self._listen_calibration_input, daemon=True).start()

        print("[Host] Gaze tracking active (background). Status printed every 3s.")

        last_process = 0.0
        loop_count   = 0

        while True:
            now = time.perf_counter()
            wait = GAZE_INTERVAL - (now - last_process)
            if wait > 0:
                time.sleep(wait)

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.02)
                continue

            last_process = time.perf_counter()
            loop_count  += 1

            # Handle calibration trigger from terminal
            if self._trigger_calibration_flag:
                self._tracker.is_calibrated     = False
                self._tracker.calibration_samples = []
                self.is_calibrating             = True
                self._trigger_calibration_flag  = False
                print("[Vision] Calibration started — look straight at your camera...")

            _, confidence, _ = self._tracker.process_frame(frame)

            with self._gaze_lock:
                self.gaze_states["host"] = confidence

            # Calibration completion check
            if self.is_calibrating and self._tracker.is_calibrated:
                self.is_calibrating = False
                print("[Vision] Host calibration complete. Triggering clients...")
                self.broadcast_calibration()

            # Status log every 3 seconds
            if loop_count % (GAZE_FPS * 3) == 0:
                focused = confidence >= 0.40
                status  = "FOCUSED" if focused else "AWAY"
                owner   = self.active_client_ip or "host"
                print(f"[Host]  gaze={confidence:.2f} ({status}) | owner={owner}")

    # ── Entry point ───────────────────────────────────────────────────────────

    def start(self):
        print(f"Starting Host (camera={self.camera_source}) ...")
        threading.Thread(target=self.accept_tcp_clients, daemon=True).start()
        threading.Thread(target=self.accept_gaze_clients, daemon=True).start()
        threading.Thread(target=self.run_vision,          daemon=True).start()
        threading.Thread(target=self.focus_arbiter,       daemon=True).start()

        # Start in Host mode
        self.switch_focus(None)

        print("[Host] Running in background. Press ENTER to calibrate, Ctrl+C to quit.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Host] Shutting down.")


if __name__ == "__main__":
    import sys

    camera_source = 0
    if len(sys.argv) > 1:
        camera_source = sys.argv[1]

    host = HostController(camera_source)
    host.start()
