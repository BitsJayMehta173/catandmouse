import socket
import threading
import time
import cv2
import ctypes
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

# Make all coordinate operations (SetCursorPos, GetSystemMetrics) use
# physical pixels consistently, regardless of display DPI scaling.
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

class HostController:
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.active_client_ip = None
        self.gaze_states = {"host": 0.0}  # ip -> float confidence (0.0-1.0)
        
        self.mouse_listener = None
        self.vx = 0.5
        self.vy = 0.5
        self.sensitivity = 1.5
        self._gaze_lock = threading.Lock()  # protects gaze_states across threads
        
        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # TCP Server for Clicks/Commands
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.tcp_server.bind(('0.0.0.0', network_utils.TCP_PORT))
            self.tcp_server.listen(5)
            print(f"[Server] TCP Click/Command server listening on port {network_utils.TCP_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind TCP Click/Command server: {e}")
            raise
        
        # TCP Server for Gaze State
        self.gaze_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.gaze_server.bind(('0.0.0.0', network_utils.GAZE_PORT))
            self.gaze_server.listen(5)
            print(f"[Server] Gaze state server listening on port {network_utils.GAZE_PORT}")
        except Exception as e:
            print(f"[ERROR] Failed to bind Gaze state server: {e}")
            raise

        self.client_tcp_sockets = {} # IP -> socket

        # Mouse tracking for deltas
        self.last_pos = None
        self.mouse_controller = mouse.Controller()
        
        # Calibration state
        self.is_calibrating = False

    def _win32_filter_client_mode(self, msg, data):
        """In CLIENT mode: handle EVERYTHING here.
        Key insight: with suppress=True, Windows does NOT advance its internal cursor
        position after a suppressed event. So computing dx from data.pt drifts over time.
        Fix: snap cursor back to screen-center after every real event so Windows always
        starts the NEXT delta from center. Skip the resulting injected event via LLMHF_INJECTED.
        """
        LLMHF_INJECTED = 0x01

        if msg == 0x0200:  # WM_MOUSEMOVE
            # Skip events we injected ourselves (snap-back)
            if data.flags & LLMHF_INJECTED:
                return False

            user32 = ctypes.windll.user32
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
            cx, cy = sw // 2, sh // 2

            # Delta is always relative to center because we snap back there after each event
            dx = data.pt.x - cx
            dy = data.pt.y - cy

            if (dx != 0 or dy != 0) and self.active_client_ip:
                self.vx = max(0.0, min(1.0, self.vx + (dx / sw) * self.sensitivity))
                self.vy = max(0.0, min(1.0, self.vy + (dy / sh) * self.sensitivity))
                try:
                    pkt = network_utils.pack_move(self.vx, self.vy)
                    self.udp_sock.sendto(pkt, (self.active_client_ip, network_utils.UDP_PORT))
                except Exception:
                    pass

            # Snap back to center — keeps Windows cursor state consistent for next delta.
            # This fires an injected WM_MOUSEMOVE caught by the LLMHF_INJECTED check above.
            user32.SetCursorPos(cx, cy)

        elif msg == 0x0201: self.send_manual_click(1, True)
        elif msg == 0x0202: self.send_manual_click(1, False)
        elif msg == 0x0204: self.send_manual_click(2, True)
        elif msg == 0x0205: self.send_manual_click(2, False)
        elif msg == 0x0207: self.send_manual_click(3, True)
        elif msg == 0x0208: self.send_manual_click(3, False)

        elif msg == 0x020A:  # WM_MOUSEWHEEL (vertical)
            delta = ctypes.c_short(data.mouseData >> 16).value / 120
            self._send_scroll_to_client(0, delta)
        elif msg == 0x020E:  # WM_MOUSEHWHEEL (horizontal)
            delta = ctypes.c_short(data.mouseData >> 16).value / 120
            self._send_scroll_to_client(delta, 0)

        return False  # Suppress ALL host input

    def _win32_filter_host_mode(self, msg, data):
        """HOST mode: pass everything through normally."""
        return True

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
                try: sock.sendall(data)
                except Exception as e: print(f"Error: {e}")

    def _restart_listener(self, suppress):
        """Restart the mouse listener with the appropriate suppress mode."""
        if self.mouse_listener and self.mouse_listener.running:
            self.mouse_listener.stop()
            # No join() — stopping is async but the new listener starts cleanly

        win32_filter = self._win32_filter_client_mode if suppress else self._win32_filter_host_mode
        self.mouse_listener = mouse.Listener(
            suppress=suppress,
            win32_event_filter=win32_filter
        )
        self.mouse_listener.start()

    def switch_focus(self, target_ip):
        if self.active_client_ip == target_ip:
            return

        self.active_client_ip = target_ip

        if target_ip:
            print(f"[*] Switching to CLIENT mode — host input fully suppressed")
            # Park cursor at center so the first real delta is computed from center
            user32 = ctypes.windll.user32
            cx = user32.GetSystemMetrics(0) // 2
            cy = user32.GetSystemMetrics(1) // 2
            user32.SetCursorPos(cx, cy)
            self.vx, self.vy = 0.5, 0.5
            self._restart_listener(suppress=True)
        else:
            print("[*] Switching to HOST mode — host input restored")
            self._restart_listener(suppress=False)

    def focus_arbiter(self):
        """
        Runs every 100 ms.  Rule: whichever device has the highest confidence
        score claims the mouse, as long as it beats the current owner by at
        least MARGIN (hysteresis so we don't flip-flop on noise).
        """
        MARGIN   = 0.06   # challenger must beat owner by this much to take over
        INTERVAL = 0.10   # 10 Hz
        loop_count = 0

        while True:
            time.sleep(INTERVAL)
            loop_count += 1

            with self._gaze_lock:
                states = dict(self.gaze_states)

            if loop_count % 10 == 0:
                print(f"[Arbiter] {states} | owner={self.active_client_ip or 'host'}")

            # Current owner's confidence
            owner_key  = "host" if self.active_client_ip is None else self.active_client_ip
            owner_conf = states.get(owner_key, 0.0)

            # Find the device with the highest confidence (excluding current owner)
            best_ip   = self.active_client_ip   # default: keep owner
            best_conf = owner_conf + MARGIN      # challenger must exceed this

            for ip, conf in states.items():
                dev_ip = None if ip == "host" else ip
                if dev_ip == self.active_client_ip:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_ip   = dev_ip

            self.switch_focus(best_ip)

    def start(self):
        print(f"Starting Host Server (Camera: {self.camera_source})...")
        threading.Thread(target=self.accept_tcp_clients, daemon=True).start()
        threading.Thread(target=self.accept_gaze_clients, daemon=True).start()
        threading.Thread(target=self.run_vision,      daemon=True).start()
        threading.Thread(target=self.focus_arbiter,   daemon=True).start()

        # Start in Host Mode — listener in normal (suppress=False) state
        self.switch_focus(None)

        print("[Host] Gaze-based focus switching active. Press ESC in camera window to quit.")
        # Keep main thread alive
        while True:
            time.sleep(1)

    def accept_tcp_clients(self):
        while True:
            client, addr = self.tcp_server.accept()
            print(f"[TCP] Client connected: {addr[0]}")
            self.client_tcp_sockets[addr[0]] = client

    def accept_gaze_clients(self):
        while True:
            client, addr = self.gaze_server.accept()
            print(f"[Gaze] Client connected: {addr[0]}")
            threading.Thread(target=self.handle_gaze_client, args=(client, addr[0]), daemon=True).start()

    def handle_gaze_client(self, client, ip):
        GAZE_SIZE = 4  # float = 4 bytes
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
                # NOTE: focus_arbiter handles all switching at 10 Hz.
        except Exception as e:
            print(f"[Gaze] Lost connection with {ip}: {e}")
        finally:
            client.close()
            with self._gaze_lock:
                self.gaze_states.pop(ip, None)


    def run_vision(self):
        print(f"[Vision] Initializing camera source: {self.camera_source}...")
        
        source = self.camera_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        cap = None
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Could not open camera source: {source}")
            return

        tracker = GazeTracker()
        print("Press 'C' to start coordinated calibration.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            annotated_image, confidence, angles = tracker.process_frame(frame)
            with self._gaze_lock:
                self.gaze_states["host"] = confidence
            # NOTE: do NOT call update_focus here — focus_arbiter does it at 10 Hz.

            focused = confidence >= 0.40
            text  = f"HOST FOCUSED  ({confidence:.2f})" if focused else f"HOST AWAY  ({confidence:.2f})"
            color = (0, 255, 0) if focused else (0, 0, 255)
            cv2.putText(annotated_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(annotated_image, "Press 'C' to Calibrate", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

            if self.is_calibrating:
                cv2.putText(annotated_image, "CALIBRATING HOST...", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if tracker.is_calibrated:
                    self.is_calibrating = False
                    print("[Vision] Host calibration complete. Triggering Clients...")
                    self.broadcast_calibration()

            cv2.imshow('Host Gaze Tracker', annotated_image)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('c'):
                print("[Vision] Starting calibration...")
                tracker.is_calibrated = False
                tracker.calibration_samples = []
                self.is_calibrating = True
            elif key == 27: # ESC
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def broadcast_calibration(self):
        data = network_utils.pack_control(1) # 1 = Start Calibration
        for ip, sock in self.client_tcp_sockets.items():
            try:
                sock.sendall(data)
                print(f"[Control] Sent calibration command to {ip}")
            except Exception as e:
                print(f"[Control] Failed to send to {ip}: {e}")

if __name__ == "__main__":
    import sys

    camera_source = 0
    if len(sys.argv) > 1:
        camera_source = sys.argv[1]

    host = HostController(camera_source)
    host.start()
