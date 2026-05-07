"""
client_webcam.py — CLIENT using the machine's built-in / default webcam.

Runs fully in the background (no camera window).
Gaze is processed at 8 fps on 640x480 frames for low CPU usage and fast switching.

Usage:
    python client_webcam.py <HOST_IP> [CAMERA_INDEX]

Examples:
    python client_webcam.py 192.168.1.10          # camera index 0 (default)
    python client_webcam.py 192.168.1.10 1        # camera index 1

For DroidCam / phone camera, use client.py instead.
"""

import socket
import threading
import time
import cv2
import struct
import ctypes
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# ── Performance constants ──────────────────────────────────────────────────────
GAZE_FPS      = 8
GAZE_INTERVAL = 1.0 / GAZE_FPS
PROCESS_W     = 640
PROCESS_H     = 480
SEND_THRESHOLD = 0.02   # Send gaze update when confidence changes by this much
# ──────────────────────────────────────────────────────────────────────────────


class ClientController:
    def __init__(self, host_ip, camera_index=0):
        self.host_ip      = host_ip
        self.camera_index = int(camera_index)
        self.mouse_controller   = mouse.Controller()
        self.should_calibrate   = False

        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind(('0.0.0.0', network_utils.UDP_PORT))

        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.gaze_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        print(f"[Client] Connecting to Host {self.host_ip}...")

        while True:
            try:
                print(f"[TCP] Connecting to {self.host_ip}:{network_utils.TCP_PORT}...")
                self.tcp_sock.connect((self.host_ip, network_utils.TCP_PORT))
                print("[TCP] Connected.")
                break
            except Exception as e:
                print(f"[TCP] Failed: {e}. Retrying in 2s...")
                time.sleep(2)

        while True:
            try:
                print(f"[Gaze] Connecting to {self.host_ip}:{network_utils.GAZE_PORT}...")
                self.gaze_sock.connect((self.host_ip, network_utils.GAZE_PORT))
                print("[Gaze] Connected.")
                break
            except Exception as e:
                print(f"[Gaze] Failed: {e}. Retrying in 2s...")
                time.sleep(2)

        threading.Thread(target=self.listen_udp, daemon=True).start()
        threading.Thread(target=self.listen_tcp, daemon=True).start()

        self.run_vision()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _recv_exact(self, sock, n):
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed mid-packet")
            buf += chunk
        return buf

    def _send_click(self, button_id, pressed):
        INPUT_MOUSE = 0
        flags = {
            (1, True):  0x0002, (1, False): 0x0004,
            (2, True):  0x0008, (2, False): 0x0010,
            (3, True):  0x0020, (3, False): 0x0040,
        }
        flag = flags.get((button_id, pressed), 0x0002)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long),
                        ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong),
                        ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong))]

        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [('mi', MOUSEINPUT)]
            _anonymous_ = ('_input',)
            _fields_ = [('type', ctypes.c_ulong), ('_input', _INPUT)]

        inp = INPUT(type=INPUT_MOUSE)
        inp.mi.dwFlags = flag
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_scroll(self, dx, dy):
        WHEEL_DELTA = 120

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [('dx', ctypes.c_long), ('dy', ctypes.c_long),
                        ('mouseData', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong),
                        ('time', ctypes.c_ulong), ('dwExtraInfo', ctypes.POINTER(ctypes.c_ulong))]

        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [('mi', MOUSEINPUT)]
            _anonymous_ = ('_input',)
            _fields_ = [('type', ctypes.c_ulong), ('_input', _INPUT)]

        if dy != 0:
            inp = INPUT(type=0)
            inp.mi.dwFlags = 0x0800
            inp.mi.mouseData = ctypes.c_ulong(int(dy * WHEEL_DELTA))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if dx != 0:
            inp = INPUT(type=0)
            inp.mi.dwFlags = 0x01000
            inp.mi.mouseData = ctypes.c_ulong(int(dx * WHEEL_DELTA))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    # ── Network listeners ─────────────────────────────────────────────────────

    def listen_udp(self):
        print("[UDP] Listening for mouse movement...")
        user32 = ctypes.windll.user32
        while True:
            try:
                data, _ = self.udp_sock.recvfrom(1024)
                if not data:
                    continue
                if struct.unpack('!B', data[:1])[0] == 1:
                    px, py = network_utils.unpack_move(data)
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)
                    px = max(0.0, min(1.0, px))
                    py = max(0.0, min(1.0, py))
                    user32.SetCursorPos(int(px * (sw - 1)), int(py * (sh - 1)))
            except Exception as e:
                print(f"[UDP] Error: {e}")

    def listen_tcp(self):
        print("[TCP] Listening for clicks/scrolls...")
        TOTAL_SIZE = {2: 3, 3: 9, 4: 2}
        while True:
            try:
                type_byte   = self._recv_exact(self.tcp_sock, 1)
                packet_type = struct.unpack('!B', type_byte)[0]
                remaining   = TOTAL_SIZE.get(packet_type, 0) - 1
                rest        = self._recv_exact(self.tcp_sock, remaining) if remaining > 0 else b''
                data        = type_byte + rest

                if packet_type == 2:
                    button_id, pressed = network_utils.unpack_click(data)
                    self._send_click(button_id, pressed)
                elif packet_type == 3:
                    dx, dy = network_utils.unpack_scroll(data)
                    self._send_scroll(dx, dy)
                elif packet_type == 4:
                    if network_utils.unpack_control(data) == 1:
                        print("[Control] Calibration command received from Host")
                        self.should_calibrate = True
                else:
                    print(f"[TCP] Unknown packet type: {packet_type}")
            except ConnectionError as e:
                print(f"[TCP] Connection closed: {e}")
                break
            except Exception as e:
                print(f"[TCP] Error: {e}")
                break

    # ── Vision (headless, background capture) ─────────────────────────────────

    def run_vision(self):
        print(f"[Vision] Opening webcam index {self.camera_index} (background mode)...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Cannot open camera {self.camera_index}.")
            print("  Run 'python check_cameras.py' to list available indices.")
            return

        print(f"[Vision] Camera ready. Processing gaze at {GAZE_FPS} fps ({PROCESS_W}x{PROCESS_H}).")

        # ── Background capture thread — drops stale frames, keeps only latest ──
        latest_frame = [None]
        frame_lock   = threading.Lock()

        def _capture_loop():
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
                    with frame_lock:
                        latest_frame[0] = small

        threading.Thread(target=_capture_loop, daemon=True).start()

        tracker = GazeTracker(headless=True)
        print("[Client] Waiting for calibration signal from Host...")

        last_gaze_conf = -1.0
        last_process   = 0.0
        loop_count     = 0

        try:
            while True:
                now  = time.time()
                wait = GAZE_INTERVAL - (now - last_process)
                if wait > 0:
                    time.sleep(wait)

                # Handle calibration request from host
                if self.should_calibrate:
                    print("[Vision] Starting calibration — look straight at camera...")
                    tracker.is_calibrated      = False
                    tracker.calibration_samples = []
                    self.should_calibrate      = False

                with frame_lock:
                    frame = latest_frame[0]

                if frame is None:
                    time.sleep(0.02)
                    continue

                last_process = time.time()
                loop_count  += 1

                _, confidence, _ = tracker.process_frame(frame)

                # Send gaze update only when confidence changes meaningfully
                if abs(confidence - last_gaze_conf) >= SEND_THRESHOLD:
                    try:
                        self.gaze_sock.sendall(network_utils.pack_gaze(confidence))
                        last_gaze_conf = confidence
                    except Exception as e:
                        print(f"[Gaze] Send error: {e}")

                # Status log every 5 seconds
                if loop_count % (GAZE_FPS * 5) == 0:
                    focused = confidence >= 0.40
                    status  = "FOCUSED" if focused else "AWAY"
                    calib   = "calibrated" if tracker.is_calibrated else "NOT calibrated"
                    print(f"[Client] gaze={confidence:.2f} ({status}) | {calib}")

        except KeyboardInterrupt:
            print("\n[Client] Shutting down.")
        finally:
            cap.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("=" * 55)
        print("  Cat & Mouse — Client (Built-in Webcam)")
        print("=" * 55)
        print("\nUsage:")
        print("  python client_webcam.py <HOST_IP> [CAMERA_INDEX]")
        print("\nExamples:")
        print("  python client_webcam.py 192.168.1.10")
        print("  python client_webcam.py 192.168.1.10 1")
        print("\nRun 'python check_cameras.py' to list available camera indices.")
        print("Run 'python check_host.py' on the HOST to find its IP address.")
        print("\nFor DroidCam / phone camera, use client.py instead.")
        sys.exit(1)

    host_ip      = sys.argv[1]
    camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    client = ClientController(host_ip, camera_index)
    client.start()
