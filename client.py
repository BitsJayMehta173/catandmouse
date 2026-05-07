import socket
import threading
import time
import cv2
import struct
import ctypes
from pynput import mouse
import network_utils
from gaze_tracker import GazeTracker

# Make all coordinate operations consistent with physical pixels,
# regardless of Windows DPI scaling (100%, 125%, 150%, etc.)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

class ClientController:
    def __init__(self, host_ip, camera_source=0):
        self.host_ip = host_ip
        self.camera_source = camera_source
        self.mouse_controller = mouse.Controller()
        self.should_calibrate = False
        self.tracker = None
        
        # Sockets
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind(('0.0.0.0', network_utils.UDP_PORT))
        
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.gaze_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        print(f"Connecting to Host {self.host_ip}...")
        
        # Connect TCP for clicks
        while True:
            try:
                print(f"Attempting TCP connection to {self.host_ip}:{network_utils.TCP_PORT}...")
                self.tcp_sock.connect((self.host_ip, network_utils.TCP_PORT))
                print("[TCP] Connected for clicks.")
                break
            except Exception as e:
                print(f"[TCP] Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

        # Connect TCP for gaze
        while True:
            try:
                print(f"Attempting Gaze connection to {self.host_ip}:{network_utils.GAZE_PORT}...")
                self.gaze_sock.connect((self.host_ip, network_utils.GAZE_PORT))
                print("[Gaze] Connected to send gaze data.")
                break
            except Exception as e:
                print(f"[Gaze] Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

        # Start listening threads
        threading.Thread(target=self.listen_udp, daemon=True).start()
        threading.Thread(target=self.listen_tcp, daemon=True).start()
        
        # Start Vision processing in main thread
        self.run_vision()

    def _recv_exact(self, sock, n):
        """Read exactly n bytes from sock, blocking until all arrive.
        Returns bytes or raises ConnectionError if socket closes early.
        """
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed mid-packet")
            buf += chunk
        return buf

    def _send_click(self, button_id, pressed):
        """Inject a click via SendInput so it goes through the same Win32
        pathway as SetCursorPos (avoids pynput/SendInput level mismatch)."""
        INPUT_MOUSE = 0
        MOUSEEVENTF_LEFTDOWN   = 0x0002
        MOUSEEVENTF_LEFTUP     = 0x0004
        MOUSEEVENTF_RIGHTDOWN  = 0x0008
        MOUSEEVENTF_RIGHTUP    = 0x0010
        MOUSEEVENTF_MIDDLEDOWN = 0x0020
        MOUSEEVENTF_MIDDLEUP   = 0x0040

        if button_id == 1:
            flag = MOUSEEVENTF_LEFTDOWN if pressed else MOUSEEVENTF_LEFTUP
        elif button_id == 2:
            flag = MOUSEEVENTF_RIGHTDOWN if pressed else MOUSEEVENTF_RIGHTUP
        else:
            flag = MOUSEEVENTF_MIDDLEDOWN if pressed else MOUSEEVENTF_MIDDLEUP

        # MOUSEINPUT structure: dx, dy, mouseData, dwFlags, time, dwExtraInfo
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
        """Inject a scroll wheel event via SendInput."""
        MOUSEEVENTF_WHEEL  = 0x0800
        MOUSEEVENTF_HWHEEL = 0x01000
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
            inp.mi.dwFlags = MOUSEEVENTF_WHEEL
            inp.mi.mouseData = ctypes.c_ulong(int(dy * WHEEL_DELTA))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

        if dx != 0:
            inp = INPUT(type=0)
            inp.mi.dwFlags = MOUSEEVENTF_HWHEEL
            inp.mi.mouseData = ctypes.c_ulong(int(dx * WHEEL_DELTA))
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def listen_udp(self):
        print("[UDP] Listening for mouse movement...")
        user32 = ctypes.windll.user32
        while True:
            try:
                data, addr = self.udp_sock.recvfrom(1024)
                if not data:
                    continue
                packet_type = struct.unpack('!B', data[:1])[0]
                if packet_type == 1:
                    px, py = network_utils.unpack_move(data)

                    # Use GetSystemMetrics in physical-pixel mode (guaranteed by
                    # SetProcessDPIAware above) to map the 0–1 fraction to screen coords.
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)

                    # Clamp to valid range before converting
                    px = max(0.0, min(1.0, px))
                    py = max(0.0, min(1.0, py))

                    target_x = int(px * (sw - 1))
                    target_y = int(py * (sh - 1))

                    user32.SetCursorPos(target_x, target_y)
            except Exception as e:
                print(f"[UDP] Error: {e}")

    def listen_tcp(self):
        print("[TCP] Listening for clicks/scrolls...")
        # Packet sizes by type byte:
        #   type 2 (click):   3 bytes total
        #   type 3 (scroll):  9 bytes total
        #   type 4 (control): 2 bytes total
        TOTAL_SIZE = {2: 3, 3: 9, 4: 2}

        while True:
            try:
                # Read the type byte first (always 1 byte)
                type_byte = self._recv_exact(self.tcp_sock, 1)
                packet_type = struct.unpack('!B', type_byte)[0]

                remaining = TOTAL_SIZE.get(packet_type, 0) - 1
                rest = self._recv_exact(self.tcp_sock, remaining) if remaining > 0 else b''
                data = type_byte + rest

                if packet_type == 2:
                    button_id, pressed = network_utils.unpack_click(data)
                    self._send_click(button_id, pressed)
                elif packet_type == 3:
                    dx, dy = network_utils.unpack_scroll(data)
                    self._send_scroll(dx, dy)
                elif packet_type == 4:
                    cmd_id = network_utils.unpack_control(data)
                    if cmd_id == 1:
                        print("[Control] Received calibration command from Host")
                        self.should_calibrate = True
                else:
                    print(f"[TCP] Unknown packet type: {packet_type}, skipping")
            except ConnectionError as e:
                print(f"[TCP] Connection closed: {e}")
                break
            except Exception as e:
                print(f"[TCP] Error: {e}")
                break

    def run_vision(self):
        print(f"[Vision] Initializing camera source: {self.camera_source}...")
        
        source = self.camera_source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        cap = None
        if isinstance(source, int):
            print(f"[Vision] Using local camera index {source} with DSHOW")
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            # It's a URL. Try FFMPEG first as it's usually better for network streams
            print(f"[Vision] Attempting to open URL {source} with FFMPEG...")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                print(f"[Vision] FFMPEG failed, trying default backend...")
                cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            print(f"[Vision] Error: Could not open camera source: {source}")
            print("[Vision] Troubleshooting tips:")
            print("1. Make sure DroidCam is running on your phone.")
            print(f"2. Try running: python test_connection.py {source.rsplit('/', 1)[0] if isinstance(source, str) else ''}")
            return

        self.tracker = GazeTracker()
        
        print("Waiting for Host to start calibration (Press 'C' on Host)...")
        
        last_gaze_conf     = -1.0  # last sent confidence value
        SEND_THRESHOLD      = 0.03  # only send when confidence changes by this much
        consecutive_failures = 0
        
        while cap.isOpened():
            if self.should_calibrate:
                print("[Vision] Starting calibration...")
                self.tracker.is_calibrated = False
                self.tracker.calibration_samples = []
                self.should_calibrate = False

            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print("[Vision] Error: Consecutive frame grab failures. Camera might be disconnected.")
                    break
                continue
            
            consecutive_failures = 0

            annotated_image, confidence, angles = self.tracker.process_frame(frame)

            if not self.tracker.is_calibrated:
                cv2.putText(annotated_image, "CALIBRATING CLIENT...", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Send confidence to host only when it changes meaningfully
            if abs(confidence - last_gaze_conf) >= SEND_THRESHOLD:
                try:
                    self.gaze_sock.sendall(network_utils.pack_gaze(confidence))
                    last_gaze_conf = confidence
                except Exception as e:
                    print(f"Failed to send gaze confidence: {e}")
            
            try:
                cv2.imshow('Client Gaze Tracker', annotated_image)
            except Exception as e:
                print(f"[Vision] Warning: Could not display window: {e}")
                
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <HOST_IP> [CAMERA_SOURCE]")
        print("Examples:")
        print("  python client.py 192.168.1.10")
        print("  python client.py 192.168.1.10 1")
        print("  python client.py 192.168.1.10 http://192.168.1.9:4747/video")
        sys.exit(1)
        
    host_ip = sys.argv[1]
    camera_source = sys.argv[2] if len(sys.argv) > 2 else 0
    
    client = ClientController(host_ip, camera_source)
    client.start()
