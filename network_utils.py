import struct

# Ports
UDP_PORT = 8080    # For mouse movement
TCP_PORT = 8081    # For clicks, keys, and control
GAZE_PORT = 8082   # For client sending gaze status to host

# Packet Structures
# We use struct to pack data efficiently

# Movement Packet: type(1 byte), x(float), y(float)
# type 1 = move
MOVE_FMT = '!Bff'

# Click Packet: type(1 byte), button(1 byte), pressed(1 byte)
# type 2 = click
CLICK_FMT = '!BBB'
# Buttons: 1=Left, 2=Right, 3=Middle

# Scroll Packet: type(1 byte), dx(int), dy(int)
# type 3 = scroll
SCROLL_FMT = '!Bii'

# Gaze Packet: confidence score (float 0.0-1.0)
GAZE_FMT = '!f'

# Control Packet: type(1 byte), cmd_id(1 byte)
# type 4 = control
# cmd_id: 1 = Start Calibration
CONTROL_FMT = '!BB'

def pack_move(x, y):
    return struct.pack(MOVE_FMT, 1, float(x), float(y))

def unpack_move(data):
    _, x, y = struct.unpack(MOVE_FMT, data)
    return x, y

def pack_click(button_id, pressed):
    return struct.pack(CLICK_FMT, 2, button_id, 1 if pressed else 0)

def unpack_click(data):
    _, button_id, pressed = struct.unpack(CLICK_FMT, data)
    return button_id, bool(pressed)

def pack_scroll(dx, dy):
    return struct.pack(SCROLL_FMT, 3, int(dx), int(dy))

def unpack_scroll(data):
    _, dx, dy = struct.unpack(SCROLL_FMT, data)
    return dx, dy

def pack_gaze(confidence: float) -> bytes:
    return struct.pack(GAZE_FMT, float(confidence))

def unpack_gaze(data: bytes) -> float:
    return struct.unpack(GAZE_FMT, data)[0]

def pack_control(cmd_id):
    return struct.pack(CONTROL_FMT, 4, int(cmd_id))

def unpack_control(data):
    _, cmd_id = struct.unpack(CONTROL_FMT, data)
    return cmd_id
