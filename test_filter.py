import time
from pynput import mouse

def win32_event_filter(msg, data):
    # Suppress clicks, allow moves
    if msg in (0x0201, 0x0202, 0x0204, 0x0205, 0x0207, 0x0208):
        print(f"Blocked click: {msg}")
        return False
    return True

print("Starting listener...")
listener = mouse.Listener(win32_event_filter=win32_event_filter)
listener.start()
time.sleep(3)
listener.stop()
print("Done")
