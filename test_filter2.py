import time
from pynput import mouse

def win32_event_filter(msg, data):
    if msg in (0x0201, 0x0202, 0x0204, 0x0205, 0x0207, 0x0208):
        return False
    return True

def on_click(x, y, button, pressed):
    print(f"Clicked: {button}")

print("Starting listener...")
listener = mouse.Listener(win32_event_filter=win32_event_filter, on_click=on_click)
listener.start()
# simulate a click programmatically to see if it fires
m = mouse.Controller()
time.sleep(1)
m.click(mouse.Button.left)
time.sleep(1)
listener.stop()
