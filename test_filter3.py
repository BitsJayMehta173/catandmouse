import time
from pynput import mouse

def win32_event_filter(msg, data):
    if msg == 0x0201:
        print("Intercepted LDOWN! Sending to client...")
        return False
    elif msg == 0x0202:
        print("Intercepted LUP! Sending to client...")
        return False
    return True

print("Starting listener...")
listener = mouse.Listener(win32_event_filter=win32_event_filter)
listener.start()

m = mouse.Controller()
time.sleep(1)
m.click(mouse.Button.left)
time.sleep(1)
listener.stop()
