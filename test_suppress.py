import time, threading
from pynput import mouse

def move_mouse():
    m = mouse.Controller()
    time.sleep(1)
    print("Moving 10, 0")
    m.move(10, 0)
    time.sleep(0.5)
    print("Moving 10, 0")
    m.move(10, 0)
    time.sleep(0.5)
    listener.stop()

def on_move(x, y):
    print(f"MOVE: {x}, {y}")

threading.Thread(target=move_mouse).start()
listener = mouse.Listener(on_move=on_move, suppress=True)
listener.start()
listener.join()
