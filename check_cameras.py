import cv2

def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

print("Checking for available cameras...")
cameras = list_cameras()
if not cameras:
    print("No cameras found!")
else:
    print(f"Available camera indices: {cameras}")
    print("If index 0 is not working, you can try changing VideoCapture(0) to another index in client.py")
