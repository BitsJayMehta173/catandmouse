import cv2
import sys
import time

def test_url(url, backend_name, backend_const):
    print(f"Testing {url} with {backend_name}...")
    try:
        if backend_const is not None:
            cap = cv2.VideoCapture(url, backend_const)
        else:
            cap = cv2.VideoCapture(url)
            
        if not cap.isOpened():
            print(f"  [X] Failed to open.")
            return False
        
        # Try to read a few frames
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"  [OK] Successfully read frame {i+1}!")
                cap.release()
                return True
            time.sleep(0.1)
            
        print("  [!] Opened but could not read frames.")
        cap.release()
        return False
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_connection.py <IP_BASE_URL>")
        print("Example: python test_connection.py http://192.168.1.9:4747")
        sys.exit(1)
        
    base_url = sys.argv[1].rstrip('/')
    
    urls = [
        f"{base_url}/video",
        f"{base_url}/mjpegfeed",
        f"{base_url}/video.force?1280x720", # DroidCam specific force resolution
        base_url
    ]
    
    backends = [
        ("Default", None),
        ("FFMPEG", cv2.CAP_FFMPEG),
    ]
    
    found_any = False
    for url in urls:
        for b_name, b_const in backends:
            if test_url(url, b_name, b_const):
                print(f"\n>>> SUCCESS! Use this URL: {url}")
                if b_const == cv2.CAP_FFMPEG:
                    print(">>> And make sure to use cv2.CAP_FFMPEG")
                found_any = True
                break
        if found_any: break
        
    if not found_any:
        print("\nAll attempts failed. Please check if DroidCam is running and on the same WiFi.")
        print("Also ensure you don't have a firewall blocking Python.")
