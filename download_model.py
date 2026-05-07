import os
import urllib.request

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping.")
        return
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    # MediaPipe Face Landmarker model
    FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    download_file(FACE_LANDMARKER_MODEL_URL, "face_landmarker.task")
