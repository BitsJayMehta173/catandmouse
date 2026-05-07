# 🐱 Cat and Mouse — Gaze-Driven Cross-Laptop Mouse Sharing

Suppose you have to work on two laptop simultaneously now you want to work with the single mouse and you want to look in a specific laptop and use the same mouse for both the laptop you can do it with this so what we are doing is looking to the specific laptop and using a single mouse for both, one laptops mouse controls will be shifted to another one.

While I have two laptops and a single wireless or wired mouse and want to use the mouse for both the laptop by just seeing the screen of the specific laptop we will use this for it.

A system that automatically shares your mouse between multiple Windows laptops based on **where you're looking**. Using real-time facial gaze tracking (MediaPipe), the host laptop detects which screen you're facing and seamlessly hands off mouse control — including movement, clicks, and scrolling — to the appropriate client machine over a local Wi-Fi network.

---

## 📁 Project Structure

```
cnm/
├── host.py               # Run on the HOST machine (the one that owns the mouse)
│
├── client_webcam.py      # CLIENT — uses built-in/USB webcam (simpler, recommended)
├── client.py             # CLIENT — uses DroidCam / IP camera URL (phone as webcam)
│
├── gaze_tracker.py       # MediaPipe-based face gaze detection (shared module)
├── network_utils.py      # Packet serialization / port constants (shared module)
├── face_landmarker.task  # MediaPipe Face Landmarker ML model (required)
├── download_model.py     # Script to download face_landmarker.task if missing
├── check_cameras.py      # Diagnostic: list available camera indices
├── check_host.py         # Diagnostic: find your host machine's LAN IP + port check
├── test_connection.py    # Diagnostic: test DroidCam / IP camera URL
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ How It Works

```
HOST (webcam + mouse)          CLIENT (webcam + no mouse)
─────────────────────          ──────────────────────────
GazeTracker reads face    ←→   GazeTracker reads face
Sends confidence score ───────►  Receives confidence from host
                                 Sends its own score ─────────►
Focus Arbiter picks winner ◄─────────────────────────────────
Mouse suppressed on host  ───► Mouse moved/clicked on client
       (UDP/TCP packets)
```

- **UDP port 8080** — mouse movement (high-frequency, low-latency)
- **TCP port 8081** — clicks, scroll events, and control commands
- **TCP port 8082** — gaze confidence scores from client → host

---

## 🛠️ Prerequisites

- **OS:** Windows 10 / 11 (required — uses Win32 APIs for mouse suppression)
- **Python:** 3.9 – 3.11 recommended
- **Webcam:** One per machine (built-in laptop webcam works perfectly)
- **Network:** All machines on the **same local Wi-Fi network**
- **Firewall:** Python must be allowed through Windows Firewall on all machines

---

## 🚀 Setup & Installation

### Step 1 — Clone / copy the project

Copy the entire `cnm/` folder to **every laptop** that will participate.

### Step 2 — Install Python dependencies

Run this on **every machine**:

```powershell
pip install -r requirements.txt
```

> **Note:** `pyzmq` is listed in requirements.txt for optional future use but is not actively used by the current code. All other packages are required.

### Step 3 — Download the ML model

The file `face_landmarker.task` (~3.6 MB) must exist in the project folder. If it's missing:

```powershell
python download_model.py
```

This downloads it from Google's MediaPipe model repository. If it already exists, the script skips it.

### Step 4 — Find the Host machine's IP address

Run this on the **HOST machine**:

```powershell
python check_host.py
```

Look for the line:
```
Primary IP: 192.168.x.x  <-- USE THIS IP ON THE CLIENT!
```

Note this IP — you'll need it in Step 6.

### Step 5 — (Optional) Identify your camera index

If you're unsure which camera index to use (especially with external webcams):

```powershell
python check_cameras.py
```

Default is `0` (built-in webcam). Use the listed index if needed.

---

## ▶️ Running the Application

> **All scripts run in the background — no camera window is shown.**
> Gaze is processed at **8 fps on 640×480 frames** for maximum speed.
> Status is printed to the terminal every 5 seconds.

### Step 1 — On the HOST machine

```powershell
python host.py
```

Or with a specific camera index:

```powershell
python host.py 1
```

The host will:
- Start listening for client connections on all 3 ports
- Run gaze tracking silently in the background
- Print status like `gaze=0.82 (FOCUSED) | owner=host` every 5 seconds
- **Press `ENTER` in the terminal** to start calibration (replaces the old `C` key)

---

### Step 2 — On each CLIENT machine

**Choose ONE of the two client scripts depending on your camera setup:**

---

#### 🖥️ Option A — Built-in / USB Webcam  `client_webcam.py`

> Use this if your client machine has a built-in laptop webcam or a USB webcam.

```powershell
python client_webcam.py <HOST_IP>
```

Example:

```powershell
python client_webcam.py 192.168.1.10
```

With a specific camera index (if index 0 doesn't work):

```powershell
python client_webcam.py 192.168.1.10 1
```

---

#### 📱 Option B — DroidCam / Phone Camera  `client.py`

> Use this if you are using your phone as the webcam via the DroidCam app.

1. Install [DroidCam](https://www.dev47apps.com/) on your Android phone.
2. Connect the phone to the **same Wi-Fi** as your laptops.
3. Open DroidCam — note the URL shown (e.g., `http://192.168.1.9:4747`).
4. Test that the URL works:
   ```powershell
   python test_connection.py http://192.168.1.9:4747
   ```
5. Run the client with the DroidCam URL:
   ```powershell
   python client.py 192.168.1.10 http://192.168.1.9:4747/video
   ```

---

Both client scripts will:
- Connect to the host's TCP and gaze ports
- Run gaze tracking silently in the background (no camera window)
- Wait for calibration signal from host
- Print status like `gaze=0.76 (FOCUSED) | calibrated` every 5 seconds

---

## 🎯 Calibration

Calibration is **coordinated** — the host triggers it for all machines simultaneously.

1. Make sure **all clients are connected** and their camera windows are open.
2. On the **HOST machine**, look straight at the camera and press **`C`**.
3. The host will collect 40 frames to establish a baseline yaw angle.
4. Once the host finishes, it automatically sends a calibration command to all clients.
5. Each client then does its own 40-frame calibration — look straight at your webcam.
6. After calibration, gaze-based focus switching becomes active.

> **Tip:** For best results, keep your head fairly still and look directly at the camera during calibration.

---

## 🖱️ Focus Switching Behavior

| You look at... | What happens |
|----------------|-------------|
| HOST screen | Mouse stays on host; host input is normal |
| CLIENT screen | Mouse is suppressed on host; movement/clicks forwarded to client |

- **Hysteresis margin (0.06):** The challenger must beat the current owner's confidence by 6% to prevent flickering during transitions.
- **Arbiter runs at 10 Hz** — focus switches happen within ~100 ms of a gaze change.
- **Mouse sensitivity:** Default `1.5×`. Adjust `self.sensitivity` in `host.py` line 26.

### Controls

| Key | Action |
|-----|--------|
| `C` | Start calibration (HOST window only) |
| `ESC` | Exit the application |

---

## 🔧 Troubleshooting

### Client can't connect to host

1. Run `python check_host.py` on the host to verify the IP and check port availability.
2. Add a Windows Firewall inbound rule to allow Python on ports **8080, 8081, 8082**.
3. Make sure both machines are on the **same Wi-Fi network** (not guest/isolated networks).

### Camera not opening

```powershell
python check_cameras.py
```

Try different indices (0, 1, 2...) until you find the working one.

### Using phone as webcam (DroidCam)

1. Install [DroidCam](https://www.dev47apps.com/) on your Android phone.
2. Connect phone to the **same Wi-Fi** as your laptops.
3. Note the URL shown in the DroidCam app (e.g., `http://192.168.1.9:4747`).
4. Test the connection:
   ```powershell
   python test_connection.py http://192.168.1.9:4747
   ```
5. Use `client.py` (not `client_webcam.py`) with the confirmed URL:
   ```powershell
   python client.py <HOST_IP> http://192.168.1.9:4747/video
   ```

### Gaze detection not working / always switching

- Re-calibrate by pressing `C` on the host.
- Ensure adequate, consistent lighting on your face.
- Avoid having bright light sources directly behind you.
- Try adjusting `YAW_SIGMA` (line 30) and `PITCH_SIGMA` (line 31) in `gaze_tracker.py` — larger values make the system more lenient.

### Mouse feels choppy on client

- Both machines must be on the same Wi-Fi. A wired LAN connection for the host is ideal.
- Reduce other network load.
- The host sends UDP packets only when the cursor moves — make sure nothing is throttling UDP traffic.

---

## 📡 Network Ports Summary

| Port | Protocol | Purpose |
|------|----------|---------|
| 8080 | UDP | Mouse movement (host → client) |
| 8081 | TCP | Clicks, scroll, calibration commands (host → client) |
| 8082 | TCP | Gaze confidence score (client → host) |

All three ports must be open and unblocked on the **host machine's** firewall.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Camera capture and frame display |
| `mediapipe` | Face landmark detection & gaze estimation |
| `numpy` | Numerical operations for head pose math |
| `pynput` | Low-level mouse listener and suppression (Win32) |
| `pyzmq` | (Installed, not actively used) |

---

## 📝 License

This project is for personal/educational use. MediaPipe models are subject to [Google's MediaPipe Terms](https://developers.google.com/mediapipe).
