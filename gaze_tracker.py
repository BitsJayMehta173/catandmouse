import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# MediaPipe curved mesh connections for visualization
try:
    _TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION
    _CONTOURS    = mp.solutions.face_mesh.FACEMESH_CONTOURS
except Exception:
    _TESSELATION = set()
    _CONTOURS    = set()

# solvePnP landmark indices: Nose, Chin, L-Eye, R-Eye, L-Mouth, R-Mouth
POSE_IDX = [1, 152, 33, 263, 61, 291]

FACE_3D_MODEL = np.array([
    (  0.0,    0.0,    0.0),
    (  0.0, -330.0,  -65.0),
    (-225.0,  170.0, -135.0),
    ( 225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    ( 150.0, -150.0, -125.0),
], dtype=np.float64)

# Tighter sigma so side-by-side cameras (small angle diff) register a clear winner
YAW_SIGMA   = 6.0
PITCH_SIGMA = 10.0

CALIB_FRAMES      = 40
LOOKING_THRESHOLD = 0.50

# Asymmetry feature landmarks
NOSE_IDX    = 1
L_EYE_IDX   = 33
R_EYE_IDX   = 263
L_CHEEK_IDX = 234   # approx left boundary of face
R_CHEEK_IDX = 454   # approx right boundary of face

# ── Motion Cache constants ────────────────────────────────────────────────────
DIFF_W, DIFF_H = 160, 120        # Downscale for diff check (4x smaller than 640x480)
MOTION_THRESH  = 0.4              # Mean pixel diff below which we skip detection
STILL_FRAMES_MAX = 30             # After this many skipped frames, force a re-check
# ──────────────────────────────────────────────────────────────────────────────

# Pre-allocate reusable buffers to avoid GC pressure
_face2d_buf = np.zeros((6, 2), dtype=np.float64)
_dist_buf   = np.zeros((4, 1), dtype=np.float64)


class GazeTracker:
    """
    Produces a 0-1 confidence score for 'how directly is the face pointing at
    this camera'. Uses TWO independent features:

      1. Yaw angle (solvePnP) compared to calibrated baseline — very precise
         for small angle differences when cameras are side by side.

      2. Nose-cheek asymmetry — calibration-free geometric feature. When the
         face turns toward a camera, the nose moves toward the center of the
         visible face. Compares nose-to-left-cheek vs nose-to-right-cheek.

    Final score = 0.6 * yaw_gaussian + 0.4 * symmetry_score.
    Both features agree when the face is clearly pointed one way, giving a
    decisive winner even for small inter-camera angles.
    """

    def __init__(self, headless=False):
        """
        headless=True: skips all cv2 drawing operations (mesh, overlays, labels).
        Use this for background / no-window mode for maximum speed.
        """
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"'{model_path}' not found. Run download_model.py first.")

        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(opts)

        self.baseline_yaw   = None
        self.is_calibrated  = False
        self.calibration_samples: list = []
        self.headless       = headless

        # ── Motion Cache (optimised) ─────────────────────────────────────────
        self._prev_gray_small = None       # Downscaled grayscale for diff
        self._last_results    = (None, 0.0, (0.0, 0.0, 0.0))
        self._still_count     = 0          # Consecutive "still" frames
        # ─────────────────────────────────────────────────────────────────────

        # Pre-allocated camera matrix (updated per-frame if resolution changes)
        self._cam_matrix = None
        self._last_w     = 0
        self._last_h     = 0

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray):
        """
        Returns (annotated_frame_or_None, confidence, angles).
        Includes an optimised motion cache: if frame delta is tiny, skips
        the expensive MediaPipe landmarker and returns cached results.
        """
        img_h, img_w = frame.shape[:2]

        # ── Motion cache check (very fast: operates on 160x120 grayscale) ────
        if self.is_calibrated:
            small = cv2.resize(frame, (DIFF_W, DIFF_H))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if self._prev_gray_small is not None:
                diff = cv2.absdiff(gray_small, self._prev_gray_small)
                motion_score = diff.mean()   # numpy mean is faster than cv2.mean for small arrays

                if motion_score < MOTION_THRESH and self._still_count < STILL_FRAMES_MAX:
                    self._still_count += 1
                    return self._last_results
                else:
                    self._still_count = 0

            self._prev_gray_small = gray_small

        # ── MediaPipe inference ──────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        out        = None if self.headless else frame.copy()
        confidence = 0.0
        angles     = (0.0, 0.0, 0.0)

        if not result.face_landmarks:
            if not self.headless:
                cv2.putText(out, "No face", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            self._last_results = (out, 0.0, angles)
            return self._last_results

        landmarks = result.face_landmarks[0]

        angles, nose_px = self._head_pose(landmarks, img_w, img_h)
        pitch, yaw, roll = angles

        # ---------- symmetry score (always available, no calibration) ----------
        sym_score = self._nose_symmetry_score(landmarks, img_w, img_h)

        # ---------- calibration ----------
        if not self.is_calibrated:
            self.calibration_samples.append(yaw)
            if not self.headless:
                pct = int(len(self.calibration_samples) / CALIB_FRAMES * 100)
                cv2.putText(out, f"Calibrating {pct}% – look straight at camera",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                self._draw_mesh(out, landmarks, img_w, img_h, confidence=0.0)
            if len(self.calibration_samples) >= CALIB_FRAMES:
                self.baseline_yaw = float(np.median(self.calibration_samples))
                self.is_calibrated = True
                print(f"[GazeTracker] Calibrated. Baseline yaw: {self.baseline_yaw:.1f}°")
            return out, sym_score * 0.4, angles   # partial score from symmetry only

        # ---------- yaw Gaussian score ----------
        d_yaw     = yaw - self.baseline_yaw
        yaw_score = float(np.exp(-0.5 * (d_yaw / YAW_SIGMA) ** 2))

        # ---------- combined confidence ----------
        confidence = 0.6 * yaw_score + 0.4 * sym_score

        # ---------- overlays (skipped in headless mode) ----------
        if not self.headless:
            self._draw_mesh(out, landmarks, img_w, img_h, confidence)
            self._draw_arrow(out, nose_px, yaw, pitch)
            self._draw_bar(out, confidence, img_h)
            self._draw_labels(out, confidence, yaw, d_yaw, sym_score)

        self._last_results = (out, confidence, angles)
        return self._last_results

    # ------------------------------------------------------------------
    # Feature: nose–cheek lateral symmetry
    # ------------------------------------------------------------------
    def _nose_symmetry_score(self, landmarks, img_w, img_h):
        """
        Calibration-free asymmetry metric.

        Measures the ratio of:
          dist(nose_tip → left cheek boundary)
          dist(nose_tip → right cheek boundary)

        When the face looks straight at this camera the ratio is ~1.
        When turned away the near side shrinks and the far side grows.
        Score = 1 when ratio == 1, drops toward 0 as asymmetry increases.
        """
        def px(idx):
            lm = landmarks[idx]
            return np.array([lm.x * img_w, lm.y * img_h])

        nose   = px(NOSE_IDX)
        l_chk  = px(L_CHEEK_IDX)
        r_chk  = px(R_CHEEK_IDX)

        d_left  = np.linalg.norm(nose - l_chk)
        d_right = np.linalg.norm(nose - r_chk)

        if d_left + d_right == 0:
            return 0.5

        ratio = d_left / (d_right + 1e-6)   # > 1 face turned right, < 1 turned left
        # Score peaks at ratio=1 (symmetric), falls off on either side
        # Using a Gaussian around log(ratio) = 0
        log_ratio = np.log(ratio + 1e-6)
        score = float(np.exp(-0.5 * (log_ratio / 0.35) ** 2))  # sigma≈0.35 in log space
        return score

    # ------------------------------------------------------------------
    # Head pose via solvePnP (optimised with pre-allocated buffers)
    # ------------------------------------------------------------------
    def _head_pose(self, landmarks, img_w, img_h):
        # Fill pre-allocated buffer
        for i, idx in enumerate(POSE_IDX):
            _face2d_buf[i, 0] = landmarks[idx].x * img_w
            _face2d_buf[i, 1] = landmarks[idx].y * img_h

        # Re-use camera matrix if resolution hasn't changed
        if img_w != self._last_w or img_h != self._last_h:
            focal = img_w
            self._cam_matrix = np.array(
                [[focal, 0, img_w / 2],
                 [0, focal, img_h / 2],
                 [0,     0,        1]], dtype=np.float64)
            self._last_w = img_w
            self._last_h = img_h

        ok, rvec, _ = cv2.solvePnP(FACE_3D_MODEL, _face2d_buf, self._cam_matrix, _dist_buf)
        rmat, _     = cv2.Rodrigues(rvec)
        euler, *_   = cv2.RQDecomp3x3(rmat)

        pitch = euler[0] * 360
        yaw   = euler[1] * 360
        roll  = euler[2] * 360
        return (pitch, yaw, roll), _face2d_buf[0]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _draw_mesh(self, img, landmarks, img_w, img_h, confidence: float):
        """Draw the actual curved face mesh (tessellation + contours)."""
        def lm_px(idx):
            lm = landmarks[idx]
            return (int(lm.x * img_w), int(lm.y * img_h))

        # Tessellation — fine inner mesh, faint
        if _TESSELATION:
            alpha = 0.3 + 0.5 * confidence  # brighter when more confident
            overlay = img.copy()
            g = int(80 + 160 * confidence)
            r = int(80 * (1 - confidence))
            for conn in _TESSELATION:
                p1 = lm_px(conn[0])
                p2 = lm_px(conn[1])
                cv2.line(overlay, p1, p2, (r, g, 40), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha * 0.4, img, 1 - alpha * 0.4, 0, img)

        # Contours — bold outer boundary
        if _CONTOURS:
            g2 = int(120 + 130 * confidence)
            r2 = int(120 * (1 - confidence))
            for conn in _CONTOURS:
                p1 = lm_px(conn[0])
                p2 = lm_px(conn[1])
                cv2.line(img, p1, p2, (r2, g2, 60), 1, cv2.LINE_AA)

    def _draw_arrow(self, img, nose_px, yaw, pitch):
        L  = 90
        dx = int(L * np.sin(np.radians(yaw)))
        dy = int(-L * np.sin(np.radians(pitch)))
        tip = (int(nose_px[0]) + dx, int(nose_px[1]) + dy)
        cv2.arrowedLine(img, (int(nose_px[0]), int(nose_px[1])), tip,
                        (0, 255, 255), 2, tipLength=0.25)

    def _draw_bar(self, img, confidence, img_h):
        bx  = img.shape[1] - 28
        bt, bb = 20, img_h - 20
        bh  = bb - bt
        fh  = int(bh * confidence)
        cv2.rectangle(img, (bx, bt), (bx + 16, bb), (40, 40, 40), -1)
        if fh > 0:
            cv2.rectangle(img, (bx, bb - fh), (bx + 16, bb),
                          (0, int(255*confidence), int(200*(1-confidence))), -1)
        cv2.rectangle(img, (bx, bt), (bx + 16, bb), (140, 140, 140), 1)

    def _draw_labels(self, img, confidence, yaw, d_yaw, sym_score):
        focused = confidence >= LOOKING_THRESHOLD
        color   = (0, 255, 0) if focused else (0, 80, 255)
        status  = "FOCUSED" if focused else "away"
        cv2.putText(img, status, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(img, f"conf:{confidence:.2f}  yaw:{yaw:+.1f}(d{d_yaw:+.1f})  sym:{sym_score:.2f}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    src = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    tracker = GazeTracker()
    print("Look straight at camera to calibrate.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        out, conf, _ = tracker.process_frame(frame)
        cv2.imshow("GazeTracker", out)
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()
