"""
native_mouse.py — High-performance mouse interpolator using Win32 APIs directly.

Uses:
  - timeBeginPeriod(1) for 1ms timer resolution (default Windows is 15.6ms!)
  - Dedicated high-priority thread for cursor updates
  - Exponential smoothing (EMA) for natural-feeling movement
  - Velocity prediction for zero-latency feel during fast swipes

This module bypasses Python's time.sleep() inaccuracy by using
Windows multimedia timers for sub-millisecond precision.
"""

import ctypes
import ctypes.wintypes
import threading
import time

# ── Win32 API setup ──────────────────────────────────────────────────────────
user32   = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Enable high-resolution timers (1ms instead of 15.6ms default)
try:
    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
except Exception:
    pass

# Pre-cache screen dimensions (avoid repeated syscalls)
_screen_w = user32.GetSystemMetrics(0)
_screen_h = user32.GetSystemMetrics(1)


class NativeMouseInterpolator:
    """
    High-frequency mouse interpolation engine.

    Architecture:
      - UDP listener sets target_x/target_y (normalised 0.0–1.0)
      - This engine runs at ~240 Hz in a tight native loop
      - Uses exponential moving average (EMA) with velocity prediction
      - Calls SetCursorPos directly via ctypes (no pynput overhead)

    The result is a cursor that feels like a local mouse, not a remote one.
    """

    def __init__(self, update_hz=240, smoothing=0.45, velocity_weight=0.15):
        """
        Args:
            update_hz:       How often to update the cursor (Hz). 240 is ideal.
            smoothing:       EMA factor (0.0=frozen, 1.0=instant). 0.45 is natural.
            velocity_weight: How much to predict ahead using velocity (0.0–0.5).
        """
        self.update_hz       = update_hz
        self.interval        = 1.0 / update_hz
        self.smoothing       = smoothing
        self.velocity_weight = velocity_weight

        # Target position (set by UDP listener)
        self.target_x = 0.5
        self.target_y = 0.5
        self._lock    = threading.Lock()

        # Internal state
        self._curr_x  = 0.5
        self._curr_y  = 0.5
        self._vel_x   = 0.0   # velocity estimate (normalised units / tick)
        self._vel_y   = 0.0
        self._prev_tx = 0.5   # previous target (for velocity estimation)
        self._prev_ty = 0.5

        # Screen dimensions (cached)
        self._sw = _screen_w
        self._sh = _screen_h

        # Last pixel position sent to SetCursorPos (avoid redundant calls)
        self._last_px = -1
        self._last_py = -1

        self._running = False
        self._thread  = None

    def update_target(self, x_norm, y_norm):
        """Called by UDP listener. Thread-safe."""
        with self._lock:
            self.target_x = max(0.0, min(1.0, x_norm))
            self.target_y = max(0.0, min(1.0, y_norm))

    def start(self):
        """Start the interpolation thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the interpolation thread."""
        self._running = False

    def _run_loop(self):
        """
        Main interpolation loop. Runs at update_hz using high-res sleep.
        Uses EMA + velocity prediction for natural cursor movement.
        """
        sm  = self.smoothing
        vw  = self.velocity_weight
        sw1 = self._sw - 1
        sh1 = self._sh - 1

        # Raise thread priority for consistent timing
        try:
            handle = kernel32.GetCurrentThread()
            kernel32.SetThreadPriority(handle, 2)  # THREAD_PRIORITY_HIGHEST
        except Exception:
            pass

        while self._running:
            t0 = time.perf_counter()

            # Read target
            with self._lock:
                tx, ty = self.target_x, self.target_y

            # Estimate velocity from target changes
            self._vel_x = 0.8 * self._vel_x + 0.2 * (tx - self._prev_tx)
            self._vel_y = 0.8 * self._vel_y + 0.2 * (ty - self._prev_ty)
            self._prev_tx = tx
            self._prev_ty = ty

            # Predicted target = target + velocity * weight
            pred_x = tx + self._vel_x * vw * self.update_hz
            pred_y = ty + self._vel_y * vw * self.update_hz

            # Clamp prediction
            pred_x = max(0.0, min(1.0, pred_x))
            pred_y = max(0.0, min(1.0, pred_y))

            # EMA smoothing toward predicted target
            self._curr_x += (pred_x - self._curr_x) * sm
            self._curr_y += (pred_y - self._curr_y) * sm

            # Convert to pixels
            px = int(self._curr_x * sw1)
            py = int(self._curr_y * sh1)

            # Only call SetCursorPos if position actually changed
            if px != self._last_px or py != self._last_py:
                user32.SetCursorPos(px, py)
                self._last_px = px
                self._last_py = py

            # High-precision sleep
            elapsed = time.perf_counter() - t0
            remaining = self.interval - elapsed
            if remaining > 0.001:
                time.sleep(remaining - 0.0005)  # Sleep most of it
                # Spin-wait the last 0.5ms for precision
                while time.perf_counter() - t0 < self.interval:
                    pass
            elif remaining > 0:
                while time.perf_counter() - t0 < self.interval:
                    pass
