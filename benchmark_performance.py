import time
import numpy as np
import cv2
import os
from gaze_tracker import GazeTracker

def benchmark_gaze_tracker():
    print("Starting GazeTracker Performance Benchmark...")
    
    # 1. Setup
    tracker = GazeTracker(headless=True)
    tracker.is_calibrated = True  # Force calibration for motion cache test
    # Generate a dummy frame (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add a white square to simulate some visual features
    cv2.rectangle(frame, (200, 200), (440, 280), (255, 255, 255), -1)
    
    # 2. Warmup
    print("Warming up...")
    for _ in range(20):
        tracker.process_frame(frame)
    
    # 3. Test Raw Inference Speed (Motion Cache MISS)
    print("Measuring Raw Inference Latency (Motion Cache MISS)...")
    latencies = []
    for i in range(50):
        # Slightly modify frame to force a cache miss
        test_frame = frame.copy()
        test_frame[0, 0] = i % 255 
        
        t0 = time.perf_counter()
        tracker.process_frame(test_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    
    avg_latency = np.mean(latencies)
    print(f"   Avg Latency: {avg_latency:.2f} ms")

    # 4. Test Motion Cache Efficiency (Motion Cache HIT)
    print("\nMeasuring Motion Cache Efficiency (Motion Cache HIT)...")
    cache_latencies = []
    # Use the EXACT same frame to trigger hit
    still_frame = frame.copy()
    tracker.process_frame(still_frame) # Seed the cache
    
    for _ in range(100):
        t0 = time.perf_counter()
        tracker.process_frame(still_frame)
        t1 = time.perf_counter()
        cache_latencies.append((t1 - t0) * 1000)
    
    avg_cache_latency = np.mean(cache_latencies)
    print(f"   Avg Cache Latency: {avg_cache_latency:.4f} ms")

    # 5. Native Timer Check
    print("\nChecking System Timer Resolution...")
    import ctypes
    t_samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        time.sleep(0.001)
        t1 = time.perf_counter()
        t_samples.append((t1 - t0) * 1000)
    
    avg_sleep = np.mean(t_samples)
    print(f"   sleep(1ms) actually took: {avg_sleep:.2f} ms")
    if avg_sleep < 2.0:
        print("   High-resolution timers are ENABLED.")
    else:
        print("   High-resolution timers are DISABLED (Windows default).")

    print("\n" + "="*40)
    print("BENCHMARK SUMMARY")
    print("="*40)
    print(f"Raw Inference:  {avg_latency:.1f}ms")
    print(f"Cached Hit:     {avg_cache_latency:.3f}ms")
    print(f"Timer Res:      {avg_sleep:.2f}ms")
    print("="*40)

if __name__ == "__main__":
    benchmark_gaze_tracker()
