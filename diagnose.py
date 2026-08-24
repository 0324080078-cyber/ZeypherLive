"""ZeypherLive — Diagnostic Script (run this first)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  ZeypherLive — System Diagnostics")
print("=" * 60)

# 1. Python
print(f"\n[1] Python: {sys.version}")

# 2. OpenCV
try:
    import cv2
    print(f"[2] OpenCV: {cv2.__version__}")
    print(f"    Backends: DSHOW={cv2.CAP_DSHOW}, MSMF={cv2.CAP_MSMF}")
except Exception as e:
    print(f"[2] OpenCV: FAILED - {e}")

# 3. Camera detection
print("\n[3] Scanning cameras...")
found_any = False
for i in range(5):
    for backend_name, backend_id in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
        try:
            cap = cv2.VideoCapture(i, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend_name_actual = cap.getBackendName()
                if ret and frame is not None:
                    print(f"    Camera {i} [{backend_name}]: {w}x{h} @ {fps:.0f}fps - WORKING (frame={frame.shape})")
                    found_any = True
                else:
                    print(f"    Camera {i} [{backend_name}]: Opened but cannot read frame")
                cap.release()
                break
            else:
                cap.release()
        except Exception as e:
            pass
if not found_any:
    print("    NO CAMERAS FOUND!")

# 4. MediaPipe
try:
    import mediapipe as mp
    print(f"\n[4] MediaPipe: {mp.__version__}")
    from mediapipe.tasks import python as mp_python
    print("    Tasks API: OK")
except Exception as e:
    print(f"\n[4] MediaPipe: FAILED - {e}")

# 5. Model files
print("\n[5] Model files:")
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
for f in ["pose_landmarker_heavy.task", "inswapper_128.onnx", "detection_10g.onnx"]:
    path = os.path.join(models_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"    {f}: {size:.1f} MB")
    else:
        print(f"    {f}: MISSING")

# 6. PyQt5
try:
    from PyQt5.QtWidgets import QApplication
    print("\n[6] PyQt5: OK")
except Exception as e:
    print(f"\n[6] PyQt5: FAILED - {e}")

# 7. aiohttp / websockets
try:
    import aiohttp
    print(f"\n[7] aiohttp: {aiohttp.__version__}")
except Exception as e:
    print(f"\n[7] aiohttp: FAILED - {e}")

try:
    import websockets
    print(f"    websockets: {websockets.__version__}")
except Exception as e:
    print(f"    websockets: FAILED - {e}")

# 8. pyvirtualcam
try:
    import pyvirtualcam
    print(f"\n[8] pyvirtualcam: {pyvirtualcam.__version__}")
except Exception as e:
    print(f"\n[8] pyvirtualcam: FAILED - {e}")

# 9. Test camera capture loop
if found_any:
    print("\n[9] Testing camera capture (2 seconds)...")
    import time
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if cap.isOpened():
        frames = 0
        start = time.time()
        while time.time() - start < 2.0:
            ret, frame = cap.read()
            if ret and frame is not None:
                frames += 1
            else:
                print(f"    Frame read failed at count={frames}")
                break
        elapsed = time.time() - start
        print(f"    Captured {frames} frames in {elapsed:.1f}s = {frames/elapsed:.0f} FPS")
        cap.release()
    else:
        print("    Cannot open camera for test")

print("\n" + "=" * 60)
print("  Diagnostics complete.")
print("=" * 60)
