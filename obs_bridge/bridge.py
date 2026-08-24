"""ZeypherLive — OBS Virtual Camera Bridge"""
import cv2
import numpy as np
import threading
import time
from typing import Optional
from config.settings import CONFIG

try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except ImportError:
    HAS_PYVIRTUALCAM = False


class OBSBridge:
    def __init__(self, config=None):
        self.config = config or CONFIG.obs
        self._cam = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._source = None
        self._lock = threading.Lock()
        self._frames_sent = 0
        self._fps_actual = 0.0
        self._last_time = time.time()
        self._frame_count = 0

    def available(self) -> bool:
        return HAS_PYVIRTUALCAM

    def start(self) -> bool:
        if self.running:
            return True
        if not HAS_PYVIRTUALCAM:
            return False
        try:
            fmt = pyvirtualcam.PixelFormat.BGR
            if self.config.pixel_format.upper() == "RGB":
                fmt = pyvirtualcam.PixelFormat.RGB
            self._cam = pyvirtualcam.Camera(
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
                fmt=fmt,
                device=self.config.virtual_cam_name,
                print_fps=False,
            )
            self.running = True
            self._thread = threading.Thread(target=self._send_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[OBSBridge] Failed to start: {e}")
            return False

    def set_source(self, source):
        self._source = source

    def _send_loop(self):
        frame_interval = 1.0 / self.config.fps
        while self.running:
            start = time.time()
            if self._source is not None:
                frame = self._source.read() if hasattr(self._source, 'read') else None
                if frame is not None:
                    self.send_frame(frame)
            elapsed = time.time() - start
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

    def send_frame(self, frame: np.ndarray) -> bool:
        if not self.running or self._cam is None:
            return False
        try:
            target_w = self.config.width
            target_h = self.config.height
            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
            self._cam.send(frame)
            self._cam.sleep_until_next_frame()
            self._frames_sent += 1
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_time
            if elapsed >= 1.0:
                self._fps_actual = self._frame_count / elapsed
                self._frame_count = 0
                self._last_time = now
            return True
        except Exception:
            return False

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cam:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None

    @property
    def stats(self) -> dict:
        return {
            "available": HAS_PYVIRTUALCAM,
            "running": self.running,
            "frames_sent": self._frames_sent,
            "fps": round(self._fps_actual, 1),
            "device": self.config.virtual_cam_name,
        }

    def __del__(self):
        self.stop()
