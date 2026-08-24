"""ZeypherLive Core — Camera Capture (Final)"""
import cv2
import numpy as np
import threading
import time
import subprocess
from typing import Optional, Callable
from config.settings import CONFIG


class CameraCapture:
    def __init__(self, config=None):
        self.config = config or CONFIG.camera
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.fps_actual = 0.0
        self._callbacks: list[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._last_time = time.time()
        self._device_name = ""
        self._opened_backend = ""
        self._is_ip_stream = False
        self._ip_url = ""

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def _notify_callbacks(self, frame: np.ndarray):
        for cb in self._callbacks:
            try:
                cb(frame)
            except Exception:
                pass

    def open(self, device_id: int = None) -> bool:
        if device_id is None:
            device_id = self.config.device_id
        self.stop()
        backends = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("ANY", cv2.CAP_ANY),
        ]
        for name, backend in backends:
            try:
                cap = cv2.VideoCapture(device_id, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test = cap.read()
                    if ret and test is not None:
                        self.cap = cap
                        self._opened_backend = name
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[Camera] Opened device {device_id} via {name}: {w}x{h}")
                        return True
                    cap.release()
            except Exception as e:
                print(f"[Camera] Backend {name} failed: {e}")
        print(f"[Camera] FAILED to open device {device_id}")
        return False

    def open_url(self, url: str) -> bool:
        self.stop()
        self._is_ip_stream = True
        self._ip_url = url
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    self.cap = cap
                    self._device_name = url
                    self._opened_backend = "IP"
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[Camera] IP stream opened: {w}x{h} from {url[:50]}")
                    return True
                cap.release()
            print(f"[Camera] IP stream failed: {url[:50]}")
            return False
        except Exception as e:
            print(f"[Camera] IP stream error: {e}")
            return False

    def list_ip_sources(self) -> list[dict]:
        sources = []
        for port in [8080, 8888, 4747, 11111, 5000]:
            url = f"http://127.0.0.1:{port}/video"
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
                if cap.isOpened():
                    ret, test = cap.read()
                    if ret:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        sources.append({"url": url, "width": w, "height": h, "source": f"Local IP Cam :{port}"})
                    cap.release()
            except Exception:
                pass
        return sources

    def start(self):
        if self.running:
            return
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                raise RuntimeError("Cannot open camera")
        self.running = True
        self._last_time = time.time()
        self.frame_count = 0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        fail_count = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                fail_count += 1
                if fail_count > 200:
                    self.running = False
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                fail_count += 1
                time.sleep(0.005)
                continue
            fail_count = 0
            with self.lock:
                self.frame = frame.copy()
                self.frame_count += 1
            self._notify_callbacks(frame)
            now = time.time()
            elapsed = now - self._last_time
            if elapsed >= 1.0:
                self.fps_actual = self.frame_count / elapsed
                self.frame_count = 0
                self._last_time = now

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        time.sleep(0.1)

    def _get_dshow_device_names(self) -> dict:
        names = {}
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_PnPEntity | Where-Object {$_.Name -like '*camera*' -or $_.Name -like '*webcam*' -or $_.Name -like '*video*'} | Select-Object Name | ForEach-Object { $_.Name }"],
                capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            for i, line in enumerate(result.stdout.strip().split('\n')):
                if line.strip():
                    names[i] = line.strip()
        except Exception:
            pass
        return names

    def list_devices(self) -> list[dict]:
        devices = []
        tested = set()
        for i in range(10):
            if i in tested:
                continue
            for backend_name, backend_id in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
                try:
                    cap = cv2.VideoCapture(i, backend_id)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            devices.append({
                                "id": i,
                                "name": f"Camera {i}",
                                "width": w,
                                "height": h,
                                "fps": 30,
                                "backend": backend_name,
                            })
                            tested.add(i)
                            cap.release()
                            break
                        cap.release()
                except Exception:
                    pass
        print(f"[Camera] Found {len(devices)} cameras: {[d['id'] for d in devices]}")
        return devices

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
