"""ZeypherLive — fal.ai Face Swap Engine (cloud AI, cheaper than Lucy)"""
import cv2
import numpy as np
import base64
import time
import threading
import json
import urllib.request
import urllib.error
from typing import Optional, Callable


class FalFaceSwap:
    MODELS = {
        "instantid": {"id": "fal-ai/instant-id", "name": "InstantID", "cost_per_sec": 0.03},
        "ip-adapter": {"id": "fal-ai/ip-adapter-face-id", "name": "IP-Adapter Face ID", "cost_per_sec": 0.02},
        "face-swap": {"id": "fal-ai/face-swap", "name": "Face Swap", "cost_per_sec": 0.01},
        "face-couple": {"id": "fal-ai/face-couple", "name": "Face Couple", "cost_per_sec": 0.015},
        "face-morph": {"id": "fal-ai/face-morph", "name": "Face Morph", "cost_per_sec": 0.01},
    }

    def __init__(self):
        self.api_key = ""
        self.connected = False
        self._callbacks: list[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._output_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._frames_sent = 0
        self._frames_received = 0
        self._model = "face-swap"
        self._last_error = ""
        self._reference_image: Optional[str] = None
        self._prompt = "Substitute the character in the video with the person in the reference image."
        self._strength = 0.75
        self._last_result_url = ""

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def set_model(self, model: str):
        if model in self.MODELS:
            self._model = model

    def set_reference(self, image_path: str):
        with open(image_path, "rb") as f:
            self._reference_image = base64.b64encode(f.read()).decode()

    def set_reference_from_frame(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._reference_image = base64.b64encode(buf).decode()
        return self._reference_image

    def set_prompt(self, prompt: str):
        self._prompt = prompt

    def connect(self, api_key: str) -> bool:
        if not api_key:
            self._last_error = "No API key"
            return False
        self.api_key = api_key
        try:
            req = urllib.request.Request(
                "https://fal.run/fal-ai/face-swap",
                headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
                data=json.dumps({"image_url": "data:image/jpeg;base64," + (self._reference_image or ""), "prompt": "test"}).encode(),
            )
            self.connected = True
            print(f"[FalAI] Connected with model: {self._model}")
            return True
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            if "Invalid" in err or "401" in str(e.code):
                self._last_error = "Invalid API key"
            elif "402" in str(e.code) or "credit" in err.lower():
                self._last_error = "No credits"
            else:
                self._last_error = f"HTTP {e.code}: {err[:100]}"
            print(f"[FalAI] Connect error: {self._last_error}")
            return False
        except Exception as e:
            self._last_error = str(e)[:100]
            print(f"[FalAI] Connect error: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.connected or not self._reference_image:
            return None
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_b64 = base64.b64encode(buf).decode()

            model_info = self.MODELS[self._model]
            payload = {
                "image_url": "data:image/jpeg;base64," + frame_b64,
                "reference_image_url": "data:image/jpeg;base64," + self._reference_image,
                "prompt": self._prompt,
                "strength": self._strength,
            }

            req = urllib.request.Request(
                f"https://fal.run/{model_info['id']}",
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload).encode(),
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())

            self._frames_sent += 1
            output_url = result.get("image", {}).get("url") or result.get("output", [None])[0] if isinstance(result.get("output"), list) else None

            if output_url:
                self._last_result_url = output_url
                img_req = urllib.request.Request(output_url)
                with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                    img_data = np.frombuffer(img_resp.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        self._frames_received += 1
                        with self._lock:
                            self._output_frame = img
                        return img

            return None
        except Exception as e:
            print(f"[FalAI] Process error: {e}")
            return None

    def push_frame(self, frame: np.ndarray):
        if not self.connected:
            return
        threading.Thread(target=self._process_async, args=(frame,), daemon=True).start()

    def _process_async(self, frame: np.ndarray):
        result = self.process_frame(frame)
        if result is not None:
            for cb in self._callbacks:
                try:
                    cb(result)
                except Exception:
                    pass

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._output_frame.copy() if self._output_frame is not None else None

    def disconnect(self):
        self.connected = False
        self._stop.set()
        print("[FalAI] Disconnected")

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def stats(self) -> dict:
        return {
            "connected": self.connected,
            "model": self._model,
            "model_name": self.MODELS.get(self._model, {}).get("name", ""),
            "frames_sent": self._frames_sent,
            "frames_received": self._frames_received,
            "cost_per_sec": self.MODELS.get(self._model, {}).get("cost_per_sec", 0),
        }
