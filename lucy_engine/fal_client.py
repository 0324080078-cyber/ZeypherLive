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
        "instantid": {"id": "fal-ai/instant-id", "name": "InstantID"},
        "ip-adapter": {"id": "fal-ai/ip-adapter-face-id", "name": "IP-Adapter Face ID"},
        "face-swap": {"id": "fal-ai/face-swap", "name": "Face Swap"},
        "face-couple": {"id": "fal-ai/face-couple", "name": "Face Couple"},
        "face-morph": {"id": "fal-ai/face-morph", "name": "Face Morph"},
    }

    def __init__(self):
        self.api_key = ""
        self.connected = False
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
        self._busy = False
        self._last_call_time = 0
        self._min_interval = 1.0

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

    def connect(self, api_key: str) -> bool:
        if not api_key:
            self._last_error = "No API key"
            return False
        self.api_key = api_key
        self.connected = True
        self._last_error = ""
        print(f"[FalAI] Connected — model: {self.MODELS[self._model]['name']}")
        return True

    def _extract_image_url(self, result: dict) -> Optional[str]:
        url = None
        if isinstance(result.get("image"), dict):
            url = result["image"].get("url")
        if not url and isinstance(result.get("output"), list) and len(result["output"]) > 0:
            first = result["output"][0]
            if isinstance(first, dict):
                url = first.get("url") or first.get("image_url")
            elif isinstance(first, str) and first.startswith("http"):
                url = first
        if not url and isinstance(result.get("images"), list) and len(result["images"]) > 0:
            first = result["images"][0]
            if isinstance(first, dict):
                url = first.get("url")
            elif isinstance(first, str) and first.startswith("http"):
                url = first
        return url

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not self.connected or not self._reference_image:
            return None
        if self._busy:
            return None
        now = time.time()
        if now - self._last_call_time < self._min_interval:
            return None

        self._busy = True
        self._last_call_time = now
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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

            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                result = json.loads(raw)

            self._frames_sent += 1
            output_url = self._extract_image_url(result)

            if output_url:
                self._last_result_url = output_url
                img_req = urllib.request.Request(output_url)
                with urllib.request.urlopen(img_req, timeout=15) as img_resp:
                    img_data = np.frombuffer(img_resp.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is not None:
                        self._frames_received += 1
                        with self._lock:
                            self._output_frame = img
                        return img
            else:
                print(f"[FalAI] No image in response: {json.dumps(result)[:200]}")

            return None
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:200]
            print(f"[FalAI] HTTP {e.code}: {err}")
            self._last_error = f"HTTP {e.code}"
            return None
        except Exception as e:
            print(f"[FalAI] Process error: {e}")
            self._last_error = str(e)[:100]
            return None
        finally:
            self._busy = False

    def push_frame(self, frame: np.ndarray):
        if not self.connected or self._busy:
            return
        threading.Thread(target=self._process_async, args=(frame,), daemon=True).start()

    def _process_async(self, frame: np.ndarray):
        self.process_frame(frame)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._output_frame.copy() if self._output_frame is not None else None

    def disconnect(self):
        self.connected = False
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
        }
