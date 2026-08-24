"""ZeypherLive — Lucy 2.5 Client (Official SDK, WebRTC via LiveKit)"""
import asyncio
import cv2
import numpy as np
import time
import threading
import os
from typing import Optional, Callable
from config.settings import CONFIG


class LucyEngine:
    def __init__(self, config=None):
        self.config = config or CONFIG.lucy
        self.connected = False
        self._callbacks: list[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._output_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._last_latency = 0.0
        self._frames_sent = 0
        self._frames_received = 0
        self._stop = threading.Event()
        self._source = None
        self._video_track = None
        self._last_error = ""

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        print("[Lucy] Thread started")

    def _run_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_run())
        except Exception as e:
            print(f"[Lucy] Thread error: {e}")
        finally:
            if self._loop and self._loop.is_running():
                self._loop.close()
            self._loop = None

    async def _connect_and_run(self):
        try:
            from decart.realtime.client import RealtimeClient
            from decart.realtime.types import RealtimeConnectOptions
            from decart import models, ModelState, Prompt
            from livekit import rtc
        except ImportError as e:
            print(f"[Lucy] Missing deps: {e}")
            return

        if not self.config.api_key:
            print("[Lucy] No API key")
            return

        try:
            print("[Lucy] Setting up video source...")
            w, h = self.config.resolution
            source = rtc.VideoSource(w, h)
            self._source = source
            self._video_track = rtc.LocalVideoTrack.create_video_track("zeypher_camera", source)
            print(f"[Lucy] Video source ready: {w}x{h}")

            print("[Lucy] Connecting to Decart...")
            initial_state = None
            if self.config.reference_image and os.path.exists(self.config.reference_image):
                with open(self.config.reference_image, "rb") as f:
                    img_bytes = f.read()
                prompt_text = self.config.prompt or "Substitute the character in the video with the person in the reference image."
                initial_state = ModelState(
                    prompt=Prompt(text=prompt_text, enhance=True),
                    image=img_bytes,
                )
                print(f"[Lucy] Reference: {os.path.basename(self.config.reference_image)}")
            elif self.config.prompt:
                initial_state = ModelState(
                    prompt=Prompt(text=self.config.prompt, enhance=True),
                )

            realtime = await RealtimeClient.connect(
                base_url="wss://api3.decart.ai",
                api_key=self.config.api_key,
                local_track=self._video_track,
                options=RealtimeConnectOptions(
                    model=models.realtime("lucy-2.5"),
                    on_remote_stream=self._on_remote_track,
                    initial_state=initial_state,
                    resolution="720p",
                ),
            )

            self.connected = True
            print(f"[Lucy] Connected! Session: {realtime.session_id}")

            while not self._stop.is_set():
                await asyncio.sleep(0.1)

            print("[Lucy] Disconnecting...")
            await realtime.disconnect()

        except Exception as e:
            err = str(e)
            if "Invalid API key" in err or "401" in err:
                self._last_error = "Invalid API key"
            elif "Insufficient credits" in err:
                self._last_error = "No credits — top up at decart.ai"
            elif "timeout" in err.lower():
                self._last_error = "Connection timed out"
            elif "permission denied" in err.lower():
                self._last_error = "Permission denied"
            else:
                self._last_error = err[:100]
            print(f"[Lucy] Error: {self._last_error}")
            import traceback
            traceback.print_exc()
            self.connected = False

    def _on_remote_track(self, track, publication=None, participant=None):
        print(f"[Lucy] Remote track: kind={track.kind}, sid={getattr(track, 'sid', '?')}")
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._subscribe_track(track), self._loop)
        else:
            print("[Lucy] No event loop — cannot subscribe to remote track")

    async def _subscribe_track(self, track):
        from livekit import rtc
        try:
            stream = rtc.VideoStream(track)
            print(f"[Lucy] Subscribing to remote stream...")
            async for event in stream:
                try:
                    frame = event.frame
                    w, h = frame.width, frame.height
                    frame_type = frame.type
                    data = bytes(frame.data)
                    if frame_type == 5:
                        img = self._yuv420_to_bgr(data, w, h)
                    elif frame_type == 4:
                        rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    elif frame_type in (6, 7):
                        rgba = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
                        img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    else:
                        rgb = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    with self._lock:
                        self._output_frame = img
                    self._frames_received += 1
                    for cb in self._callbacks:
                        try:
                            cb(img)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[Lucy] Frame convert error: {e}")
        except Exception as e:
            print(f"[Lucy] Stream error: {e}")

    def _yuv420_to_bgr(self, data: bytes, w: int, h: int) -> np.ndarray:
        y_size = w * h
        uv_size = y_size // 4
        y_plane = np.frombuffer(data[:y_size], dtype=np.uint8).reshape(h, w)
        u_plane = np.frombuffer(data[y_size:y_size + uv_size], dtype=np.uint8).reshape(h // 2, w // 2)
        v_plane = np.frombuffer(data[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape(h // 2, w // 2)
        u_up = cv2.resize(u_plane, (w, h), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(v_plane, (w, h), interpolation=cv2.INTER_LINEAR)
        yuv = np.stack([y_plane, u_up, v_up], axis=-1)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def push_frame(self, frame: np.ndarray):
        if not self.connected or self._source is None:
            return
        try:
            from livekit import rtc
            target_w, target_h = self.config.resolution
            h, w = frame.shape[:2]
            if w != target_w or h != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = rgb.tobytes()
            video_frame = rtc.VideoFrame(target_w, target_h, 4, data)
            self._source.capture_frame(video_frame)
            self._frames_sent += 1
            if self._frames_sent % 30 == 1:
                print(f"[Lucy] Sent {self._frames_sent} frames, received {self._frames_received}")
        except Exception as e:
            if self._frames_sent == 0:
                print(f"[Lucy] Push frame error: {e}")

    def send_frame_sync(self, frame: np.ndarray, prompt: str = None, reference_image: str = None) -> bool:
        if self.connected:
            self.push_frame(frame)
            return True
        return False

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._output_frame.copy() if self._output_frame is not None else None

    def stop(self):
        self.connected = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._source = None
        self._video_track = None
        self._loop = None
        print("[Lucy] Stopped")

    def set_prompt(self, prompt: str):
        self.config.prompt = prompt

    def set_reference_image(self, image_path: str):
        self.config.reference_image = image_path

    def set_reference_from_frame(self, frame: np.ndarray) -> str:
        import base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode('utf-8')

    @property
    def stats(self) -> dict:
        return {
            "connected": self.connected,
            "frames_sent": self._frames_sent,
            "frames_received": self._frames_received,
            "last_latency_ms": round(self._last_latency, 1),
        }

    def process(self, frame: np.ndarray, timestamp: float = None) -> Optional[np.ndarray]:
        if not self.connected:
            return None
        self.push_frame(frame)
        time.sleep(0.03)
        return self.read()
