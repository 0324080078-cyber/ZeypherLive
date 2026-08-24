"""ZeypherLive — Real-time Lip Sync (Audio Energy → Mouth Landmarks)"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class LipSyncEngine:
    MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
    LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._input_device = None
        self._sample_rate = 16000
        self._chunk_size = 1024
        self._callbacks: list[Callable] = []
        self._audio_level = 0.0
        self._peak_level = 0.0
        self._smooth_level = 0.0
        self._speech_detected = False
        self._speech_threshold = 0.02
        self._energy_history = []
        self._history_len = 10
        self._mouth_openness = 0.0
        self._formants = [0.0, 0.0, 0.0]
        self._lock = threading.Lock()

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def list_devices(self) -> list[dict]:
        if not HAS_AUDIO:
            return []
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append({"id": i, "name": d["name"], "channels": d["max_input_channels"], "rate": int(d["default_samplerate"])})
        return result

    def set_device(self, device_id=None):
        self._input_device = device_id

    def start(self) -> bool:
        if self.running:
            return True
        if not HAS_AUDIO:
            print("[LipSync] sounddevice not available")
            return False
        try:
            self.running = True
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()
            print("[LipSync] Started")
            return True
        except Exception as e:
            print(f"[LipSync] Start error: {e}")
            self.running = False
            return False

    def _audio_loop(self):
        try:
            with sd.InputStream(
                device=self._input_device,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                dtype="float32",
            ) as inp:
                while self.running:
                    data, overflowed = inp.read(self._chunk_size)
                    audio = data[:, 0].copy()
                    self._analyze_audio(audio)
                    for cb in self._callbacks:
                        try:
                            cb(self._get_state())
                        except Exception:
                            pass
        except Exception as e:
            print(f"[LipSync] Audio loop error: {e}")
            self.running = False

    def _analyze_audio(self, audio: np.ndarray):
        level = float(np.abs(audio).mean())
        peak = float(np.abs(audio).max())

        self._energy_history.append(level)
        if len(self._energy_history) > self._history_len:
            self._energy_history.pop(0)

        avg_energy = np.mean(self._energy_history) if self._energy_history else 0
        variance = np.var(self._energy_history) if self._energy_history else 0

        self._speech_detected = level > self._speech_threshold

        if self._speech_detected:
            raw_openness = min(1.0, (level / 0.15) * 2.0)
            raw_openness *= (1.0 + variance * 10.0)
            self._mouth_openness = self._smooth_level * 0.3 + raw_openness * 0.7
        else:
            self._mouth_openness = self._smooth_level * 0.8

        self._smooth_level = self._smooth_level * 0.7 + level * 0.3
        self._audio_level = level
        self._peak_level = peak

        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self._sample_rate)
        for i, band in enumerate([(200, 600), (600, 2500), (2500, 5000)]):
            mask = (freqs >= band[0]) & (freqs <= band[1])
            self._formants[i] = float(np.mean(fft[mask])) if np.any(mask) else 0.0

    def get_mouth_deformation(self, landmarks, frame_shape) -> dict:
        if not landmarks or len(landmarks) < 468:
            return {"openness": 0, "width": 0, "points": []}

        h, w = frame_shape[:2]
        mouth_pts = []
        for idx in self.MOUTH_LANDMARKS:
            if idx < len(landmarks):
                lm = landmarks[idx]
                mouth_pts.append((int(lm.x * w), int(lm.y * h)))

        if len(mouth_pts) < 2:
            return {"openness": 0, "width": 0, "points": []}

        top_lip = []
        for idx in self.UPPER_LIP:
            if idx < len(landmarks):
                lm = landmarks[idx]
                top_lip.append((lm.x * w, lm.y * h))

        bottom_lip = []
        for idx in self.LOWER_LIP:
            if idx < len(landmarks):
                lm = landmarks[idx]
                bottom_lip.append((lm.x * w, lm.y * h))

        openness = self._mouth_openness
        width = 0
        if mouth_pts:
            xs = [p[0] for p in mouth_pts]
            width = (max(xs) - min(xs)) / w if w > 0 else 0

        return {
            "openness": openness,
            "width": width,
            "points": mouth_pts,
            "top_lip": top_lip,
            "bottom_lip": bottom_lip,
            "speech": self._speech_detected,
        }

    def apply_lip_sync_to_frame(self, frame: np.ndarray, face_landmarks=None) -> np.ndarray:
        if face_landmarks is None:
            return frame
        h, w = frame.shape[:2]
        result = frame.copy()

        mouth_deform = self.get_mouth_deformation(face_landmarks, frame.shape)
        if not mouth_deform["points"]:
            return result

        openness = mouth_deform["openness"]

        if openness > 0.1:
            mouth_center = np.mean(mouth_deform["points"], axis=0).astype(int)
            radius = int(openness * 25)
            overlay = result.copy()
            cv2.ellipse(
                overlay,
                tuple(mouth_center),
                (radius, max(2, int(openness * radius * 0.8))),
                0, 0, 360,
                (0, 0, 0), -1, cv2.LINE_AA
            )
            alpha = min(0.6, openness * 0.8)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        if self._speech_detected:
            for pt in mouth_deform["points"]:
                cv2.circle(result, pt, 2, (0, 200, 255), -1, cv2.LINE_AA)

        return result

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[LipSync] Stopped")

    def _get_state(self) -> dict:
        return {
            "audio_level": round(self._audio_level, 4),
            "peak_level": round(self._peak_level, 4),
            "mouth_openness": round(self._mouth_openness, 3),
            "speech_detected": self._speech_detected,
            "formants": [round(f, 2) for f in self._formants],
        }

    @property
    def stats(self) -> dict:
        return {
            "running": self.running,
            **self._get_state(),
        }
