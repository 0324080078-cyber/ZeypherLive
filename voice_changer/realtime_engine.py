"""ZeypherLive — Real-time Voice Changer (RVC-style, no external deps beyond torch)"""
import numpy as np
import threading
import time
import os
from typing import Optional, Callable

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class RealtimeVoiceChanger:
    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._input_device = None
        self._output_device = None
        self._sample_rate = 48000
        self._chunk_size = 960
        self._pitch_shift = 4.0
        self._formant_shift = 0.3
        self._denoise = False
        self._model_path = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []
        self._input_level = 0.0
        self._output_level = 0.0
        self._latency_ms = 0.0
        self._rvc_model = None
        self._hubert = None
        self._f0_extractor = None
        self._index = None

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

    def set_devices(self, input_id=None, output_id=None):
        if input_id is not None:
            self._input_device = input_id
        if output_id is not None:
            self._output_device = output_id

    def load_rvc_model(self, model_path: str) -> bool:
        if not os.path.exists(model_path):
            print(f"[VoiceChanger] Model not found: {model_path}")
            return False
        try:
            import torch
            state = torch.load(model_path, map_location="cpu")
            self._rvc_model = state
            print(f"[VoiceChanger] RVC model loaded: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"[VoiceChanger] Model load error: {e}")
            return False

    def start(self) -> bool:
        if self.running:
            return True
        if not HAS_AUDIO:
            print("[VoiceChanger] sounddevice not available")
            return False
        try:
            self.running = True
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()
            print("[VoiceChanger] Started")
            return True
        except Exception as e:
            print(f"[VoiceChanger] Start error: {e}")
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
            ) as inp, sd.OutputStream(
                device=self._output_device,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                dtype="float32",
            ) as out:
                while self.running:
                    data, overflowed = inp.read(self._chunk_size)
                    audio = data[:, 0].copy()
                    self._input_level = float(np.abs(audio).mean())

                    processed = self._process_chunk(audio)

                    self._output_level = float(np.abs(processed).mean())
                    out.write(processed.reshape(-1, 1))

                    for cb in self._callbacks:
                        try:
                            cb(processed)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[VoiceChanger] Audio loop error: {e}")
            self.running = False

    def _process_chunk(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()

        if self._pitch_shift != 0.0:
            result = self._pitch_shift_resample(result, self._pitch_shift)

        if self._formant_shift != 0.0:
            result = self._formant_shift_fft(result, self._formant_shift)

        peak = np.abs(result).max()
        if peak > 0.95:
            result = result * (0.95 / peak)

        return result.astype(np.float32)

    def _pitch_shift_resample(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if semitones == 0:
            return audio
        factor = 2 ** (semitones / 12.0)
        n = len(audio)
        old_idx = np.arange(n)
        new_idx = np.clip(old_idx * factor, 0, n - 1)
        return np.interp(new_idx, old_idx, audio).astype(np.float32)

    def _formant_shift_fft(self, audio: np.ndarray, shift: float) -> np.ndarray:
        if shift == 0:
            return audio
        n = len(audio)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / self._sample_rate)
        for f0 in [500, 1500, 2500, 3500]:
            f_shifted = f0 * (1.0 + shift)
            mask = np.exp(-((freqs - f_shifted) ** 2) / (2 * (f0 * 0.3) ** 2))
            fft *= (1.0 + 0.5 * mask)
        return np.fft.irfft(fft, n=n).astype(np.float32)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[VoiceChanger] Stopped")

    @property
    def stats(self) -> dict:
        return {
            "running": self.running,
            "input_level": round(self._input_level, 4),
            "output_level": round(self._output_level, 4),
            "pitch": self._pitch_shift,
            "formant": self._formant_shift,
            "sample_rate": self._sample_rate,
            "chunk_size": self._chunk_size,
        }
