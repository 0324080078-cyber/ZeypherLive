"""ZeypherLive — Real-time Voice Changer (routes to any app via virtual device)"""
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
        self._sample_rate = 44100
        self._chunk_size = 1024
        self._pitch_shift = 4.0
        self._formant_shift = 0.3
        self._denoise_enabled = False
        self._gate_threshold = 0.01
        self._gate_open = True
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []
        self._input_level = 0.0
        self._output_level = 0.0
        self._latency_ms = 0.0
        self._rms_level = 0.0
        self._peak_level = 0.0
        self._effect_stack = []
        self._noise_profile = None
        self._denoise_amount = 0.5

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def list_devices(self) -> list[dict]:
        if not HAS_AUDIO:
            return []
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                result.append({
                    "id": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "rate": int(d["default_samplerate"]),
                    "is_input": d["max_input_channels"] > 0,
                    "is_output": d["max_output_channels"] > 0,
                })
        return result

    def list_output_devices(self) -> list[dict]:
        if not HAS_AUDIO:
            return []
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                result.append({
                    "id": i,
                    "name": d["name"],
                    "channels": d["max_output_channels"],
                    "rate": int(d["default_samplerate"]),
                })
        return result

    def set_devices(self, input_id=None, output_id=None):
        if input_id is not None:
            self._input_device = input_id
        if output_id is not None:
            self._output_device = output_id

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
            inp_params = {}
            if self._input_device is not None:
                inp_params["device"] = self._input_device
            out_params = {}
            if self._output_device is not None:
                out_params["device"] = self._output_device

            with sd.InputStream(
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                dtype="float32",
                **inp_params,
            ) as inp, sd.OutputStream(
                channels=1,
                samplerate=self._sample_rate,
                blocksize=self._chunk_size,
                dtype="float32",
                **out_params,
            ) as out:
                while self.running:
                    data, overflowed = inp.read(self._chunk_size)
                    audio = data[:, 0].copy()

                    self._input_level = float(np.abs(audio).mean())
                    self._peak_level = float(np.abs(audio).max())
                    self._rms_level = float(np.sqrt(np.mean(audio ** 2)))

                    self._apply_noise_gate(audio)

                    if self._gate_open:
                        processed = self._process_audio(audio)
                        self._output_level = float(np.abs(processed).mean())
                        out.write(processed.reshape(-1, 1))
                    else:
                        silence = np.zeros(self._chunk_size, dtype=np.float32)
                        self._output_level = 0.0
                        out.write(silence.reshape(-1, 1))

                    for cb in self._callbacks:
                        try:
                            cb({
                                "input_level": self._input_level,
                                "output_level": self._output_level,
                                "peak_level": self._peak_level,
                                "gate_open": self._gate_open,
                            })
                        except Exception:
                            pass
        except Exception as e:
            print(f"[VoiceChanger] Audio loop error: {e}")
            self.running = False

    def _apply_noise_gate(self, audio: np.ndarray):
        level = float(np.abs(audio).mean())
        if level < self._gate_threshold:
            if self._gate_open:
                self._gate_open = False
        else:
            self._gate_open = True

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()

        if self._pitch_shift != 0.0:
            result = self._pitch_shift_wsola(result, self._pitch_shift)

        if self._formant_shift != 0.0:
            result = self._formant_shift_fft(result, self._formant_shift)

        if self._denoise_enabled:
            result = self._simple_denoise(result)

        peak = np.abs(result).max()
        if peak > 0.95:
            result = result * (0.95 / peak)

        return result.astype(np.float32)

    def _pitch_shift_wsola(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if semitones == 0:
            return audio
        factor = 2 ** (semitones / 12.0)
        n = len(audio)
        hop = self._chunk_size // 2
        window = np.hanning(hop)
        output = np.zeros(n, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)

        read_pos = 0.0
        write_pos = 0
        while int(read_pos) + hop <= n and write_pos + hop <= n:
            start = int(read_pos)
            chunk = audio[start:start + hop].copy()
            windowed = chunk * window
            output[write_pos:write_pos + hop] += windowed
            weight[write_pos:write_pos + hop] += window ** 2
            read_pos += hop * factor
            write_pos += hop

        nonzero = weight > 1e-8
        output[nonzero] /= weight[nonzero]
        return output

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

    def _simple_denoise(self, audio: np.ndarray) -> np.ndarray:
        if self._noise_profile is None:
            self._noise_profile = np.abs(np.fft.rfft(audio)).mean(axis=0) if audio.ndim > 1 else np.abs(np.fft.rfft(audio))
            return audio
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        noise_est = self._noise_profile[:len(magnitude)] if len(self._noise_profile) >= len(magnitude) else np.pad(self._noise_profile, (0, len(magnitude) - len(self._noise_profile)))
        clean_mag = np.maximum(magnitude - noise_est * self._denoise_amount, 0)
        clean_fft = clean_mag * np.exp(1j * phase)
        return np.fft.irfft(clean_fft, n=len(audio)).astype(np.float32)

    def calibrate_noise(self, audio: np.ndarray):
        self._noise_profile = np.abs(np.fft.rfft(audio))
        print("[VoiceChanger] Noise profile calibrated")

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
            "peak_level": round(self._peak_level, 4),
            "rms_level": round(self._rms_level, 4),
            "gate_open": self._gate_open,
            "pitch": self._pitch_shift,
            "formant": self._formant_shift,
            "sample_rate": self._sample_rate,
            "chunk_size": self._chunk_size,
        }
