"""ZeypherLive — Voice Changer Engine"""
import numpy as np
import pyaudio
import threading
import time
from typing import Optional, Callable
from config.settings import CONFIG


class VoiceChanger:
    EFFECTS = {
        "none": None,
        "female": "female",
        "male": "male",
        "robot": "robot",
        "deep": "deep",
        "high": "high",
        "alien": "alien",
        "demon": "demon",
        "chipmunk": "chipmunk",
        "echo": "echo",
        "reverb": "reverb",
        "telephone": "telephone",
        "megaphone": "megaphone",
    }

    def __init__(self, config=None):
        self.config = config or CONFIG.voice
        self.running = False
        self._pa: Optional[pyaudio.PyAudio] = None
        self._input_stream = None
        self._output_stream = None
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable] = []
        self._input_level = 0.0
        self._output_level = 0.0
        self._latency_ms = 0.0
        self._effect_params = {}
        self._noise_gate_open = True
        self._noise_floor = 0.0
        self._sample_buffer = np.array([], dtype=np.float32)

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def initialize(self) -> bool:
        try:
            self._pa = pyaudio.PyAudio()
            return True
        except Exception:
            return False

    def list_devices(self) -> list[dict]:
        if self._pa is None:
            return []
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            devices.append({
                "id": i,
                "name": info["name"],
                "inputs": info["maxInputChannels"],
                "outputs": info["maxOutputChannels"],
                "sample_rate": int(info["defaultSampleRate"]),
            })
        return devices

    def set_device(self, input_device: int = -1, output_device: int = -1):
        if input_device >= 0:
            self.config.input_device = input_device
        if output_device >= 0:
            self.config.output_device = output_device

    def start(self):
        if self.running:
            return
        if self._pa is None:
            if not self.initialize():
                return
        try:
            input_idx = self.config.input_device if self.config.input_device >= 0 else self._pa.get_default_input_device_info()["index"]
            output_idx = self.config.output_device if self.config.output_device >= 0 else self._pa.get_default_output_device_info()["index"]
            self._input_stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=input_idx,
                frames_per_buffer=self.config.chunk_size,
            )
            self._output_stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                output_device_index=output_idx,
                frames_per_buffer=self.config.chunk_size,
            )
            self.running = True
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()
        except Exception:
            self.running = False

    def _audio_loop(self):
        while self.running:
            try:
                data = self._input_stream.read(self.config.chunk_size, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.float32).copy()
                self._input_level = np.abs(audio).mean()
                processed = self._apply_effects(audio)
                self._noise_gate(processed)
                if self._noise_gate_open:
                    self._output_stream.write(processed.tobytes())
                    self._output_level = np.abs(processed).mean()
                else:
                    silence = np.zeros_like(processed)
                    self._output_stream.write(silence.tobytes())
                    self._output_level = 0.0
            except Exception:
                if not self.running:
                    break
                time.sleep(0.01)

    def _apply_effects(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()
        pitch = self.config.pitch_shift
        formant = self.config.formant_shift
        effect = self.config.effect
        if pitch != 0.0:
            result = self._shift_pitch(result, pitch)
        if formant != 0.0:
            result = self._shift_formant(result, formant)
        if effect != "none":
            result = self._apply_effect(result, effect)
        dry_wet = self.config.dry_wet
        result = audio * (1.0 - dry_wet) + result * dry_wet
        peak = np.abs(result).max()
        if peak > 0.95:
            result = result * (0.95 / peak)
        return result.astype(np.float32)

    def _shift_pitch(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if semitones == 0:
            return audio
        factor = 2 ** (semitones / 12.0)
        n = len(audio)
        indices = np.floor(np.arange(n) * factor).astype(int) % n
        return audio[indices]

    def _shift_formant(self, audio: np.ndarray, shift: float) -> np.ndarray:
        if shift == 0:
            return audio
        n = len(audio)
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n, 1.0 / self.config.sample_rate)
        formant_freqs = [500, 1500, 2500, 3500]
        for f0 in formant_freqs:
            f_shifted = f0 * (1.0 + shift)
            mask = np.exp(-((freqs - f_shifted) ** 2) / (2 * (f0 * 0.3) ** 2))
            fft *= (1.0 + 0.5 * mask)
        return np.fft.irfft(fft, n=n).astype(np.float32)

    def _apply_effect(self, audio: np.ndarray, effect: str) -> np.ndarray:
        n = len(audio)
        sr = self.config.sample_rate
        if effect == "female":
            shifted = self._shift_pitch_resample(audio, 4.0)
            formanted = self._shift_formant(shifted, 0.3)
            return np.clip(formanted * 1.1, -1.0, 1.0).astype(np.float32)
        elif effect == "male":
            shifted = self._shift_pitch_resample(audio, -3.0)
            formanted = self._shift_formant(shifted, -0.2)
            return np.clip(formanted * 1.1, -1.0, 1.0).astype(np.float32)
        elif effect == "robot":
            carrier = np.sin(2 * np.pi * 100 * np.arange(n) / sr)
            return (audio * carrier * 2).astype(np.float32)
        elif effect == "deep":
            return self._shift_pitch_resample(audio, -4).astype(np.float32)
        elif effect == "high":
            return self._shift_pitch_resample(audio, 6).astype(np.float32)
        elif effect == "chipmunk":
            return self._shift_pitch_resample(audio, 12).astype(np.float32)
        elif effect == "demon":
            shifted = self._shift_pitch_resample(audio, -7)
            return (shifted * 0.8).astype(np.float32)
        elif effect == "alien":
            lfo = np.sin(2 * np.pi * 5 * np.arange(n) / sr)
            modulated = audio * (1.0 + 0.5 * lfo)
            return self._shift_pitch_resample(modulated, 3).astype(np.float32)
        elif effect == "echo":
            delay_samples = int(0.15 * sr)
            echo = np.zeros(n)
            if delay_samples < n:
                echo[delay_samples:] = audio[:n - delay_samples] * 0.4
            return (audio + echo).astype(np.float32)
        elif effect == "reverb":
            delays = [int(d * sr) for d in [0.03, 0.05, 0.08, 0.12]]
            gains = [0.3, 0.2, 0.15, 0.1]
            reverb = np.zeros(n)
            for delay, gain in zip(delays, gains):
                if delay < n:
                    reverb[delay:] += audio[:n - delay] * gain
            return (audio + reverb).astype(np.float32)
        elif effect == "telephone":
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            mask = np.where((freqs >= 300) & (freqs <= 3400), 1.0, 0.0)
            fft *= mask
            return np.fft.irfft(fft, n=n).astype(np.float32)
        elif effect == "megaphone":
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(n, 1.0 / sr)
            mask = np.where((freqs >= 200) & (freqs <= 4000), 1.0, 0.0)
            fft *= mask
            result = np.fft.irfft(fft, n=n)
            result = np.clip(result * 2.0, -1.0, 1.0)
            return result.astype(np.float32)
        return audio

    def _shift_pitch_resample(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if semitones == 0:
            return audio
        factor = 2 ** (semitones / 12.0)
        n = len(audio)
        old_indices = np.arange(n)
        new_indices = old_indices * factor
        new_indices = np.clip(new_indices, 0, n - 1)
        result = np.interp(new_indices, old_indices, audio)
        return result.astype(np.float32)

    def _noise_gate(self, audio: np.ndarray):
        level = np.abs(audio).mean()
        threshold = 10 ** (self.config.noise_gate_db / 20.0)
        if level < threshold:
            self._noise_gate_open = False
            self._noise_floor = self._noise_floor * 0.99 + level * 0.01
        else:
            self._noise_gate_open = True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except Exception:
                pass
        if self._output_stream:
            try:
                self._output_stream.stop_stream()
                self._output_stream.close()
            except Exception:
                pass
        self._input_stream = None
        self._output_stream = None

    def set_pitch(self, semitones: float):
        self.config.pitch_shift = max(-24.0, min(24.0, semitones))

    def set_formant(self, shift: float):
        self.config.formant_shift = max(-1.0, min(1.0, shift))

    def set_effect(self, effect: str):
        if effect in self.EFFECTS:
            self.config.effect = effect

    def set_dry_wet(self, value: float):
        self.config.dry_wet = max(0.0, min(1.0, value))

    @property
    def stats(self) -> dict:
        return {
            "running": self.running,
            "input_level": round(self._input_level, 4),
            "output_level": round(self._output_level, 4),
            "effect": self.config.effect,
            "pitch": self.config.pitch_shift,
            "formant": self.config.formant_shift,
            "dry_wet": self.config.dry_wet,
            "noise_gate_open": self._noise_gate_open,
        }

    def __del__(self):
        self.stop()
        if self._pa:
            self._pa.terminate()
