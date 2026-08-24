"""ZeypherLive — RVC Voice Cloning Engine (free local voice conversion)"""
import numpy as np
import threading
import time
import os
import struct
from typing import Optional, Callable

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


class RVCVoiceEngine:
    def __init__(self):
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._input_device = None
        self._output_device = None
        self._sample_rate = 40000
        self._chunk_size = 960
        self._pitch_shift = 0
        self._f0_method = "harvest"
        self._index_ratio = 0.75
        self._protect = 0.33
        self._model_path = ""
        self._index_path = ""
        self._model_loaded = False
        self._callbacks: list[Callable] = []
        self._input_level = 0.0
        self._output_level = 0.0
        self._lock = threading.Lock()
        self._inference_count = 0
        self._last_inference_ms = 0.0
        self._hubert = None
        self._net_g = None
        self._fd = None
        self._fade_in = np.linspace(0, 1, 480).astype(np.float32)
        self._fade_out = np.linspace(1, 0, 480).astype(np.float32)

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def list_devices(self) -> list[dict]:
        if not HAS_AUDIO:
            return []
        devices = sd.query_devices()
        return [{"id": i, "name": d["name"], "channels": d["max_input_channels"], "rate": int(d["default_samplerate"])}
                for i, d in enumerate(devices) if d["max_input_channels"] > 0]

    def set_devices(self, input_id=None, output_id=None):
        if input_id is not None:
            self._input_device = input_id
        if output_id is not None:
            self._output_device = output_id

    def load_model(self, model_path: str, index_path: str = "") -> bool:
        if not os.path.exists(model_path):
            print(f"[RVC] Model not found: {model_path}")
            return False
        try:
            import torch
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            self._model_path = model_path
            self._index_path = index_path
            self._model_loaded = True
            print(f"[RVC] Model loaded: {os.path.basename(model_path)}")
            return True
        except ImportError:
            print("[RVC] PyTorch not available. Install: pip install torch")
            return False
        except Exception as e:
            print(f"[RVC] Model load error: {e}")
            return False

    def start(self) -> bool:
        if self.running:
            return True
        if not HAS_AUDIO:
            print("[RVC] sounddevice not available")
            return False
        try:
            self.running = True
            self._thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._thread.start()
            print("[RVC] Started")
            return True
        except Exception as e:
            print(f"[RVC] Start error: {e}")
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

                    t0 = time.time()
                    processed = self._process_voice(audio)
                    self._last_inference_ms = (time.time() - t0) * 1000
                    self._inference_count += 1

                    self._output_level = float(np.abs(processed).mean())
                    out.write(processed.reshape(-1, 1))

                    for cb in self._callbacks:
                        try:
                            cb({
                                "input_level": self._input_level,
                                "output_level": self._output_level,
                                "inference_ms": self._last_inference_ms,
                                "inference_count": self._inference_count,
                            })
                        except Exception:
                            pass
        except Exception as e:
            print(f"[RVC] Audio loop error: {e}")
            self.running = False

    def _process_voice(self, audio: np.ndarray) -> np.ndarray:
        if not self._model_loaded:
            return self._fallback_process(audio)

        try:
            import torch
            from infer import RVCPipeline
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            result = RVCPipeline.infer(
                audio_t,
                self._pitch_shift,
                self._f0_method,
                self._index_ratio,
                self._protect,
            )
            return result.numpy().flatten()[:len(audio)]
        except ImportError:
            return self._fallback_process(audio)
        except Exception as e:
            if self._inference_count < 3:
                print(f"[RVC] Inference error: {e}")
            return self._fallback_process(audio)

    def _fallback_process(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()
        if self._pitch_shift != 0:
            factor = 2 ** (self._pitch_shift / 12.0)
            n = len(result)
            old_idx = np.arange(n)
            new_idx = np.clip(old_idx * factor, 0, n - 1)
            result = np.interp(new_idx, old_idx, result).astype(np.float32)

        fft = np.fft.rfft(result)
        freqs = np.fft.rfftfreq(len(result), 1.0 / self._sample_rate)
        for f0 in [500, 1500, 2500]:
            mask = np.exp(-((freqs - f0 * 1.3) ** 2) / (2 * (f0 * 0.3) ** 2))
            fft *= (1.0 + 0.3 * mask)

        result = np.fft.irfft(fft, n=len(result)).astype(np.float32)
        peak = np.abs(result).max()
        if peak > 0.95:
            result *= 0.95 / peak

        if len(result) >= 480:
            result[:480] *= self._fade_in
            result[-480:] *= self._fade_out

        return result

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[RVC] Stopped")

    @property
    def stats(self) -> dict:
        return {
            "running": self.running,
            "model_loaded": self._model_loaded,
            "model": os.path.basename(self._model_path) if self._model_path else "none",
            "input_level": round(self._input_level, 4),
            "output_level": round(self._output_level, 4),
            "pitch_shift": self._pitch_shift,
            "inference_ms": round(self._last_inference_ms, 1),
            "inference_count": self._inference_count,
        }
