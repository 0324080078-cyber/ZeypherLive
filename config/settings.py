"""ZeypherLive — Central Configuration"""
import os
import json
from dataclasses import dataclass, field
from typing import Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "zeypher_config.json")

@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "dshow"

@dataclass
class LucyConfig:
    api_key: str = ""
    model_id: str = "lucy-2.5"
    ws_url: str = "wss://api3.decart.ai/v1/stream"
    resolution: tuple = (1280, 720)
    fps: int = 20
    prompt: str = ""
    reference_image: Optional[str] = None
    auto_reconnect: bool = True
    max_retries: int = 5
    credit_limit: int = 1000

@dataclass
class FaceSwapConfig:
    enabled: bool = True
    model_path: str = os.path.join(os.path.dirname(__file__), "..", "models", "inswapper_128.onnx")
    provider: str = "cpu"
    confidence_threshold: float = 0.5
    swap_method: str = "lucy"
    local_fallback: bool = True

@dataclass
class BodyTrackConfig:
    enabled: bool = True
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = True
    smooth_landmarks: bool = True
    show_skeleton: bool = False
    body_swap_enabled: bool = False

@dataclass
class VoiceConfig:
    enabled: bool = False
    input_device: int = -1
    output_device: int = -1
    pitch_shift: float = 0.0
    formant_shift: float = 0.0
    effect: str = "none"
    dry_wet: float = 0.5
    noise_gate_db: float = -40.0
    sample_rate: int = 44100
    chunk_size: int = 1024

@dataclass
class OBSConfig:
    enabled: bool = True
    virtual_cam_name: str = "OBS Virtual Camera"
    width: int = 1280
    height: int = 720
    fps: int = 30
    pixel_format: str = "BGR"

@dataclass
class CloudConfig:
    enabled: bool = False
    server_host: str = "0.0.0.0"
    server_port: int = 8765
    max_connections: int = 5
    use_ssl: bool = False
    cert_path: str = ""
    key_path: str = ""

@dataclass
class AndroidConfig:
    server_ip: str = "192.168.1.100"
    server_port: int = 8765
    preview_width: int = 640
    preview_height: int = 480
    preview_fps: int = 15

@dataclass
class ZeypherConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    lucy: LucyConfig = field(default_factory=LucyConfig)
    face_swap: FaceSwapConfig = field(default_factory=FaceSwapConfig)
    body_track: BodyTrackConfig = field(default_factory=BodyTrackConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    obs: OBSConfig = field(default_factory=OBSConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    android: AndroidConfig = field(default_factory=AndroidConfig)
    log_level: str = "INFO"
    theme: str = "dark"

    def save(self):
        with open(CONFIG_PATH, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls) -> "ZeypherConfig":
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    data = json.load(f)
            except Exception:
                return cls()
            if not isinstance(data, dict):
                return cls()
            cfg = cls()
            section_map = {
                "camera": cfg.camera,
                "lucy": cfg.lucy,
                "face_swap": cfg.face_swap,
                "body_track": cfg.body_track,
                "voice": cfg.voice,
                "obs": cfg.obs,
                "cloud": cfg.cloud,
                "android": cfg.android,
            }
            for key, obj in section_map.items():
                if key in data and isinstance(data[key], dict):
                    for k, v in data[key].items():
                        if hasattr(obj, k):
                            setattr(obj, k, v)
            if "log_level" in data and isinstance(data["log_level"], str):
                cfg.log_level = data["log_level"]
            if "theme" in data and isinstance(data["theme"], str):
                cfg.theme = data["theme"]
            return cfg
        return cls()


CONFIG = ZeypherConfig.load()
