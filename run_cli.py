"""ZeypherLive — CLI Entry Point"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CONFIG
from core.camera import CameraCapture
from core.pipeline import FramePipeline
from body_track.tracker import BodyTracker
from face_swap.engine import FaceSwapEngine
from voice_changer.engine import VoiceChanger
from obs_bridge.bridge import OBSBridge
from cloud_sync.server import CloudSyncServer

try:
    from lucy_engine.lucy_client import LucyEngine
except ImportError:
    LucyEngine = None


def main():
    parser = argparse.ArgumentParser(description="ZeypherLive CLI")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS")
    parser.add_argument("--face-swap", action="store_true", help="Enable face swap")
    parser.add_argument("--body-track", action="store_true", help="Enable body tracking")
    parser.add_argument("--voice", action="store_true", help="Enable voice changer")
    parser.add_argument("--obs", action="store_true", help="Enable OBS bridge")
    parser.add_argument("--lucy", action="store_true", help="Enable Lucy 2.5")
    parser.add_argument("--cloud", action="store_true", help="Enable cloud server")
    parser.add_argument("--lucy-key", type=str, default="", help="Lucy API key")
    parser.add_argument("--lucy-prompt", type=str, default="", help="Lucy prompt")
    parser.add_argument("--face-source", type=str, default="", help="Source face image")
    parser.add_argument("--voice-effect", type=str, default="none", help="Voice effect")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch shift semitones")
    args = parser.parse_args()

    CONFIG.camera.device_id = args.camera
    CONFIG.camera.width = args.width
    CONFIG.camera.height = args.height
    CONFIG.camera.fps = args.fps

    camera = CameraCapture()
    pipeline = FramePipeline()
    pipeline.set_source(camera)

    if args.body_track:
        tracker = BodyTracker()
        pipeline.add_processor(tracker)
        print("[ZeypherLive] Body tracking enabled")

    face_engine = None
    if args.face_swap:
        face_engine = FaceSwapEngine()
        face_engine.initialize()
        if args.face_source:
            face_engine.set_source_face(args.face_source)
        pipeline.add_processor(face_engine)
        print("[ZeypherLive] Face swap enabled")

    lucy = None
    if args.lucy and LucyEngine:
        lucy = LucyEngine()
        CONFIG.lucy.api_key = args.lucy_key
        CONFIG.lucy.prompt = args.lucy_prompt
        if face_engine:
            face_engine.set_lucy_client(lucy)
        pipeline.add_processor(lucy)
        print("[ZeypherLive] Lucy 2.5 enabled")

    voice = None
    if args.voice:
        voice = VoiceChanger()
        CONFIG.voice.enabled = True
        CONFIG.voice.effect = args.voice_effect
        CONFIG.voice.pitch_shift = args.pitch
        voice.set_effect(args.voice_effect)
        voice.set_pitch(args.pitch)
        voice.start()
        print(f"[ZeypherLive] Voice enabled: {args.voice_effect}, pitch={args.pitch}")

    obs = None
    if args.obs:
        obs = OBSBridge()
        obs.set_source(pipeline)
        obs.start()
        print("[ZeypherLive] OBS bridge enabled")

    cloud = None
    if args.cloud:
        cloud = CloudSyncServer()
        cloud.set_source(pipeline)
        cloud.start()
        print(f"[ZeypherLive] Cloud server on port {CONFIG.cloud.server_port}")

    if not camera.open():
        print("[ZeypherLive] ERROR: Cannot open camera")
        return

    camera.start()
    pipeline.start()
    if lucy:
        lucy.start()

    print("[ZeypherLive] Running. Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[ZeypherLive] Shutting down...")
    finally:
        camera.stop()
        pipeline.stop()
        if voice:
            voice.stop()
        if obs:
            obs.stop()
        if cloud:
            cloud.stop()
        if lucy:
            lucy.stop()
        CONFIG.save()
        print("[ZeypherLive] Stopped.")


if __name__ == "__main__":
    main()
