# ZeypherLive — Real-Time Body & Face Swap Engine

Full-stack real-time video manipulation engine. Face swap, body swap, voice changer, OBS integration, cloud sync, and Android mobile client. Built from scratch.

## Features

| Module | Description |
|--------|-------------|
| **Face Swap** | InsightFace local inference + Lucy 2.5 cloud API hybrid. SeamlessClone blending. Bidirectional swap. |
| **Body Swap** | MediaPipe 33-point pose tracking + Lucy 2.5 body replacement. Real-time skeleton overlay. |
| **Voice Changer** | 10 effects: robot, deep, high, alien, demon, chipmunk, echo, reverb, telephone, megaphone. Pitch ±24 semitones. Formant shift. Noise gate. |
| **Lucy 2.5** | Decart AI WebRTC API. Character swap, background replacement, VFX, style transfer. 720p realtime. |
| **OBS Bridge** | pyvirtualcam → OBS Virtual Camera. Configurable resolution/FPS. Zero-copy frame forwarding. |
| **Cloud Sync** | WebSocket server for mobile phones and remote control. Real-time frame streaming. Command protocol. |
| **Android Client** | Kivy mobile app. Remote preview, preset buttons, module toggles. Buildozer APK build. |
| **GUI** | PyQt5 dark-theme interface. 7 tabs: Camera, Face Swap, Body Track, Voice, OBS, Lucy, Cloud/Android. |

## Quick Start

### Windows
```bat
build.bat
python run_desktop.py
```

### Linux / Mac
```bash
chmod +x build.sh
./build.sh
python run_desktop.py
```

### CLI
```bash
python run_cli.py --camera 0 --face-swap --body-track --obs --lucy --lucy-key YOUR_KEY
```

### Android
```bash
cd android_client
pip install buildozer
buildozer android debug
```

## Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
mediapipe>=0.10.0
PyQt5>=5.15.0
pyaudio>=0.2.13
scipy>=1.11.0
pyvirtualcam>=0.14.0
aiohttp>=3.8.0
websockets>=12.0
onnxruntime>=1.16.0
```

## Models

Download required models:
```bash
python models/download_models.py
```

Required:
- `inswapper_128.onnx` — InsightFace face swap model (~500MB)
- `detection_10g.onnx` — InsightFace face detection (~17MB)
- `pose_landmarker_heavy.task` — MediaPipe pose (~25MB)

## Configuration

Config file: `config/zeypher_config.json`

All settings configurable via GUI or config file:
- Camera device, resolution, FPS
- Lucy 2.5 API key and prompts
- Face swap method (local/lucy), confidence threshold
- Body tracking complexity, confidence
- Voice effects, pitch, formant, devices
- OBS resolution, FPS, device name
- Cloud server host, port, SSL
- Android connection IP, port

## Architecture

```
ZeypherLive/
├── config/           # Central configuration
├── core/             # Camera capture, frame pipeline
├── body_track/       # MediaPipe pose estimation
├── face_swap/        # InsightFace + Lucy hybrid
├── voice_changer/    # Pitch, formant, effects
├── lucy_engine/      # Lucy 2.5 WebRTC client
├── obs_bridge/       # OBS Virtual Camera bridge
├── cloud_sync/       # WebSocket server
├── gui/              # PyQt5 interface
├── android_client/   # Kivy mobile app
├── models/           # ONNX model files
├── assets/           # Icons, presets
├── build.bat         # Windows build
├── build.sh          # Linux/Mac build
└── setup.py          # Python package
```

## License

MIT
