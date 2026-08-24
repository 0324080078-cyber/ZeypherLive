"""ZeypherLive — Model Download Script (Fixed)"""
import os
import sys
import requests

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = {
    "mediapipe_pose": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "filename": "pose_landmarker_heavy.task",
        "size": "~25MB",
        "auth": False,
    },
    "inswapper_128": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "filename": "inswapper_128.onnx",
        "size": "~500MB",
        "auth": False,
    },
    "detection_10g": {
        "url": "https://huggingface.co/ezioruan/insightface/resolve/main/det_10g.onnx",
        "filename": "detection_10g.onnx",
        "size": "~17MB",
        "auth": False,
    },
}

FALLBACK_URLS = {
    "inswapper_128": [
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "https://huggingface.co/fonyor/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    ],
    "detection_10g": [
        "https://huggingface.co/ezioruan/insightface/resolve/main/det_10g.onnx",
        "https://huggingface.co/fonyor/insightface/resolve/main/det_10g.onnx",
    ],
}


def download_file(url, dest, token=None, chunk_size=8192):
    print(f"  Downloading: {os.path.basename(dest)}")
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = (downloaded / total) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(f"\r  Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        print(f"\n  Saved: {dest}")
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def main():
    print("=" * 50)
    print("  ZeypherLive — Model Downloader")
    print("=" * 50)
    print()

    token = os.environ.get("HF_TOKEN", None)
    if not token:
        hf_token_path = os.path.expanduser("~/.huggingface/token")
        if os.path.exists(hf_token_path):
            with open(hf_token_path) as f:
                token = f.read().strip()

    for name, info in MODELS.items():
        dest = os.path.join(MODELS_DIR, info["filename"])
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  [SKIP] {info['filename']} (already exists)")
            continue
        print(f"  [{name}] Size: {info['size']}")
        success = download_file(info["url"], dest, token)
        if not success and name in FALLBACK_URLS:
            for alt_url in FALLBACK_URLS[name]:
                print(f"  Trying alternate source...")
                success = download_file(alt_url, dest, token)
                if success:
                    break
        if not success:
            print(f"  [WARN] Failed to download {info['filename']}")
            print(f"  You can download manually and place in: {MODELS_DIR}")
        print()

    print("Done. Models saved to:", MODELS_DIR)
    print()
    print("If download failed, you can:")
    print("  1. Get HF token: huggingface-cli login")
    print("  2. Set env: set HF_TOKEN=your_token_here")
    print("  3. Re-run this script")
    print("  4. Or place .onnx files manually in models/")


if __name__ == "__main__":
    main()
