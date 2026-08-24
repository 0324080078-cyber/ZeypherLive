#!/bin/bash
echo "===================================="
echo "  ZeypherLive - Linux/Mac Build"
echo "===================================="
echo

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install pyvirtualcam onnxruntime

echo "[3/4] Build complete!"
echo
echo "To run desktop:  python run_desktop.py"
echo "To run CLI:      python run_cli.py --help"
echo
echo "To build Android:"
echo "  cd android_client"
echo "  buildozer android debug"
echo
