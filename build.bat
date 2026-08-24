@echo off
echo ====================================
echo   ZeypherLive - Windows Build
echo ====================================
echo.

cd /d "%~dp0"
echo Working directory: %CD%
echo.

if exist ".venv" (
    echo Removing old venv from wrong location...
    rmdir /s /q ".venv" 2>nul
)

echo [1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [!] Failed. Trying admin...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%~dp0\" && python -m venv .venv' -Verb RunAs"
    timeout /t 5
)

if not exist ".venv\Scripts\activate.bat" (
    echo [!] Venv not created. Run as Administrator manually.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Venv: %CD%\.venv
echo.

echo [2/4] Setting pip to Chinese mirror...
python -m pip install --upgrade pip
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo.

echo [3/4] Installing packages...
python -m pip install opencv-python numpy mediapipe PyQt5 scipy Pillow requests aiohttp websockets pyaudio pyvirtualcam onnxruntime
echo.

echo [4/4] Done!
echo.
echo ====================================
echo   Run: python run_desktop.py
echo ====================================
pause
