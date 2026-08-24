@echo off
echo ============================================
echo   ZeypherLive — Build Installer
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Installing PyInstaller...
.venv\Scripts\pip.exe install pyinstaller 2>nul

echo [2/3] Building executable...
.venv\Scripts\pyinstaller.exe ^
    --name ZeypherLive ^
    --onedir ^
    --noconsole ^
    --icon=NONE ^
    --add-data "config;config" ^
    --add-data "models;models" ^
    --add-data "saas_backend;saas_backend" ^
    --hidden-import=PyQt5 ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=mediapipe ^
    --hidden-import=pyaudio ^
    --hidden-import=pyvirtualcam ^
    --hidden-import=fastapi ^
    --hidden-import=uvicorn ^
    --hidden-import=passlib ^
    --hidden-import=jose ^
    run_desktop.py

echo [3/3] Build complete!
echo Output: dist\ZeypherLive\ZeypherLive.exe
echo.
pause
