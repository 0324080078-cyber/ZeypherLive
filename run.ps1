# ZeypherLive — PowerShell Activation
# Run this in PowerShell:

cd C:\Users\pc\Desktop\ZeypherLive
.\.venv\Scripts\Activate.ps1

# If activation policy blocks you, run this first (one time):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

python models/download_models.py
python run_desktop.py
