@echo off
echo ========================================================
echo [AEROFLOOD AI] Securely booting the FastAPI framework...
echo ========================================================

set PYTHON=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe

echo Installing/verifying dependencies...
%PYTHON% -m pip install fastapi uvicorn tensorflow pillow numpy requests pydantic scikit-learn xgboost --quiet

echo Starting backend...
cd /d %~dp0
%PYTHON% app.py

pause
