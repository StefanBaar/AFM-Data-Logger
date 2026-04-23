@echo off
REM AFM Logger — Windows launcher
REM Double-click this file to start

cd /d "%~dp0"

echo.
echo   AFM Logger starting at http://localhost:8000
echo   Close this window to stop
echo.

REM Install dependencies if needed
python -c "import uvicorn" 2>nul || pip install -r requirements.txt

start "" "http://localhost:8000"
uvicorn server:app --port 8000
pause
