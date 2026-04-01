@echo off
echo Starting AFM Logger...
cd /d "%~dp0"
uvicorn server:app --reload --port 8000
pause
