@echo off
echo Starting Iris Classifier Web App...
cd /d "%~dp0"
call .venv\Scripts\activate
python app.py
pause
