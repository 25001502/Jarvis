@echo off
echo Starting JARVIS GUI Interface...
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Please run:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment and run GUI
.venv\Scripts\python.exe jarvis_gui.py %*
