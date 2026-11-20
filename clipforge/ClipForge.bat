@echo off
setlocal
cd /d "%~dp0"
title ClipForge

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
  echo and tick "Add Python to PATH" during setup, then run this file again.
  pause
  exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found.
  choice /m "Install it now with winget"
  if not errorlevel 2 (
    winget install Gyan.FFmpeg
    echo.
    echo ffmpeg was installed. Please CLOSE this window and double-click ClipForge.bat again.
    pause
    exit /b 0
  )
  echo Continuing without ffmpeg. The app will not be able to make clips until it is installed.
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: creating a private Python environment...
  python -m venv .venv
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)

if not exist ".venv\installed.flag" (
  echo Installing libraries, this can take a few minutes the first time...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 ( echo Install failed. Check your internet connection and try again. & pause & exit /b 1 )
  echo done> ".venv\installed.flag"
)

echo.
echo Starting ClipForge. Keep this window open while you use it. Close it to stop the app.
".venv\Scripts\python.exe" app.py
pause
