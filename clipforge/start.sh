#!/usr/bin/env bash
# macOS / Linux launcher: creates a private environment on first run, then starts ClipForge.
cd "$(dirname "$0")" || exit 1
command -v python3 >/dev/null || { echo "Python 3.10+ is required."; exit 1; }
command -v ffmpeg  >/dev/null || echo "Warning: ffmpeg not found (brew install ffmpeg  /  sudo apt install ffmpeg)."
[ -d .venv ] || python3 -m venv .venv
if [ ! -f .venv/installed.flag ]; then
  .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt && touch .venv/installed.flag || exit 1
fi
exec .venv/bin/python app.py "$@"
