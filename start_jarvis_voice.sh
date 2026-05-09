#!/bin/bash

# Navigate to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Python virtual environment not found at .venv/bin/python"
    echo ""
    echo "Create it with Python 3.12 and install dependencies:"
    echo "  python3.12 -m venv .venv"
    echo "  .venv/bin/python -m pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "Starting Jarvis voice mode using .venv..."
.venv/bin/python jarvis.py "$@"
