#!/bin/bash
echo "Starting JARVIS GUI Interface..."
echo

# Check if virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "Virtual environment not found. Please run:"
    echo "  python3.12 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and run GUI
.venv/bin/python jarvis_gui.py "$@"
