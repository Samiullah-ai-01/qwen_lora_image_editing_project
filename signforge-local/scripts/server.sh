#!/bin/bash
# Start the SignForge server

HOST=${1:-"0.0.0.0"}
PORT=${2:-8000}

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting SignForge server on $HOST:$PORT..."
python -m signforge.server.app
