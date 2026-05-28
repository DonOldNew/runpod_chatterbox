#!/bin/bash
# Start the AI Call Engine for n8n integration
# Usage: ./start_engine.sh [port]

PORT=${1:-5050}
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting AI Call Engine on port $PORT..."
echo "Directory: $DIR"

cd "$DIR"

# Check if already running
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "Port $PORT is already in use. Stopping existing process..."
    kill $(lsof -ti:$PORT) 2>/dev/null
    sleep 1
fi

# Start the engine
python3 call_engine.py serve --port $PORT
