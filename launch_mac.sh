#!/usr/bin/env bash
# AFM Logger — macOS / Linux launcher
# Double-click this file, or run: bash launch_mac.sh
set -e
cd "$(dirname "$0")"

# Install dependencies if needed
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "  AFM Logger starting at http://localhost:8000"
echo "  Press Ctrl+C to stop"
echo ""
open "http://localhost:8000" 2>/dev/null || true
uvicorn server:app --port 8000
