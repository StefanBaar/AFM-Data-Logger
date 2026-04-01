#!/usr/bin/env bash
echo "Starting AFM Logger..."
cd "$(dirname "$0")"
uvicorn server:app --reload --port 8000
