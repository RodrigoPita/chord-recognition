#!/bin/sh
set -e

echo "[start.sh] Container is running"
echo "[start.sh] Python: $(/app/.venv/bin/python --version)"
echo "[start.sh] Working dir: $(pwd)"
echo "[start.sh] PORT: ${PORT:-7860}"
echo "[start.sh] Launching uvicorn..."

exec /app/.venv/bin/uvicorn chord_rec.api:app \
    --host 0.0.0.0 \
    --port "${PORT:-7860}" \
    --log-level debug
