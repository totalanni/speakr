#!/bin/bash
set -e

# Create necessary directories
mkdir -p /data/uploads /data/instance
chmod 755 /data/uploads /data/instance

# Initialize the database if it doesn't exist
set +e
if [ ! -f /data/instance/transcriptions.db ]; then
    echo "Database doesn't exist. Creating new database..."
    python -c "from src.app import app, db; app.app_context().push(); db.create_all()"
    echo "Database created successfully."
else
    echo "Database exists. Checking for schema updates..."
    python -c "from src.app import app; app.app_context().push()" || true
fi
set -e

# Check if we need to create an admin user (regardless of whether the database exists)
if [ -n "$ADMIN_USERNAME" ] && [ -n "$ADMIN_EMAIL" ] && [ -n "$ADMIN_PASSWORD" ]; then
    echo "Creating admin user using environment variables..."
    cd /app && python scripts/docker_create_admin.py
fi

# ---------------------------------------------------------------------------
# Start the NATIVE NeMo-Speech.cpp CPU runtime (in-container)
#   asr:  Nemotron 3.5 ASR 0.6B (q8_0 GGUF, RNNT)
#   diar: Nemotron 3 Diarization (q8_0 GGUF)
# This is the CPU-only ASR engine Speakr's /asr blueprint talks to.
# ---------------------------------------------------------------------------
echo ">>> ENTERING NEMO BLOCK" >&2
NEMO_BIN=/opt/nemo-speech/bin/nemo-speech
NEMO_LIB=/opt/nemo-speech/lib
NEMO_ASR_MODEL=/opt/nemo-speech/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
NEMO_DIAR_MODEL=/opt/nemo-speech/models/Nemotron-3-Diarization.q8_0.gguf
NEMO_PORT=${NEMO_ASR_PORT:-8080}

if [ -x "$NEMO_BIN" ] && [ -f "$NEMO_ASR_MODEL" ]; then
    set +e
    echo "Starting NeMo-Speech.cpp serve (port $NEMO_PORT)..."
    # Kill any stale instance first (idempotent across restarts)
    pkill -f "nemo-speech serve" 2>/dev/null || true
    sleep 1
    # Auto-detect GPU: if CUDA lib exists and --gpus is passed to container, use cuda
    CUDA_DEV="cpu"
    if [ -f "$NEMO_LIB/libggml-cuda.so.0" ] && [ -f "$NEMO_LIB/libcudart.so.12" ]; then
        CUDA_DEV="cuda"
    fi
    echo "Using device: $CUDA_DEV"
    NEMO_SVC_ARGS="--host 127.0.0.1 --port $NEMO_PORT --device $CUDA_DEV --asr-model $NEMO_ASR_MODEL --diar-model $NEMO_DIAR_MODEL --threads 8 --no-ui"
    if [ "$CUDA_DEV" = "cpu" ]; then
        NEMO_SVC_ARGS="$NEMO_SVC_ARGS --threads 32"
    fi
    # Reduce VRAM for CUDA: fewer state arena slots + disable VMM
    NEMO_SVC_ARGS="$NEMO_SVC_ARGS --asr.batching.state_arena_slots 1"
    export GGML_CUDA_VMM=0
    nohup env LD_LIBRARY_PATH="$NEMO_LIB" $NEMO_BIN serve $NEMO_SVC_ARGS \
        > /tmp/nemo_serve.log 2>&1 &
    echo "NeMo serve PID: $!"

    # Wait for NeMo to be healthy (models load on boot; give it time)
    for i in $(seq 1 180); do
        if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${NEMO_PORT}/health', timeout=3)" 2>/dev/null; then
            echo "NeMo serve is healthy."
            break
        fi
        sleep 2 || true
    done || true
    set -e
else
    echo "WARNING: NeMo-Speech.cpp runtime not found — /asr will fail. Skipping."
fi

# Speakr uses its OWN native /asr pipeline (Nemotron + wespeaker voiceprints)
echo "DEBUG: setting env vars" >&2
export TRANSCRIPTION_CONNECTOR=asr_endpoint
export ASR_BASE_URL=http://127.0.0.1:8899
export ASR_DIARIZE=true
export ASR_RETURN_SPEAKER_EMBEDDINGS=true
export ASR_TIMEOUT=1800
export NEMO_ASR_URL=http://127.0.0.1:${NEMO_PORT}
echo "DEBUG: env vars set" >&2

# Start the application
exec "$@"
