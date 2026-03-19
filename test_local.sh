#!/usr/bin/env bash
# test_local.sh — run the parity worker locally for testing
# ───────────────────────────────────────────────────────────
# Prerequisites:
#   1. ./prepare_models.sh        (populate models/ directory)
#   2. Edit test_input.json       (set real audio_url and gist_token)
#   3. Run this script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create venv if not present
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3.11 -m venv venv 2>/dev/null || python3 -m venv venv
fi

echo "Activating venv..."
# shellcheck disable=SC1091
source venv/bin/activate

# Install uv if needed
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing dependencies..."
uv pip install -r requirements.txt

# Run handler (RunPod test mode reads test_input.json via RUNPOD_WEBHOOK_GET_JOB)
export RUNPOD_WEBHOOK_GET_JOB="local_test"
export DIARIZATION_MODEL_DIR="$SCRIPT_DIR/models"
export WHISPER_CACHE_DIR="$SCRIPT_DIR/local_whisper_cache"

echo "Starting worker..."
echo "(send a job via test_input.json — press Ctrl+C to stop)"
python3 rp_handler.py

deactivate