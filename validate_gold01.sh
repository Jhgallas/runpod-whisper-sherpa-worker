#!/usr/bin/env bash
# validate_gold01.sh — submit GOLD-01 to RunPod and report cluster count + RTF
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./validate_gold01.sh <RUNPOD_API_KEY> <GIST_TOKEN> [GOLD01_PATH_OR_URL]
#
# GOLD01_PATH_OR_URL:
#   - Local file path  → uploaded to private GCS bucket (whisper-files-jh), signed URL passed to RunPod
#   - https:// URL     → used directly (skip upload)
#   - Omitted          → looks for GOLD-01 opus in ../golden_dataset/GOLD-01/
#
# Optional env vars:
#   GCS_KEY_FILE  — path to GCP service account JSON key
#                   (default: ../athefact-jhgallas-bec6b8fb5abd.json)
#
# Success criteria (from agent prompt):
#   num_speakers : 100 – 250  (target ~170)
#   rtf_total    : < 0.15
#   gist_url     : must be present

set -euo pipefail

RUNPOD_API_KEY="${1:?Usage: $0 <RUNPOD_API_KEY> <GIST_TOKEN> [GOLD01_PATH_OR_URL]}"
GIST_TOKEN="${2:?Usage: $0 <RUNPOD_API_KEY> <GIST_TOKEN> [GOLD01_PATH_OR_URL]}"
GOLD01_ARG="${3:-}"

ENDPOINT_ID="zcvayklfqbh1ov"
GCS_BUCKET="whisper-files-jh"

# Default GOLD-01 path relative to this script (worker repo lives inside the workspace)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_GOLD01_PATH="$(dirname "$SCRIPT_DIR")/golden_dataset/GOLD-01/GOLD-01_rec39_gsk-meeting_56min.opus"

# GCS key file: env var → default path next to gcs_upload.py
GCS_KEY_FILE="${GCS_KEY_FILE:-$(dirname "$SCRIPT_DIR")/athefact-jhgallas-bec6b8fb5abd.json}"
if [ ! -f "$GCS_KEY_FILE" ]; then
    echo "ERROR: GCS key file not found: $GCS_KEY_FILE" >&2
    echo "Set GCS_KEY_FILE env var or place the key at the default location." >&2
    exit 1
fi

# Track GCS object name so we can clean it up after the job
GCS_OBJECT_NAME=""

cleanup_gcs() {
    if [[ -n "$GCS_OBJECT_NAME" ]]; then
        echo "Cleaning up GCS object: $GCS_OBJECT_NAME" >&2
        python3 "$SCRIPT_DIR/gcs_upload.py" delete "$GCS_OBJECT_NAME" \
            --key-file "$GCS_KEY_FILE" --bucket "$GCS_BUCKET" || true
    fi
}
trap cleanup_gcs EXIT

# ── Resolve audio URL ─────────────────────────────────────────────────────────
if [[ -z "$GOLD01_ARG" ]]; then
    GOLD01_ARG="$DEFAULT_GOLD01_PATH"
fi

if [[ "$GOLD01_ARG" == http* ]]; then
    AUDIO_URL="$GOLD01_ARG"
    echo "Using provided URL: $AUDIO_URL"
else
    if [ ! -f "$GOLD01_ARG" ]; then
        echo "ERROR: File not found: $GOLD01_ARG" >&2
        echo "Pass a valid file path or https:// URL as the third argument." >&2
        echo "Expected default path: $DEFAULT_GOLD01_PATH" >&2
        exit 1
    fi
    FILE_SIZE_MB=$(du -m "$GOLD01_ARG" 2>/dev/null | cut -f1 || echo "?")
    echo "Uploading ${FILE_SIZE_MB}MB GOLD-01 audio to GCS bucket ${GCS_BUCKET}..."
    AUDIO_URL=$(python3 "$SCRIPT_DIR/gcs_upload.py" upload "$GOLD01_ARG" \
        --key-file "$GCS_KEY_FILE" --bucket "$GCS_BUCKET" --expiry 2)
    if [[ "$AUDIO_URL" != http* ]]; then
        echo "ERROR: GCS upload failed. Response: $AUDIO_URL" >&2
        exit 1
    fi
    # Record the object name (predictable path) so the EXIT trap can delete it
    GCS_OBJECT_NAME="runpod-audio/$(basename "$GOLD01_ARG")"
    echo "Signed URL obtained (2h expiry): ${AUDIO_URL:0:80}..."
fi

# ── Submit job ────────────────────────────────────────────────────────────────
echo ""
echo "Submitting GOLD-01 validation job to endpoint $ENDPOINT_ID..."
JOB_RESPONSE=$(curl -s -X POST \
  "https://api.runpod.io/v2/${ENDPOINT_ID}/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d "{
    \"input\": {
      \"audio_url\": \"${AUDIO_URL}\",
      \"ext\": \"opus\",
      \"language\": \"en\",
      \"language_candidates\": [\"en\", \"pt\"],
      \"model_size\": \"base.en\",
      \"cluster_threshold\": 0.5,
      \"diarization_mode\": \"whole_day\",
      \"recording_id\": \"GOLD-01-validation\",
      \"gist_token\": \"${GIST_TOKEN}\"
    }
  }")

JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c \
    "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || true)
if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: Failed to get job ID. Full response:" >&2
    echo "$JOB_RESPONSE" >&2
    exit 1
fi
echo "Job ID: $JOB_ID"
echo ""

# ── Poll for completion ───────────────────────────────────────────────────────
echo "Polling /status/${JOB_ID} every 15s (timeout: 60 min)..."
LAST_STATE=""
ELAPSED=0
MAX_WAIT=3600  # 60 min — GOLD-01 is 56 min × RTF 0.08 ≈ 4.5 min pipeline + margin

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    STATUS=$(curl -s \
      "https://api.runpod.io/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}")
    STATE=$(echo "$STATUS" | python3 -c \
        "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "UNKNOWN")

    if [[ "$STATE" != "$LAST_STATE" ]]; then
        echo "  $(date '+%H:%M:%S')  Status: $STATE"
        LAST_STATE="$STATE"
    fi

    if [ "$STATE" = "COMPLETED" ]; then
        GIST_URL=$(echo "$STATUS" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('output',{}).get('gist_url','NOT_FOUND'))" \
            2>/dev/null || echo "NOT_FOUND")
        NUM_SPEAKERS=$(echo "$STATUS" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('output',{}).get('num_speakers','?'))" \
            2>/dev/null || echo "?")
        RTF=$(echo "$STATUS" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('output',{}).get('timing',{}).get('rtf_total','?'))" \
            2>/dev/null || echo "?")
        COST=$(echo "$STATUS" | python3 -c \
            "import sys,json; d=json.load(sys.stdin); print(d.get('output',{}).get('cost_estimate_usd','?'))" \
            2>/dev/null || echo "?")

        echo ""
        echo "╔══════════════════════════════════════════╗"
        echo "║       GOLD-01 VALIDATION COMPLETE        ║"
        echo "╠══════════════════════════════════════════╣"
        printf "║  %-12s %-27s ║\n" "num_speakers" "$NUM_SPEAKERS   (target: ~170, pass: 100-250)"
        printf "║  %-12s %-27s ║\n" "rtf_total" "$RTF   (target: < 0.15)"
        printf "║  %-12s %-27s ║\n" "cost_est_usd" "$COST"
        echo "╠══════════════════════════════════════════╣"
        printf "║  Gist: %-35s ║\n" "$GIST_URL"
        echo "╚══════════════════════════════════════════╝"
        echo ""

        # Pass/fail on cluster count
        if python3 -c "n=int('$NUM_SPEAKERS'); exit(0 if 100<=n<=250 else 1)" 2>/dev/null; then
            echo "  PASS — cluster count in acceptable range [100, 250]"
        else
            echo "  WARN — cluster count $NUM_SPEAKERS outside [100, 250] — inspect Gist for root cause"
        fi

        echo ""
        echo "Next — compute LAA against ground truth:"
        echo "  python3 scripts/compute_t9_laa.py \\"
        echo "    --pred <GOLD-01-validation_diarization.json from Gist> \\"
        echo "    --gt golden_dataset/GOLD-01/ground_truth.json"
        exit 0

    elif [ "$STATE" = "FAILED" ]; then
        echo ""
        echo "=== JOB FAILED ==="
        echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"
        exit 1
    fi

    sleep 15
    ELAPSED=$((ELAPSED + 15))
done

echo "ERROR: Timed out after ${MAX_WAIT}s waiting for job $JOB_ID" >&2
exit 1
