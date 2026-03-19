#!/usr/bin/env bash
# prepare_models.sh — populate models/ directory for Docker build
# ─────────────────────────────────────────────────────────────────
# Run this once from the repo root before building the Docker image.
# Two modes:
#   Option A (recommended): copy from Android APK assets (exact parity)
#   Option B: download from sherpa-onnx GitHub releases
#
# Usage:
#   ./prepare_models.sh             # auto-detect: tries assets first, then download
#   ./prepare_models.sh --from-assets
#   ./prepare_models.sh --download
# ─────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
ASSETS_DIR="$SCRIPT_DIR/../app/src/main/assets/diarization"

MODE="${1:---auto}"

mkdir -p "$MODELS_DIR/pyannote-segmentation-3-0"

copy_from_assets() {
    echo "Copying diarization models from Android assets..."
    if [ ! -d "$ASSETS_DIR" ]; then
        echo "ERROR: assets directory not found: $ASSETS_DIR"
        return 1
    fi

    local seg_src="$ASSETS_DIR/pyannote-segmentation-3-0/model.int8.onnx"
    local emb_src="$ASSETS_DIR/nemo_en_titanet_small.onnx"

    if [ ! -f "$seg_src" ] || [ ! -f "$emb_src" ]; then
        echo "ERROR: one or more model files not found under $ASSETS_DIR"
        return 1
    fi

    cp "$seg_src" "$MODELS_DIR/pyannote-segmentation-3-0/model.int8.onnx"
    cp "$emb_src" "$MODELS_DIR/nemo_en_titanet_small.onnx"
    echo "Copied:"
    echo "  model.int8.onnx → $(du -sh "$MODELS_DIR/pyannote-segmentation-3-0/model.int8.onnx" | cut -f1)"
    echo "  nemo_en_titanet_small.onnx → $(du -sh "$MODELS_DIR/nemo_en_titanet_small.onnx" | cut -f1)"
}

download_from_releases() {
    echo "Downloading diarization models from sherpa-onnx releases..."
    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        echo "ERROR: wget or curl is required for download mode"
        exit 1
    fi

    # Versions pinned to match the Android AAR (sherpa-onnx v1.12.x model set)
    # pyannote/segmentation-3.0 int8 ONNX
    local SEG_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    # NeMo TitaNet Small
    local EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recog-models/sherpa-onnx-nemo-en-titanet-small.tar.bz2"

    local TMP_DIR
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT

    _fetch() {
        local url="$1" dest="$2"
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "$dest" "$url"
        else
            curl -L --progress-bar -o "$dest" "$url"
        fi
    }

    echo "  Downloading segmentation model..."
    _fetch "$SEG_URL" "$TMP_DIR/seg.tar.bz2"
    tar -xjf "$TMP_DIR/seg.tar.bz2" -C "$TMP_DIR"
    local seg_onnx
    seg_onnx="$(find "$TMP_DIR" -name "model.int8.onnx" | head -1)"
    if [ -z "$seg_onnx" ]; then
        echo "ERROR: could not find model.int8.onnx in downloaded archive"
        exit 1
    fi
    cp "$seg_onnx" "$MODELS_DIR/pyannote-segmentation-3-0/model.int8.onnx"

    echo "  Downloading embedding model..."
    _fetch "$EMB_URL" "$TMP_DIR/emb.tar.bz2"
    tar -xjf "$TMP_DIR/emb.tar.bz2" -C "$TMP_DIR"
    local emb_onnx
    emb_onnx="$(find "$TMP_DIR" -name "*.onnx" | grep -i titanet | head -1)"
    if [ -z "$emb_onnx" ]; then
        # Fallback: any onnx file that isn't the segmentation model
        emb_onnx="$(find "$TMP_DIR" -name "*.onnx" | grep -v pyannote | grep -v model.int8 | head -1)"
    fi
    if [ -z "$emb_onnx" ]; then
        echo "ERROR: could not find embedding onnx in downloaded archive"
        exit 1
    fi
    cp "$emb_onnx" "$MODELS_DIR/nemo_en_titanet_small.onnx"

    echo "Downloaded:"
    echo "  model.int8.onnx → $(du -sh "$MODELS_DIR/pyannote-segmentation-3-0/model.int8.onnx" | cut -f1)"
    echo "  nemo_en_titanet_small.onnx → $(du -sh "$MODELS_DIR/nemo_en_titanet_small.onnx" | cut -f1)"
    echo
    echo "IMPORTANT: Verify the downloaded model checksums match the Android assets"
    echo "before using for parity testing. Mismatched models will produce different"
    echo "clustering results."
}

case "$MODE" in
    --from-assets)
        copy_from_assets
        ;;
    --download)
        download_from_releases
        ;;
    --auto)
        if copy_from_assets 2>/dev/null; then
            echo "Models copied from Android assets (exact parity guaranteed)."
        else
            echo "Android assets not found — falling back to download."
            download_from_releases
        fi
        ;;
    *)
        echo "Usage: $0 [--from-assets | --download | --auto]"
        exit 1
        ;;
esac

echo
echo "Models ready in $MODELS_DIR:"
find "$MODELS_DIR" -name "*.onnx" -exec ls -lh {} \;
