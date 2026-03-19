# RunPod Parity Worker — whisper.cpp + Sherpa ONNX + ORT CUDA equivalent
# ─────────────────────────────────────────────────────────────────────────
# GPU-accelerated: CUDA 12.1 runtime base provides libcublas.so.12 and cuDNN 8
# so ctranslate2 (faster-whisper) uses GPU automatically at runtime.
#
# Diarization engine priority:
#   1. onnxruntime-gpu:
#        - CUDAExecutionProvider if GPU supports it (Kepler+ / sm_37+)
#        - Falls back to CPUExecutionProvider transparently
#   2. sherpa-onnx CPU fallback (statically-linked CPU onnxruntime inside)
#
# Build prerequisites — populate models/ in the build context:
#   Option A (recommended): copy from Android assets
#     cp app/src/main/assets/diarization/pyannote-segmentation-3-0/model.int8.onnx \
#        runpod-worker-whisper-diarization-master/models/pyannote-segmentation-3-0/
#     cp app/src/main/assets/diarization/nemo_en_titanet_small.onnx \
#        runpod-worker-whisper-diarization-master/models/
#   Option B: run ./prepare_models.sh (downloads from sherpa-onnx releases)
#
# Build:
#   cd runpod-worker-whisper-diarization-master
#   docker build -t whisper-parity-worker .

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ── uv for fast package installation ──────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ── Python dependencies ────────────────────────────────────────────────────────
# torch with CUDA 12.1 — needed for torch.cuda.is_available() detection
# ctranslate2 (bundled with faster-whisper) handles GPU inference via libcublas
RUN uv pip install --system --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system --no-cache-dir \
    "faster-whisper>=1.0.0" \
    "sherpa-onnx>=1.12.0" \
    "onnxruntime-gpu>=1.18.0" \
    "kaldi-native-fbank>=1.17" \
    runpod \
    requests \
    soundfile \
    "numpy<2.0" \
    langdetect

# ── Bake Whisper models (downloads at build time → baked into image layer) ─────
# Models are cached to /app/whisper_models to avoid runtime downloads.
ENV WHISPER_CACHE_DIR=/app/whisper_models
# Force device=cpu at build time — build machines have no GPU
RUN python3 -c "from faster_whisper import WhisperModel; \
    WhisperModel('tiny.en', device='cpu', download_root='/app/whisper_models')"
RUN python3 -c "from faster_whisper import WhisperModel; \
    WhisperModel('base.en', device='cpu', download_root='/app/whisper_models')"
RUN python3 -c "from faster_whisper import WhisperModel; \
    WhisperModel('small.en', device='cpu', download_root='/app/whisper_models')"

# ── Diarization ONNX models ────────────────────────────────────────────────────
# Source: app/src/main/assets/diarization/ (or downloaded via prepare_models.sh)
# Checksums must match the Android AAR assets for numerical parity.
COPY models/pyannote-segmentation-3-0/ /app/models/pyannote-segmentation-3-0/
COPY models/nemo_en_titanet_small.onnx /app/models/
ENV DIARIZATION_MODEL_DIR=/app/models

# ── Application ────────────────────────────────────────────────────────────────
COPY rp_handler.py /app/

CMD ["python3", "-u", "/app/rp_handler.py"]
