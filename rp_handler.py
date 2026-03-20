"""
RunPod Parity Worker — whisper.cpp + Sherpa ONNX equivalent
============================================================
Produces output format-identical to the local Android pipeline
(whisper.cpp + Sherpa ONNX).

Architecture:
  - Transcription:     faster-whisper (CPU-first; GPU if ctranslate2 detects CUDA)
  - Diarization:       sherpa-onnx Python package — same C++ library as Android AAR,
                       same FastClustering implementation and model weights.
  - Speaker alignment: exact port of DiarizationService.assignSpeakersToLines()
  - Outputs:           {recording_id}_transcript_{model}.txt
                       {recording_id}_diarization.json
                       {recording_id}_labeled_{model}.txt
                       {recording_id}_segment_metadata_{model}.json
                       {recording_id}_language_flags_{model}.json
                       {recording_id}_run_summary.json
  All outputs are uploaded to a single GitHub Gist per job.

Language detection (based on native-lib.cpp micro_detect_language()):
  The Android implementation runs per-utterance constrained language detection:
    - Uses first 3 seconds of each voice chunk
    - Picks the best language from a constrained candidate set (e.g. ["en", "pt"])
    - Confidence threshold: 0.5 (returns "" → keeps previous language if below)
    - Per-utterance: each voice chunk may switch language independently
  Python approximation (faster-whisper limitation):
    - Single-pass detection from the first 30 seconds of the concatenated audio
    - Restricted to provided language_candidates with the same 0.5 threshold
    - Falls back to the first candidate if confidence is below threshold
    - Per-utterance switching is not reproducible via the faster-whisper API;
      this is an expected and documented divergence.

Diarization parameters (matching DiarizationService.kt):
  min_duration_on  = 0.3s   (minDurationOn in Android)
  min_duration_off = 0.5s   (minDurationOff in Android)
  cluster_threshold = 0.5   (confirmed best for Android C1 model combination)
  embedding model: nemo_en_titanet_small.onnx
  segmentation model: pyannote-segmentation-3-0/model.int8.onnx (int8 = C1 default)
"""

import datetime
import json
import logging
import os
import resource
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import requests
import runpod
import soundfile as sf
from faster_whisper import WhisperModel

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    logging.warning("sherpa_onnx not available — diarization will be skipped")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logging.warning("onnxruntime not available — ORT diarization path disabled")

try:
    import kaldi_native_fbank as knf
    KNF_AVAILABLE = True
except ImportError:
    KNF_AVAILABLE = False
    logging.warning("kaldi_native_fbank not available — falling back to scipy mel features")

try:
    from langdetect import detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available — language flags will be empty")

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000

# Env-configurable paths (set in Dockerfile for baked-in models)
DIARIZATION_MODEL_DIR: str = os.environ.get(
    "DIARIZATION_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)
WHISPER_CACHE_DIR: Optional[str] = os.environ.get("WHISPER_CACHE_DIR")

# Diarization parameters — must match DiarizationService.kt exactly
MIN_DURATION_ON: float = 0.3    # minDurationOn
MIN_DURATION_OFF: float = 0.5   # minDurationOff
DEFAULT_THRESHOLD: float = 0.5  # confirmed best for C1 combination

# Set DIARIZATION_DEBUG=1 to enable extra assertions (L2 norm check) in the ORT path.
DIARIZATION_DEBUG: bool = os.environ.get("DIARIZATION_DEBUG", "0") == "1"

# Per-hour GPU prices on RunPod serverless (used for cost_estimate_usd in run_summary).
# Source: https://www.runpod.io/serverless-gpu-pricing — check periodically.
GPU_PRICE_PER_HOUR: dict = {
    "NVIDIA GeForce RTX 4090": 0.59,
    "NVIDIA RTX A5000": 0.58,
    "NVIDIA L4": 0.58,
    "NVIDIA RTX 3090": 0.58,
    "NVIDIA A100": 1.99,
    "NVIDIA L40S": 1.22,
    "NVIDIA H100": 3.49,
}

# Android OOM threshold: 40 M samples ≈ 41.7 min at 16 kHz
FIXED_CHUNK_SAMPLES: int = 40_000_000

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Runtime detection ──────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    """Detect CUDA via ctranslate2 (faster-whisper's actual inference backend)."""
    try:
        import ctranslate2  # type: ignore
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        pass
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False

CPU_ONLY: bool = not _gpu_available()
COMPUTE_TYPE: str = "int8"  # int8 works on all CUDA devices; int8_float16 requires Volta+
log.info("Runtime: %s (compute_type=%s)", "CPU" if CPU_ONLY else "GPU/CUDA", COMPUTE_TYPE)


def _get_worker_info() -> dict:
    """
    Collect RunPod worker environment and GPU hardware info for cost/billing analysis.

    Cost calculation:
      billed_s  = executionTime_ms / 1000  (from RunPod API response — caller-side only)
      cost_usd  = billed_s * gpu_price_per_second
      GPU price per second: look up gpu_name at https://www.runpod.io/serverless-gpu-pricing

    Note: delayTime (queue wait) and executionTime are only visible in the RunPod API
    response envelope and cannot be read from inside the worker container.
    """
    info: dict = {}

    # RunPod-injected environment variables
    for env_key, out_key in [
        ("RUNPOD_POD_ID",      "pod_id"),
        ("RUNPOD_ENDPOINT_ID", "endpoint_id"),
        ("RUNPOD_GPU_COUNT",   "gpu_count_env"),
        ("RUNPOD_CPU_COUNT",   "cpu_count_env"),
        ("RUNPOD_MEM_GB",      "mem_gb_env"),
    ]:
        val = os.environ.get(env_key, "")
        if val:
            info[out_key] = val

    # Python-visible CPU count
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass

    # GPU hardware via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "memory_mb": int(parts[1]) if parts[1].isdigit() else parts[1],
                        "driver_version": parts[2],
                        "cuda_version": parts[3],
                    })
            if gpus:
                info["gpus"] = gpus
    except Exception as exc:
        log.debug("nvidia-smi unavailable: %s", exc)

    return info


# ── Worker info (collected once at module load) ────────────────────────────────

WORKER_INFO: dict = _get_worker_info()
log.info("Worker info: %s", WORKER_INFO)


def _log_vram(label: str) -> None:
    """Log current VRAM utilisation via nvidia-smi (best-effort, no-op on CPU)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            log.info("VRAM [%s]: %s (used/free/total MiB)", label, r.stdout.strip())
    except Exception:
        pass


def _estimate_cost(pipeline_s: float) -> Optional[float]:
    """
    Return a USD cost estimate using pipeline_s as a proxy for billed executionTime.

    Uses the GPU name captured in WORKER_INFO at startup against GPU_PRICE_PER_HOUR.
    Returns None when the GPU model is not in the table — log the GPU name so the
    caller can look it up manually.
    """
    gpus = WORKER_INFO.get("gpus", [])
    if not gpus:
        return None
    gpu_name = gpus[0].get("name", "")
    price = GPU_PRICE_PER_HOUR.get(gpu_name)
    if price is None:
        log.warning("GPU '%s' not in GPU_PRICE_PER_HOUR — cost_estimate_usd=null", gpu_name)
        return None
    return round((pipeline_s / 3600.0) * price, 4)


# ── Lazily-cached Whisper models (loaded once per worker process) ──────────────

_whisper_cache: dict[str, WhisperModel] = {}


def get_whisper_model(model_size: str) -> WhisperModel:
    """Return a cached WhisperModel, loading it on first use."""
    if model_size not in _whisper_cache:
        log.info("Loading WhisperModel '%s' (device=auto, compute=%s)...", model_size, COMPUTE_TYPE)
        t0 = time.time()
        kwargs: dict = {"device": "auto", "compute_type": COMPUTE_TYPE}
        if WHISPER_CACHE_DIR:
            kwargs["download_root"] = WHISPER_CACHE_DIR
        _whisper_cache[model_size] = WhisperModel(model_size, **kwargs)
        log.info("WhisperModel '%s' loaded in %.1fs", model_size, time.time() - t0)
    return _whisper_cache[model_size]


# ── Gist utilities ─────────────────────────────────────────────────────────────

def _gist_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def verify_gist_upload(token: str) -> None:
    """
    Create a test Gist, confirm creation, then delete it.
    Raises on any failure.  Call this BEFORE starting audio processing.
    """
    headers = _gist_headers(token)
    payload = {
        "description": "RunPod worker pre-flight connectivity check (delete me)",
        "public": False,
        "files": {"preflight.txt": {"content": "ok"}},
    }
    resp = requests.post(
        "https://api.github.com/gists",
        json=payload, headers=headers, timeout=30,
    )
    resp.raise_for_status()
    gist_id = resp.json().get("id")
    if not gist_id:
        raise RuntimeError("Gist creation returned no ID")
    requests.delete(
        f"https://api.github.com/gists/{gist_id}",
        headers=headers, timeout=30,
    ).raise_for_status()
    log.info("Gist connectivity verified")


def upload_to_gist(
    token: str,
    files: dict[str, str],
    description: str,
    timing: dict,
) -> str:
    """
    Upload files dict to a GitHub Gist with 3 retries (exponential backoff).
    Returns the Gist HTML URL.
    """
    headers = _gist_headers(token)
    payload = {
        "description": description,
        "public": False,
        "files": {name: {"content": content or "(empty)"} for name, content in files.items()},
    }
    t0 = time.time()
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.github.com/gists",
                json=payload, headers=headers, timeout=120,
            )
            resp.raise_for_status()
            gist_url: str = resp.json()["html_url"]
            timing["gist_upload_s"] = round(time.time() - t0, 3)
            log.info("Gist uploaded: %s", gist_url)
            return gist_url
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                log.warning("Gist upload attempt %d failed: %s — retrying in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
    raise last_exc


# ── File metadata ──────────────────────────────────────────────────────────────

def get_file_metadata(path: str) -> dict:
    """Extract duration, creation_time, size via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.warning("ffprobe failed for %s: %s", path, result.stderr[:200])
        return {"filename": os.path.basename(path), "duration_s": 0.0}
    try:
        fmt = json.loads(result.stdout)["format"]
        return {
            "filename": os.path.basename(path),
            "duration_s": float(fmt.get("duration", 0)),
            "creation_time_utc": fmt.get("tags", {}).get("creation_time"),
            "size_bytes": int(fmt.get("size", 0)),
            "format_name": fmt.get("format_name"),
        }
    except Exception as exc:
        log.warning("ffprobe parse error for %s: %s", path, exc)
        return {"filename": os.path.basename(path), "duration_s": 0.0}


# ── Audio utilities ────────────────────────────────────────────────────────────

def download_audio(url: str, ext: str, dest_dir: str) -> str:
    """Download an audio file from URL. Returns local path."""
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    # Honour extension embedded in URL if present
    url_basename = url.split("?")[0].rstrip("/").split("/")[-1]
    if "." in url_basename:
        ext = url_basename.rsplit(".", 1)[-1]
    fd, path = tempfile.mkstemp(suffix=f".{ext}", dir=dest_dir)
    with os.fdopen(fd, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65_536):
            f.write(chunk)
    log.info("Downloaded %s → %.1f MB", url_basename, os.path.getsize(path) / (1024 * 1024))
    return path


def concat_to_wav(input_paths: list[str], output_path: str) -> None:
    """
    Concatenate audio files into a single 16 kHz mono WAV using ffmpeg.
    No gap is inserted between files — timestamps are continuous.
    (The +1.0 s inter-file gap is applied later in normalizeMultiFileTranscript()
    on the Kotlin side and must NOT be replicated here.)
    """
    if len(input_paths) == 1:
        cmd = [
            "ffmpeg", "-y", "-i", input_paths[0],
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {r.stderr[-500:]}")
        return

    # Multi-file: write concat list file then use ffmpeg concat demuxer
    fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            for p in input_paths:
                escaped = p.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1_800)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-500:]}")
    finally:
        if os.path.exists(list_path):
            os.unlink(list_path)


def load_wav_as_float32(path: str) -> np.ndarray:
    """Load a 16 kHz mono WAV as a float32 numpy array for Sherpa ONNX."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


# ── Timestamp formatting ───────────────────────────────────────────────────────

def seconds_to_hms(secs: float) -> str:
    """Convert float seconds to [HH:MM:SS] (Android canonical transcript format)."""
    s = int(secs)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"[{h:02d}:{m:02d}:{sec:02d}]"


# ── Language detection ─────────────────────────────────────────────────────────

def detect_constrained_language(
    audio_path: str,
    model: WhisperModel,
    candidates: list[str],
    fallback: str,
    confidence_threshold: float = 0.5,
) -> tuple[str, float]:
    """
    Detect recording language from the first 30 seconds, restricted to candidates.
    Approximates Android's micro_detect_language() per-utterance detection.

    Android implementation (native-lib.cpp):
      - Uses first 3 seconds of each voice utterance (DETECT_SAMPLES = 3 * 16000)
      - Picks best language from constrained lang-ID set (candidate IDs only)
      - Confidence threshold 0.5; returns "" (→ keep previous) when below threshold
      - Per-utterance: language can switch mid-recording

    Python approximation (limitation of faster-whisper transcription API):
      - Single pass from first 30 seconds of audio (not per-utterance)
      - Same 0.5 threshold and candidate restriction logic
      - Per-utterance switching cannot be reproduced; this divergence is documented

    Returns: (detected_language_code, confidence_float)
    """
    try:
        segs_gen, info = model.transcribe(
            audio_path,
            duration=30.0,
            language=None,          # unconstrained — we do the restriction ourselves
            beam_size=1,
            condition_on_previous_text=False,
        )
        # Consume the generator to trigger the detection pass
        for _ in segs_gen:
            break
        detected = info.language
        confidence = info.language_probability
    except Exception as exc:
        log.warning("Language detection failed: %s — using fallback '%s'", exc, fallback)
        return fallback, 0.0

    if confidence < confidence_threshold:
        log.info(
            "Language confidence too low (%.3f < %.2f) — using fallback '%s'",
            confidence, confidence_threshold, fallback,
        )
        return fallback, confidence

    if detected in candidates:
        log.info("Detected language: %s (p=%.3f)", detected, confidence)
        return detected, confidence

    # Romance-family fallback: map to 'pt' if that is a candidate
    PT_FAMILY = {"es", "it", "fr", "ro", "ca", "gl"}
    if "pt" in candidates and detected in PT_FAMILY:
        remapped_conf = confidence * 0.7
        log.info(
            "Detected '%s' (romance family) → remapped to 'pt' (conf=%.3f)", detected, remapped_conf,
        )
        return "pt", remapped_conf

    log.info(
        "Detected '%s' not in candidates %s — using first candidate '%s'",
        detected, candidates, candidates[0],
    )
    return candidates[0], confidence * 0.5


# ── Transcription ──────────────────────────────────────────────────────────────

def transcribe_audio(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    timing: dict,
) -> tuple[list[dict], dict, str, float]:
    """
    Transcribe audio with faster-whisper.

    Transcription parameters are set to match whisper.cpp Android defaults:
      beam_size=5                    (whisper.cpp default)
      condition_on_previous_text=True (whisper.cpp default)
      vad_filter=True                (energy-based VAD approximation)
      min_silence_duration_ms=2000   (matches Android SILENCE_GAP_SAMPLES ~2s)
      speech_pad_ms=500              (matches Android MIN_SPEECH_REGION_MS)

    Returns:
      transcript_lines     List of {"timestamp_s", "timestamp_str", "text"}
      segment_metadata     Dict with per-segment Whisper confidence signals
      detected_language    Language code string
      language_probability Float confidence
    """
    model = get_whisper_model(model_size)
    log.info(
        "Transcribing with model=%s language=%s beam=5 cpu=%s",
        model_size, language or "auto", CPU_ONLY,
    )
    t0 = time.time()
    segs_iter, info = model.transcribe(
        audio_path,
        language=language or None,
        beam_size=5,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=2_000,
            speech_pad_ms=500,
        ),
    )

    transcript_lines: list[dict] = []
    segment_metadata_list: list[dict] = []
    for seg in segs_iter:
        ts = seconds_to_hms(seg.start)
        transcript_lines.append({
            "timestamp_s": seg.start,
            "timestamp_str": ts,
            "text": seg.text.strip(),
        })
        segment_metadata_list.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "avg_logprob": round(seg.avg_logprob, 4),
            "compression_ratio": round(seg.compression_ratio, 4),
            "temperature": round(seg.temperature, 4) if seg.temperature is not None else None,
            "text": seg.text.strip(),
        })

    elapsed = time.time() - t0
    model_key = model_size.replace(".", "_")
    timing[f"transcription_{model_key}_s"] = round(elapsed, 3)
    log.info(
        "Transcription done in %.1fs — %d segments, detected=%s (p=%.2f)",
        elapsed, len(transcript_lines), info.language, info.language_probability,
    )

    segment_metadata = {
        "model": model_size,
        "detected_language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": segment_metadata_list,
    }
    return transcript_lines, segment_metadata, info.language, info.language_probability


# ── ORT CUDA diarization ──────────────────────────────────────────────────────

# Cached ORT sessions (loaded once per worker process)
_ort_seg_session = None
_ort_emb_session = None
_ort_provider: Optional[str] = None


def _release_ort_sessions() -> None:
    """Reset cached ORT sessions so VRAM is freed. Sessions reload on next call."""
    global _ort_seg_session, _ort_emb_session, _ort_provider
    _ort_seg_session = None
    _ort_emb_session = None
    _ort_provider = None


def _ort_cuda_available() -> bool:
    """Return True if onnxruntime CUDAExecutionProvider is accessible."""
    if not ORT_AVAILABLE:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _get_ort_sessions():
    """
    Load (once) ORT inference sessions for segmentation + embedding models.
    Returns (seg_session, emb_session, provider_str) where provider_str is
    "CUDA" or "CPU".

    Model selection for the ORT path:
      - Segmentation: model.onnx (FP32) preferred over model.int8.onnx.
        INT8 quantised ops are not fully supported by the CUDA ExecutionProvider
        and cause silent node-level fallback to CPU, defeating GPU acceleration.
        FP32 runs entirely on CUDA with no fallbacks.
      - Embedding: nemo_en_titanet_small.onnx (already FP32 in the Android AAR).
    """
    global _ort_seg_session, _ort_emb_session, _ort_provider
    if _ort_seg_session is not None:
        return _ort_seg_session, _ort_emb_session, _ort_provider

    # Prefer FP32 segmentation model for full CUDA EP coverage.
    seg_path_fp32 = os.path.join(
        DIARIZATION_MODEL_DIR, "pyannote-segmentation-3-0", "model.onnx"
    )
    seg_path_int8 = os.path.join(
        DIARIZATION_MODEL_DIR, "pyannote-segmentation-3-0", "model.int8.onnx"
    )
    if os.path.exists(seg_path_fp32):
        seg_path = seg_path_fp32
        log.info("ORT: using FP32 segmentation model (full CUDA coverage)")
    else:
        seg_path = seg_path_int8
        log.warning(
            "ORT: FP32 model not found at %s — using INT8. "
            "INT8 ops may silently fall back to CPU inside CUDA EP. "
            "Copy model.onnx to %s for guaranteed GPU usage.",
            seg_path_fp32,
            os.path.dirname(seg_path_fp32),
        )
    emb_path = os.path.join(DIARIZATION_MODEL_DIR, "nemo_en_titanet_small.onnx")

    cuda_ok = _ort_cuda_available()
    # Explicit CUDA provider options: device 0, arena-based allocator
    cuda_provider_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    providers = (
        [("CUDAExecutionProvider", cuda_provider_opts), "CPUExecutionProvider"]
        if cuda_ok
        else ["CPUExecutionProvider"]
    )
    log.info("ORT: initialising sessions (preferred providers: %s)", providers)

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4

    seg_session = ort.InferenceSession(seg_path, sess_options=sess_opts, providers=providers)
    emb_session = ort.InferenceSession(emb_path, sess_options=sess_opts, providers=providers)

    # Log actual providers chosen by ORT
    seg_prov = seg_session.get_providers()
    emb_prov = emb_session.get_providers()
    log.info("ORT segmentation providers: %s", seg_prov)
    log.info("ORT embedding   providers: %s", emb_prov)

    # Log model metadata so timing data is unambiguous
    seg_meta = seg_session.get_modelmeta().custom_metadata_map
    emb_meta = emb_session.get_modelmeta().custom_metadata_map
    log.info("ORT seg model metadata: %s", seg_meta)
    log.info("ORT emb model metadata: %s", emb_meta)

    for inp in seg_session.get_inputs():
        log.info("ORT seg  input  %-20s shape=%s type=%s", inp.name, inp.shape, inp.type)
    for out in seg_session.get_outputs():
        log.info("ORT seg  output %-20s shape=%s type=%s", out.name, out.shape, out.type)
    for inp in emb_session.get_inputs():
        log.info("ORT emb  input  %-20s shape=%s type=%s", inp.name, inp.shape, inp.type)
    for out in emb_session.get_outputs():
        log.info("ORT emb  output %-20s shape=%s type=%s", out.name, out.shape, out.type)

    actual = "CUDA" if (cuda_ok and seg_prov[0] == "CUDAExecutionProvider") else "CPU"
    if cuda_ok and actual != "CUDA":
        log.warning(
            "ORT: CUDA was requested but session is running on %s. "
            "Actual providers: seg=%s emb=%s. "
            "This usually means the segmentation model contains INT8 ops unsupported "
            "by the CUDA EP — ensure model.onnx (FP32) is present in the models dir.",
            actual, seg_prov, emb_prov,
        )
    _ort_provider = actual
    _ort_seg_session = seg_session
    _ort_emb_session = emb_session
    log.info("ORT: diarization will run on %s", actual)
    return _ort_seg_session, _ort_emb_session, _ort_provider


def _build_powerset_map(num_speakers: int, powerset_max_classes: int) -> list:
    """
    Map each powerset class index → frozenset of active speaker indices.

    Class 0 = empty set (no speaker active).
    Follows the pyannote-audio PowersetActivation enumeration used when exporting
    pyannote/segmentation-3.0 to ONNX (same scheme as sherpa-onnx expects).
    """
    from itertools import combinations
    mapping: list = [frozenset()]
    for k in range(1, powerset_max_classes + 1):
        for combo in combinations(range(num_speakers), k):
            mapping.append(frozenset(combo))
    return mapping


def _run_pyannote_ort(
    seg_session,
    samples: np.ndarray,
    num_speakers: int,
    powerset_max_classes: int,
    window_size: int,
    window_shift: int,
    sr: int = SAMPLE_RATE,
) -> list:
    """
    Slide a window over `samples`, run the pyannote segmentation ONNX model in
    each window, decode powerset output to per-speaker binary activity, and
    return a flat list of (start_s, end_s, unique_speaker_id) tuples.

    unique_speaker_id is window-local: it uniquely identifies a speaker within
    a window but NOT across windows.  FastClustering later assigns global IDs
    based on embedding similarity.
    """
    powerset_map = _build_powerset_map(num_speakers, powerset_max_classes)
    input_name = seg_session.get_inputs()[0].name

    n = len(samples)
    raw_segs: list = []
    speaker_id_offset = 0
    window_count = 0
    start_idx = 0

    while start_idx < n:
        end_idx = min(start_idx + window_size, n)
        chunk = samples[start_idx:end_idx].copy()
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)), mode="constant")

        chunk_start_s = start_idx / sr

        # Run segmentation model: input [1, 1, window_size]
        x = chunk[np.newaxis, np.newaxis, :].astype(np.float32)
        output = seg_session.run(None, {input_name: x})
        probs = output[0][0]  # [T, num_classes]
        T = probs.shape[0]
        # seconds per output frame
        frame_step_s = (window_size / sr) / T  # ≈ 17 ms for pyannote v3

        # Decode powerset classes → per-speaker binary activity [T, num_speakers]
        classes = np.argmax(probs, axis=-1)  # [T]
        activity = np.zeros((T, num_speakers), dtype=bool)
        for t_idx, cls in enumerate(classes):
            if cls < len(powerset_map):
                for spk in powerset_map[cls]:
                    activity[t_idx, spk] = True

        # Convert frame-level activity to time intervals per local speaker
        for local_spk in range(num_speakers):
            spk_act = activity[:, local_spk]
            padded = np.concatenate([[False], spk_act, [False]])
            diff = np.diff(padded.astype(np.int8))
            seg_starts = np.where(diff == 1)[0]
            seg_ends = np.where(diff == -1)[0]
            for fs, fe in zip(seg_starts, seg_ends):
                s = chunk_start_s + fs * frame_step_s
                e = chunk_start_s + fe * frame_step_s
                e = min(e, n / sr)
                if e > s:
                    raw_segs.append((s, e, speaker_id_offset + local_spk))

        speaker_id_offset += num_speakers
        window_count += 1
        if end_idx >= n:
            break
        start_idx += window_shift

    log.info(
        "ORT segmentation: %d windows → %d raw speaker intervals",
        window_count, len(raw_segs),
    )
    return raw_segs


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Triangular mel filterbank matrix [n_mels, n_fft//2+1] using HTK mel scale."""
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = np.array([mel_to_hz(m) for m in mel_pts])
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bins = np.array([np.argmin(np.abs(freqs - f)) for f in hz_pts], dtype=np.int32)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        if center > left:
            fb[m, left : center + 1] = np.linspace(0.0, 1.0, center - left + 1)
        if right > center:
            fb[m, center : right + 1] = np.linspace(1.0, 0.0, right - center + 1)
    return fb


def _compute_nemo_fbank(
    audio: np.ndarray,
    n_mels: int = 80,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Log mel filterbank features matching sherpa-onnx's NeMo/TitaNet feature
    extraction (kaldi-style: hanning window, 25 ms frames, 10 ms shift,
    power spectrogram, per-feature mean/std normalisation).

    Uses kaldi_native_fbank when available (most accurate match to
    sherpa-onnx's internal implementation).  Falls back to a scipy/numpy
    re-implementation otherwise.

    Returns: float32 [n_mels, n_frames]
    """
    audio_f32 = audio.astype(np.float32)

    if KNF_AVAILABLE:
        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hanning"
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.samp_freq = float(sr)
        opts.frame_opts.snip_edges = True
        opts.frame_opts.round_to_power_of_two = True
        opts.mel_opts.num_bins = n_mels
        opts.mel_opts.high_freq = float(sr) / 2.0
        opts.mel_opts.low_freq = 20.0
        opts.use_power = True
        opts.use_log_fbank = True
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(sr, audio_f32.tolist())
        fb.input_finished()
        n_frames = fb.num_frames_ready
        if n_frames == 0:
            return np.zeros((n_mels, 1), dtype=np.float32)
        features = np.array(
            [fb.get_frame(i) for i in range(n_frames)], dtype=np.float32
        ).T  # [n_mels, n_frames]
    else:
        # Scipy fallback ──────────────────────────────────────────────────────
        win_len = int(0.025 * sr)   # 400 samples
        hop_len = int(0.010 * sr)   # 160 samples
        n_fft = 512
        window = np.hanning(win_len).astype(np.float32)
        n_frames = max(1, 1 + (len(audio_f32) - win_len) // hop_len)
        frames = np.zeros((n_frames, n_fft), dtype=np.float32)
        for i in range(n_frames):
            s = i * hop_len
            e = s + win_len
            frame = audio_f32[s : min(e, len(audio_f32))]
            if len(frame) < win_len:
                frame = np.pad(frame, (0, win_len - len(frame)))
            frames[i, :win_len] = frame * window
        power = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1)) ** 2  # [T, n_fft//2+1]
        fb_matrix = _mel_filterbank(sr, n_fft, n_mels, fmin=20.0, fmax=sr / 2.0)
        features = np.log(np.maximum(fb_matrix @ power.T, 1e-9)).astype(np.float32)
        # features: [n_mels, n_frames]

    # Per-feature (per-mel-bin) mean/std normalisation — matches sherpa-onnx NeMo impl
    mean = features.mean(axis=-1, keepdims=True)
    std = features.std(axis=-1, keepdims=True) + 1e-9
    return ((features - mean) / std).astype(np.float32)


def _extract_embeddings_ort(
    emb_session,
    samples: np.ndarray,
    segments: list,
    sr: int = SAMPLE_RATE,
    n_mels: int = 80,
    min_seg_s: float = 0.1,
) -> tuple:
    """
    Extract L2-normalised TitaNet speaker embeddings for each segment.

    Returns (embeddings [N, D], valid_segments) where valid_segments is the
    subset of `segments` for which a valid embedding was obtained.
    """
    input_infos = emb_session.get_inputs()
    inp_name = input_infos[0].name
    has_length = len(input_infos) > 1
    len_name = input_infos[1].name if has_length else None

    valid_segs: list = []
    embeds: list = []

    for start_s, end_s, spk_id in segments:
        if (end_s - start_s) < min_seg_s:
            continue
        s0 = max(0, int(start_s * sr))
        s1 = min(len(samples), int(end_s * sr))
        seg_audio = samples[s0:s1]
        if len(seg_audio) < 1:
            continue

        mel = _compute_nemo_fbank(seg_audio, n_mels=n_mels, sr=sr)  # [n_mels, T]
        T = mel.shape[-1]
        if T < 2:
            continue

        feed = {inp_name: mel[np.newaxis, :, :]}  # [1, n_mels, T]
        if has_length:
            feed[len_name] = np.array([T], dtype=np.int64)

        try:
            out = emb_session.run(None, feed)
        except Exception as exc:
            log.warning(
                "ORT embedding failed [%.2f-%.2f]: %s", start_s, end_s, exc
            )
            continue

        emb = out[0][0] if out[0].ndim > 1 else out[0]  # [D]
        norm = np.linalg.norm(emb)
        if norm > 1e-8:  # 1e-8 matches sherpa-onnx C++ epsilon (was 1e-9)
            emb = emb / norm
        valid_segs.append((start_s, end_s, spk_id))
        embeds.append(emb.astype(np.float32))

    if not embeds:
        return np.zeros((0, 1), dtype=np.float32), []
    result = np.stack(embeds, axis=0)
    if DIARIZATION_DEBUG:
        norms = np.linalg.norm(result, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            log.error(
                "L2 normalisation check FAILED: min=%.6f max=%.6f mean=%.6f "
                "— embeddings may not be unit-normalised (set DIARIZATION_DEBUG=0 to suppress)",
                norms.min(), norms.max(), norms.mean(),
            )
    return result, valid_segs


def _fast_clustering(
    embeddings: np.ndarray,
    threshold: float = 0.5,
    num_clusters: int = -1,
) -> np.ndarray:
    """
    Centroid-linkage agglomerative clustering matching sherpa-onnx FastClustering.

    Repeatedly merges the most cosine-similar pair until:
      • threshold mode (num_clusters < 0): max similarity < threshold, OR
      • fixed-count mode: exactly num_clusters remain.

    All embeddings must be L2-normalised before calling.
    Returns integer cluster labels [N].
    """
    N = len(embeddings)
    if N == 0:
        return np.array([], dtype=np.int32)
    if N == 1:
        return np.array([0], dtype=np.int32)

    # Initial cosine-similarity matrix (dot product of unit vectors)
    centroids = embeddings.copy().astype(np.float64)
    sim = (centroids @ centroids.T).astype(np.float64)  # [N, N]
    np.fill_diagonal(sim, -np.inf)

    labels = np.arange(N, dtype=np.int32)
    sizes = np.ones(N, dtype=np.int64)
    active = np.ones(N, dtype=bool)

    target = max(1, num_clusters) if num_clusters > 0 else 1
    current_n = N

    while current_n > target:
        # Mask inactive rows/cols and find best merge candidate
        active_idx = np.where(active)[0]
        sub = sim[np.ix_(active_idx, active_idx)]
        max_val = float(sub.max())

        if num_clusters < 0 and max_val < threshold:
            break

        flat = np.argmax(sub)
        i_loc, j_loc = np.unravel_index(flat, sub.shape)
        i, j = int(active_idx[i_loc]), int(active_idx[j_loc])

        # Merge j into i with weighted centroid
        ni, nj = int(sizes[i]), int(sizes[j])
        new_c = (ni * centroids[i] + nj * centroids[j]) / (ni + nj)
        norm = np.linalg.norm(new_c)
        if norm > 1e-9:
            new_c /= norm
        centroids[i] = new_c
        sizes[i] = ni + nj

        # Update entire column/row i in similarity matrix
        col = centroids @ new_c  # [N] — similarity of every centroid to merged one
        sim[i, :] = col
        sim[:, i] = col
        sim[i, i] = -np.inf

        # Deactivate j
        active[j] = False
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        labels[labels == j] = i
        current_n -= 1

    # Remap to contiguous 0-based labels
    unique = sorted(set(labels.tolist()))
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[l] for l in labels.tolist()], dtype=np.int32)


def run_diarization_ort(
    samples: np.ndarray,
    threshold: float,
    num_speakers: Optional[int],
    timing: dict,
) -> tuple:
    """
    Full ORT diarization pipeline — drop-in replacement for
    run_diarization_whole_day().

    Stage 1: pyannote segmentation via ORT (CUDA if available)
    Stage 2: TitaNet embedding extraction via ORT (CUDA if available)
    Stage 3: FastClustering — pure numpy, CPU

    Sets timing["diarization_provider"] = "CUDA" | "CPU (ort)".
    """
    seg_session, emb_session, provider = _get_ort_sessions()
    timing["diarization_provider"] = provider

    duration_s = len(samples) / SAMPLE_RATE
    log.info(
        "ORT diarization: %.0fs audio, provider=%s, threshold=%.2f, num_speakers=%s",
        duration_s, provider, threshold, num_speakers,
    )

    # ── Read parameters from model metadata ──────────────────────────────────
    seg_meta = seg_session.get_modelmeta().custom_metadata_map
    emb_meta = emb_session.get_modelmeta().custom_metadata_map

    window_size = int(seg_meta.get("window_size", "160000"))
    # Use window_size as stride (non-overlapping) when metadata doesn't specify,
    # matching sherpa-onnx offline mode behaviour.
    window_shift = int(seg_meta.get("window_shift", str(window_size)))
    model_num_speakers = int(seg_meta.get("num_speakers", "3"))
    powerset_max = int(seg_meta.get("powerset_max_classes", "2"))
    n_mels = int(
        emb_meta.get("feature_dim", emb_meta.get("feat_dim", emb_meta.get("n_mels", "80")))
    )
    log.info(
        "ORT params: window=%d shift=%d speakers=%d powerset_max=%d n_mels=%d",
        window_size, window_shift, model_num_speakers, powerset_max, n_mels,
    )

    t_total = time.time()

    # ── Stage 1: segmentation ─────────────────────────────────────────────────
    t0 = time.time()
    raw_segs = _run_pyannote_ort(
        seg_session, samples, model_num_speakers, powerset_max,
        window_size, window_shift,
    )
    timing["diarization_seg_s"] = round(time.time() - t0, 3)
    log.info("ORT seg done: %d raw intervals in %.1fs", len(raw_segs), timing["diarization_seg_s"])

    if not raw_segs:
        timing["diarization_s"] = round(time.time() - t_total, 3)
        return [], 0

    # Pre-filter: remove segments below min_duration_on before embedding
    cand_segs = [(s, e, spk) for s, e, spk in raw_segs if (e - s) >= MIN_DURATION_ON]
    log.info("ORT: %d segments after min_duration_on filter", len(cand_segs))

    # ── Stage 2: embedding extraction ────────────────────────────────────────
    t0 = time.time()
    embeddings, valid_segs = _extract_embeddings_ort(
        emb_session, samples, cand_segs, n_mels=n_mels,
    )
    timing["diarization_emb_s"] = round(time.time() - t0, 3)
    log.info(
        "ORT emb done: %d/%d → shape=%s in %.1fs",
        len(valid_segs), len(cand_segs), embeddings.shape, timing["diarization_emb_s"],
    )

    if len(valid_segs) == 0:
        timing["diarization_s"] = round(time.time() - t_total, 3)
        return [], 0

    # ── Stage 3: FastClustering ───────────────────────────────────────────────
    t0 = time.time()
    n_clust = num_speakers if num_speakers is not None else -1
    labels = _fast_clustering(embeddings, threshold=threshold, num_clusters=n_clust)
    timing["diarization_clust_s"] = round(time.time() - t0, 3)
    n_detected = len(set(labels.tolist()))
    log.info(
        "ORT clustering: %d segs → %d speakers in %.1fs",
        len(valid_segs), n_detected, timing["diarization_clust_s"],
    )

    # ── Build output segments ─────────────────────────────────────────────────
    out_segs = [
        {
            "start": round(s, 3),
            "end": round(e, 3),
            "speaker": f"speaker_{int(lbl):02d}",
        }
        for (s, e, _), lbl in zip(valid_segs, labels)
    ]
    out_segs.sort(key=lambda x: x["start"])

    # Apply min_duration_off: merge short gaps between consecutive same-speaker segs
    if out_segs:
        merged: list = [out_segs[0]]
        for seg in out_segs[1:]:
            prev = merged[-1]
            if (
                seg["speaker"] == prev["speaker"]
                and seg["start"] - prev["end"] < MIN_DURATION_OFF
            ):
                merged[-1] = {"start": prev["start"], "end": seg["end"], "speaker": prev["speaker"]}
            else:
                merged.append(seg)
        out_segs = merged

    elapsed = time.time() - t_total
    timing["diarization_s"] = round(elapsed, 3)
    num_speakers_detected = len({s["speaker"] for s in out_segs})

    log.info(
        "ORT diarization done: %.1fs | %d segs | %d speakers | RTF=%.3f | provider=%s",
        elapsed, len(out_segs), num_speakers_detected, elapsed / max(duration_s, 1), provider,
    )

    try:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info("Peak RSS after ORT diarization: %.0f MB", rss_mb)
    except Exception:
        pass

    return out_segs, num_speakers_detected


# ── Sherpa-onnx diarization ────────────────────────────────────────────────────

def _build_diarizer(
    threshold: float,
    num_speakers: Optional[int],
) -> "sherpa_onnx.OfflineSpeakerDiarization":
    """Build a Sherpa ONNX diarizer matching DiarizationService.kt C1 config."""
    seg_model = os.path.join(
        DIARIZATION_MODEL_DIR, "pyannote-segmentation-3-0", "model.int8.onnx",
    )
    emb_model = os.path.join(DIARIZATION_MODEL_DIR, "nemo_en_titanet_small.onnx")

    if not os.path.exists(seg_model):
        raise FileNotFoundError(
            f"Segmentation model not found: {seg_model}\n"
            f"Run prepare_models.sh to populate {DIARIZATION_MODEL_DIR}"
        )
    if not os.path.exists(emb_model):
        raise FileNotFoundError(
            f"Embedding model not found: {emb_model}\n"
            f"Run prepare_models.sh to populate {DIARIZATION_MODEL_DIR}"
        )

    if num_speakers is not None:
        clustering = sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers)
    else:
        clustering = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_model,
            )
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model),
        clustering=clustering,
        min_duration_on=MIN_DURATION_ON,
        min_duration_off=MIN_DURATION_OFF,
    )
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def _segments_from_result(result, time_offset_s: float = 0.0, speaker_id_offset: int = 0) -> list[dict]:
    """Convert sherpa_onnx diarization result segments to dict list.

    The sherpa-onnx Python API returns an OfflineSpeakerDiarizationResult from
    diarizer.process(). Segments are retrieved via result.sort_by_start_time(),
    which returns a list of OfflineSpeakerDiarizationSegment objects each with
    .start, .end, and .speaker (integer index).
    """
    result_type = type(result).__name__
    public_attrs = [a for a in dir(result) if not a.startswith("_")]
    log.info("Diarization result type: %s | public attrs: %s", result_type, public_attrs)

    # Primary API: result.sort_by_start_time() — official sherpa-onnx Python examples
    if hasattr(result, "sort_by_start_time"):
        raw_segs = result.sort_by_start_time()
    elif hasattr(result, "segments"):
        raw_segs = result.segments
    elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes, np.ndarray)):
        raw_segs = list(result)
    else:
        raise AttributeError(
            f"Cannot extract segments from sherpa_onnx result of type {result_type}. "
            f"Available attrs: {public_attrs}"
        )

    out = []
    for seg in raw_segs:
        out.append({
            "start": round(float(seg.start) + time_offset_s, 3),
            "end": round(float(seg.end) + time_offset_s, 3),
            "speaker": f"speaker_{speaker_id_offset + int(seg.speaker):02d}",
        })
    return out


def run_diarization_whole_day(
    samples: np.ndarray,
    threshold: float,
    num_speakers: Optional[int],
    timing: dict,
) -> tuple[list[dict], int]:
    """
    Run Sherpa ONNX diarization over the full concatenated audio (whole_day mode).
    One pass produces a single global speaker namespace across the entire day.
    The Android OOM limit (~41 min) does not apply on a cloud instance.
    """
    if not SHERPA_AVAILABLE:
        raise RuntimeError("sherpa_onnx package not available — cannot diarize")

    duration_s = len(samples) / SAMPLE_RATE
    log.info("Diarizing %.0fs (whole_day, threshold=%.2f, num_speakers=%s)", duration_s, threshold, num_speakers)

    diarizer = _build_diarizer(threshold, num_speakers)
    t0 = time.time()
    result = diarizer.process(samples)
    elapsed = time.time() - t0
    timing["diarization_s"] = round(elapsed, 3)
    timing["diarization_provider"] = "CPU (sherpa-onnx)"

    # Log peak RSS for RAM sizing decisions
    try:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        log.info("Peak RSS after diarization: %.0f MB", rss_mb)
    except Exception:
        pass

    segments = _segments_from_result(result)
    num_speakers_detected = len({s["speaker"] for s in segments})
    log.info("Diarization done in %.1fs — %d segments, %d speakers", elapsed, len(segments), num_speakers_detected)
    return segments, num_speakers_detected


def run_diarization_fixed_chunk(
    samples: np.ndarray,
    threshold: float,
    num_speakers: Optional[int],
    timing: dict,
) -> tuple[list[dict], int]:
    """
    Run Sherpa ONNX diarization in fixed 40 M-sample chunks with no overlap
    (fixed_chunk mode).  Reproduces the Android OOM-driven forced-chunk behaviour
    (T9 failure mode: global LAA 10.9%, completeness 13.6% on gold dataset).
    Each chunk gets a cold-start Sherpa call with no cross-chunk speaker continuity.
    """
    if not SHERPA_AVAILABLE:
        raise RuntimeError("sherpa_onnx package not available — cannot diarize")

    chunks = [
        samples[i : i + FIXED_CHUNK_SAMPLES]
        for i in range(0, len(samples), FIXED_CHUNK_SAMPLES)
    ]
    log.info("Fixed-chunk diarization: %d chunk(s) of ≤%.0fs", len(chunks), FIXED_CHUNK_SAMPLES / SAMPLE_RATE)

    all_segments: list[dict] = []
    id_offset = 0
    t0 = time.time()

    for idx, chunk in enumerate(chunks):
        chunk_start_s = idx * (FIXED_CHUNK_SAMPLES / SAMPLE_RATE)
        log.info("  Chunk %d/%d: %.0fs at t=%.0fs", idx + 1, len(chunks), len(chunk) / SAMPLE_RATE, chunk_start_s)
        diarizer = _build_diarizer(threshold, num_speakers)
        result = diarizer.process(chunk)
        chunk_segs = _segments_from_result(result, time_offset_s=chunk_start_s, speaker_id_offset=id_offset)
        all_segments.extend(chunk_segs)
        id_offset += len({seg.speaker for seg in result.segments})

    elapsed = time.time() - t0
    timing["diarization_s"] = round(elapsed, 3)

    num_speakers_detected = len({s["speaker"] for s in all_segments})
    log.info("Fixed-chunk diarization done in %.1fs — %d segments, %d speakers", elapsed, len(all_segments), num_speakers_detected)
    return all_segments, num_speakers_detected


# ── Speaker alignment ──────────────────────────────────────────────────────────

def assign_speakers_to_lines(
    transcript_lines: list[dict],
    segments: list[dict],
    audio_duration_s: float,
) -> list[str]:
    """
    Exact port of DiarizationService.assignSpeakersToLines() + findSegmentForTime().

    Algorithm (matches Android):
      For each transcript line timestamp:
        1. Exact containment: find first segment where seg.start <= ts <= seg.end
        2. Nearest fallback: if no exact match, find nearest segment by
           min(|ts - seg.start|, |ts - seg.end|); assign if distance < 5.0 s
        3. Otherwise: assign "[unknown]"

    Args:
      transcript_lines: [{"timestamp_s": float, "timestamp_str": str, "text": str}]
      segments:         [{"start": float, "end": float, "speaker": str}]
      audio_duration_s: total audio duration (kept for API symmetry; not used here)

    Returns:
      ["[HH:MM:SS] speaker_XX: text", ...]  — parser-significant format
    """
    sorted_segs = sorted(segments, key=lambda s: s["start"])

    def _find_segment(time_s: float) -> Optional[dict]:
        # Step 1: exact containment
        for seg in sorted_segs:
            if seg["start"] <= time_s <= seg["end"]:
                return seg
        # Step 2: nearest within 5 s
        if not sorted_segs:
            return None
        nearest = min(
            sorted_segs,
            key=lambda s: min(abs(time_s - s["start"]), abs(time_s - s["end"])),
        )
        dist = min(abs(time_s - nearest["start"]), abs(time_s - nearest["end"]))
        return nearest if dist < 5.0 else None

    result: list[str] = []
    for line in transcript_lines:
        seg = _find_segment(line["timestamp_s"])
        label = seg["speaker"] if seg else "[unknown]"
        result.append(f"{line['timestamp_str']} {label}: {line['text']}")
    return result


# ── Language flags ─────────────────────────────────────────────────────────────

def compute_language_flags(
    transcript_lines: list[dict],
    target_language: Optional[str],
    model_name: str,
    window_size: int = 20,
) -> dict:
    """
    Slide a window of ~20 lines over the transcript, detect language per window,
    and flag windows where the detected language differs from target_language.
    Pure post-hoc metadata — does NOT modify the transcript.
    """
    base: dict = {
        "model": model_name,
        "overall_detected_language": target_language or "unknown",
        "overall_confidence": 0.0,
        "window_flags": [],
    }
    if not LANGDETECT_AVAILABLE:
        base["note"] = "langdetect not installed"
        return base

    texts = [line["text"] for line in transcript_lines if line.get("text", "").strip()]
    if not texts:
        return base

    # Overall detection from first ~50 lines
    try:
        overall_langs = detect_langs(" ".join(texts[:50]))
        overall_lang = overall_langs[0].lang if overall_langs else (target_language or "unknown")
        overall_conf = round(overall_langs[0].prob, 3) if overall_langs else 0.0
    except LangDetectException:
        overall_lang = target_language or "unknown"
        overall_conf = 0.0
    base["overall_detected_language"] = overall_lang
    base["overall_confidence"] = overall_conf

    # For null-language recordings, use detected majority as the reference
    effective_target = target_language if target_language else overall_lang

    window_flags: list[dict] = []
    step = max(1, window_size // 2)
    t0 = time.time()
    for i in range(0, len(texts), step):
        if time.time() - t0 > 1.0 and window_flags:
            log.info("Language flag time limit reached — truncating at line %d", i)
            break
        window_text = " ".join(texts[i : i + window_size])
        try:
            langs = detect_langs(window_text)
            detected = langs[0].lang if langs else overall_lang
            confidence = round(langs[0].prob, 3) if langs else 0.0
        except LangDetectException:
            detected = overall_lang
            confidence = 0.0
        window_flags.append({
            "start_line": i,
            "end_line": min(i + window_size, len(texts)) - 1,
            "detected_language": detected,
            "confidence": confidence,
            "flag": "POSSIBLE_LANGUAGE_MISMATCH" if detected != effective_target else None,
        })

    base["window_flags"] = window_flags
    return base


# ── Main handler ───────────────────────────────────────────────────────────────

def handler(event: dict) -> dict:
    """
    RunPod serverless job handler.
    Input / output contract: see agent_prompt_runpod_parity_worker.md
    """
    job_start = time.time()
    job_id: str = event.get("id", "unknown")
    log.info("=== Job start === (job_id=%s)", job_id)

    # ── Input validation ──────────────────────────────────────────────────────
    if not event or "input" not in event:
        return {"error": "Event must contain 'input' field."}

    inp = event["input"]

    # audio_urls takes precedence over audio_url
    audio_urls: Optional[list[str]] = inp.get("audio_urls")
    if not audio_urls:
        single = inp.get("audio_url")
        if single:
            audio_urls = [single]
    if not audio_urls:
        return {"error": "Either 'audio_url' or 'audio_urls' input is required."}

    ext: str = inp.get("ext", "opus")
    language_input: Optional[str] = inp.get("language")
    language_candidates: list[str] = inp.get("language_candidates", ["en", "pt"])

    # model_sizes (array) is canonical; model_size (string) is a convenience alias.
    # Default: ["base.en", "small.en"] — both always run unless caller restricts.
    if "model_sizes" in inp:
        model_sizes: list[str] = list(inp["model_sizes"])
    elif "model_size" in inp:
        model_sizes = [str(inp["model_size"])]
    else:
        model_sizes = ["base.en", "small.en"]

    if not model_sizes:
        return {"error": "'model_sizes' must be a non-empty array."}

    cluster_threshold: float = float(inp.get("cluster_threshold", DEFAULT_THRESHOLD))
    num_speakers: Optional[int] = inp.get("num_speakers")
    diarization_mode: str = inp.get("diarization_mode", "whole_day")
    recording_id: str = inp.get("recording_id", "recording")
    gist_token: str = inp.get("gist_token", "")

    log.info(
        "Params: recording_id=%s models=%s language=%s files=%d mode=%s",
        recording_id, model_sizes, language_input, len(audio_urls), diarization_mode,
    )

    if diarization_mode not in ("whole_day", "fixed_chunk"):
        return {
            "error": f"Unknown diarization_mode '{diarization_mode}'.",
            "detail": "Supported modes: 'whole_day' (default), 'fixed_chunk'.",
        }

    # ── Gist pre-flight (mandatory — refuse to run without it) ───────────────
    if not gist_token:
        return {
            "error": "Gist connectivity check failed",
            "detail": "gist_token is required. The worker uploads all results to Gist "
                      "and will refuse to process audio without a valid token.",
        }
    try:
        verify_gist_upload(gist_token)
    except Exception as exc:
        return {"error": "Gist connectivity check failed", "detail": str(exc)}

    # ── Audio processing ──────────────────────────────────────────────────────
    timing: dict = {}

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Download all audio files
        log.info("Downloading %d audio file(s)...", len(audio_urls))
        t0 = time.time()
        local_paths: list[str] = []
        file_metadata_list: list[dict] = []
        for url in audio_urls:
            lp = download_audio(url, ext, tmp_dir)
            local_paths.append(lp)
            file_metadata_list.append(get_file_metadata(lp))
        timing["audio_download_s"] = round(time.time() - t0, 3)
        file_durations_s = [m.get("duration_s", 0.0) for m in file_metadata_list]
        log.info("Downloaded %d file(s) in %.1fs", len(local_paths), timing["audio_download_s"])

        # Concatenate / convert to 16 kHz mono WAV
        concat_wav = os.path.join(tmp_dir, "concat.wav")
        log.info("Converting to 16 kHz mono WAV...")
        t0 = time.time()
        concat_to_wav(local_paths, concat_wav)
        timing["audio_concat_s"] = round(time.time() - t0, 3)
        log.info(
            "Concat WAV: %.0f MB in %.1fs",
            os.path.getsize(concat_wav) / (1024 * 1024), timing["audio_concat_s"],
        )

        # Audio duration for RTF calculations
        wav_meta = get_file_metadata(concat_wav)
        audio_duration_s = wav_meta.get("duration_s") or sum(file_durations_s) or 1.0
        log.info("Audio duration: %.0fs (%.1f min)", audio_duration_s, audio_duration_s / 60)

        # Language detection — use first model in list; result shared across all models
        first_model = model_sizes[0]
        t0 = time.time()
        if language_input is None or language_input == "auto":
            detected_language, lang_confidence = detect_constrained_language(
                concat_wav, get_whisper_model(first_model),
                candidates=language_candidates,
                fallback=language_candidates[0] if language_candidates else "en",
            )
        elif "," in language_input:
            candidates = [c.strip() for c in language_input.split(",")]
            detected_language, lang_confidence = detect_constrained_language(
                concat_wav, get_whisper_model(first_model),
                candidates=candidates,
                fallback=candidates[0],
            )
        else:
            detected_language = language_input
            lang_confidence = 1.0
        timing["language_detection_s"] = round(time.time() - t0, 3)
        log.info("Language: %s (p=%.2f)", detected_language, lang_confidence)

        # ── Transcription — one pass per model ───────────────────────────────
        per_model: dict[str, dict] = {}  # keyed by model_size string
        for ms in model_sizes:
            log.info("--- Transcribing with model: %s ---", ms)
            t_lines, seg_meta, _, _ = transcribe_audio(
                concat_wav, ms, detected_language, timing,
            )
            per_model[ms] = {
                "transcript_lines": t_lines,
                "segment_metadata": seg_meta,
            }
            log.info("  %s: %d lines", ms, len(t_lines))

        # ── Diarization — once, shared across all models ──────────────────────
        # Transcription is complete. Free cached Whisper models from VRAM before
        # loading ORT diarization sessions, reducing peak VRAM pressure.
        if not CPU_ONLY:
            _log_vram("pre-diarization (whisper cached)")
            _whisper_cache.clear()
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
            _log_vram("pre-diarization (whisper freed)")
        log.info("Loading WAV samples for diarization...")
        samples = load_wav_as_float32(concat_wav)

        # Engine selection: ORT (CUDA if available) → sherpa-onnx CPU fallback
        if diarization_mode == "whole_day":
            if ORT_AVAILABLE:
                try:
                    diarization_segments, num_speakers_detected = run_diarization_ort(
                        samples, cluster_threshold, num_speakers, timing,
                    )
                except Exception as ort_exc:
                    log.warning(
                        "ORT diarization failed (%s) — falling back to sherpa-onnx", ort_exc
                    )
                    diarization_segments, num_speakers_detected = run_diarization_whole_day(
                        samples, cluster_threshold, num_speakers, timing,
                    )
            else:
                diarization_segments, num_speakers_detected = run_diarization_whole_day(
                    samples, cluster_threshold, num_speakers, timing,
                )
        else:
            diarization_segments, num_speakers_detected = run_diarization_fixed_chunk(
                samples, cluster_threshold, num_speakers, timing,
            )

        del samples  # release RAM before alignment + Gist upload

        # Diarization complete. Release ORT sessions from VRAM.
        if not CPU_ONLY:
            _log_vram("post-diarization (ort loaded)")
            _release_ort_sessions()
            try:
                import torch as _torch
                _torch.cuda.empty_cache()
            except Exception:
                pass
            _log_vram("post-diarization (ort freed)")

        # ── Per-model alignment, language flags, formatted artifacts ─────────
        gist_files: dict[str, str] = {}

        for ms in model_sizes:
            model_key = ms.replace(".", "_")
            t_lines = per_model[ms]["transcript_lines"]
            seg_meta = per_model[ms]["segment_metadata"]

            # Speaker alignment
            t0 = time.time()
            labeled_lines = assign_speakers_to_lines(t_lines, diarization_segments, audio_duration_s)
            timing[f"alignment_{model_key}_s"] = round(time.time() - t0, 3)

            # Language flags
            t0 = time.time()
            effective_target = detected_language if (language_input and language_input != "auto") else None
            lang_flags = compute_language_flags(t_lines, effective_target, ms)
            timing[f"language_flags_{model_key}_s"] = round(time.time() - t0, 3)

            # Format transcript: [HH:MM:SS]  text  (TWO spaces — parser-significant)
            transcript_text = "\n".join(
                f"{line['timestamp_str']}  {line['text']}" for line in t_lines
            )
            labeled_text = "\n".join(labeled_lines)

            # Store formatted artifacts for response and Gist
            per_model[ms]["transcript_text"] = transcript_text
            per_model[ms]["labeled_text"] = labeled_text
            per_model[ms]["lang_flags"] = lang_flags

            # Gist files for this model
            gist_files[f"{recording_id}_transcript_{ms}.txt"] = transcript_text
            gist_files[f"{recording_id}_labeled_{ms}.txt"] = labeled_text
            gist_files[f"{recording_id}_segment_metadata_{ms}.json"] = json.dumps(seg_meta, indent=2)
            gist_files[f"{recording_id}_language_flags_{ms}.json"] = json.dumps(lang_flags, indent=2)

        # Diarization JSON — shared, one copy per Gist
        diarization_json_obj = {
            "segments": diarization_segments,
            "num_speakers": num_speakers_detected,
            "threshold": cluster_threshold,
            "model": "pyannote-seg-3.0_titanet-small",
            "mode": diarization_mode,
        }
        gist_files[f"{recording_id}_diarization.json"] = json.dumps(diarization_json_obj, indent=2)

        # RTF and totals
        total_s = time.time() - job_start
        timing["total_s"] = round(total_s, 3)
        for ms in model_sizes:
            model_key = ms.replace(".", "_")
            tkey = f"transcription_{model_key}_s"
            timing[f"rtf_transcription_{model_key}"] = round(
                timing.get(tkey, 0) / audio_duration_s, 4
            )
        timing["rtf_total"] = round(total_s / audio_duration_s, 4)

        run_summary = {
            "recording_id": recording_id,
            "model_sizes": model_sizes,
            "detected_language": detected_language,
            "language_probability": round(lang_confidence, 4),
            "audio_duration_s": round(audio_duration_s, 2),
            "file_durations_s": [round(d, 2) for d in file_durations_s],
            "file_metadata": file_metadata_list,
            "num_files": len(audio_urls),
            "num_speakers": num_speakers_detected,
            "cluster_threshold": cluster_threshold,
            "diarization_mode": diarization_mode,
            "cpu_only": CPU_ONLY,
            "timing": timing,
            "job_id": job_id,
            "worker": WORKER_INFO,
            "cost_estimate_usd": _estimate_cost(total_s),
            "sherpa_onnx_version": getattr(sherpa_onnx, "__version__", "unknown") if SHERPA_AVAILABLE else "unavailable",
        }
        gist_files[f"{recording_id}_run_summary.json"] = json.dumps(run_summary, indent=2)

        log.info(
            "Total: %.0fs | diarization: %.0fs | RTF total: %.3fx",
            total_s, timing.get("diarization_s", 0), timing["rtf_total"],
        )

        # ── Gist upload ────────────────────────────────────────────────────────
        today = datetime.date.today().isoformat()
        gist_description = f"{recording_id}_{today}"
        gist_url = ""

        try:
            gist_url = upload_to_gist(gist_token, gist_files, gist_description, timing)
        except Exception as exc:
            log.error("Gist upload failed after retries: %s", exc)
            # Embed raw artifacts so caller can recover without re-running
            recovery = {
                "error": "Gist upload failed",
                "detail": str(exc),
                "recording_id": recording_id,
                "model_sizes": model_sizes,
                "detected_language": detected_language,
                "language_probability": round(lang_confidence, 4),
                "audio_duration_s": round(audio_duration_s, 2),
                "file_durations_s": [round(d, 2) for d in file_durations_s],
                "num_speakers": num_speakers_detected,
                "cluster_threshold": cluster_threshold,
                "timing": timing,
                "cpu_only": CPU_ONLY,
                "diarization_segments": diarization_segments,
                "models": {},
            }
            for ms in model_sizes:
                recovery["models"][ms] = {
                    "transcript_lines": len(per_model[ms]["transcript_lines"]),
                    "transcript": per_model[ms]["transcript_text"],
                    "labeled_transcript": per_model[ms]["labeled_text"],
                }
            return recovery

    # ── Response ───────────────────────────────────────────────────────────────
    models_out: dict[str, dict] = {}
    for ms in model_sizes:
        models_out[ms] = {
            "transcript_lines": len(per_model[ms]["transcript_lines"]),
            "transcript": per_model[ms]["transcript_text"],
            "labeled_transcript": per_model[ms]["labeled_text"],
        }

    return {
        "recording_id": recording_id,
        "model_sizes": model_sizes,
        "detected_language": detected_language,
        "language_probability": round(lang_confidence, 4),
        "num_speakers": num_speakers_detected,
        "cluster_threshold": cluster_threshold,
        "audio_duration_s": round(audio_duration_s, 2),
        "file_durations_s": [round(d, 2) for d in file_durations_s],
        "cpu_only": CPU_ONLY,
        "gist_url": gist_url,
        "diarization_segments": diarization_segments,
        "models": models_out,
        "timing": timing,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
