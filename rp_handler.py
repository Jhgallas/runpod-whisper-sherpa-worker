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
COMPUTE_TYPE: str = "int8" if CPU_ONLY else "int8_float16"
log.info("Runtime: %s (compute_type=%s)", "CPU" if CPU_ONLY else "GPU/CUDA", COMPUTE_TYPE)

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
    timing["transcription_s"] = round(elapsed, 3)
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


# ── Diarization ────────────────────────────────────────────────────────────────

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
    log.info("=== Job start ===")

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
    language_input: Optional[str] = inp.get("language")            # None or "auto" = auto-detect
    language_candidates: list[str] = inp.get("language_candidates", ["en", "pt"])
    model_size: str = inp.get("model_size", "base.en")
    cluster_threshold: float = float(inp.get("cluster_threshold", DEFAULT_THRESHOLD))
    num_speakers: Optional[int] = inp.get("num_speakers")          # None = auto-detect
    diarization_mode: str = inp.get("diarization_mode", "whole_day")
    recording_id: str = inp.get("recording_id", "recording")
    gist_token: str = inp.get("gist_token", "")

    log.info(
        "Params: recording_id=%s model=%s language=%s files=%d mode=%s",
        recording_id, model_size, language_input, len(audio_urls), diarization_mode,
    )

    if diarization_mode not in ("whole_day", "fixed_chunk"):
        return {
            "error": f"Unknown diarization_mode '{diarization_mode}'.",
            "detail": "Supported modes: 'whole_day' (default), 'fixed_chunk'. "
                      "'vad_silence_chunk' is deferred (not yet implemented).",
        }

    # ── Gist pre-flight (mandatory — refuse to run without it) ───────────────
    if not gist_token:
        return {
            "error": "Gist connectivity check failed",
            "detail": "gist_token is required in input. The worker uploads all results "
                      "to Gist and will refuse to process audio without a valid token.",
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

        # Language detection
        model = get_whisper_model(model_size)
        t0 = time.time()
        if language_input is None or language_input == "auto":
            detected_language, lang_confidence = detect_constrained_language(
                concat_wav, model,
                candidates=language_candidates,
                fallback=language_candidates[0] if language_candidates else "en",
            )
        elif "," in language_input:
            # Comma-joined multi-language setting (e.g. "en,pt") — constrained detect
            candidates = [c.strip() for c in language_input.split(",")]
            detected_language, lang_confidence = detect_constrained_language(
                concat_wav, model,
                candidates=candidates,
                fallback=candidates[0],
            )
        else:
            # Fixed single language — no detection pass needed
            detected_language = language_input
            lang_confidence = 1.0
        timing["language_detection_s"] = round(time.time() - t0, 3)

        # Transcription
        transcript_lines, segment_metadata, _, _ = transcribe_audio(
            concat_wav, model_size, detected_language, timing,
        )

        # Audio duration for RTF
        wav_meta = get_file_metadata(concat_wav)
        timing["metadata_extraction_s"] = 0.0  # extracted inline above
        audio_duration_s = wav_meta.get("duration_s") or sum(file_durations_s) or 1.0
        log.info(
            "Audio duration: %.0fs (%.1f min), transcript: %d lines",
            audio_duration_s, audio_duration_s / 60, len(transcript_lines),
        )

        # Load WAV as float32 for sherpa_onnx
        samples = load_wav_as_float32(concat_wav)

        # Diarization
        if diarization_mode == "whole_day":
            diarization_segments, num_speakers_detected = run_diarization_whole_day(
                samples, cluster_threshold, num_speakers, timing,
            )
        else:  # fixed_chunk
            diarization_segments, num_speakers_detected = run_diarization_fixed_chunk(
                samples, cluster_threshold, num_speakers, timing,
            )

        del samples  # release RAM before Gist upload

        # Speaker alignment
        t0 = time.time()
        labeled_lines = assign_speakers_to_lines(transcript_lines, diarization_segments, audio_duration_s)
        timing["alignment_s"] = round(time.time() - t0, 3)
        log.info("Alignment done in %.3fs", timing["alignment_s"])

        # Language flags
        t0 = time.time()
        effective_target_for_flags = detected_language if (language_input and language_input != "auto") else None
        language_flags = compute_language_flags(
            transcript_lines, effective_target_for_flags, model_size,
        )
        timing["language_flags_s"] = round(time.time() - t0, 3)

        # ── Format output artifacts ───────────────────────────────────────────

        # Transcript: [HH:MM:SS]  text  (TWO spaces — parser-significant in GeminiService.kt)
        transcript_text = "\n".join(
            f"{line['timestamp_str']}  {line['text']}" for line in transcript_lines
        )

        # Labeled transcript: [HH:MM:SS] speaker_XX: text
        labeled_transcript_text = "\n".join(labeled_lines)

        # Diarization JSON
        diarization_json_obj = {
            "segments": diarization_segments,
            "num_speakers": num_speakers_detected,
            "threshold": cluster_threshold,
            "model": "pyannote-seg-3.0_titanet-small",
            "mode": diarization_mode,
        }

        # RTF and totals
        total_s = time.time() - job_start
        timing["total_s"] = round(total_s, 3)
        timing["rtf_transcription"] = round(timing.get("transcription_s", 0) / audio_duration_s, 4)
        timing["rtf_total"] = round(total_s / audio_duration_s, 4)

        run_summary = {
            "recording_id": recording_id,
            "model_size": model_size,
            "detected_language": detected_language,
            "language_probability": round(lang_confidence, 4),
            "audio_duration_s": round(audio_duration_s, 2),
            "file_durations_s": [round(d, 2) for d in file_durations_s],
            "file_metadata": file_metadata_list,
            "num_files": len(audio_urls),
            "num_speakers": num_speakers_detected,
            "cluster_threshold": cluster_threshold,
            "diarization_mode": diarization_mode,
            "transcript_lines": len(transcript_lines),
            "cpu_only": CPU_ONLY,
            "timing": timing,
            "sherpa_onnx_version": getattr(sherpa_onnx, "__version__", "unknown") if SHERPA_AVAILABLE else "unavailable",
        }

        log.info(
            "Total: %.0fs | RTF transcription: %.3fx | RTF total: %.3fx",
            total_s, timing["rtf_transcription"], timing["rtf_total"],
        )

        # ── Gist upload ────────────────────────────────────────────────────────
        today = datetime.date.today().isoformat()
        gist_description = f"{recording_id}_{model_size}_{today}"
        gist_files = {
            f"{recording_id}_transcript_{model_size}.txt": transcript_text,
            f"{recording_id}_diarization.json": json.dumps(diarization_json_obj, indent=2),
            f"{recording_id}_labeled_{model_size}.txt": labeled_transcript_text,
            f"{recording_id}_segment_metadata_{model_size}.json": json.dumps(segment_metadata, indent=2),
            f"{recording_id}_language_flags_{model_size}.json": json.dumps(language_flags, indent=2),
            f"{recording_id}_run_summary.json": json.dumps(run_summary, indent=2),
        }

        try:
            gist_url = upload_to_gist(gist_token, gist_files, gist_description, timing)
        except Exception as exc:
            log.error("Gist upload failed after retries: %s", exc)
            # Embed raw data in response so caller can recover without re-running
            return {
                "error": "Gist upload failed",
                "detail": str(exc),
                "recording_id": recording_id,
                "model_size": model_size,
                "detected_language": detected_language,
                "language_probability": round(lang_confidence, 4),
                "audio_duration_s": round(audio_duration_s, 2),
                "file_durations_s": [round(d, 2) for d in file_durations_s],
                "transcript_lines": len(transcript_lines),
                "num_speakers": num_speakers_detected,
                "cluster_threshold": cluster_threshold,
                "timing": timing,
                "cpu_only": CPU_ONLY,
                # Raw artifacts for manual recovery
                "transcript": transcript_text,
                "diarization_segments": diarization_segments,
                "labeled_transcript": labeled_transcript_text,
            }

    # ── Response ───────────────────────────────────────────────────────────────
    return {
        "recording_id": recording_id,
        "model_size": model_size,
        "detected_language": detected_language,
        "language_probability": round(lang_confidence, 4),
        "transcript_lines": len(transcript_lines),
        "num_speakers": num_speakers_detected,
        "cluster_threshold": cluster_threshold,
        "audio_duration_s": round(audio_duration_s, 2),
        "file_durations_s": [round(d, 2) for d in file_durations_s],
        "processing_time_s": round(total_s, 2),
        "cpu_only": CPU_ONLY,
        "gist_url": gist_url,
        "transcript": transcript_text,
        "diarization_segments": diarization_segments,
        "labeled_transcript": labeled_transcript_text,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
