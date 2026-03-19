# RunPod Parity Worker — whisper + Sherpa ONNX

Produces output **format-identical** to the local Android pipeline
(`whisper.cpp` + Sherpa ONNX). Primary purpose: transcribe ~20 days of
real-life recordings for scale testing of chaptering quality and the sampling
pipeline. Secondary purpose: multi-model comparison data (`base.en` vs
`small.en`) on real-life recordings.

---

## Architecture

| Stage | Tool | Notes |
|---|---|---|
| Transcription | `faster-whisper` (CTranslate2) | `device="auto"`, CPU-first |
| Diarization | `sherpa-onnx` Python package | Same C++ library as Android AAR |
| Segmentation model | `pyannote/segmentation-3.0` int8 ONNX | Matches Android C1 combo |
| Embedding model | NeMo TitaNet Small ONNX | Matches Android C1 combo |
| Clustering | `FastClustering` (cosine agglomerative) | `threshold=0.5`, auto-detect |
| Speaker alignment | `assign_speakers_to_lines()` | Exact port of `DiarizationService.kt` |

**Diarization parameters** (identical to Android `DiarizationService.kt`):
- `min_duration_on = 0.3 s`
- `min_duration_off = 0.5 s`
- default `threshold = 0.5`

---

## Outputs (per job)

All outputs are uploaded to a single GitHub Gist. The Gist URL is returned
in the job response.

| File | Description |
|---|---|
| `{id}_transcript_{model}.txt` | Raw Whisper output — `[HH:MM:SS]  text` (two spaces, parser-significant) |
| `{id}_diarization.json` | Raw Sherpa diarization segments |
| `{id}_labeled_{model}.txt` | Merged output — `[HH:MM:SS] speaker_XX: text` |
| `{id}_segment_metadata_{model}.json` | Per-segment `no_speech_prob`, `avg_logprob`, `compression_ratio` |
| `{id}_language_flags_{model}.json` | Sliding-window language detection flags |
| `{id}_run_summary.json` | Timing, RTF, file metadata, run parameters |

---

## Input / Output Contract

### Input

```json
{
  "input": {
    "audio_url":           "https://...",
    "audio_urls":          ["https://...", "https://..."],
    "ext":                 "opus",
    "language":            "en",
    "language_candidates": ["en", "pt"],
    "model_size":          "base.en",
    "cluster_threshold":   0.5,
    "num_speakers":        null,
    "diarization_mode":    "whole_day",
    "recording_id":        "2026-03-01",
    "gist_token":          "ghp_..."
  }
}
```

- `audio_urls` takes precedence over `audio_url`. Multiple files are concatenated
  (no gap added) before transcription so timestamps are continuous across the day.
- `language`: `null` / `"auto"` = constrained detection. Single code forces language.
  Comma-joined (e.g. `"en,pt"`) = constrained detection over those candidates.
- `diarization_mode`: `"whole_day"` (default) or `"fixed_chunk"`.
- **`gist_token` is mandatory.** The worker refuses to process audio if the Gist
  pre-flight connectivity check fails.

### Output

```json
{
  "recording_id":          "2026-03-01",
  "model_size":            "base.en",
  "detected_language":     "en",
  "language_probability":  0.97,
  "transcript_lines":      487,
  "num_speakers":          4,
  "cluster_threshold":     0.5,
  "audio_duration_s":      19440.3,
  "file_durations_s":      [3340.5, 1892.3],
  "processing_time_s":     312.4,
  "cpu_only":              true,
  "gist_url":              "https://gist.github.com/...",
  "transcript":            "[00:00:18]  hello everyone...\n...",
  "diarization_segments":  [{"start": 18.3, "end": 24.1, "speaker": "speaker_00"}],
  "labeled_transcript":    "[00:00:18] speaker_00: hello everyone...\n..."
}
```

---

## Supported models

| `model_size` | CTranslate2 | Android equivalent | Image size |
|---|---|---|---|
| `tiny.en` | tiny.en int8 | ggml-tiny.en-q5_1 | ~75 MB |
| `base.en` | base.en int8 | ggml-base.en-q5_1 | ~148 MB |
| `small.en` | small.en int8 | ggml-small.en-q5_1 | ~244 MB |

**Do not include `medium.en`** in this deployment.

---

## Diarization modes

| Mode | Description |
|---|---|
| `whole_day` (default) | Single Sherpa pass over full concatenated audio. Global speaker namespace across the entire day. Recommended for 20-day batch run. |
| `fixed_chunk` | Fixed 40 M-sample chunks (≈41.7 min), no cross-chunk speaker continuity. Reproduces Android OOM-forced splitting (T9 failure mode). Use only to compare against Android output on the same recording. |
| `vad_silence_chunk` | Not yet implemented (deferred). |

---

## Build

### 1. Populate diarization models

```bash
cd runpod-worker-whisper-diarization-master

# Option A — copy from Android APK assets (recommended, exact parity)
./prepare_models.sh --from-assets

# Option B — download from sherpa-onnx GitHub releases
./prepare_models.sh --download
# THEN verify checksums match the Android assets before parity testing
```

### 2. Build Docker image

```bash
docker build -t whisper-parity-worker .
```

Whisper models (`tiny.en`, `base.en`, `small.en`) are baked in at build time.
No network downloads at job startup. The image works on both CPU-only and GPU
instances — `faster-whisper`/ctranslate2 detects CUDA automatically at runtime.

---

## Local testing

```bash
./prepare_models.sh
# Edit test_input.json — set real audio_url and gist_token
./test_local.sh
```

---

## Language detection

**Android** (`native-lib.cpp → micro_detect_language()`): Per-utterance
constrained detection — for each voice chunk runs Whisper language ID on the
first 3 seconds, picks the best language from the candidate set, confidence
threshold 0.5. Below threshold → keeps previous language.

**Python approximation**: Single-pass detection from the first 30 seconds of
the concatenated audio, same 0.5 threshold and candidate restriction. Per-
utterance switching cannot be reproduced via the faster-whisper API — this is a
documented, acceptable divergence.

---

## Validation checklist

Before the 20-day batch run:

- [ ] **Clustering parity** — Run on `GOLD-01`. LAA within ±5% of Android T2 (70.1%).
- [ ] **Transcript format** — `[HH:MM:SS]  text` (two spaces), monotonically increasing, starts near 00:00:00.
- [ ] **Multi-file continuity** — Second file's first timestamp ≈ first file's duration (no gap, no reset).
- [ ] **Labeled format** — lowercase `speaker_00`, every line labeled or `[unknown]`.
- [ ] **CPU timing** — Run `GOLD-01` (56 min) CPU-only. Log RTF as cost baseline.
