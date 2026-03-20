"""
Tests for _fast_clustering() — determinism, sanity, and parity checks.

Run with:  python -m pytest tests/test_clustering.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from rp_handler import _fast_clustering


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_embeddings(
    n_speakers: int = 4,
    n_per_speaker: int = 50,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    Generate unit-normalised embeddings with n_speakers distinct clusters.

    Returns (embeddings [N, 192], ground_truth_labels [N]).
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_speakers, 192))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    labels_gt = np.repeat(np.arange(n_speakers), n_per_speaker)
    noise = rng.normal(scale=noise_scale, size=(len(labels_gt), 192))
    emb = centres[labels_gt] + noise
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb, labels_gt


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_clustering_determinism():
    """Same input must produce identical output on repeated calls."""
    emb, _ = _make_embeddings()
    labels_a = _fast_clustering(emb.copy(), threshold=0.5)
    labels_b = _fast_clustering(emb.copy(), threshold=0.5)
    np.testing.assert_array_equal(
        labels_a, labels_b,
        err_msg="_fast_clustering produced different results on the same input",
    )


def test_clustering_expected_speaker_count():
    """4-centre test case with low noise should cluster into exactly 4 speakers."""
    emb, _ = _make_embeddings(n_speakers=4, noise_scale=0.05)
    labels = _fast_clustering(emb, threshold=0.5)
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == 4, (
        f"Expected 4 clusters on 4-centre synthetic data (noise=0.05), got {n_clusters}. "
        "If this changed, inspect the clustering threshold calibration."
    )


def test_clustering_contiguous_labels():
    """Output labels must be zero-based contiguous integers (no gaps)."""
    emb, _ = _make_embeddings(n_speakers=3)
    labels = _fast_clustering(emb, threshold=0.5)
    unique = sorted(set(labels.tolist()))
    assert unique == list(range(len(unique))), (
        f"Labels not contiguous 0-based: got {unique}"
    )


def test_clustering_single_segment():
    """Single embedding must produce label [0]."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(1, 192))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = _fast_clustering(emb, threshold=0.5)
    assert labels.tolist() == [0]


def test_clustering_empty_input():
    """Zero-length input must return empty array without error."""
    emb = np.zeros((0, 192), dtype=np.float32)
    labels = _fast_clustering(emb, threshold=0.5)
    assert labels.shape == (0,)


def test_clustering_fixed_count_mode():
    """num_clusters > 0 must produce exactly that many clusters when N > num_clusters."""
    emb, _ = _make_embeddings(n_speakers=6, noise_scale=0.02)
    labels = _fast_clustering(emb, threshold=0.5, num_clusters=3)
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == 3, (
        f"Fixed-count mode (num_clusters=3) produced {n_clusters} clusters"
    )


def test_clustering_high_threshold_merges_all():
    """threshold=1.0 should merge everything into one cluster (all cosine sims < 1)."""
    emb, _ = _make_embeddings(n_speakers=5)
    labels = _fast_clustering(emb, threshold=1.0)
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == 1, (
        f"threshold=1.0 should merge all segments, got {n_clusters} clusters"
    )


def test_clustering_low_threshold_keeps_all():
    """threshold=0.0 should keep every segment as its own cluster (no pair ≥ 0)."""
    emb, _ = _make_embeddings(n_speakers=4, noise_scale=0.05, seed=1)
    labels = _fast_clustering(emb, threshold=0.0)
    n_clusters = len(set(labels.tolist()))
    assert n_clusters == len(emb), (
        f"threshold=0.0 should keep all {len(emb)} segments separate, got {n_clusters}"
    )


def test_embeddings_are_unit_normalised_before_call():
    """
    Parity guard: clustering assumes unit-norm inputs.
    Verify the synthetic test embeddings satisfy this so tests are valid.
    """
    emb, _ = _make_embeddings()
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), (
        f"Test fixture embeddings are not unit-normalised: "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )
