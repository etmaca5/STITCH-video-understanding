"""Focused tests for frame-selection methods."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frame_selection import (
    _allocate_segment_budget,
    _detect_segment_boundaries,
    _map_windows_to_chunks,
    _smooth_signal,
    select_frames_from_windows,
)


def _embeddings_from_relevance(relevance):
    """Build unit-length 2D embeddings with a target cosine to [1, 0]."""
    relevance = np.clip(np.asarray(relevance, dtype=np.float32), -0.99, 0.99)
    ortho = np.sqrt(np.maximum(1.0 - relevance ** 2, 0.0)).astype(np.float32)
    return np.stack([relevance, ortho], axis=1)


def test_random_selection_is_deterministic_for_seed():
    window_embeddings = np.arange(48, dtype=np.float32).reshape(12, 4)
    window_times = np.arange(12, dtype=np.float32) * 2.0
    query_embedding = np.ones(4, dtype=np.float32)

    result_a = select_frames_from_windows(
        method="random",
        window_embeddings=window_embeddings,
        window_times=window_times,
        query_embedding=query_embedding,
        n_frames=5,
        duration_sec=24.0,
        fps=2.0,
        sample_interval=2.0,
        seed=7,
    )
    result_b = select_frames_from_windows(
        method="random",
        window_embeddings=window_embeddings,
        window_times=window_times,
        query_embedding=query_embedding,
        n_frames=5,
        duration_sec=24.0,
        fps=2.0,
        sample_interval=2.0,
        seed=7,
    )

    assert result_a["frame_indices"] == result_b["frame_indices"]
    assert len(result_a["frame_indices"]) == 5
    assert len(set(result_a["frame_indices"])) == 5
    assert result_a["frame_indices"] == sorted(result_a["frame_indices"])
    assert all(0 <= idx < 48 for idx in result_a["frame_indices"])
    assert result_a["frame_times"] == [idx / 2.0 for idx in result_a["frame_indices"]]


def test_segment_adaptive_detects_major_relevance_shifts():
    relevance = np.array([
        0.10, 0.10, 0.12, 0.11,
        0.82, 0.85, 0.83, 0.80,
        0.20, 0.18, 0.22, 0.19,
        0.88, 0.90, 0.87, 0.89,
    ], dtype=np.float32)

    smoothed = _smooth_signal(relevance, window=3)
    boundaries = _detect_segment_boundaries(
        smoothed,
        boundary_threshold=0.5,
        min_segment_size=2,
    )

    assert any(abs(boundary - 4) <= 1 for boundary in boundaries)
    assert any(abs(boundary - 8) <= 1 for boundary in boundaries)
    assert any(abs(boundary - 12) <= 1 for boundary in boundaries)


def test_segment_adaptive_budget_preserves_coverage_floor():
    allocations = _allocate_segment_budget(
        [0.9, 0.6, 0.2],
        n_frames=5,
        coverage_floor=True,
    )

    assert allocations == [2, 2, 1]
    assert sum(allocations) == 5


def test_segment_adaptive_selects_across_distinct_segments():
    relevance = np.array([
        0.08, 0.10, 0.09, 0.11,
        0.92, 0.88, 0.84, 0.80,
        0.12, 0.15, 0.11, 0.13,
        0.89, 0.91, 0.86, 0.83,
    ], dtype=np.float32)
    window_embeddings = _embeddings_from_relevance(relevance)
    window_times = np.arange(len(relevance), dtype=np.float32) * 2.0
    query_embedding = np.array([1.0, 0.0], dtype=np.float32)

    result = select_frames_from_windows(
        method="segment_adaptive",
        window_embeddings=window_embeddings,
        window_times=window_times,
        query_embedding=query_embedding,
        n_frames=4,
        duration_sec=32.0,
        fps=2.0,
        sample_interval=2.0,
        smoothing_window=3,
        boundary_threshold=0.5,
        min_segment_size=2,
        intra_segment_method="topk",
    )

    selected = result["window_indices"]
    assert len(selected) == 4
    assert selected == sorted(selected)
    assert any(4 <= idx <= 7 for idx in selected)
    assert any(12 <= idx <= 15 for idx in selected)


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------

def _make_test_scenario():
    """4 chunks, 20 windows at 0.5s intervals, 2fps, 10s video (20 frames total)."""
    fps = 2.0
    chunks = [(0, 5), (5, 10), (10, 15), (15, 20)]  # frames
    # chunk 0: 0-2.5s (windows 0-4), chunk 1: 2.5-5s (windows 5-9)
    # chunk 2: 5-7.5s (windows 10-14), chunk 3: 7.5-10s (windows 15-19)
    n_windows = 20
    window_times = np.arange(n_windows, dtype=np.float32) * 0.5
    relevance = np.array([
        0.3, 0.4, 0.5, 0.6, 0.9,   # chunk 0
        0.2, 0.3, 0.1, 0.2, 0.15,   # chunk 1
        0.7, 0.8, 0.85, 0.6, 0.5,   # chunk 2
        0.4, 0.3, 0.5, 0.2, 0.1,    # chunk 3
    ], dtype=np.float32)
    window_embeddings = _embeddings_from_relevance(relevance)
    query_embedding = np.array([1.0, 0.0], dtype=np.float32)
    return dict(
        window_embeddings=window_embeddings,
        window_times=window_times,
        query_embedding=query_embedding,
        n_frames=4,
        duration_sec=10.0,
        chunks=chunks,
        fps=fps,
        sample_interval=0.5,
    )


def test_map_windows_to_chunks():
    window_times = np.arange(20, dtype=np.float32) * 0.5
    chunks = [(0, 5), (5, 10), (10, 15), (15, 20)]
    fps = 2.0
    mapping = _map_windows_to_chunks(window_times, chunks, fps)
    assert mapping[0] == 0   # t=0.0s, chunk 0: 0-2.5s
    assert mapping[4] == 0   # t=2.0s, chunk 0: 0-2.5s
    assert mapping[5] == 1   # t=2.5s, chunk 1: 2.5-5s
    assert mapping[10] == 2  # t=5.0s, chunk 2: 5-7.5s
    assert mapping[15] == 3  # t=7.5s, chunk 3: 7.5-10s


# ---------------------------------------------------------------------------
# mmr_chunk_penalty tests
# ---------------------------------------------------------------------------

def test_mmr_chunk_penalty_returns_correct_count():
    scenario = _make_test_scenario()
    result = select_frames_from_windows(method="mmr_chunk_penalty", **scenario)
    assert len(result["window_indices"]) == 4
    assert result["window_indices"] == sorted(result["window_indices"])


def test_mmr_chunk_penalty_spreads_across_chunks():
    """With high chunk_weight, MMR should spread more than without it."""
    scenario = _make_test_scenario()
    result_with = select_frames_from_windows(
        method="mmr_chunk_penalty", chunk_weight=0.8, **scenario,
    )
    result_without = select_frames_from_windows(
        method="mmr_chunk_penalty", chunk_weight=0.0, **scenario,
    )
    mapping = _map_windows_to_chunks(
        scenario["window_times"], scenario["chunks"], scenario["fps"],
    )
    chunks_with = set(mapping[i] for i in result_with["window_indices"])
    chunks_without = set(mapping[i] for i in result_without["window_indices"])
    assert len(chunks_with) >= len(chunks_without)


def test_mmr_chunk_penalty_zero_weight_matches_mmr():
    """With chunk_weight=0 and temporal_weight=0, should match plain MMR."""
    scenario = _make_test_scenario()
    result_penalty = select_frames_from_windows(
        method="mmr_chunk_penalty", chunk_weight=0.0,
        temporal_weight=0.0, **scenario,
    )
    result_mmr = select_frames_from_windows(method="mmr", **scenario)
    assert result_penalty["window_indices"] == result_mmr["window_indices"]


# ---------------------------------------------------------------------------
# mmr_chunk_constrained tests
# ---------------------------------------------------------------------------

def test_mmr_chunk_constrained_returns_correct_count():
    scenario = _make_test_scenario()
    result = select_frames_from_windows(
        method="mmr_chunk_constrained", **scenario,
    )
    assert len(result["window_indices"]) == 4
    assert result["window_indices"] == sorted(result["window_indices"])


def test_mmr_chunk_constrained_one_per_chunk():
    """With 4 chunks and 4 frames, should pick exactly one from each chunk."""
    scenario = _make_test_scenario()
    result = select_frames_from_windows(
        method="mmr_chunk_constrained", **scenario,
    )
    mapping = _map_windows_to_chunks(
        scenario["window_times"], scenario["chunks"], scenario["fps"],
    )
    selected_chunks = [mapping[i] for i in result["window_indices"]]
    assert len(set(selected_chunks)) == 4


def test_mmr_chunk_constrained_fallback_when_more_frames_than_chunks():
    """When n_frames > n_chunks, should still return n_frames frames."""
    scenario = _make_test_scenario()
    scenario["n_frames"] = 8
    result = select_frames_from_windows(
        method="mmr_chunk_constrained", **scenario,
    )
    assert len(result["window_indices"]) == 8


# ---------------------------------------------------------------------------
# intra_chunk_greedy tests
# ---------------------------------------------------------------------------

def test_intra_chunk_greedy_returns_correct_count():
    scenario = _make_test_scenario()
    result = select_frames_from_windows(method="intra_chunk_greedy", **scenario)
    assert len(result["window_indices"]) == 4
    assert result["window_indices"] == sorted(result["window_indices"])


def test_intra_chunk_greedy_picks_peak_window_per_chunk_when_frames_equal_chunks():
    """With 4 chunks and 4 frames, each chunk contributes its single top-relevance window."""
    scenario = _make_test_scenario()
    result = select_frames_from_windows(method="intra_chunk_greedy", **scenario)
    # Peaks: chunk 0 → window 4 (0.9), chunk 1 → window 6 (0.3),
    #        chunk 2 → window 12 (0.85), chunk 3 → window 17 (0.5).
    assert set(result["window_indices"]) == {4, 6, 12, 17}


def test_intra_chunk_greedy_fewer_frames_than_chunks_picks_top_peak_chunks():
    """With 2 frames and 4 chunks, top 2 chunks by peak each get 1 window."""
    scenario = _make_test_scenario()
    scenario["n_frames"] = 2
    result = select_frames_from_windows(method="intra_chunk_greedy", **scenario)
    # Top-2 peaks are chunk 0 (0.9) and chunk 2 (0.85).
    assert set(result["window_indices"]) == {4, 12}


def test_intra_chunk_greedy_distributes_remainder_to_top_peak_chunks():
    """With 6 frames and 4 chunks: base=1 each, remainder=2 goes to top-2-peak chunks."""
    scenario = _make_test_scenario()
    scenario["n_frames"] = 6
    result = select_frames_from_windows(method="intra_chunk_greedy", **scenario)
    mapping = _map_windows_to_chunks(
        scenario["window_times"], scenario["chunks"], scenario["fps"],
    )
    from collections import Counter
    chunk_counts = Counter(mapping[i] for i in result["window_indices"])
    # Chunks 0 and 2 (highest peaks) each get 2; chunks 1 and 3 get 1.
    assert chunk_counts[0] == 2
    assert chunk_counts[2] == 2
    assert chunk_counts[1] == 1
    assert chunk_counts[3] == 1


def test_intra_chunk_greedy_picks_intra_chunk_top_k_by_relevance():
    """When a chunk gets 2 frames, it picks its top-2 windows by relevance, not MMR."""
    scenario = _make_test_scenario()
    scenario["n_frames"] = 6
    result = select_frames_from_windows(method="intra_chunk_greedy", **scenario)
    # Chunk 0 windows 0-4, relevances [0.3, 0.4, 0.5, 0.6, 0.9] → top-2 are windows 4 and 3.
    # Chunk 2 windows 10-14, relevances [0.7, 0.8, 0.85, 0.6, 0.5] → top-2 are windows 12 and 11.
    selected = set(result["window_indices"])
    assert {3, 4}.issubset(selected)
    assert {11, 12}.issubset(selected)
