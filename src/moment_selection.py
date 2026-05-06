"""Moment selection strategies for choosing which chunks to return.

Three methods are provided so they can be compared directly:

  penalized_dp  — PELT-inspired DP that trades off total reward against
                  a per-moment penalty.
  top_gap       — Keep every chunk whose score is within ``gap`` of the
                  best chunk, then merge adjacent selected chunks.
  score_drop    — Sort scores descending, find the largest gap between
                  consecutive scores, keep everything above the gap.
                  Zero parameters.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _merge_adjacent(selected_indices, chunks, scores, fps):
    """Merge runs of adjacent selected chunk indices into moments."""
    if len(selected_indices) == 0:
        return []
    selected_indices = sorted(selected_indices)
    runs = []
    start = selected_indices[0]
    end = selected_indices[0]
    for idx in selected_indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            runs.append((start, end))
            start = idx
            end = idx
    runs.append((start, end))

    predictions = []
    for si, ei in runs:
        predictions.append({
            "start": chunks[si][0] / fps,
            "end": chunks[ei][1] / fps,
            "score": float(scores[si:ei + 1].max()),
        })
    predictions.sort(key=lambda p: p["score"], reverse=True)
    return predictions


# ---------------------------------------------------------------------------
# Method 1: penalized DP
# ---------------------------------------------------------------------------

def _penalized_dp(chunks, scores, fps, penalty, penalty_factor,
                  max_moment_sec):
    n = len(scores)
    if penalty is not None:
        threshold = float(penalty)
    else:
        threshold = (float(np.std(scores)) * penalty_factor
                     * np.sqrt(2.0 * np.log(max(n, 2))))

    centered = scores - float(np.mean(scores))
    prefix = np.concatenate([[0.0], np.cumsum(centered)])

    dp = np.zeros(n + 1, dtype=np.float64)
    back = np.full(n + 1, -1, dtype=np.int32)

    for j in range(1, n + 1):
        dp[j] = dp[j - 1]
        for i in range(j):
            if max_moment_sec is not None:
                dur = (chunks[j - 1][1] - chunks[i][0]) / fps
                if dur > max_moment_sec:
                    continue
            reward = float(prefix[j] - prefix[i])
            candidate = dp[i] + reward - threshold
            if candidate > dp[j] + 1e-9:
                dp[j] = candidate
                back[j] = i

    moments = []
    cursor = n
    while cursor > 0:
        start = int(back[cursor])
        if start < 0:
            cursor -= 1
        else:
            moments.append((start, cursor - 1))
            cursor = start
    moments.reverse()

    predictions = []
    for si, ei in moments:
        predictions.append({
            "start": chunks[si][0] / fps,
            "end": chunks[ei][1] / fps,
            "score": float(scores[si:ei + 1].max()),
        })
    predictions.sort(key=lambda p: p["score"], reverse=True)

    return predictions, {
        "penalty_used": threshold,
        "penalty_auto": penalty is None,
    }


# ---------------------------------------------------------------------------
# Method 2: top-gap threshold
# ---------------------------------------------------------------------------

def _top_gap(chunks, scores, fps, gap):
    n = len(scores)
    best = float(scores.max())
    cutoff = best - gap
    selected_set = set(i for i in range(n) if scores[i] >= cutoff)
    selected = _merge_adjacent(sorted(selected_set), chunks, scores, fps)
    remaining = [
        {"start": chunks[i][0] / fps, "end": chunks[i][1] / fps,
         "score": float(scores[i])}
        for i in range(n) if i not in selected_set
    ]
    remaining.sort(key=lambda p: p["score"], reverse=True)
    predictions = selected + remaining
    return predictions, {
        "gap": gap, "cutoff": cutoff, "best_score": best,
        "num_selected": len(selected),
    }


# ---------------------------------------------------------------------------
# Method 3: score-drop (largest gap in sorted scores)
# ---------------------------------------------------------------------------

def _score_drop(chunks, scores, fps):
    n = len(scores)
    if n <= 1:
        return _merge_adjacent(list(range(n)), chunks, scores, fps), {}

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    gaps = sorted_scores[:-1] - sorted_scores[1:]
    cut_pos = int(np.argmax(gaps)) + 1
    cutoff = float(sorted_scores[cut_pos - 1])

    selected = [i for i in range(n) if scores[i] >= cutoff]
    predictions = _merge_adjacent(selected, chunks, scores, fps)
    return predictions, {
        "cutoff": cutoff,
        "max_gap": float(gaps.max()),
        "cut_after_rank": cut_pos,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def select_moments(chunks, chunk_scores, fps, method="penalized_dp",
                   penalty=None, penalty_factor=1.0,
                   max_moment_sec=None, gap=0.05):
    """Select moments from scored chunks.

    Args:
        chunks: list of (start_frame, end_frame) tuples.
        chunk_scores: per-chunk similarity scores.
        fps: video frames per second.
        method: one of "penalized_dp", "top_gap", "score_drop".
        penalty: (penalized_dp) fixed per-moment penalty, or None for auto.
        penalty_factor: (penalized_dp) multiplier for auto penalty.
        max_moment_sec: (penalized_dp) optional moment duration cap.
        gap: (top_gap) max allowed score difference from the best chunk.

    Returns:
        (predictions, diagnostics).
    """
    scores = np.asarray(chunk_scores, dtype=np.float32).ravel()
    n = len(scores)
    if n == 0:
        return [], {"method": method, "num_chunks": 0, "num_moments": 0}

    if method == "penalized_dp":
        predictions, extra = _penalized_dp(
            chunks, scores, fps, penalty, penalty_factor, max_moment_sec)
    elif method == "top_gap":
        predictions, extra = _top_gap(chunks, scores, fps, gap)
    elif method == "score_drop":
        predictions, extra = _score_drop(chunks, scores, fps)
    else:
        raise ValueError(f"Unknown moment selection method: {method}")

    diagnostics = {
        "method": method,
        "num_chunks": n,
        "num_moments": len(predictions),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        **extra,
    }
    return predictions, diagnostics
