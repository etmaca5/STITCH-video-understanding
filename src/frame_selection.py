"""Window-embedding-based frame selection methods.

Each method takes dense window embeddings across a video and selects
N frame positions that balance query relevance with temporal coverage
or embedding diversity.
"""

import logging

import numpy as np
import torch
from torch.nn.functional import cosine_similarity as torch_cos_sim

log = logging.getLogger(__name__)

WINDOW_METHODS = {
    "weighted", "mmr", "temporal", "coverage", "rdmv",
    "nudge", "stretch", "constrained", "focused", "segment_adaptive",
    "uniform_topk", "random",
    "mmr_chunk_penalty", "mmr_chunk_constrained",
    "intra_chunk_greedy",
}


def select_frames_from_windows(
    method: str,
    window_embeddings: np.ndarray,
    window_times: np.ndarray,
    query_embedding: np.ndarray,
    n_frames: int,
    duration_sec: float,
    chunks: list[tuple[int, int]] | None = None,
    fps: float | None = None,
    sample_interval: float = 1.0,
    **method_kwargs,
) -> dict:
    """Select n_frames window positions using the specified method.

    Returns dict with:
        window_indices: selected indices into window arrays, sorted by time
        frame_indices: corresponding video frame indices
        frame_times: corresponding times in seconds
        scores: per-selected-window query relevance scores
    """
    if method not in WINDOW_METHODS:
        raise ValueError(f"Unknown window selection method: {method}")

    n_windows = len(window_embeddings)

    query_t = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
    emb_t = torch.tensor(window_embeddings, dtype=torch.float32)
    relevance = torch_cos_sim(
        query_t.expand(n_windows, -1), emb_t,
    ).numpy()

    if method == "random":
        return _select_random_frames(
            relevance=relevance,
            window_times=window_times,
            n_frames=n_frames,
            duration_sec=duration_sec,
            fps=fps,
            sample_interval=sample_interval,
            seed=method_kwargs.get("seed", 42),
        )

    n_frames = min(n_frames, n_windows)

    selector = {
        "weighted": _select_weighted,
        "mmr": _select_mmr,
        "temporal": _select_temporal,
        "coverage": _select_coverage,
        "rdmv": _select_rdmv,
        "nudge": _select_nudge,
        "stretch": _select_stretch,
        "constrained": _select_constrained,
        "focused": _select_focused,
        "segment_adaptive": _select_segment_adaptive,
        "uniform_topk": _select_uniform_topk,
        "random": _select_random,
        "mmr_chunk_penalty": _select_mmr_chunk_penalty,
        "mmr_chunk_constrained": _select_mmr_chunk_constrained,
        "intra_chunk_greedy": _select_intra_chunk_greedy,
    }[method]

    selected = selector(
        relevance=relevance,
        embeddings=emb_t,
        window_times=window_times,
        n_frames=n_frames,
        duration_sec=duration_sec,
        chunks=chunks,
        fps=fps,
        **method_kwargs,
    )
    selected = sorted(selected)

    scores = relevance[selected].tolist()
    half_interval = sample_interval / 2.0

    frame_times = [float(window_times[i] + half_interval) for i in selected]
    if fps is not None and fps > 0:
        total_frames = int(duration_sec * fps)
        frame_indices = [
            min(int(frame_times[j] * fps), total_frames - 1)
            for j in range(len(selected))
        ]
    else:
        frame_indices = [int(ft) for ft in frame_times]

    return {
        "window_indices": selected,
        "frame_indices": frame_indices,
        "frame_times": frame_times,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Method 1: weighted — relevance-weighted chunk allocation
# ---------------------------------------------------------------------------

def _select_weighted(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, **kwargs,
) -> list[int]:
    """Allocate frames across chunks proportional to relevance."""
    if chunks is None or fps is None:
        return _select_weighted_no_chunks(relevance, window_times, n_frames)

    n_chunks = len(chunks)
    chunk_best_windows = []
    chunk_best_scores = []
    for start_f, end_f in chunks:
        s_sec = start_f / fps
        e_sec = end_f / fps
        mask = (window_times >= s_sec) & (window_times < e_sec)
        if mask.any():
            indices = np.flatnonzero(mask)
            best_in_chunk = indices[np.argmax(relevance[indices])]
            chunk_best_windows.append(best_in_chunk)
            chunk_best_scores.append(float(relevance[best_in_chunk]))
        else:
            nearest = int(np.abs(window_times - (s_sec + e_sec) / 2).argmin())
            chunk_best_windows.append(nearest)
            chunk_best_scores.append(float(relevance[nearest]))

    chunk_best_scores = np.array(chunk_best_scores)

    if n_frames >= n_chunks:
        selected = set(chunk_best_windows)
        remaining = n_frames - len(selected)
        if remaining > 0:
            ranked = np.argsort(-chunk_best_scores)
            for ci in ranked:
                if remaining <= 0:
                    break
                start_f, end_f = chunks[ci]
                s_sec, e_sec = start_f / fps, end_f / fps
                mask = (window_times >= s_sec) & (window_times < e_sec)
                candidates = np.flatnonzero(mask)
                candidates = [c for c in candidates if c not in selected]
                candidates.sort(key=lambda c: -relevance[c])
                for c in candidates:
                    if remaining <= 0:
                        break
                    selected.add(c)
                    remaining -= 1
        return list(selected)
    else:
        return _select_weighted_subsample_chunks(
            chunk_best_windows, chunk_best_scores, window_times,
            n_frames, duration_sec,
        )


def _select_weighted_no_chunks(relevance, window_times, n_frames):
    """Fallback when no chunks: divide timeline into n_frames spans."""
    n_windows = len(window_times)
    span = (window_times[-1] - window_times[0]) / n_frames if n_frames > 0 else 1.0
    selected = []
    for i in range(n_frames):
        t_start = window_times[0] + i * span
        t_end = t_start + span
        mask = (window_times >= t_start) & (window_times < t_end)
        if mask.any():
            candidates = np.flatnonzero(mask)
            best = candidates[np.argmax(relevance[candidates])]
        else:
            best = int(np.abs(window_times - (t_start + t_end) / 2).argmin())
        if best not in selected:
            selected.append(best)
    while len(selected) < n_frames and len(selected) < n_windows:
        ranked = np.argsort(-relevance)
        for idx in ranked:
            if idx not in selected:
                selected.append(idx)
                break
    return selected


def _select_weighted_subsample_chunks(
    chunk_best_windows, chunk_best_scores, window_times,
    n_frames, duration_sec,
):
    """When more chunks than frames: pick most relevant chunk per time span."""
    chunk_times = np.array([window_times[w] for w in chunk_best_windows])
    n_chunks = len(chunk_best_windows)
    span = duration_sec / n_frames if n_frames > 0 else duration_sec
    selected = []
    for i in range(n_frames):
        t_start = i * span
        t_end = t_start + span
        mask = (chunk_times >= t_start) & (chunk_times < t_end)
        if mask.any():
            candidates = np.flatnonzero(mask)
            best_ci = candidates[np.argmax(chunk_best_scores[candidates])]
        else:
            best_ci = int(np.abs(chunk_times - (t_start + t_end) / 2).argmin())
        w = chunk_best_windows[best_ci]
        if w not in selected:
            selected.append(w)
    while len(selected) < n_frames:
        ranked = np.argsort(-chunk_best_scores)
        for ci in ranked:
            w = chunk_best_windows[ci]
            if w not in selected:
                selected.append(w)
                break
    return selected


# ---------------------------------------------------------------------------
# Method 2: mmr — Maximal Marginal Relevance
# ---------------------------------------------------------------------------

def _select_mmr(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, relevance_weight=0.7, temporal_weight=0.0,
    **kwargs,
) -> list[int]:
    """Greedy MMR: balance query relevance against redundancy.

    temporal_weight blends temporal proximity into the penalty:
        closeness = (1-β)*emb_sim + β*(1 - |dt|/duration)
    When 0, standard embedding-only MMR. When >0, also penalizes temporal closeness.
    """
    n_windows = len(relevance)
    lam = float(relevance_weight)
    beta = float(temporal_weight)
    dur = max(float(duration_sec), 1e-6)
    times = np.asarray(window_times, dtype=float)

    first = int(np.argmax(relevance))
    selected = [first]
    selected_embs = embeddings[first].unsqueeze(0)
    selected_times = [times[first]]

    for _ in range(n_frames - 1):
        best_score = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected:
                continue
            emb_sims = torch_cos_sim(
                embeddings[i].unsqueeze(0).expand(len(selected), -1),
                selected_embs,
            )
            if beta > 0:
                t_dists = np.abs(times[i] - np.array(selected_times))
                t_prox = 1.0 - t_dists / dur
                closeness = (1.0 - beta) * emb_sims.numpy() + beta * t_prox
                max_close = float(closeness.max())
            else:
                max_close = float(emb_sims.max())
            score = lam * relevance[i] - (1.0 - lam) * max_close
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_embs = torch.cat(
            [selected_embs, embeddings[best_idx].unsqueeze(0)], dim=0,
        )
        selected_times.append(times[best_idx])

    return selected


# ---------------------------------------------------------------------------
# Shared helper: map each window to its chunk index
# ---------------------------------------------------------------------------

def _map_windows_to_chunks(window_times, chunks, fps):
    """Return an int array mapping each window index to its chunk index (-1 if none)."""
    mapping = np.full(len(window_times), -1, dtype=int)
    for ci, (start_f, end_f) in enumerate(chunks):
        s_sec = start_f / fps
        e_sec = end_f / fps
        mask = (window_times >= s_sec) & (window_times < e_sec)
        mapping[mask] = ci
    return mapping


# ---------------------------------------------------------------------------
# Method 2b: mmr_chunk_penalty — MMR with same-chunk penalty
#   USE WITH query_merge.enabled=true
# ---------------------------------------------------------------------------

def _select_mmr_chunk_penalty(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None,
    relevance_weight=0.7, temporal_weight=0.0, chunk_weight=0.3,
    **kwargs,
) -> list[int]:
    """Greedy MMR with an additive penalty for picking from the same chunk.

    chunk_weight (gamma) blends a same-chunk indicator into the closeness term:
        closeness = (1-beta-gamma)*emb_sim + beta*temporal_prox + gamma*same_chunk
    Falls back to standard MMR when chunks/fps are unavailable.
    """
    if chunks is None or fps is None:
        return _select_mmr(
            relevance, embeddings, window_times, n_frames, duration_sec,
            relevance_weight=relevance_weight, temporal_weight=temporal_weight,
        )

    n_windows = len(relevance)
    lam = float(relevance_weight)
    beta = float(temporal_weight)
    gamma = float(chunk_weight)
    dur = max(float(duration_sec), 1e-6)
    times = np.asarray(window_times, dtype=float)
    win_chunks = _map_windows_to_chunks(times, chunks, fps)

    first = int(np.argmax(relevance))
    selected = [first]
    selected_embs = embeddings[first].unsqueeze(0)
    selected_times = [times[first]]
    selected_chunks = {win_chunks[first]}

    for _ in range(n_frames - 1):
        best_score = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected:
                continue
            emb_sims = torch_cos_sim(
                embeddings[i].unsqueeze(0).expand(len(selected), -1),
                selected_embs,
            )
            max_emb_sim = float(emb_sims.max())

            t_prox_max = 0.0
            if beta > 0:
                t_dists = np.abs(times[i] - np.array(selected_times))
                t_prox_max = float((1.0 - t_dists / dur).max())

            same_chunk = 1.0 if win_chunks[i] in selected_chunks else 0.0

            closeness = (
                (1.0 - beta - gamma) * max_emb_sim
                + beta * t_prox_max
                + gamma * same_chunk
            )
            score = lam * relevance[i] - (1.0 - lam) * closeness
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_embs = torch.cat(
            [selected_embs, embeddings[best_idx].unsqueeze(0)], dim=0,
        )
        selected_times.append(times[best_idx])
        selected_chunks.add(win_chunks[best_idx])

    return selected


# ---------------------------------------------------------------------------
# Method 2c: mmr_chunk_constrained — MMR with at-most-one-per-chunk constraint
#   USE WITH query_merge.enabled=false
# ---------------------------------------------------------------------------

def _select_mmr_chunk_constrained(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None,
    relevance_weight=0.7, temporal_weight=0.0,
    **kwargs,
) -> list[int]:
    """Greedy MMR where each chunk can contribute at most one frame.

    Once a window is picked from chunk c, all other windows in c are excluded.
    If all chunks are exhausted before n_frames, the constraint is lifted and
    remaining picks use standard MMR over unused windows.
    """
    if chunks is None or fps is None:
        return _select_mmr(
            relevance, embeddings, window_times, n_frames, duration_sec,
            relevance_weight=relevance_weight, temporal_weight=temporal_weight,
        )

    n_windows = len(relevance)
    lam = float(relevance_weight)
    beta = float(temporal_weight)
    dur = max(float(duration_sec), 1e-6)
    times = np.asarray(window_times, dtype=float)
    win_chunks = _map_windows_to_chunks(times, chunks, fps)

    excluded = set()
    used_chunks: set[int] = set()

    first = int(np.argmax(relevance))
    selected = [first]
    selected_embs = embeddings[first].unsqueeze(0)
    selected_times = [times[first]]
    fc = win_chunks[first]
    if fc >= 0:
        used_chunks.add(fc)
        for w in np.flatnonzero(win_chunks == fc):
            if int(w) != first:
                excluded.add(int(w))

    constrained = True
    while len(selected) < n_frames:
        best_score = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected or (constrained and i in excluded):
                continue
            emb_sims = torch_cos_sim(
                embeddings[i].unsqueeze(0).expand(len(selected), -1),
                selected_embs,
            )
            if beta > 0:
                t_dists = np.abs(times[i] - np.array(selected_times))
                t_prox = 1.0 - t_dists / dur
                closeness = (1.0 - beta) * emb_sims.numpy() + beta * t_prox
                max_close = float(closeness.max())
            else:
                max_close = float(emb_sims.max())
            score = lam * relevance[i] - (1.0 - lam) * max_close
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            if constrained:
                constrained = False
                excluded.clear()
                continue
            break

        selected.append(best_idx)
        selected_embs = torch.cat(
            [selected_embs, embeddings[best_idx].unsqueeze(0)], dim=0,
        )
        selected_times.append(times[best_idx])

        if constrained:
            bc = win_chunks[best_idx]
            if bc >= 0:
                used_chunks.add(bc)
                for w in np.flatnonzero(win_chunks == bc):
                    if int(w) not in selected:
                        excluded.add(int(w))

    return selected


# ---------------------------------------------------------------------------
# Method 2d: intra_chunk_greedy — allocate across chunks, then top-k per chunk
#   USE WITH query_merge.enabled=true
# ---------------------------------------------------------------------------

def _select_intra_chunk_greedy(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None,
    **kwargs,
) -> list[int]:
    """Allocate the frame budget across chunks, then pick top-k windows per chunk.

    Allocation (non-empty chunks only, ranked by peak relevance):
      - If n_frames <= n_valid_chunks: the top n_frames chunks each get 1 frame.
      - Otherwise: every chunk gets floor(n_frames / n_valid_chunks) frames,
        and the top (n_frames mod n_valid_chunks) chunks each get +1.

    This matches round-robin filling by peak-relevance order: round 1 gives
    every chunk 1 frame, round 2 gives every chunk a 2nd frame (stopping when
    budget is exhausted), etc.

    Within each allocated chunk: pick that chunk's top-b windows by raw
    relevance (no diversity term).

    Fallback behavior:
      - No chunks/fps passed, or no non-empty chunks: global top-k by relevance.
      - Chunk has fewer windows than its allocation: take what's available;
        leftover slots are filled from the globally-highest-relevance unused
        windows.
    """
    if chunks is None or fps is None:
        ranked = np.argsort(-relevance)
        return [int(i) for i in ranked[:n_frames]]

    times = np.asarray(window_times, dtype=float)
    n_chunks = len(chunks)

    chunk_windows: list[list[int]] = []
    chunk_peaks: list[float] = []
    for start_f, end_f in chunks:
        s_sec = start_f / fps
        e_sec = end_f / fps
        mask = (times >= s_sec) & (times < e_sec)
        indices = np.flatnonzero(mask).tolist()
        chunk_windows.append(indices)
        chunk_peaks.append(float(np.max(relevance[indices])) if indices else -np.inf)

    valid_chunks = [ci for ci in range(n_chunks) if chunk_windows[ci]]
    n_valid = len(valid_chunks)
    if n_valid == 0:
        ranked = np.argsort(-relevance)
        return [int(i) for i in ranked[:n_frames]]

    valid_peaks = np.array([chunk_peaks[ci] for ci in valid_chunks], dtype=float)
    ranked_valid = [valid_chunks[i] for i in np.argsort(-valid_peaks)]

    allocations = np.zeros(n_chunks, dtype=int)
    if n_frames <= n_valid:
        for ci in ranked_valid[:n_frames]:
            allocations[ci] = 1
    else:
        base = n_frames // n_valid
        extra = n_frames % n_valid
        for ci in valid_chunks:
            allocations[ci] = base
        for ci in ranked_valid[:extra]:
            allocations[ci] += 1

    selected: list[int] = []
    selected_set: set[int] = set()
    for ci in range(n_chunks):
        budget = int(allocations[ci])
        windows = chunk_windows[ci]
        if budget <= 0 or not windows:
            continue
        local = _select_segment_topk(windows, relevance, budget)
        for idx in local:
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))

    if len(selected) < n_frames:
        ranked = np.argsort(-relevance)
        for idx in ranked:
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= n_frames:
                break

    return selected[:n_frames]


# ---------------------------------------------------------------------------
# Method 3: temporal — temporal-penalized relevance
# ---------------------------------------------------------------------------

def _select_temporal(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, spacing_factor=2.0, duration_scale=False,
    **kwargs,
) -> list[int]:
    """Greedy selection penalizing temporal proximity to already-selected.

    duration_scale: when True, raise the penalty to a duration-dependent
    exponent so longer videos get a harsher proximity penalty.
    """
    min_spacing = duration_sec / (float(spacing_factor) * n_frames) if n_frames > 0 else 0.0
    exponent = 1.0
    if duration_scale:
        exponent = 1.0 + np.log10(max(duration_sec / 60.0, 1.0))

    first = int(np.argmax(relevance))
    selected = [first]
    selected_times = [float(window_times[first])]

    n_windows = len(relevance)
    for _ in range(n_frames - 1):
        best_score = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected:
                continue
            t = float(window_times[i])
            dist = min(abs(t - st) for st in selected_times)
            temporal_factor = min(dist / min_spacing, 1.0) if min_spacing > 0 else 1.0
            temporal_factor = temporal_factor ** exponent
            score = relevance[i] * temporal_factor
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_times.append(float(window_times[best_idx]))

    return selected


# ---------------------------------------------------------------------------
# Method 3b: constrained — hard minimum distance, greedy by relevance
# ---------------------------------------------------------------------------

def _select_constrained(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, k=2.0, **kwargs,
) -> list[int]:
    """Greedy relevance selection with a hard minimum distance constraint.

    min_dist = duration / n_frames / k
    This is uniform spacing divided by k, so k=1 ≈ uniform, k→∞ ≈ pure relevance.

    If the constraint is too tight to fill n_frames (e.g. short videos where
    window spacing > min_dist), the constraint is progressively halved until
    all frames can be placed.
    """
    times = np.asarray(window_times, dtype=float)
    n_windows = len(times)
    ranked = np.argsort(-relevance)

    min_dist = duration_sec / n_frames / float(k) if n_frames > 0 else 0.0
    if n_windows > 1 and n_frames > 1:
        max_feasible = (times[-1] - times[0]) / (n_frames - 1)
        min_dist = min(min_dist, max_feasible)

    while True:
        selected = []
        selected_times = []
        for idx in ranked:
            if len(selected) >= n_frames:
                break
            t = times[int(idx)]
            if selected_times and min(abs(t - st) for st in selected_times) < min_dist:
                continue
            selected.append(int(idx))
            selected_times.append(t)

        if len(selected) >= n_frames or min_dist <= 0:
            break
        min_dist *= 0.5

    return selected


# ---------------------------------------------------------------------------
# Method 4: coverage — facility location + relevance
# ---------------------------------------------------------------------------

def _select_coverage(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, relevance_weight=0.5, **kwargs,
) -> list[int]:
    """Greedy submodular maximization: coverage + relevance.

    f(S) = sum_i max_{s in S} sim(w_i, w_s) + lambda * sum_{s in S} rel(s)
    """
    n_windows = len(relevance)
    lam = float(relevance_weight)

    sim_matrix = torch_cos_sim(
        embeddings.unsqueeze(1).expand(-1, n_windows, -1).reshape(-1, embeddings.shape[1]),
        embeddings.unsqueeze(0).expand(n_windows, -1, -1).reshape(-1, embeddings.shape[1]),
    ).reshape(n_windows, n_windows).numpy()

    max_sim_to_selected = np.full(n_windows, -np.inf)

    selected = []
    for _ in range(n_frames):
        best_gain = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected:
                continue
            new_coverage = np.maximum(max_sim_to_selected, sim_matrix[:, i])
            coverage_gain = float(new_coverage.sum() - max_sim_to_selected.clip(min=0).sum())
            gain = coverage_gain + lam * relevance[i]
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim_matrix[:, best_idx])

    return selected


# ---------------------------------------------------------------------------
# Method 5: rdmv — Relevance-Diversity Max-Volume (AdaRD-Key inspired)
# ---------------------------------------------------------------------------

def _select_rdmv(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None,
    lambda_min=0.05, lambda_max=0.6, alpha=2.0, gate_threshold=0.4,
    **kwargs,
) -> list[int]:
    """Greedy RD-MV: relevance + log-det diversity with adaptive lambda."""
    n_windows = len(relevance)
    emb_np = embeddings.numpy()
    eps = 1e-6

    lam = _compute_adaptive_lambda(
        relevance, n_windows, n_frames,
        lambda_min=lambda_min, lambda_max=lambda_max, alpha=alpha,
    )

    use_relevance = float(np.max(relevance)) >= gate_threshold

    gram_inv = np.eye(0, dtype=np.float64)
    log_det_val = 0.0
    selected = []
    selected_embs = np.zeros((0, emb_np.shape[1]), dtype=np.float64)

    for _ in range(n_frames):
        best_gain = -float("inf")
        best_idx = -1
        for i in range(n_windows):
            if i in selected:
                continue
            fi = emb_np[i].astype(np.float64)
            if len(selected) == 0:
                diversity_gain = float(np.log(np.dot(fi, fi) + eps))
            else:
                r = selected_embs @ fi
                schur = float(np.dot(fi, fi) + eps - r @ gram_inv @ r)
                if schur <= 0:
                    diversity_gain = -1e12
                else:
                    diversity_gain = float(np.log(schur))

            if use_relevance:
                gain = float(relevance[i]) + lam * diversity_gain
            else:
                gain = diversity_gain
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)

        fi = emb_np[best_idx].astype(np.float64)
        if len(selected) == 1:
            val = float(np.dot(fi, fi) + eps)
            gram_inv = np.array([[1.0 / val]], dtype=np.float64)
            selected_embs = fi.reshape(1, -1)
        else:
            r = selected_embs @ fi
            schur = float(np.dot(fi, fi) + eps - r @ gram_inv @ r)
            if schur <= 1e-12:
                schur = 1e-12
            v = gram_inv @ r
            k = len(selected) - 1
            new_inv = np.zeros((k + 1, k + 1), dtype=np.float64)
            new_inv[:k, :k] = gram_inv + np.outer(v, v) / schur
            new_inv[:k, k] = -v / schur
            new_inv[k, :k] = -v / schur
            new_inv[k, k] = 1.0 / schur
            gram_inv = new_inv
            selected_embs = np.vstack([selected_embs, fi.reshape(1, -1)])

    return selected


def _compute_adaptive_lambda(
    relevance, n_windows, n_frames,
    lambda_min=0.05, lambda_max=0.6, alpha=2.0,
):
    """VB-Scale: adaptive lambda from relevance variability and budget ratio."""
    rho = n_windows / max(n_frames, 1)
    rho_cap = 8.0

    mean_r = float(np.mean(relevance))
    std_r = float(np.std(relevance))
    cv = std_r / (mean_r + 1e-8)

    lambda_var = lambda_min + lambda_max / (1.0 + alpha * cv)

    lambda_bud = lambda_max * min(1.0, np.log(rho + 1e-8) / np.log(rho_cap))
    lambda_bud = max(lambda_bud, 0.0)

    w = 1.0 / (1.0 + np.exp(-(rho - 1.0)))
    lam = w * lambda_bud + (1.0 - w) * lambda_var
    return float(np.clip(lam, lambda_min, lambda_max))


# ---------------------------------------------------------------------------
# Method 6: nudge — uniform positions refined by local relevance
# ---------------------------------------------------------------------------

def _select_nudge(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, radius_factor=0.5, **kwargs,
) -> list[int]:
    """Start with uniform positions, nudge each to the most relevant nearby window.

    radius_factor: search radius as a fraction of uniform spacing.
        0.5 = half the spacing (adjacent frames can't swap past each other).
    """
    times = np.asarray(window_times, dtype=float)
    dur = max(float(duration_sec), 1e-6)
    spacing = dur / n_frames
    radius = spacing * float(radius_factor)

    uniform_times = np.linspace(0, dur - spacing, n_frames) + spacing / 2.0
    selected = []
    used = set()
    for target_t in uniform_times:
        mask = np.abs(times - target_t) <= radius
        candidates = np.flatnonzero(mask)
        candidates = [c for c in candidates if c not in used]
        if candidates:
            best = max(candidates, key=lambda c: relevance[c])
        else:
            best = int(np.abs(times - target_t).argmin())
            if best in used:
                all_unused = [i for i in range(len(times)) if i not in used]
                if all_unused:
                    best = min(all_unused, key=lambda i: abs(times[i] - target_t))
        selected.append(best)
        used.add(best)

    return selected


# ---------------------------------------------------------------------------
# Method 7: stretch — relevance-biased spacing across full timeline
# ---------------------------------------------------------------------------

def _select_stretch(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, relevance_bias=0.5, **kwargs,
) -> list[int]:
    """Space frames denser in high-relevance regions, sparser in low.

    relevance_bias: 0 = pure uniform, 1 = fully relevance-weighted.
    Always covers the full video timeline.
    """
    times = np.asarray(window_times, dtype=float)
    n_windows = len(times)
    bias = float(relevance_bias)

    rel = np.clip(relevance, 0, None)
    uniform_density = np.ones(n_windows, dtype=float)
    density = (1.0 - bias) * uniform_density + bias * (rel / (rel.sum() + 1e-8) * n_windows)
    density = np.clip(density, 0.1, None)

    cdf = np.cumsum(density)
    cdf = cdf / cdf[-1]

    targets = np.linspace(0.5 / n_frames, 1.0 - 0.5 / n_frames, n_frames)
    selected = []
    used = set()
    for t in targets:
        idx = int(np.searchsorted(cdf, t))
        idx = min(idx, n_windows - 1)
        if idx in used:
            for offset in range(1, n_windows):
                for candidate in [idx + offset, idx - offset]:
                    if 0 <= candidate < n_windows and candidate not in used:
                        idx = candidate
                        break
                else:
                    continue
                break
        selected.append(idx)
        used.add(idx)

    return selected


# ---------------------------------------------------------------------------
# Method 8: focused — half query-relevant, half video-overview
# ---------------------------------------------------------------------------

def _select_focused(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, context_mode="uniform", merge_threshold=0.85,
    **kwargs,
) -> list[int]:
    """Split budget 50/50: focused (query-relevant) + context (video overview).

    context_mode:
        "uniform" — place context frames uniformly across non-focused regions.
        "representative" — merge adjacent similar windows into segments,
            pick one representative per segment.
    """
    n_focused = n_frames // 2
    n_context = n_frames - n_focused

    focused = _pick_focused_frames(relevance, n_focused)

    if context_mode == "representative":
        context = _pick_context_representative(
            embeddings, window_times, n_context, focused, duration_sec,
            merge_threshold,
        )
    else:
        context = _pick_context_uniform(
            window_times, n_context, focused, duration_sec,
        )

    return focused + context


def _pick_focused_frames(relevance, n_focused):
    """Greedy top-relevance selection, requiring >= 2 window indices apart."""
    ranked = np.argsort(-relevance)
    selected = []
    for idx in ranked:
        if len(selected) >= n_focused:
            break
        idx = int(idx)
        if any(abs(idx - s) < 2 for s in selected):
            continue
        selected.append(idx)
    return selected


def _pick_context_uniform(window_times, n_context, focused_indices, duration_sec):
    """Place context frames uniformly, avoiding focused regions."""
    times = np.asarray(window_times, dtype=float)
    n_windows = len(times)

    excluded = set()
    for idx in focused_indices:
        for offset in range(-1, 2):
            ei = idx + offset
            if 0 <= ei < n_windows:
                excluded.add(ei)

    available = sorted(i for i in range(n_windows) if i not in excluded)
    if not available:
        available = sorted(i for i in range(n_windows) if i not in set(focused_indices))
    if not available or n_context <= 0:
        return []

    targets = np.linspace(0, duration_sec, n_context + 2)[1:-1]

    selected = []
    used = set()
    for target_t in targets:
        best = None
        best_dist = float("inf")
        for i in available:
            if i in used:
                continue
            d = abs(times[i] - target_t)
            if d < best_dist:
                best_dist = d
                best = i
        if best is not None:
            selected.append(best)
            used.add(best)

    return selected


def _pick_context_representative(
    embeddings, window_times, n_context, focused_indices, duration_sec,
    merge_threshold,
):
    """Pick context frames from semantically distinct segments of the video."""
    times = np.asarray(window_times, dtype=float)
    n_windows = len(times)

    excluded = set()
    for idx in focused_indices:
        for offset in range(-1, 2):
            ei = idx + offset
            if 0 <= ei < n_windows:
                excluded.add(ei)

    available = sorted(i for i in range(n_windows) if i not in excluded)
    if not available or n_context <= 0:
        return []

    segments = _merge_adjacent_windows(available, embeddings, merge_threshold)

    representatives = []
    for seg in segments:
        centroid = embeddings[seg].mean(dim=0)
        sims = torch_cos_sim(
            embeddings[seg],
            centroid.unsqueeze(0).expand(len(seg), -1),
        )
        representatives.append(seg[int(sims.argmax())])

    if n_context <= len(representatives):
        pick_idx = np.round(
            np.linspace(0, len(representatives) - 1, n_context)
        ).astype(int)
        return [representatives[int(i)] for i in pick_idx]

    selected = list(representatives)
    selected_set = set(selected)
    extras = n_context - len(selected)

    seg_by_size = sorted(range(len(segments)), key=lambda i: -len(segments[i]))
    added = 0
    cycle = 0
    while added < extras and cycle < extras + len(segments):
        si = seg_by_size[cycle % len(seg_by_size)]
        seg = segments[si]
        unused = [w for w in seg if w not in selected_set]
        if unused:
            best = max(unused, key=lambda w: min(abs(w - s) for s in selected))
            selected.append(best)
            selected_set.add(best)
            added += 1
        cycle += 1

    return selected


def _merge_adjacent_windows(available, embeddings, threshold):
    """Merge consecutive available windows with similar embeddings into segments."""
    if not available:
        return []
    segments = [[available[0]]]
    for i in range(1, len(available)):
        curr = available[i]
        centroid = embeddings[segments[-1]].mean(dim=0)
        sim = float(torch_cos_sim(
            embeddings[curr].unsqueeze(0),
            centroid.unsqueeze(0),
        ))
        if sim >= threshold:
            segments[-1].append(curr)
        else:
            segments.append([curr])
    return segments


# ---------------------------------------------------------------------------
# Method 9: segment_adaptive — segment first, then allocate and select locally
# ---------------------------------------------------------------------------

def _select_segment_adaptive(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None,
    smoothing_window=5,
    boundary_threshold=0.75,
    min_segment_size=None,
    max_segments=None,
    coverage_floor=True,
    duration_weight=0.25,
    mean_relevance_weight=0.35,
    peak_relevance_weight=0.40,
    intra_segment_method="mmr",
    mmr_relevance_weight=0.7,
    **kwargs,
) -> list[int]:
    """Select frames by segmenting the relevance timeline first.

    The method builds a smoothed 1D relevance signal, detects a small set of
    temporal change points, allocates the frame budget across the resulting
    segments, and then selects frames locally within each segment.
    """
    n_windows = len(relevance)
    if n_windows == 0 or n_frames <= 0:
        return []

    n_frames = min(int(n_frames), n_windows)
    if n_frames >= n_windows:
        return list(range(n_windows))

    if min_segment_size is None:
        min_segment_size = max(2, n_windows // max(2 * n_frames, 1))
    min_segment_size = max(int(min_segment_size), 1)

    smoothed = _smooth_signal(relevance, smoothing_window)
    boundaries = _detect_segment_boundaries(
        smoothed,
        boundary_threshold=boundary_threshold,
        min_segment_size=min_segment_size,
        max_segments=max_segments,
    )
    segments = _build_segments(n_windows, boundaries)

    segment_scores = _score_segments(
        relevance,
        segments,
        duration_weight=duration_weight,
        mean_relevance_weight=mean_relevance_weight,
        peak_relevance_weight=peak_relevance_weight,
    )
    allocations = _allocate_segment_budget(
        segment_scores,
        n_frames=n_frames,
        coverage_floor=coverage_floor,
    )

    selected: list[int] = []
    selected_set: set[int] = set()
    for segment, budget in zip(segments, allocations):
        if budget <= 0:
            continue
        if intra_segment_method == "topk":
            local = _select_segment_topk(segment, relevance, budget)
        else:
            local = _select_segment_mmr(
                segment,
                relevance,
                embeddings,
                budget,
                relevance_weight=mmr_relevance_weight,
            )
        for idx in local:
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))

    # Small segments can exhaust their candidates. Fill any budget shortfall
    # with globally relevant unused windows so the selector always returns n_frames.
    if len(selected) < n_frames:
        ranked = np.argsort(-relevance)
        for idx in ranked:
            idx = int(idx)
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= n_frames:
                break

    return selected[:n_frames]


def _smooth_signal(signal, window):
    """Return a moving-average-smoothed copy of a 1D signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.size <= 2:
        return signal.copy()

    window = max(int(window), 1)
    if window <= 1:
        return signal.copy()

    window = min(window, signal.size)
    kernel = np.ones(window, dtype=float) / float(window)
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(signal, (left_pad, right_pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _detect_segment_boundaries(
    smoothed_relevance,
    boundary_threshold=0.75,
    min_segment_size=1,
    max_segments=None,
):
    """Detect change-point boundaries from a smoothed relevance signal."""
    smoothed_relevance = np.asarray(smoothed_relevance, dtype=float)
    if smoothed_relevance.size <= 1:
        return []

    change = np.abs(np.diff(smoothed_relevance))
    if change.size == 0:
        return []

    threshold = float(change.mean() + float(boundary_threshold) * change.std())
    candidates: list[tuple[float, int]] = []
    for i, value in enumerate(change):
        left = change[i - 1] if i > 0 else -float("inf")
        right = change[i + 1] if i + 1 < change.size else -float("inf")
        if value < threshold:
            continue
        if value < left or value < right:
            continue
        candidates.append((float(value), i + 1))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    max_boundaries = None
    if max_segments is not None:
        max_boundaries = max(int(max_segments) - 1, 0)

    selected: list[int] = []
    for _strength, boundary in candidates:
        if boundary <= 0 or boundary >= smoothed_relevance.size:
            continue
        if any(abs(boundary - existing) < int(min_segment_size) for existing in selected):
            continue
        selected.append(int(boundary))
        if max_boundaries is not None and len(selected) >= max_boundaries:
            break

    return sorted(selected)


def _build_segments(n_windows, boundaries):
    """Convert boundary indices into half-open window segments."""
    starts = [0] + [int(b) for b in boundaries]
    ends = [int(b) for b in boundaries] + [int(n_windows)]
    return [
        list(range(start, end))
        for start, end in zip(starts, ends)
        if start < end
    ]


def _score_segments(
    relevance,
    segments,
    duration_weight=0.25,
    mean_relevance_weight=0.35,
    peak_relevance_weight=0.40,
):
    """Score segments using duration and relevance statistics."""
    weights = np.array(
        [duration_weight, mean_relevance_weight, peak_relevance_weight],
        dtype=float,
    )
    if float(weights.sum()) <= 0:
        weights = np.array([1.0, 1.0, 1.0], dtype=float)
    weights = weights / weights.sum()

    total = max(len(relevance), 1)
    scores: list[float] = []
    for segment in segments:
        segment_rel = np.asarray([relevance[i] for i in segment], dtype=float)
        duration_score = len(segment) / float(total)
        mean_score = float(segment_rel.mean()) if segment_rel.size else 0.0
        peak_score = float(segment_rel.max()) if segment_rel.size else 0.0
        score = (
            weights[0] * duration_score
            + weights[1] * mean_score
            + weights[2] * peak_score
        )
        scores.append(float(score))
    return scores


def _allocate_segment_budget(segment_scores, n_frames, coverage_floor=True):
    """Allocate a frame budget across segments using softmax weights."""
    n_segments = len(segment_scores)
    if n_segments == 0 or n_frames <= 0:
        return []

    scores = np.asarray(segment_scores, dtype=float)
    allocations = np.zeros(n_segments, dtype=int)

    if coverage_floor:
        if n_frames >= n_segments:
            allocations += 1
        else:
            ranked = np.argsort(-scores)
            allocations[ranked[:n_frames]] = 1
            return allocations.tolist()

    remaining = int(n_frames) - int(allocations.sum())
    if remaining <= 0:
        return allocations.tolist()

    stable_scores = scores - float(scores.max())
    weights = np.exp(stable_scores)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones(n_segments, dtype=float)
    weights = weights / weights.sum()

    targets = weights * float(remaining)
    extra = np.floor(targets).astype(int)
    allocations += extra

    leftover = int(remaining - extra.sum())
    if leftover > 0:
        fractional = targets - extra
        ranked = np.argsort(-fractional)
        for idx in ranked[:leftover]:
            allocations[int(idx)] += 1

    return allocations.tolist()


def _select_segment_topk(segment, relevance, k):
    """Select the top-k most relevant windows within a segment."""
    ranked = sorted(segment, key=lambda idx: (-float(relevance[idx]), int(idx)))
    return [int(idx) for idx in ranked[: max(int(k), 0)]]


def _select_segment_mmr(segment, relevance, embeddings, k, relevance_weight=0.7):
    """Run MMR locally within a segment."""
    if not segment or k <= 0:
        return []

    k = min(int(k), len(segment))
    if k >= len(segment):
        return [int(idx) for idx in segment]

    lam = float(relevance_weight)
    first = max(segment, key=lambda idx: (float(relevance[idx]), -int(idx)))
    selected = [int(first)]
    selected_embs = embeddings[first].unsqueeze(0)

    for _ in range(k - 1):
        best_idx = -1
        best_score = -float("inf")
        for idx in segment:
            idx = int(idx)
            if idx in selected:
                continue
            sims = torch_cos_sim(
                embeddings[idx].unsqueeze(0).expand(len(selected), -1),
                selected_embs,
            )
            redundancy = float(sims.max())
            score = lam * float(relevance[idx]) - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected_embs = torch.cat(
            [selected_embs, embeddings[best_idx].unsqueeze(0)],
            dim=0,
        )

    return selected


# ---------------------------------------------------------------------------
# Method 10: uniform_topk — half uniform spacing + half top-relevance
# ---------------------------------------------------------------------------

def _select_uniform_topk(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, **kwargs,
) -> list[int]:
    """Half uniformly spaced, half highest-relevance (non-overlapping)."""
    n_windows = len(window_times)
    n_uniform = n_frames // 2
    n_topk = n_frames - n_uniform

    uniform_indices = np.linspace(0, n_windows - 1, n_uniform, dtype=int).tolist()
    uniform_set = set(uniform_indices)

    ranked = np.argsort(-relevance)
    topk = []
    for idx in ranked:
        if len(topk) >= n_topk:
            break
        idx = int(idx)
        if idx not in uniform_set:
            topk.append(idx)

    return uniform_indices + topk


# ---------------------------------------------------------------------------
# Method 11: random — uniformly random windows
# ---------------------------------------------------------------------------

def _select_random(
    relevance, embeddings, window_times, n_frames, duration_sec,
    chunks=None, fps=None, seed=42, **kwargs,
) -> list[int]:
    """Select distinct random windows uniformly across the video."""
    n_windows = len(window_times)
    if n_windows == 0 or n_frames <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    count = min(int(n_frames), n_windows)
    return rng.choice(n_windows, size=count, replace=False).tolist()


def _select_random_frames(
    relevance,
    window_times,
    n_frames,
    duration_sec,
    fps,
    sample_interval,
    seed=42,
):
    """Select actual random frame indices uniformly from the full video."""
    if fps is None or fps <= 0:
        raise ValueError("random frame selection requires a positive fps")

    total_frames = max(int(duration_sec * fps), 1)
    count = min(int(n_frames), total_frames)
    rng = np.random.default_rng(int(seed))
    frame_indices = np.sort(rng.choice(total_frames, size=count, replace=False)).tolist()
    frame_times = [float(idx) / float(fps) for idx in frame_indices]

    half_interval = float(sample_interval) / 2.0
    window_midpoints = np.asarray(window_times, dtype=float) + half_interval
    window_indices = [
        int(np.abs(window_midpoints - frame_time).argmin())
        for frame_time in frame_times
    ]
    scores = [float(relevance[idx]) for idx in window_indices]

    return {
        "window_indices": window_indices,
        "frame_indices": [int(idx) for idx in frame_indices],
        "frame_times": frame_times,
        "scores": scores,
    }
