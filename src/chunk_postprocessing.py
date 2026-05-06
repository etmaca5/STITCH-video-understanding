import numpy as np
import torch
from torch.nn.functional import cosine_similarity


class TransitionDetector:
    """Identifies transition regions as dips in the change signal.

    A transition is a contiguous period where the signal drops below the
    stability threshold (the same threshold used for initial chunking) and
    stays there long enough to not be a simple cut. Short dips are cuts;
    longer dips are transitions.
    """

    def __init__(self, fps):
        self.fps = fps

    def find_transitions(self, signal_times, signal_values, threshold,
                         smooth_window=3, min_transition_sec=1.0,
                         high_is_change=False):
        """Find time intervals where the signal crosses the boundary threshold.

        For embedding chunking (high_is_change=False), transitions are where
        the signal stays below threshold. For content_detector and surprise
        (high_is_change=True), transitions are where the signal stays above.

        Args:
            signal_times: sample times in seconds
            signal_values: signal values at those times
            threshold: stability threshold (reused from chunking)
            smooth_window: moving average window size for noise reduction
            min_transition_sec: regions shorter than this are cuts, not transitions
            high_is_change: if True, signal above threshold = change/transition

        Returns:
            List of (start_sec, end_sec) for each transition region.
        """
        signal = np.asarray(signal_values, dtype=float)
        times = np.asarray(signal_times, dtype=float)

        if len(signal) < smooth_window:
            return []

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(signal, kernel, mode="same")

        below = smoothed > threshold if high_is_change else smoothed < threshold

        regions = []
        start_idx = None
        for i, is_below in enumerate(below):
            if is_below and start_idx is None:
                start_idx = i
            elif not is_below and start_idx is not None:
                regions.append((times[start_idx], times[i - 1]))
                start_idx = None
        if start_idx is not None:
            regions.append((times[start_idx], times[len(below) - 1]))

        regions = [(s, e) for s, e in regions if (e - s) >= min_transition_sec]
        return regions

    def label_chunks(self, chunks, transition_regions, total_frames):
        """Overlay transition regions onto initial chunks.

        Transition boundaries take precedence: initial chunk boundaries
        that fall inside a transition region are removed (consecutive
        transition segments are merged). The output is a flat list of
        labeled segments covering the full timeline.

        Args:
            chunks: list of (start_frame, end_frame) from initial chunking
            transition_regions: list of (start_sec, end_sec) from find_transitions
            total_frames: total frames in video

        Returns:
            List of {"chunk": (start_frame, end_frame), "label": "stable"/"transition"}
        """
        transition_frames = [
            (int(s * self.fps), int(e * self.fps))
            for s, e in transition_regions
        ]

        cuts = {0, total_frames}
        for start, end in chunks:
            cuts.add(start)
            cuts.add(end)
        for start, end in transition_frames:
            cuts.add(max(0, start))
            cuts.add(min(total_frames, end))

        cuts = sorted(cuts)

        segments = []
        for i in range(len(cuts) - 1):
            seg_start, seg_end = cuts[i], cuts[i + 1]
            if seg_start >= seg_end:
                continue
            mid = (seg_start + seg_end) / 2
            is_trans = any(ts <= mid <= te for ts, te in transition_frames)
            segments.append({
                "chunk": (seg_start, seg_end),
                "label": "transition" if is_trans else "stable",
            })

        # Merge consecutive transition segments — initial chunk boundaries
        # within a transition region don't matter (transitions take precedence)
        merged = [segments[0]] if segments else []
        for seg in segments[1:]:
            if seg["label"] == "transition" and merged[-1]["label"] == "transition":
                merged[-1] = {
                    "chunk": (merged[-1]["chunk"][0], seg["chunk"][1]),
                    "label": "transition",
                }
            else:
                merged.append(seg)

        return merged


class ChunkMerger:
    """Absorbs short stable chunks into their most similar neighbor.

    Based on the boundary refinement strategy from Barbara & Maalouf (2025),
    "Prompts to Summaries" (arXiv:2506.10807). Short scenes are iteratively
    merged into the adjacent stable chunk with the highest cosine similarity.
    Transition chunks are never merged.
    """

    def __init__(self, fps):
        self.fps = fps

    def merge(self, labeled_chunks, chunk_embeddings,
              min_chunk_sec=5.0, max_chunk_sec=60.0):
        """Iteratively absorb short stable chunks into their most similar neighbor.

        Each iteration finds the shortest stable chunk below min_chunk_sec
        and merges it into whichever adjacent stable chunk has the highest
        cosine similarity. Repeats until no short stable chunks remain.

        Args:
            labeled_chunks: list of {"chunk": (start, end), "label": ...}
            chunk_embeddings: array of shape (n_chunks, embed_dim)
            min_chunk_sec: stable chunks shorter than this get absorbed
            max_chunk_sec: never create chunks longer than this

        Returns:
            List of {"chunk": (start_frame, end_frame), "label": "stable"/"transition"}
        """
        if len(labeled_chunks) <= 1:
            return list(labeled_chunks)

        chunks = [dict(c) for c in labeled_chunks]
        embs = [np.asarray(e, dtype=np.float32) for e in chunk_embeddings]
        skip = set()

        while True:
            shortest_idx = None
            shortest_dur = float("inf")
            for i, item in enumerate(chunks):
                if i in skip or item["label"] != "stable":
                    continue
                dur = (item["chunk"][1] - item["chunk"][0]) / self.fps
                if dur < min_chunk_sec and dur < shortest_dur:
                    shortest_dur = dur
                    shortest_idx = i

            if shortest_idx is None:
                break

            best_neighbor = None
            best_sim = -1.0
            for j in [shortest_idx - 1, shortest_idx + 1]:
                if j < 0 or j >= len(chunks):
                    continue
                if chunks[j]["label"] != "stable":
                    continue
                neighbor_dur = (chunks[j]["chunk"][1] - chunks[j]["chunk"][0]) / self.fps
                if shortest_dur + neighbor_dur > max_chunk_sec:
                    continue
                sim = self._cosine_sim(embs[shortest_idx], embs[j])
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = j

            if best_neighbor is None:
                skip.add(shortest_idx)
                continue

            a, b = sorted([shortest_idx, best_neighbor])
            dur_a = (chunks[a]["chunk"][1] - chunks[a]["chunk"][0]) / self.fps
            dur_b = (chunks[b]["chunk"][1] - chunks[b]["chunk"][0]) / self.fps
            merged_dur = dur_a + dur_b
            w_a, w_b = dur_a / merged_dur, dur_b / merged_dur

            chunks[a] = {
                "chunk": (chunks[a]["chunk"][0], chunks[b]["chunk"][1]),
                "label": "stable",
            }
            # TODO: recompute embedding from video frames instead of averaging
            embs[a] = w_a * embs[a] + w_b * embs[b]

            del chunks[b]
            del embs[b]
            # Reset skip set — indices shifted after deletion, and
            # the merge may have made previously-skipped chunks mergeable
            skip = set()

        return chunks

    @staticmethod
    def aggregate_embeddings(window_times, window_embeddings, labeled_chunks, fps):
        """Average per-window embeddings into per-chunk embeddings.

        For each chunk, averages the window embeddings whose timestamps
        fall within the chunk's time range.

        Args:
            window_times: array of sliding-window **start** times in seconds
                (same convention as chunking: one timestamp per window, the
                left edge of that window’s span). Windows are assigned to a
                chunk when ``start_sec <= t < end_sec`` (half-open).
            window_embeddings: array of shape (n_windows, embed_dim)
            labeled_chunks: list of {"chunk": (start_frame, end_frame), ...}
            fps: video frame rate

        Returns:
            numpy array of shape (n_chunks, embed_dim)
        """
        wt = np.asarray(window_times)
        we = np.asarray(window_embeddings)

        chunk_embs = []
        for item in labeled_chunks:
            start, end = item["chunk"] if isinstance(item, dict) else item
            start_sec, end_sec = start / fps, end / fps
            mask = (wt >= start_sec) & (wt < end_sec)
            if mask.any():
                chunk_embs.append(we[mask].mean(axis=0))
            else:
                nearest = np.argmin(np.abs(wt - (start_sec + end_sec) / 2))
                chunk_embs.append(we[nearest])

        return np.stack(chunk_embs)

    @staticmethod
    def _cosine_sim(a, b):
        a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        return cosine_similarity(a_t, b_t).item()


class QueryChunkMerger:
    """Merges adjacent chunks with similar relevance to a text query.

    Iteratively finds the adjacent pair whose cosine similarities to the
    query differ the least and merges them, until no pair has a difference
    below ``similarity_threshold``.
    """

    def __init__(self, fps):
        self.fps = fps

    def merge(self, chunks, chunk_embeddings, query_embedding,
              similarity_threshold=0.05, max_chunk_sec=120.0,
              embed_fn=None):
        """Merge adjacent chunks with similar query relevance.

        Args:
            chunks: list of (start_frame, end_frame) tuples
            chunk_embeddings: array-like of shape (n_chunks, embed_dim)
            query_embedding: array-like of shape (embed_dim,)
            similarity_threshold: max allowed difference in query cosine
                similarity between adjacent chunks for a merge
            max_chunk_sec: never create chunks longer than this
            embed_fn: callable (start_frame, end_frame) -> embedding array.
                When provided, merged chunks are re-embedded from video
                frames instead of using a weighted average.

        Returns:
            Tuple of (merged_chunks, merged_embeddings) where
            merged_chunks is a list of (start_frame, end_frame) and
            merged_embeddings is a list of numpy arrays.
        """
        if len(chunks) <= 1:
            return list(chunks), [np.asarray(e, dtype=np.float32)
                                  for e in chunk_embeddings]

        chunks = list(chunks)
        embs = [np.asarray(e, dtype=np.float32) for e in chunk_embeddings]
        q = np.asarray(query_embedding, dtype=np.float32)

        while len(chunks) > 1:
            sims = [self._cosine_sim(q, e) for e in embs]

            best_idx = None
            best_diff = float("inf")
            for i in range(len(chunks) - 1):
                merged_dur = (chunks[i + 1][1] - chunks[i][0]) / self.fps
                if merged_dur > max_chunk_sec:
                    continue
                diff = abs(sims[i] - sims[i + 1])
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is None or best_diff >= similarity_threshold:
                break

            i = best_idx
            old_chunk_a = chunks[i]
            chunks[i] = (chunks[i][0], chunks[i + 1][1])

            if embed_fn is not None:
                embs[i] = np.asarray(embed_fn(chunks[i]), dtype=np.float32)
            else:
                dur_a = (old_chunk_a[1] - old_chunk_a[0]) / self.fps
                dur_b = (chunks[i + 1][1] - chunks[i + 1][0]) / self.fps
                total = dur_a + dur_b
                embs[i] = (dur_a / total) * embs[i] + (dur_b / total) * embs[i + 1]

            del chunks[i + 1]
            del embs[i + 1]

        return chunks, embs

    @staticmethod
    def _cosine_sim(a, b):
        a_t = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float32).unsqueeze(0)
        return cosine_similarity(a_t, b_t).item()


def postprocess_chunks(chunking_result, detect_transitions=False,
                       transition_params=None, merge_params=None,
                       chunk_embeddings=None, return_stages=False):
    """Post-processing pipeline: transition detection (optional) then merging.

    Args:
        chunking_result: dict from Chunking.chunk() with keys:
            chunks, signal_times, signal_values, fps, total_frames
            (+ window_times, window_embeddings for embedding method)
        detect_transitions: whether to run transition detection
        transition_params: kwargs for TransitionDetector.find_transitions
            (smooth_window, min_transition_sec)
        merge_params: kwargs for ChunkMerger.merge (min_chunk_sec, max_chunk_sec)
        chunk_embeddings: pre-computed (n_chunks, embed_dim) array.
            Only used when window_embeddings is not in chunking_result
            and detect_transitions is False. When window_embeddings is
            available, per-chunk embeddings are computed automatically
            even after transition detection restructures chunks.
        return_stages: if True, return (final_labeled, stages_dict) where
            stages_dict has keys "initial", "after_transitions", "after_merge".

    Returns:
        If return_stages is False:
            List of {"chunk": (start_frame, end_frame), "label": "stable"/"transition"}
        If return_stages is True:
            Tuple of (final_labeled, stages_dict)
    """
    fps = chunking_result["fps"]
    chunks = chunking_result["chunks"]
    total_frames = chunking_result["total_frames"]

    stages = {}
    if return_stages:
        stages["initial"] = [{"chunk": c, "label": "stable"} for c in chunks]

    has_threshold = chunking_result.get("threshold") is not None
    # NOTE: threshold-free chunkers like kernel_cpd / pelt currently bypass
    # transition detection because this stage reuses a scalar threshold.
    if detect_transitions and has_threshold:
        detector = TransitionDetector(fps)
        high_is_change = chunking_result.get("high_is_change", False)
        tp = dict(transition_params or {})
        tp["high_is_change"] = high_is_change
        regions = detector.find_transitions(
            chunking_result["signal_times"],
            chunking_result["signal_values"],
            chunking_result["threshold"],
            **tp,
        )
        labeled = detector.label_chunks(chunks, regions, total_frames)
    else:
        labeled = [{"chunk": c, "label": "stable"} for c in chunks]

    if return_stages:
        stages["after_transitions"] = [dict(lc) for lc in labeled]

    if "window_embeddings" in chunking_result:
        embs = ChunkMerger.aggregate_embeddings(
            chunking_result["window_times"],
            chunking_result["window_embeddings"],
            labeled, fps,
        )
    elif chunk_embeddings is not None and not detect_transitions:
        embs = chunk_embeddings
    else:
        embs = None

    if embs is not None:
        merger = ChunkMerger(fps)
        labeled = merger.merge(labeled, embs, **(merge_params or {}))

    if return_stages:
        stages["after_merge"] = [dict(lc) for lc in labeled]
        return labeled, stages

    return labeled
