"""Evaluation entry point for the video retrieval system.

Run from the project root:
    python src/evaluate.py                                       # defaults (embedding)
    python src/evaluate.py dataset=activitynet evaluation=vlm_chunk_selection vlm=qwen3_vl_8b
    python src/evaluate.py --cfg job                             # show config
"""

import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_aggregation import aggregate as aggregate_chunk_embedding
from chunk_postprocessing import ChunkMerger, QueryChunkMerger, postprocess_chunks
from chunking import Chunking, VideoMAEv2EmbeddingWrapper, load_videomaev2_model
from chunking_cache import WindowCacheMissError
from frame_selection import WINDOW_METHODS, select_frames_from_windows
from datasets import build_dataset
from metrics import (
    compute_activitynet_qa_metrics,
    compute_gebd_f1_metrics,
    compute_lovr_pass_metrics,
    compute_longvideobench_metrics,
    compute_lvbench_metrics,
    compute_map_at_iou,
    compute_mean_iou,
    compute_mlvu_metrics,
    compute_qvhighlight_highlight_metrics,
    compute_recall_at_k,
    compute_videomme_mcq_metrics,
    extract_mcq_letter,
    parse_longvideobench_mcq_answer,
)
from moment_selection import select_moments
from plots import generate_all_plots
from results import (
    make_run_dir,
    save_run,
    save_run_crash,
    save_run_issues,
    task_type_from_mode,
)
from retrieval import InternVideo2Backend, XCLIPBackend
from temporal_abstraction import TemporalAbstractionLayer, compute_allowed_chunks
from vlm_client import VLMClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run selection helpers
# ---------------------------------------------------------------------------

def _validate_vlm_config(cfg):
    """Reject VLM config combinations that would otherwise be ignored."""
    mode = cfg.evaluation.mode
    if mode not in ("vlm_chunk_selection", "vlm_qa"):
        return

    chunk_source = cfg.evaluation.chunk_source
    preselection = cfg.evaluation.get("chunk_preselection", "none")
    task = cfg.temporal_abstraction.get("task")
    frame_method = cfg.temporal_abstraction.frame_selection.method
    query_merge_enabled = bool(cfg.postprocessing.query_merge.enabled)
    selector_cfg = cfg.postprocessing.get("moment_selection")
    moment_selection_enabled = bool(
        selector_cfg is not None and selector_cfg.get("enabled", False)
    )

    if mode == "vlm_chunk_selection" and task != "chunk_selection":
        raise ValueError(
            "evaluation=vlm_chunk_selection requires temporal_abstraction=chunk_selection "
            f"(current temporal_abstraction.task={task!r})."
        )
    if mode == "vlm_qa" and task != "qa":
        raise ValueError(
            "evaluation=vlm_qa requires temporal_abstraction=qa "
            f"(current temporal_abstraction.task={task!r})."
        )

    if preselection != "none" and chunk_source != "stable_chunks":
        raise ValueError(
            f"evaluation.chunk_preselection={preselection!r} requires "
            "evaluation.chunk_source=stable_chunks. It is not used for uniform chunks."
        )

    if frame_method == "best_window" and chunk_source != "stable_chunks":
        raise ValueError(
            "temporal_abstraction.frame_selection.method=best_window requires "
            "evaluation.chunk_source=stable_chunks."
        )

    if query_merge_enabled and chunk_source != "stable_chunks":
        raise ValueError(
            "postprocessing.query_merge.enabled=true requires "
            f"evaluation.chunk_source=stable_chunks for evaluation={mode}."
        )

    if moment_selection_enabled:
        raise ValueError(
            "postprocessing.moment_selection.enabled=true is not supported for "
            f"evaluation={mode}."
        )

def _load_failed_video_paths(run_dir):
    """Return the failed video paths recorded in a previous run directory."""
    failed_path = Path(run_dir) / "failed_videos.json"
    issues_path = Path(run_dir) / "run_issues.json"

    if failed_path.exists():
        with open(failed_path) as f:
            failed = json.load(f)
    elif issues_path.exists():
        with open(issues_path) as f:
            failed = json.load(f).get("failed_videos") or []
    else:
        raise FileNotFoundError(
            f"No failed_videos.json or run_issues.json found in {run_dir}"
        )

    video_paths = []
    for entry in failed:
        video_path = entry.get("video_path")
        if video_path:
            video_paths.append(str(video_path))

    if not video_paths:
        raise ValueError(f"No failed videos listed in {failed_path}")

    return video_paths


def _write_run_config_yaml(run_dir, cfg):
    """Persist cfg for reproducibility and resume (mirrors save_run)."""
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def _load_resume_checkpoint(resume_from):
    """Load config and checkpoint from a previous run directory for resuming."""
    resume_dir = Path(resume_from)
    config_path = resume_dir / "config.yaml"
    checkpoint_path = resume_dir / "checkpoint.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {resume_from}. "
            f"Configs are written at run start and on each checkpoint; "
            f"older runs may only have checkpoint.json — restore config.yaml "
            f"or start a new run directory."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint.json found in {resume_from}")

    saved_cfg = OmegaConf.load(config_path)
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    for key in ("per_query_results", "videos_processed"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint.json missing required key: {key}")

    return saved_cfg, checkpoint


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_chunking_models(cfg, device, ret_backend=None, embedding_device=None):
    """Return (model, processor, embedding_backend) needed by the chosen chunking method.

    The embedding_backend is any object with embed_video(frames) and num_frames
    (e.g. VideoMAEv2EmbeddingWrapper or InternVideo2Backend).

    When embedding_backend is "internvideo2" and ret_backend is already an
    InternVideo2Backend, it is reused to avoid loading the model twice.

    embedding_device, if given, overrides device for the embedding backend
    (e.g. "cpu" to keep the GPU free for another process). Note: when the
    embedding backend is reused from ret_backend, this override has no effect.
    """
    ctype = cfg.chunking.type
    emb_backend_type = cfg.chunking.get("embedding_backend", "internvideo2")
    if embedding_device is not None:
        emb_device = (
            embedding_device if isinstance(embedding_device, torch.device)
            else torch.device(embedding_device)
        )
    else:
        emb_device = device

    def _make_embedding_backend(model_dir_key="embedding_model_dir"):
        c = cfg.chunking

        if emb_backend_type == "internvideo2":
            if isinstance(ret_backend, InternVideo2Backend):
                return ret_backend
            return InternVideo2Backend(
                model_dir=c.get("embedding_model_dir", cfg.retrieval.model_dir),
                device=emb_device,
                num_frames=c.get("embedding_num_frames", cfg.retrieval.num_frames),
                orig_num_frames=c.get("embedding_orig_num_frames", cfg.retrieval.get("orig_num_frames")),
                image_size=c.get("embedding_image_size", cfg.retrieval.image_size),
                use_flash_attn=c.get(
                    "embedding_use_flash_attn",
                    cfg.retrieval.get("use_flash_attn", False),
                ),
            )
        model_dir = c.get(model_dir_key, c.get("model_dir"))
        raw_model, _ = load_videomaev2_model(model_dir, device=emb_device)
        return VideoMAEv2EmbeddingWrapper(raw_model, emb_device)

    if ctype == "content_detector":
        return None, None, _make_embedding_backend("embedding_model_dir")
    elif ctype == "embedding":
        return None, None, _make_embedding_backend("model_dir")
    elif ctype == "surprise":
        from transformers import AutoModel, AutoVideoProcessor
        path = cfg.chunking.model_path
        model = AutoModel.from_pretrained(path).to(device).eval()
        processor = AutoVideoProcessor.from_pretrained(path)
        return model, processor, _make_embedding_backend("embedding_model_dir")
    else:
        raise ValueError(f"Unknown chunking type: {ctype}")


def _load_retrieval_backend(cfg, device):
    """Instantiate the retrieval embedding backend."""
    backend = cfg.retrieval.backend
    if backend == "xclip":
        log.warning(
            "XCLIPBackend is deprecated: Microsoft's X-CLIP is a video "
            "classification model and underperforms on retrieval. "
            "Consider switching to retrieval=internvideo2."
        )
        return XCLIPBackend(model_path=cfg.retrieval.model_path, device=device)
    elif backend == "internvideo2":
        return InternVideo2Backend(
            model_dir=cfg.retrieval.model_dir,
            device=device,
            num_frames=cfg.retrieval.num_frames,
            orig_num_frames=cfg.retrieval.get("orig_num_frames"),
            image_size=cfg.retrieval.image_size,
            use_flash_attn=cfg.retrieval.get("use_flash_attn", False),
        )
    else:
        raise ValueError(f"Unknown retrieval backend: {backend}")


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _run_chunking(cfg, video_path, model, processor, device,
                   embedding_backend=None):
    """Chunk a video using the parameters from *cfg.chunking*."""
    ctype = cfg.chunking.type
    chunker = Chunking(chunking_type=ctype)
    require_cache = bool(
        cfg.evaluation.get("require_window_cache", False)
        if "evaluation" in cfg else False
    )

    if ctype == "content_detector":
        return chunker.content_detector(
            video_path,
            threshold=cfg.chunking.threshold,
            min_scene_len=cfg.chunking.min_scene_len,
            embedding_backend=embedding_backend,
            device=device,
            embedding_sample_interval=cfg.chunking.embedding_sample_interval,
        )
    elif ctype == "embedding":
        return chunker.embedding_chunking(
            video_path, embedding_backend, device,
            sample_interval=cfg.chunking.sample_interval,
            k=cfg.chunking.k,
            refine_boundaries=cfg.chunking.refine_boundaries,
            threshold_method=cfg.chunking.get("threshold_method", "kernel_cpd"),
            penalty=cfg.chunking.get("penalty"),
            min_segment_windows=cfg.chunking.get("min_segment_windows", 2),
            require_cache=require_cache,
        )
    elif ctype == "surprise":
        return chunker.surprise_chunking(
            video_path, model, processor, device,
            window_frames=cfg.chunking.window_frames,
            stride_seconds=cfg.chunking.stride_seconds,
            sample_fps=cfg.chunking.sample_fps,
            k=cfg.chunking.k,
            embedding_backend=embedding_backend,
            embedding_sample_interval=cfg.chunking.embedding_sample_interval,
            refine_boundaries=cfg.chunking.refine_boundaries,
            threshold_method=cfg.chunking.get("threshold_method", "std"),
            penalty=cfg.chunking.get("penalty"),
            min_segment_windows=cfg.chunking.get("min_segment_windows", 2),
        )
    else:
        raise ValueError(f"Unknown chunking type: {ctype}")


# ---------------------------------------------------------------------------
# GEBD boundary refinement
# ---------------------------------------------------------------------------

def _get_gebd_window_sample_interval(cfg, chunking_result):
    """Infer the window spacing used to define GEBD refinement centers."""
    window_times = chunking_result.get("window_times")
    if window_times is not None and len(window_times) >= 2:
        diffs = np.diff(np.asarray(window_times, dtype=float))
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 0:
            return float(np.median(positive_diffs))

    ctype = cfg.chunking.type
    if ctype == "embedding":
        return float(cfg.chunking.sample_interval)
    if ctype in ("content_detector", "surprise"):
        return float(cfg.chunking.embedding_sample_interval)
    raise ValueError(
        f"GEBD boundary refinement does not know how to infer the window "
        f"spacing for chunking type {ctype!r}."
    )


def _refine_gebd_boundary_frames(cfg, video_path, chunking_result, boundary_frames):
    """Snap GEBD boundaries to the max content-change frame between window centers."""
    if not boundary_frames:
        return [], []

    window_times = chunking_result.get("window_times")
    if window_times is None or len(window_times) < 2:
        raise ValueError(
            "GEBD boundary refinement requires at least two window timestamps."
        )

    fps = float(chunking_result["fps"])
    total_frames = int(chunking_result["total_frames"])
    sample_interval = _get_gebd_window_sample_interval(cfg, chunking_result)
    window_centers = np.asarray(window_times, dtype=float) + sample_interval / 2.0
    chunker = Chunking()

    refined_frames = []
    refinement_info = []
    previous_frame = -1
    for boundary_frame in boundary_frames:
        boundary_frame = int(boundary_frame)
        boundary_sec = boundary_frame / fps
        right_idx = int(np.searchsorted(window_centers, boundary_sec, side="right"))
        left_idx = right_idx - 1
        if left_idx < 0 or right_idx >= len(window_centers):
            raise ValueError(
                f"Boundary at frame {boundary_frame} in {video_path} is not "
                f"bracketed by two window centers."
            )

        search_start = int(np.floor(window_centers[left_idx] * fps))
        search_end = int(np.ceil(window_centers[right_idx] * fps))
        refined_frame = int(
            chunker._refine_boundary(
                video_path, search_start, search_end, strict=True,
            )
        )

        if refined_frame <= previous_frame:
            raise ValueError(
                f"Refined GEBD boundaries are not strictly increasing for "
                f"{video_path}."
            )
        if not 0 < refined_frame < total_frames:
            raise ValueError(
                f"Refined GEBD boundary {refined_frame} is outside the valid "
                f"frame range for {video_path}."
            )

        refined_frames.append(refined_frame)
        refinement_info.append({
            "initial_frame": boundary_frame,
            "refined_frame": refined_frame,
            "search_start_frame": search_start,
            "search_end_frame": search_end,
            "left_window_center_sec": float(window_centers[left_idx]),
            "right_window_center_sec": float(window_centers[right_idx]),
        })
        previous_frame = refined_frame

    return refined_frames, refinement_info


# ---------------------------------------------------------------------------
# Frame loading (mirrors Retrieval._load_chunk_frames)
# ---------------------------------------------------------------------------

def _load_chunk_frames(video_path, chunk, num_frames):
    """Sample *num_frames* uniform RGB frames from a chunk."""
    start_frame, end_frame = chunk
    indices = np.linspace(
        start_frame, max(start_frame, end_frame - 1), num_frames
    ).astype(int)

    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(
            f"Could not read frames from {video_path} "
            f"[{start_frame}-{end_frame}]"
        )
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def _load_full_video_frames(video_path, num_frames):
    """Sample *num_frames* uniform RGB frames from an entire video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frame count from {video_path}")

    indices = np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


# ---------------------------------------------------------------------------
# Chunk scoring helpers
# ---------------------------------------------------------------------------

SINGLE_VECTOR_METHODS = ("mean", "gem", "coherence_mean", "lse")


def _build_chunk_embs(method, stable_chunks, chunking_result,
                      ret_backend, video_path, batch_size=8,
                      aggregation_params=None):
    """Build per-chunk embeddings and per-chunk coherence.

    Returns (embs, coherence) where coherence is a list of R values
    (resultant length ||(1/n) sum e_i||) or ``None`` when no window
    embeddings are available (e.g. recompute without embedding-based
    chunking).

    Methods:
      "recompute" - re-encode chunk frames with the retrieval backend.
      "max_sim" - no chunk vector is built; scoring uses per-window MaxSim.
      single-vector methods ("mean", "gem", "coherence_mean", "lse") -
        aggregate window embeddings into one unit vector per chunk.
    """
    fps = chunking_result["fps"]
    aggregation_params = aggregation_params or {}

    if method == "recompute":
        all_frames = [
            _load_chunk_frames(video_path, chunk, ret_backend.num_frames)
            for chunk in stable_chunks
        ]
        embs = ret_backend.embed_video_batch(all_frames, batch_size=batch_size)
        coherence = _chunk_coherence_from_windows(stable_chunks, chunking_result)
        return embs, coherence

    wt = chunking_result["window_times"]
    we = chunking_result["window_embeddings"]
    embs = []
    coherence = []
    for start, end in stable_chunks:
        s_sec, e_sec = start / fps, end / fps
        mask = (wt >= s_sec) & (wt < e_sec)
        if mask.any():
            chunk_windows = we[mask]
            if method == "max_sim":
                vec = chunk_windows.mean(axis=0)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                R = float(np.linalg.norm(chunk_windows.mean(axis=0)))
            elif method in SINGLE_VECTOR_METHODS:
                vec, R = aggregate_chunk_embedding(
                    chunk_windows, method=method, **aggregation_params,
                )
            else:
                raise ValueError(
                    f"_build_chunk_embs called with unsupported method {method!r}"
                )
        else:
            idx = np.abs(wt - (s_sec + e_sec) / 2).argmin()
            vec = we[idx]
            R = 1.0
        embs.append(vec)
        coherence.append(float(R))
    return np.stack(embs), coherence


def _chunk_coherence_from_windows(stable_chunks, chunking_result):
    """Compute R for each chunk from window embeddings, or None if unavailable."""
    if "window_embeddings" not in chunking_result:
        return None
    fps = chunking_result["fps"]
    wt = chunking_result["window_times"]
    we = chunking_result["window_embeddings"]
    out = []
    for start, end in stable_chunks:
        s_sec, e_sec = start / fps, end / fps
        mask = (wt >= s_sec) & (wt < e_sec)
        if mask.any():
            mean_vec = we[mask].mean(axis=0)
            out.append(float(np.linalg.norm(mean_vec)))
        else:
            out.append(1.0)
    return out


def _score_chunks(method, query_emb, chunks, chunk_embs, chunking_result):
    """Score chunks against a query embedding.

    For single-vector methods ("recompute", "mean", "gem", "coherence_mean",
    "lse"), computes cosine similarity between the query and each chunk
    embedding. For "max_sim", computes cosine similarity of the query
    against every window embedding within each chunk and takes the maximum.
    """
    query_t = torch.tensor(query_emb).unsqueeze(0)

    if method == "recompute" or method in SINGLE_VECTOR_METHODS:
        return cosine_similarity(query_t, torch.tensor(chunk_embs)).numpy()

    scores, _ = _score_chunks_with_windows(
        query_emb,
        chunks,
        chunking_result,
        frames_per_chunk=1,
        sample_interval=_infer_sample_interval(chunking_result),
    )
    return scores


def _infer_sample_interval(chunking_result):
    """Infer the chunking window spacing from saved window start times."""
    wt = chunking_result["window_times"]
    if len(wt) >= 2:
        diffs = np.diff(wt)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) > 0:
            return float(positive_diffs[0])
    fps = chunking_result["fps"]
    return float(chunking_result["total_frames"] / fps)


def _serialize_labeled_chunks(lc_list):
    """Serialize labeled chunk records for JSON output."""
    return [
        {
            "chunk": [int(lc["chunk"][0]), int(lc["chunk"][1])],
            "label": lc["label"],
        }
        for lc in lc_list
    ]


def _select_window_sequence(window_indices, best_pos, n_windows):
    """Return a best-first window sequence with forward fill, then backfill."""
    if not window_indices or n_windows <= 0:
        return []

    selected = [window_indices[best_pos]]

    next_pos = best_pos + 1
    while len(selected) < n_windows and next_pos < len(window_indices):
        selected.append(window_indices[next_pos])
        next_pos += 1

    prev_pos = best_pos - 1
    while len(selected) < n_windows and prev_pos >= 0:
        selected.append(window_indices[prev_pos])
        prev_pos -= 1

    return selected


def _allocate_ranked_round_robin(window_indices, window_scores, n_frames):
    """Allocate frame slots to score-ranked windows in round-robin order."""
    if not window_indices or n_frames <= 0:
        return []

    ranked = sorted(
        zip(window_indices, window_scores),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    allocations = {int(idx): 0 for idx, _ in ranked}
    ranked_indices = [int(idx) for idx, _ in ranked]

    for frame_idx in range(int(n_frames)):
        allocations[ranked_indices[frame_idx % len(ranked_indices)]] += 1

    return [(int(idx), float(score), allocations[int(idx)]) for idx, score in ranked]


def _sample_window_frame_indices(start_sec, end_sec, fps, chunk, n_frames):
    """Sample ``n_frames`` interior frame indices from a window."""
    start_frame, end_frame = chunk
    max_frame = max(start_frame, end_frame - 1)
    window_start_frame = int(np.clip(np.floor(start_sec * fps), start_frame, max_frame))
    window_end_frame = int(np.clip(np.ceil(end_sec * fps), start_frame + 1, end_frame))
    window_max_frame = max(window_start_frame, window_end_frame - 1)

    available = np.arange(window_start_frame, window_max_frame + 1, dtype=int)
    n_frames = max(int(n_frames), 1)
    if len(available) == 0:
        return [max_frame] * n_frames
    if len(available) <= n_frames:
        selected = available.tolist()
        while len(selected) < n_frames:
            selected.append(selected[-1])
        return selected

    targets = np.arange(1, n_frames + 1, dtype=float) / (n_frames + 1)
    targets = window_start_frame + targets * (window_max_frame - window_start_frame)
    remaining = available.tolist()
    selected = []
    for target in targets:
        best_pos = min(
            range(len(remaining)),
            key=lambda pos: (abs(remaining[pos] - target), remaining[pos]),
        )
        selected.append(remaining.pop(best_pos))
    return selected


def _build_window_selection_metadata(
    chunk,
    fps,
    window_times,
    window_infos,
    sample_interval,
    duration_sec,
):
    """Serialize selected window info for prompt-time frame selection."""
    selected_windows = []
    selected_frames = []
    for idx, score, allocated_count in window_infos:
        window_start = float(window_times[idx])
        window_end = float(min(window_start + sample_interval, duration_sec))
        frame_indices = _sample_window_frame_indices(
            window_start,
            window_end,
            fps,
            chunk,
            allocated_count,
        )
        selected_windows.append({
            "window_index": int(idx),
            "start_sec": window_start,
            "end_sec": window_end,
            "score": float(score),
            "allocated_frame_indices": [int(frame_idx) for frame_idx in frame_indices],
        })
        for frame_idx in frame_indices:
            selected_frames.append({
                "window_index": int(idx),
                "start_sec": window_start,
                "end_sec": window_end,
                "score": float(score),
                "frame_index": int(frame_idx),
            })
    selected_frames.sort(key=lambda item: (item["frame_index"], item["window_index"]))
    return {
        "selected_windows": selected_windows,
        "selected_frames": selected_frames,
        "frame_indices": [item["frame_index"] for item in selected_frames],
    }


def _score_chunks_with_windows(
    query_emb,
    chunks,
    chunking_result,
    frames_per_chunk=1,
    sample_interval=2.0,
    best_window_strategy="ranked_round_robin",
):
    """Return max-sim scores plus the selected window sequence per chunk."""
    query_t = torch.tensor(query_emb).unsqueeze(0)
    fps = chunking_result["fps"]
    wt = chunking_result["window_times"]
    we = chunking_result["window_embeddings"]
    duration_sec = chunking_result["total_frames"] / fps

    scores = []
    metadata = []
    for start, end in chunks:
        s_sec, e_sec = start / fps, end / fps
        mask = (wt >= s_sec) & (wt < e_sec)
        if mask.any():
            matching_indices = np.flatnonzero(mask)
            windows_t = torch.tensor(we[matching_indices])
            sims = cosine_similarity(
                query_t.expand(len(windows_t), -1), windows_t
            )
            best_pos = int(torch.argmax(sims).item())
            matching_list = matching_indices.tolist()
            sim_list = sims.tolist()
            if best_window_strategy == "ranked_round_robin":
                window_infos = _allocate_ranked_round_robin(
                    matching_list,
                    sim_list,
                    frames_per_chunk,
                )
            elif best_window_strategy == "temporal_neighbors":
                selected_indices = _select_window_sequence(
                    matching_list,
                    best_pos,
                    frames_per_chunk,
                )
                sim_by_index = {
                    int(idx): float(score)
                    for idx, score in zip(matching_list, sim_list)
                }
                window_infos = [
                    (int(idx), sim_by_index[int(idx)], 1)
                    for idx in selected_indices
                ]
            else:
                raise ValueError(
                    f"Unknown best_window_strategy: {best_window_strategy}"
                )
            scores.append(float(sims[best_pos].item()))
        else:
            idx = int(np.abs(wt - (s_sec + e_sec) / 2).argmin())
            sim = cosine_similarity(
                query_t, torch.tensor(we[idx]).unsqueeze(0)
            )
            window_infos = [(idx, float(sim.item()), max(int(frames_per_chunk), 1))]
            scores.append(float(sim.item()))

        metadata.append(
            _build_window_selection_metadata(
                (start, end),
                fps,
                wt,
                window_infos,
                sample_interval,
                duration_sec,
            )
        )

    return np.array(scores), metadata


# ---------------------------------------------------------------------------
# Video info helpers
# ---------------------------------------------------------------------------

def _get_video_info(video_path):
    """Return (fps, total_frames) for a video file."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Could not read video info from {video_path}")
    return fps, total_frames


def _make_uniform_chunks(total_frames, num_chunks):
    """Return one-frame chunks at linspace-sampled video positions."""
    if num_chunks == 0:
        return []
    indices = np.linspace(0, max(total_frames - 1, 0), num_chunks, dtype=int)
    return [
        (int(idx), int(min(idx + 1, total_frames)))
        for idx in indices
    ]


def _embed_lovr_video_items(video_items, ret_backend, batch_size, desc):
    """Encode LoVR gallery/query videos in batches."""
    features = []
    kept_items = []
    failed_items = []
    batch_frames = []
    batch_items = []

    def _flush():
        nonlocal batch_frames, batch_items
        if not batch_frames:
            return
        batch_feats = ret_backend.embed_video_batch(batch_frames, batch_size=batch_size)
        for item, feat in zip(batch_items, batch_feats):
            kept_items.append(item)
            features.append(feat)
        batch_frames = []
        batch_items = []

    for item in tqdm(video_items, desc=desc, unit="vid"):
        try:
            frames = _load_full_video_frames(item["video_path"], ret_backend.num_frames)
        except Exception as exc:
            failed_items.append({
                "item_id": item["item_id"],
                "video_path": item["video_path"],
                "error": str(exc),
            })
            continue

        batch_frames.append(frames)
        batch_items.append(item)
        if len(batch_frames) >= batch_size:
            _flush()

    _flush()

    if not features:
        raise RuntimeError(f"No video features could be encoded for {desc}")
    return np.stack(features), kept_items, failed_items


def _filter_lovr_queries_by_target(queries, valid_target_ids):
    """Keep only LoVR text queries whose target remains in the gallery."""
    valid_ids = {str(item_id) for item_id in valid_target_ids}
    kept = []
    skipped = []
    for sample in queries:
        target_id = str(sample.metadata["target_id"])
        if target_id in valid_ids:
            kept.append(sample)
        else:
            skipped.append({
                "sample_id": sample.sample_id,
                "query": sample.query,
                "target_id": target_id,
                "failure_reason": "missing_target_embedding",
            })
    return kept, skipped


def _collect_lovr_topk_results(
    task_name,
    similarity_matrix,
    query_records,
    candidate_records,
    max_k,
    query_kind,
):
    """Serialize per-query ranked predictions for LoVR retrieval tasks."""
    sim = np.asarray(similarity_matrix, dtype=np.float32)
    out = []
    actual_k = min(int(max_k), len(candidate_records))
    ranked = np.argsort(-sim, axis=1)[:, :actual_k]

    for row_idx, query_record in enumerate(query_records):
        top_candidates = []
        for cand_idx in ranked[row_idx]:
            candidate = candidate_records[int(cand_idx)]
            entry = {
                "item_id": candidate["item_id"],
                "score": float(sim[row_idx, int(cand_idx)]),
            }
            if candidate.get("video_path"):
                entry["video_path"] = candidate["video_path"]
            if candidate.get("caption"):
                entry["caption"] = candidate["caption"]
            top_candidates.append(entry)

        result = {
            "task": task_name,
            "sample_id": query_record["sample_id"],
            "target_id": query_record["target_id"],
            "topk_predictions": top_candidates,
        }
        if query_kind == "text":
            result["query"] = query_record["query"]
        else:
            result["video_path"] = query_record["video_path"]
        out.append(result)

    return out


def _evaluate_lovr_retrieval(cfg, dataset, ret_backend, batch_size):
    """Evaluate LoVR's text-video retrieval benchmark."""
    topk_values = [int(k) for k in cfg.dataset.metrics.topk]
    max_k = max(topk_values)

    log.info("Encoding LoVR full videos (%d) ...", len(dataset.video_items))
    full_video_feats, kept_full_items, failed_full = _embed_lovr_video_items(
        dataset.video_items,
        ret_backend,
        batch_size=batch_size,
        desc="LoVR full videos",
    )
    log.info("Encoding LoVR clips (%d) ...", len(dataset.clip_items))
    clip_video_feats, kept_clip_items, failed_clip = _embed_lovr_video_items(
        dataset.clip_items,
        ret_backend,
        batch_size=batch_size,
        desc="LoVR clips",
    )

    kept_video_queries, skipped_video_queries = _filter_lovr_queries_by_target(
        dataset.video_text_queries,
        [item["item_id"] for item in kept_full_items],
    )
    kept_clip_queries, skipped_clip_queries = _filter_lovr_queries_by_target(
        dataset.clip_text_queries,
        [item["item_id"] for item in kept_clip_items],
    )

    if not kept_video_queries or not kept_clip_queries:
        raise RuntimeError(
            "LoVR evaluation requires at least one surviving video query and clip query"
        )

    log.info("Encoding LoVR video captions (%d) ...", len(kept_video_queries))
    video_text_feats = ret_backend.embed_text_batch(
        [sample.query for sample in kept_video_queries],
        batch_size=batch_size,
    )
    log.info("Encoding LoVR clip captions (%d) ...", len(kept_clip_queries))
    clip_text_feats = ret_backend.embed_text_batch(
        [sample.query for sample in kept_clip_queries],
        batch_size=batch_size,
    )

    clip_text_to_clip_sim = clip_text_feats @ clip_video_feats.T
    video_text_to_video_sim = video_text_feats @ full_video_feats.T
    clip_video_to_text_sim = clip_video_feats @ clip_text_feats.T
    video_video_to_text_sim = full_video_feats @ video_text_feats.T

    metrics = compute_lovr_pass_metrics(
        clip_text_to_clip_similarity=clip_text_to_clip_sim,
        video_text_to_video_similarity=video_text_to_video_sim,
        clip_video_to_text_similarity=clip_video_to_text_sim,
        video_video_to_text_similarity=video_video_to_text_sim,
        clip_query_target_ids=[
            str(sample.metadata["target_id"]) for sample in kept_clip_queries
        ],
        video_query_target_ids=[
            str(sample.metadata["target_id"]) for sample in kept_video_queries
        ],
        clip_candidate_ids=[item["item_id"] for item in kept_clip_items],
        video_candidate_ids=[item["item_id"] for item in kept_full_items],
        topk_values=topk_values,
    )

    video_query_records = [
        {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "target_id": str(sample.metadata["target_id"]),
        }
        for sample in kept_video_queries
    ]
    clip_query_records = [
        {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "target_id": str(sample.metadata["target_id"]),
        }
        for sample in kept_clip_queries
    ]
    full_video_query_records = [
        {
            "sample_id": f"video::{item['item_id']}",
            "video_path": item["video_path"],
            "target_id": item["item_id"],
        }
        for item in kept_full_items
    ]
    clip_video_query_records = [
        {
            "sample_id": f"clip::{item['item_id']}",
            "video_path": item["video_path"],
            "target_id": item["item_id"],
        }
        for item in kept_clip_items
    ]

    per_query_results = []
    per_query_results.extend(
        _collect_lovr_topk_results(
            "text_to_video",
            video_text_to_video_sim,
            video_query_records,
            kept_full_items,
            max_k,
            query_kind="text",
        )
    )
    per_query_results.extend(
        _collect_lovr_topk_results(
            "text_to_clip",
            clip_text_to_clip_sim,
            clip_query_records,
            kept_clip_items,
            max_k,
            query_kind="text",
        )
    )
    per_query_results.extend(
        _collect_lovr_topk_results(
            "video_to_text",
            video_video_to_text_sim,
            full_video_query_records,
            [
                {
                    "item_id": rec["target_id"],
                    "caption": rec["query"],
                }
                for rec in video_query_records
            ],
            max_k,
            query_kind="video",
        )
    )
    per_query_results.extend(
        _collect_lovr_topk_results(
            "clip_to_text",
            clip_video_to_text_sim,
            clip_video_query_records,
            [
                {
                    "item_id": rec["target_id"],
                    "caption": rec["query"],
                }
                for rec in clip_query_records
            ],
            max_k,
            query_kind="video",
        )
    )

    failed_items = {
        "full_videos": failed_full,
        "clips": failed_clip,
    }
    skipped_queries = skipped_video_queries + skipped_clip_queries
    summary = {
        "num_full_videos": len(kept_full_items),
        "num_clips": len(kept_clip_items),
        "num_video_text_queries": len(kept_video_queries),
        "num_clip_text_queries": len(kept_clip_queries),
    }

    return metrics, per_query_results, summary, failed_items, skipped_queries


# ---------------------------------------------------------------------------
# Shared VLM chunk preparation
# ---------------------------------------------------------------------------

def _prepare_vlm_video_context(
    cfg,
    video_path,
    tal,
    chunk_model=None,
    chunk_processor=None,
    emb_backend=None,
    device=None,
):
    """Prepare per-video chunk context shared by VLM evaluation modes."""
    chunk_source = cfg.evaluation.chunk_source

    if chunk_source == "uniform":
        fps, total_frames = _get_video_info(video_path)
        num_chunks = int(cfg.evaluation.num_uniform_chunks)
        chunks = _make_uniform_chunks(total_frames, num_chunks)
        vid_data = {
            "fps": float(fps),
            "total_frames": total_frames,
            "duration": total_frames / fps,
        }
        chunking_result = None
    elif chunk_source == "stable_chunks":
        chunking_result = _run_chunking(
            cfg, video_path, chunk_model, chunk_processor, device,
            embedding_backend=emb_backend,
        )
        fps = chunking_result["fps"]

        pp = cfg.postprocessing
        labeled, chunk_stages = postprocess_chunks(
            chunking_result,
            detect_transitions=pp.detect_transitions,
            transition_params=OmegaConf.to_container(pp.transition),
            merge_params=OmegaConf.to_container(pp.merge),
            return_stages=True,
        )

        chunks = [lc["chunk"] for lc in labeled if lc["label"] == "stable"]
        if not chunks:
            raise RuntimeError(
                f"No stable chunks produced for {video_path}. "
                f"Check chunking/postprocessing parameters."
            )

        vid_data = {
            "fps": float(fps),
            "total_frames": int(chunking_result["total_frames"]),
            "duration": chunking_result["total_frames"] / fps,
            "stages": {k: _serialize_labeled_chunks(v) for k, v in chunk_stages.items()},
        }
    else:
        raise ValueError(f"Unknown chunk_source: {chunk_source}")

    use_best_window = tal.frame_method == "best_window"
    use_window_selection = tal.frame_method in WINDOW_METHODS
    if use_best_window and chunk_source != "stable_chunks":
        raise ValueError(
            "frame_selection.method=best_window requires "
            "evaluation.chunk_source=stable_chunks"
        )
    if use_window_selection and chunk_source != "stable_chunks":
        raise ValueError(
            f"frame_selection.method={tal.frame_method} requires "
            "evaluation.chunk_source=stable_chunks (needs window embeddings)"
        )

    sample_interval = (
        _infer_sample_interval(chunking_result)
        if chunk_source == "stable_chunks"
        else None
    )
    allowed_chunks = compute_allowed_chunks(
        tal.max_chunks,
        tal.frames_per_chunk,
        tal.vlm.max_images_per_request,
    )

    return {
        "chunk_source": chunk_source,
        "chunks": chunks,
        "fps": fps,
        "vid_data": vid_data,
        "chunking_result": chunking_result,
        "use_best_window": use_best_window,
        "use_window_selection": use_window_selection,
        "sample_interval": sample_interval,
        "allowed_chunks": allowed_chunks,
    }


def _prepare_vlm_prompt_context(
    sample,
    chunks,
    fps,
    chunk_source,
    allowed_chunks,
    use_best_window,
    chunking_result=None,
    emb_backend=None,
    sample_interval=None,
    frames_per_chunk=1,
    frame_method="middle",
    best_window_strategy="ranked_round_robin",
    preselection="none",
    query_merge_cfg=None,
    use_window_selection=False,
    window_method_kwargs=None,
    n_frames=None,
):
    """Prepare per-query chunk/frame selections for VLM prompting."""

    # --- Window-based frame selection (mmr, temporal, coverage, rdmv, weighted) ---
    if use_window_selection:
        return _prepare_window_prompt_context(
            sample, chunks, fps, chunk_source, chunking_result,
            emb_backend, frame_method, n_frames=n_frames,
            window_method_kwargs=window_method_kwargs or {},
        )

    # --- Legacy chunk-based path ---
    use_preselection = (
        preselection == "query_similarity"
        and chunk_source == "stable_chunks"
    )
    use_query_merge = (
        query_merge_cfg is not None
        and query_merge_cfg.get("enabled", False)
        and chunk_source == "stable_chunks"
    )
    selector_meta = None
    prompt_frame_selections = None
    query_emb = None
    base_chunks = chunks

    if use_query_merge:
        query_emb = emb_backend.embed_text(sample.query)
        if "window_embeddings" not in chunking_result:
            raise ValueError(
                "postprocessing.query_merge for VLM QA requires chunking "
                "results with window embeddings"
            )
        chunk_embs = ChunkMerger.aggregate_embeddings(
            chunking_result["window_times"],
            chunking_result["window_embeddings"],
            base_chunks,
            fps,
        )
        query_merger = QueryChunkMerger(fps)
        chunks, _ = query_merger.merge(
            base_chunks,
            chunk_embs,
            query_emb,
            similarity_threshold=float(query_merge_cfg.similarity_threshold),
            max_chunk_sec=float(query_merge_cfg.max_chunk_sec),
        )

    if use_preselection:
        if query_emb is None:
            query_emb = emb_backend.embed_text(sample.query)
        if use_best_window:
            scores, all_frame_selections = _score_chunks_with_windows(
                query_emb,
                chunks,
                chunking_result,
                frames_per_chunk=frames_per_chunk,
                sample_interval=sample_interval,
                best_window_strategy=best_window_strategy,
            )
        else:
            scores = _score_chunks(
                "max_sim", query_emb, chunks, None, chunking_result,
            )
            all_frame_selections = None
        ranked_indices = np.argsort(-scores)
        selected_indices = ranked_indices[:allowed_chunks]
        prompt_indices = np.sort(selected_indices)
        query_chunks = [chunks[i] for i in prompt_indices]
        if use_best_window:
            prompt_frame_selections = [
                all_frame_selections[i] for i in prompt_indices.tolist()
            ]
        selector_meta = {
            "original_indices": ranked_indices.tolist(),
            "prompt_original_indices": prompt_indices.tolist(),
            "scores": scores[ranked_indices].tolist(),
        }
    else:
        query_chunks = chunks

    used_chunks = query_chunks[:allowed_chunks]
    if use_best_window:
        if query_emb is None:
            query_emb = emb_backend.embed_text(sample.query)
        if prompt_frame_selections is None:
            _, prompt_frame_selections = _score_chunks_with_windows(
                query_emb,
                used_chunks,
                chunking_result,
                frames_per_chunk=frames_per_chunk,
                sample_interval=sample_interval,
                best_window_strategy=best_window_strategy,
            )

    # Compute per-chunk cosine similarity for VLM prompt metadata.
    chunk_scores = None
    has_embeddings = (
        chunking_result is not None
        and "window_embeddings" in chunking_result
        and emb_backend is not None
    )
    if has_embeddings:
        if use_preselection:
            chunk_scores = scores[prompt_indices[: len(used_chunks)]].tolist()
        else:
            if query_emb is None:
                query_emb = emb_backend.embed_text(sample.query)
            used_scores, _ = _score_chunks_with_windows(
                query_emb,
                used_chunks,
                chunking_result,
                frames_per_chunk=frames_per_chunk,
                sample_interval=sample_interval,
                best_window_strategy=best_window_strategy,
            )
            chunk_scores = used_scores.tolist()

    vlm_prompt_metadata = {
        "chunk_source": chunk_source,
        "frame_method": frame_method,
        "best_window_strategy": best_window_strategy,
        "frames_per_chunk": int(frames_per_chunk),
        "used_chunks": [[int(s), int(e)] for s, e in used_chunks],
        "query_merge_enabled": bool(use_query_merge),
        "num_chunks_before_query_merge": len(base_chunks),
        "num_chunks_after_query_merge": len(chunks),
    }
    if use_preselection:
        vlm_prompt_metadata["original_chunk_indices"] = [
            int(i) for i in prompt_indices.tolist()[: len(used_chunks)]
        ]
    if prompt_frame_selections is not None:
        vlm_prompt_metadata["chunk_frame_selections"] = prompt_frame_selections
    if chunk_scores is not None:
        vlm_prompt_metadata["chunk_scores"] = chunk_scores

    frame_indices, frame_times = _build_legacy_prompt_frame_data(
        used_chunks,
        fps,
        frame_method=frame_method,
        frames_per_chunk=frames_per_chunk,
        prompt_frame_selections=prompt_frame_selections,
    )

    result = {
        "query_chunks": query_chunks,
        "used_chunks": used_chunks,
        "chunk_scores": chunk_scores,
        "selector_meta": selector_meta,
        "prompt_frame_selections": prompt_frame_selections,
        "vlm_prompt_metadata": vlm_prompt_metadata,
        "window_frame_indices": frame_indices,
        "window_frame_times": frame_times,
        "window_frame_scores": None,
    }
    return result


def _build_legacy_prompt_frame_data(
    used_chunks,
    fps,
    frame_method="middle",
    frames_per_chunk=1,
    prompt_frame_selections=None,
):
    """Return exact displayed frame indices/times for legacy chunk methods."""
    if frame_method == "best_window":
        if not prompt_frame_selections:
            raise ValueError("best_window QA prompting requires frame selection metadata")
        frames_per_chunk = max(int(frames_per_chunk), 1)
        frame_indices = []
        for selection in prompt_frame_selections[: len(used_chunks)]:
            frame_indices.extend(
                int(i)
                for i in selection.get("frame_indices", [])[:frames_per_chunk]
            )
    elif frame_method == "multi":
        frame_indices = []
        for start_frame, end_frame in used_chunks:
            n = max(int(frames_per_chunk), 1)
            max_frame = max(start_frame, end_frame - 1)
            positions = np.arange(1, n + 1, dtype=float) / (n + 1)
            chunk_indices = start_frame + positions * max(end_frame - start_frame, 0)
            frame_indices.extend(
                int(i)
                for i in np.clip(chunk_indices.astype(int), start_frame, max_frame).tolist()
            )
    else:
        frame_indices = [int((sf + ef) // 2) for sf, ef in used_chunks]

    frame_times = [float(idx) / fps for idx in frame_indices]
    return frame_indices, frame_times


def _prepare_window_prompt_context(
    sample, chunks, fps, chunk_source, chunking_result,
    emb_backend, frame_method, n_frames=None, window_method_kwargs=None,
):
    """Prepare prompt context using window-embedding-based frame selection."""
    window_method_kwargs = window_method_kwargs or {}
    query_emb = emb_backend.embed_text(sample.query)
    wt = chunking_result["window_times"]
    we = chunking_result["window_embeddings"]
    duration_sec = chunking_result["total_frames"] / fps
    sample_interval = _infer_sample_interval(chunking_result)

    if n_frames is None:
        n_frames = 24

    result = select_frames_from_windows(
        method=frame_method,
        window_embeddings=we,
        window_times=wt,
        query_embedding=query_emb,
        n_frames=n_frames,
        duration_sec=duration_sec,
        chunks=chunks,
        fps=fps,
        sample_interval=sample_interval,
        **window_method_kwargs,
    )

    frame_times = [float(t) for t in result["frame_times"]]
    window_indices = [int(i) for i in result["window_indices"]]
    frame_indices = [int(i) for i in result["frame_indices"]]
    frame_scores = [float(s) for s in result["scores"]]

    vlm_prompt_metadata = {
        "chunk_source": chunk_source,
        "frame_method": frame_method,
        "n_frames": int(n_frames),
        "window_indices": window_indices,
        "frame_indices": frame_indices,
        "frame_times": frame_times,
        "frame_scores": frame_scores,
        "used_chunks": [[int(s), int(e)] for s, e in chunks],
    }

    return {
        "query_chunks": chunks,
        "used_chunks": chunks,
        "chunk_scores": None,
        "selector_meta": None,
        "prompt_frame_selections": None,
        "vlm_prompt_metadata": vlm_prompt_metadata,
        "window_frame_indices": frame_indices,
        "window_frame_times": frame_times,
        "window_frame_scores": frame_scores,
    }


# ---------------------------------------------------------------------------
# Standalone per-query VLM execution (thread-safe, no GPU ops)
# ---------------------------------------------------------------------------

def _run_vlm_cs_query(tal, video_path, fps, sample, prompt_ctx,
                      max_pred_windows=None):
    """Execute one VLM chunk-selection query and build its result dict.

    Thread-safe: only disk I/O (frame loading), CPU (prompt building),
    and network I/O (API call). No GPU operations.
    """
    query_chunks = prompt_ctx["query_chunks"]
    used_chunks = prompt_ctx["used_chunks"]
    selector_meta = prompt_ctx["selector_meta"]
    prompt_frame_selections = prompt_ctx["prompt_frame_selections"]
    vlm_prompt_metadata = prompt_ctx["vlm_prompt_metadata"]
    vlm_prompt_metadata["selected_chunk_index"] = None

    result = tal.select_best_chunk(
        video_path, query_chunks, fps, sample.query,
        video_metadata=sample.metadata,
        chunk_frame_selections=prompt_frame_selections,
    )
    vlm_prompt_metadata["selected_chunk_index"] = result["chunk_index"]

    if result["decision"] == "chunk":
        sf, ef = result["chunk"]
        predictions = [{
            "start": sf / fps,
            "end": ef / fps,
            "score": 1.0,
        }]
    elif result["decision"] == "no_match":
        predictions = []
    else:
        predictions = []

    if max_pred_windows is not None:
        predictions = predictions[:int(max_pred_windows)]

    num_chunks_total = prompt_ctx.get("num_chunks_total", len(query_chunks))
    qr = {
        "sample_id": sample.sample_id,
        "query": sample.query,
        "video_path": video_path,
        "num_chunks": len(used_chunks),
        "num_chunks_total_available": num_chunks_total,
        "predictions": predictions,
        "top1_pred": predictions[0] if predictions else None,
        "gt_windows": sample.gt_windows,
        "metadata": sample.metadata,
        "vlm_raw_response": result["raw_response"],
        "vlm_decision": result["decision"],
    }
    if selector_meta is not None:
        qr["selector_metadata"] = selector_meta
    qr["vlm_prompt_metadata"] = vlm_prompt_metadata
    if result["decision"] == "parse_failure":
        qr["failure_reason"] = "vlm_parse_failure"
    elif result["decision"] == "no_match":
        qr["failure_reason"] = "vlm_no_match"

    skipped_entry = None
    if result["decision"] == "parse_failure":
        skipped_entry = {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "video_path": video_path,
            "gt_windows": sample.gt_windows,
            "vlm_raw_response": result["raw_response"],
            "reason": "vlm_parse_failure",
        }

    return qr, skipped_entry


def _run_vlm_qa_query(tal, video_path, fps, sample, prompt_ctx, vid_data,
                      num_chunks_total):
    """Execute one VLM QA query and build its result dict.

    Thread-safe: only disk I/O (frame loading), CPU (prompt building),
    and network I/O (API call). No GPU operations.
    """
    used_chunks = prompt_ctx["used_chunks"]
    selector_meta = prompt_ctx["selector_meta"]
    vlm_prompt_metadata = prompt_ctx["vlm_prompt_metadata"]

    result = tal.answer_question(
        video_path,
        used_chunks,
        fps,
        sample.query,
        video_metadata=vid_data,
        question_metadata=sample.metadata,
        window_frame_indices=prompt_ctx.get("window_frame_indices"),
        window_frame_times=prompt_ctx.get("window_frame_times"),
        window_frame_scores=prompt_ctx.get("window_frame_scores"),
    )
    qr = {
        "sample_id": sample.sample_id,
        "query": sample.metadata.get("display_query", sample.query),
        "video_path": video_path,
        "predicted_answer": result["answer"],
        "gt_answer": sample.metadata.get("gt_answer"),
        "vlm_raw_response": result["raw_response"],
        "vlm_prompt_text": result.get("prompt_text", ""),
        "num_chunks": len(used_chunks),
        "num_chunks_total_available": num_chunks_total,
        "num_chunks_before_query_merge": num_chunks_total,
        "metadata": sample.metadata,
        "vlm_prompt_metadata": vlm_prompt_metadata,
    }
    if "answer_type" in sample.metadata:
        qr["answer_type"] = sample.metadata["answer_type"]
    if sample.metadata.get("qa_format") == "mcq":
        options = list(sample.metadata.get("options") or [])
        valid_letters = "".join(chr(ord("A") + idx) for idx in range(len(options)))
        mcq_eval = sample.metadata.get("mcq_eval")
        if mcq_eval == "longvideobench":
            qr["predicted_answer_letter"] = (
                parse_longvideobench_mcq_answer(
                    result["answer"],
                    valid_letters=tuple(valid_letters),
                )
                if valid_letters
                else None
            )
        else:
            qr["predicted_answer_letter"] = extract_mcq_letter(
                result["answer"],
                valid_letters=valid_letters or "ABCD",
            )
        gt_ans = sample.metadata.get("gt_answer")
        if gt_ans is not None:
            qr["gt_answer_letter"] = extract_mcq_letter(
                gt_ans,
                valid_letters=valid_letters or "ABCD",
            )
    if selector_meta is not None:
        qr["selector_metadata"] = selector_meta
    return qr


# ---------------------------------------------------------------------------
# Per-query: iterative VLM QA
# ---------------------------------------------------------------------------

def _run_vlm_iterative_qa_query(
    tal, video_path, fps, sample, chunking_result, emb_backend,
    vid_data, num_chunks_total, iterative_config,
):
    """Execute one iterative VLM QA query and build its result dict."""
    wt = chunking_result["window_times"]
    we = chunking_result["window_embeddings"]
    duration_sec = chunking_result["total_frames"] / fps

    query_emb = emb_backend.embed_text(sample.query)

    result = tal.answer_question_iterative(
        video_path=video_path,
        fps=fps,
        question=sample.query,
        window_embeddings=we,
        window_times=wt,
        query_embedding=query_emb,
        emb_backend=emb_backend,
        iterative_config=iterative_config,
        video_metadata=vid_data,
        question_metadata=sample.metadata,
        duration_sec=duration_sec,
    )

    vlm_prompt_metadata = {
        "frame_method": "iterative",
        "iterative_queries": result["iterative_queries"],
        "iterative_rounds": result["iterative_rounds"],
        "total_frames_used": result["total_frames_used"],
        "total_api_calls": result["total_api_calls"],
        "all_frame_times": result["all_frame_times"],
        "all_frame_indices": result["all_frame_indices"],
    }

    qr = {
        "sample_id": sample.sample_id,
        "query": sample.metadata.get("display_query", sample.query),
        "video_path": video_path,
        "predicted_answer": result["answer"],
        "gt_answer": sample.metadata.get("gt_answer"),
        "vlm_raw_response": result["raw_response"],
        "vlm_prompt_text": result.get("prompt_text", ""),
        "num_chunks": num_chunks_total,
        "num_chunks_total_available": num_chunks_total,
        "num_chunks_before_query_merge": num_chunks_total,
        "metadata": sample.metadata,
        "vlm_prompt_metadata": vlm_prompt_metadata,
    }
    if "answer_type" in sample.metadata:
        qr["answer_type"] = sample.metadata["answer_type"]
    if sample.metadata.get("qa_format") == "mcq":
        options = list(sample.metadata.get("options") or [])
        valid_letters = "".join(chr(ord("A") + idx) for idx in range(len(options)))
        mcq_eval = sample.metadata.get("mcq_eval")
        if mcq_eval == "longvideobench":
            qr["predicted_answer_letter"] = (
                parse_longvideobench_mcq_answer(
                    result["answer"],
                    valid_letters=tuple(valid_letters),
                )
                if valid_letters
                else None
            )
        else:
            qr["predicted_answer_letter"] = extract_mcq_letter(
                result["answer"],
                valid_letters=valid_letters or "ABCD",
            )
        gt_ans = sample.metadata.get("gt_answer")
        if gt_ans is not None:
            qr["gt_answer_letter"] = extract_mcq_letter(
                gt_ans,
                valid_letters=valid_letters or "ABCD",
            )
    return qr


# ---------------------------------------------------------------------------
# Per-video evaluation: embedding retrieval path
# ---------------------------------------------------------------------------

def _evaluate_video_embedding(
    cfg, video_path, samples, ret_backend, chunk_model, chunk_processor,
    emb_backend, device, score_method, batch_size,
):
    """Evaluate one video using the embedding retrieval path.

    Returns (query_results, vid_data).
    """
    chunking_result = _run_chunking(
        cfg, video_path, chunk_model, chunk_processor, device,
        embedding_backend=emb_backend,
    )
    fps = chunking_result["fps"]

    pp = cfg.postprocessing
    labeled, chunk_stages = postprocess_chunks(
        chunking_result,
        detect_transitions=pp.detect_transitions,
        transition_params=OmegaConf.to_container(pp.transition),
        merge_params=OmegaConf.to_container(pp.merge),
        return_stages=True,
    )

    stable_chunks = [lc["chunk"] for lc in labeled if lc["label"] == "stable"]
    if not stable_chunks:
        raise RuntimeError(
            f"No stable chunks produced for {video_path}. "
            f"Check chunking/postprocessing parameters."
        )

    aggregation_cfg = cfg.evaluation.get("chunk_aggregation")
    aggregation_params = (
        OmegaConf.to_container(aggregation_cfg) if aggregation_cfg is not None else {}
    )
    chunk_embs, chunk_coherence = _build_chunk_embs(
        score_method, stable_chunks, chunking_result,
        ret_backend, video_path, batch_size=batch_size,
        aggregation_params=aggregation_params,
    )

    def _serialize_labeled(lc_list):
        return [{"chunk": [int(lc["chunk"][0]), int(lc["chunk"][1])],
                 "label": lc["label"]} for lc in lc_list]

    vid_data = {
        "fps": float(fps),
        "total_frames": int(chunking_result["total_frames"]),
        "duration": chunking_result["total_frames"] / fps,
        "stages": {k: _serialize_labeled(v)
                   for k, v in chunk_stages.items()},
    }
    if chunk_coherence is not None:
        vid_data["chunk_coherence"] = [float(r) for r in chunk_coherence]
    if "signal_times" in chunking_result:
        vid_data["signal_times"] = chunking_result["signal_times"].tolist()
        vid_data["signal_values"] = chunking_result["signal_values"].tolist()
        raw_threshold = chunking_result.get("threshold")
        vid_data["threshold"] = (
            float(raw_threshold) if raw_threshold is not None else None
        )

    num_frames = ret_backend.num_frames
    qm = pp.query_merge
    query_merger = (QueryChunkMerger(fps) if qm.enabled else None)

    query_embs = ret_backend.embed_text_batch(
        [s.query for s in samples], batch_size=batch_size,
    )

    query_results = []
    for sample_idx, sample in enumerate(samples):
        query_emb = query_embs[sample_idx]

        if query_merger is not None:
            embed_fn = None
            if score_method == "recompute":
                def embed_fn(chunk):
                    frames = _load_chunk_frames(
                        video_path, chunk, num_frames,
                    )
                    return ret_backend.embed_video(frames)

            q_chunks, q_embs = query_merger.merge(
                stable_chunks, chunk_embs, query_emb,
                similarity_threshold=qm.similarity_threshold,
                max_chunk_sec=qm.max_chunk_sec,
                embed_fn=embed_fn,
            )
            sims = _score_chunks(
                score_method, query_emb, q_chunks,
                np.stack(q_embs), chunking_result,
            )
        else:
            q_chunks = stable_chunks
            sims = _score_chunks(
                score_method, query_emb, q_chunks,
                chunk_embs, chunking_result,
            )

        ranked = np.argsort(-sims)

        fallback_predictions = []
        for idx in ranked:
            sf, ef = q_chunks[idx]
            fallback_predictions.append({
                "start": sf / fps,
                "end": ef / fps,
                "score": float(sims[idx]),
            })

        selector_meta = None
        selector_cfg = cfg.postprocessing.get("moment_selection")
        use_selector = (
            selector_cfg is not None
            and selector_cfg.get("enabled", False)
        )
        if use_selector:
            raw_penalty = selector_cfg.get("penalty")
            predictions, selector_meta = select_moments(
                q_chunks,
                sims,
                fps,
                method=str(selector_cfg.get("method", "penalized_dp")),
                penalty=float(raw_penalty) if raw_penalty is not None else None,
                penalty_factor=float(selector_cfg.get("penalty_factor", 1.0)),
                max_moment_sec=(
                    float(selector_cfg["max_moment_sec"])
                    if selector_cfg.get("max_moment_sec") is not None
                    else None
                ),
                gap=float(selector_cfg.get("gap", 0.05)),
            )
            if not predictions:
                predictions = fallback_predictions
        else:
            predictions = fallback_predictions

        if not predictions and fallback_predictions:
            predictions = fallback_predictions[:1]

        max_pred_windows = cfg.dataset.get("max_pred_windows", None)
        if max_pred_windows is not None:
            predictions = predictions[:int(max_pred_windows)]

        qr = {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "video_path": video_path,
            "num_chunks": len(q_chunks),
            "num_chunks_before_query_merge": len(stable_chunks),
            "predictions": predictions,
            "top1_pred": predictions[0] if predictions else None,
            "gt_windows": sample.gt_windows,
            "metadata": sample.metadata,
        }
        if selector_meta is not None:
            qr["selector_metadata"] = selector_meta
        query_results.append(qr)

    return query_results, vid_data


# ---------------------------------------------------------------------------
# Per-video evaluation: VLM chunk-selection path
# ---------------------------------------------------------------------------

def _evaluate_video_vlm(
    cfg, video_path, samples, tal,
    chunk_model=None, chunk_processor=None, emb_backend=None, device=None,
):
    """Evaluate one video using the VLM chunk-selection path.

    Returns (query_results, vid_data).
    """
    video_ctx = _prepare_vlm_video_context(
        cfg,
        video_path,
        tal,
        chunk_model=chunk_model,
        chunk_processor=chunk_processor,
        emb_backend=emb_backend,
        device=device,
    )
    chunk_source = video_ctx["chunk_source"]
    chunks = video_ctx["chunks"]
    fps = video_ctx["fps"]
    vid_data = video_ctx["vid_data"]
    chunking_result = video_ctx["chunking_result"]
    use_best_window = video_ctx["use_best_window"]
    sample_interval = video_ctx["sample_interval"]
    allowed_chunks = video_ctx["allowed_chunks"]

    preselection = cfg.evaluation.get("chunk_preselection", "none")
    query_results = []
    skipped = []
    for sample in samples:
        prompt_ctx = _prepare_vlm_prompt_context(
            sample,
            chunks,
            fps,
            chunk_source,
            allowed_chunks,
            use_best_window,
            chunking_result=chunking_result,
            emb_backend=emb_backend,
            sample_interval=sample_interval,
            frames_per_chunk=tal.frames_per_chunk,
            frame_method=tal.frame_method,
            best_window_strategy=tal.best_window_strategy,
            preselection=preselection,
            query_merge_cfg=cfg.postprocessing.query_merge,
            use_window_selection=video_ctx.get("use_window_selection", False),
            window_method_kwargs=tal.window_method_kwargs,
            n_frames=tal.n_frames,
        )
        query_chunks = prompt_ctx["query_chunks"]
        used_chunks = prompt_ctx["used_chunks"]
        chunk_scores = prompt_ctx["chunk_scores"]
        selector_meta = prompt_ctx["selector_meta"]
        prompt_frame_selections = prompt_ctx["prompt_frame_selections"]
        vlm_prompt_metadata = prompt_ctx["vlm_prompt_metadata"]
        vlm_prompt_metadata["selected_chunk_index"] = None

        result = tal.select_best_chunk(
            video_path, query_chunks, fps, sample.query,
            video_metadata=sample.metadata,
            chunk_frame_selections=prompt_frame_selections,
            chunk_scores=chunk_scores,
        )
        vlm_prompt_metadata["selected_chunk_index"] = result["chunk_index"]

        if result["decision"] == "chunk":
            sf, ef = result["chunk"]
            predictions = [{
                "start": sf / fps,
                "end": ef / fps,
                "score": 1.0,
            }]
        elif result["decision"] == "no_match":
            predictions = []
        else:
            log.warning(
                "VLM parse failure for sample %s — counts as miss",
                sample.sample_id,
            )
            predictions = []
            skipped.append({
                "sample_id": sample.sample_id,
                "query": sample.query,
                "video_path": video_path,
                "gt_windows": sample.gt_windows,
                "vlm_raw_response": result["raw_response"],
                "reason": "vlm_parse_failure",
            })

        max_pred_windows = cfg.dataset.get("max_pred_windows", None)
        if max_pred_windows is not None:
            predictions = predictions[:int(max_pred_windows)]

        qr = {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "video_path": video_path,
            "num_chunks": len(used_chunks),
            "num_chunks_total_available": len(chunks),
            "predictions": predictions,
            "top1_pred": predictions[0] if predictions else None,
            "gt_windows": sample.gt_windows,
            "metadata": sample.metadata,
            "vlm_raw_response": result["raw_response"],
            "vlm_decision": result["decision"],
        }
        if selector_meta is not None:
            qr["selector_metadata"] = selector_meta
        qr["vlm_prompt_metadata"] = vlm_prompt_metadata
        if result["decision"] == "parse_failure":
            qr["failure_reason"] = "vlm_parse_failure"
        elif result["decision"] == "no_match":
            qr["failure_reason"] = "vlm_no_match"
        query_results.append(qr)

    return query_results, skipped, vid_data


# ---------------------------------------------------------------------------
# Per-video evaluation: VLM QA path
# ---------------------------------------------------------------------------

def _evaluate_video_vlm_qa(
    cfg, video_path, samples, tal,
    chunk_model=None, chunk_processor=None, emb_backend=None, device=None,
):
    """Evaluate one video using the VLM question-answering path.

    Returns (query_results, vid_data).
    """
    video_ctx = _prepare_vlm_video_context(
        cfg,
        video_path,
        tal,
        chunk_model=chunk_model,
        chunk_processor=chunk_processor,
        emb_backend=emb_backend,
        device=device,
    )
    chunk_source = video_ctx["chunk_source"]
    chunks = video_ctx["chunks"]
    fps = video_ctx["fps"]
    vid_data = video_ctx["vid_data"]
    chunking_result = video_ctx["chunking_result"]
    use_best_window = video_ctx["use_best_window"]
    sample_interval = video_ctx["sample_interval"]
    allowed_chunks = video_ctx["allowed_chunks"]

    # --- Iterative QA path ---
    iterative_config = OmegaConf.to_container(
        cfg.temporal_abstraction.get("iterative", {}), resolve=True,
    ) if cfg.temporal_abstraction.get("iterative") else {}
    use_iterative = bool(iterative_config.get("n_queries"))

    if use_iterative:
        if chunk_source != "stable_chunks":
            raise ValueError(
                "Iterative QA requires evaluation.chunk_source=stable_chunks "
                "(needs window embeddings)"
            )
        if "window_embeddings" not in chunking_result:
            raise ValueError(
                "Iterative QA requires window embeddings in chunking result"
            )
        query_results = []
        for sample in samples:
            qr = _run_vlm_iterative_qa_query(
                tal, video_path, fps, sample, chunking_result, emb_backend,
                vid_data, len(chunks), iterative_config,
            )
            query_results.append(qr)
        return query_results, vid_data

    # --- Standard (non-iterative) QA path ---
    preselection = cfg.evaluation.get("chunk_preselection", "none")
    query_results = []
    for sample in samples:
        prompt_ctx = _prepare_vlm_prompt_context(
            sample,
            chunks,
            fps,
            chunk_source,
            allowed_chunks,
            use_best_window,
            chunking_result=chunking_result,
            emb_backend=emb_backend,
            sample_interval=sample_interval,
            frames_per_chunk=tal.frames_per_chunk,
            frame_method=tal.frame_method,
            best_window_strategy=tal.best_window_strategy,
            preselection=preselection,
            query_merge_cfg=cfg.postprocessing.query_merge,
            use_window_selection=video_ctx.get("use_window_selection", False),
            window_method_kwargs=tal.window_method_kwargs,
            n_frames=tal.n_frames,
        )
        used_chunks = prompt_ctx["used_chunks"]
        selector_meta = prompt_ctx["selector_meta"]
        vlm_prompt_metadata = prompt_ctx["vlm_prompt_metadata"]

        result = tal.answer_question(
            video_path,
            used_chunks,
            fps,
            sample.query,
            video_metadata=vid_data,
            question_metadata=sample.metadata,
            window_frame_indices=prompt_ctx.get("window_frame_indices"),
            window_frame_times=prompt_ctx.get("window_frame_times"),
            window_frame_scores=prompt_ctx.get("window_frame_scores"),
        )
        qr = {
            "sample_id": sample.sample_id,
            "query": sample.metadata.get("display_query", sample.query),
            "video_path": video_path,
            "predicted_answer": result["answer"],
            "gt_answer": sample.metadata.get("gt_answer"),
            "vlm_raw_response": result["raw_response"],
            "vlm_prompt_text": result.get("prompt_text", ""),
            "num_chunks": len(used_chunks),
            "num_chunks_total_available": len(chunks),
            "num_chunks_before_query_merge": len(chunks),
            "metadata": sample.metadata,
            "vlm_prompt_metadata": vlm_prompt_metadata,
        }
        if "answer_type" in sample.metadata:
            qr["answer_type"] = sample.metadata["answer_type"]
        if sample.metadata.get("qa_format") == "mcq":
            options = list(sample.metadata.get("options") or [])
            valid_letters = "".join(chr(ord("A") + idx) for idx in range(len(options)))
            mcq_eval = sample.metadata.get("mcq_eval")
            if mcq_eval == "longvideobench":
                qr["predicted_answer_letter"] = (
                    parse_longvideobench_mcq_answer(
                        result["answer"],
                        valid_letters=tuple(valid_letters),
                    )
                    if valid_letters
                    else None
                )
            else:
                qr["predicted_answer_letter"] = extract_mcq_letter(
                    result["answer"],
                    valid_letters=valid_letters or "ABCD",
                )
            gt_ans = sample.metadata.get("gt_answer")
            if gt_ans is not None:
                qr["gt_answer_letter"] = extract_mcq_letter(
                    gt_ans,
                    valid_letters=valid_letters or "ABCD",
                )
        if selector_meta is not None:
            qr["selector_metadata"] = selector_meta
        query_results.append(qr)

    return query_results, vid_data


# ---------------------------------------------------------------------------
# Per-video evaluation: GEBD boundary detection path
# ---------------------------------------------------------------------------

def _evaluate_video_gebd(
    cfg, video_path, samples, chunk_model, chunk_processor, emb_backend, device,
):
    """Evaluate one video for GEBD: run chunking, extract boundaries.

    Returns (video_result, vid_data) where video_result is a single dict
    (one per video, not per query).
    """
    chunking_result = _run_chunking(
        cfg, video_path, chunk_model, chunk_processor, device,
        embedding_backend=emb_backend,
    )
    fps = chunking_result["fps"]
    total_frames = chunking_result["total_frames"]

    pp = cfg.postprocessing
    labeled, chunk_stages = postprocess_chunks(
        chunking_result,
        detect_transitions=pp.detect_transitions,
        transition_params=OmegaConf.to_container(pp.transition),
        merge_params=OmegaConf.to_container(pp.merge),
        return_stages=True,
    )

    stable_chunks = [lc["chunk"] for lc in labeled if lc["label"] == "stable"]
    all_chunks = [lc["chunk"] for lc in labeled]

    boundary_frames = [int(start_frame) for start_frame, _ in all_chunks[1:]]
    refinement_info = []
    refine_cfg = pp.get("gebd_boundary_refinement")
    if (
        refine_cfg is not None
        and refine_cfg.get("enabled", False)
        and boundary_frames
    ):
        boundary_frames, refinement_info = _refine_gebd_boundary_frames(
            cfg, video_path, chunking_result, boundary_frames,
        )
    predicted_boundaries = [frame / fps for frame in boundary_frames]

    sample = samples[0]
    meta = sample.metadata
    gt_per_rater = meta.get("gt_boundaries_per_rater", [])

    def _serialize_labeled(lc_list):
        return [{"chunk": [int(lc["chunk"][0]), int(lc["chunk"][1])],
                 "label": lc["label"]} for lc in lc_list]

    vid_data = {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "duration": total_frames / fps,
        "stages": {k: _serialize_labeled(v)
                   for k, v in chunk_stages.items()},
        "predicted_boundary_frames": boundary_frames,
    }
    if refinement_info:
        vid_data["gebd_boundary_refinement"] = refinement_info
    if "signal_times" in chunking_result:
        vid_data["signal_times"] = chunking_result["signal_times"].tolist()
        vid_data["signal_values"] = chunking_result["signal_values"].tolist()
        raw_threshold = chunking_result.get("threshold")
        vid_data["threshold"] = (
            float(raw_threshold) if raw_threshold is not None else None
        )

    video_result = {
        "sample_id": sample.sample_id,
        "video_path": video_path,
        "predicted_boundaries": predicted_boundaries,
        "num_chunks": len(all_chunks),
        "num_stable_chunks": len(stable_chunks),
        "metadata": meta,
    }

    return video_result, vid_data


# ---------------------------------------------------------------------------
# Sequential evaluation loop (all modes)
# ---------------------------------------------------------------------------

def _run_sequential(
    cfg, mode, video_to_samples,
    ret_backend, chunk_model, chunk_processor, emb_backend,
    device, tal, score_method, batch_size,
    per_query_results, all_predictions, all_gt_windows,
    per_video_data, failed_videos, skipped_queries,
    checkpoint_interval, checkpoint_path,
):
    """Sequential video evaluation loop (all modes)."""
    total_videos = len(video_to_samples)
    video_times = []
    pbar = tqdm(video_to_samples.items(), desc="Videos", unit="vid",
                total=total_videos)

    for vid_idx, (video_path, samples) in enumerate(pbar):
        vid_start = time.time()
        try:
            if mode == "embedding":
                query_results, vid_data = _evaluate_video_embedding(
                    cfg, video_path, samples, ret_backend,
                    chunk_model, chunk_processor, emb_backend,
                    device, score_method, batch_size,
                )
            elif mode == "vlm_chunk_selection":
                query_results, skipped, vid_data = _evaluate_video_vlm(
                    cfg, video_path, samples, tal,
                    chunk_model=chunk_model,
                    chunk_processor=chunk_processor,
                    emb_backend=emb_backend,
                    device=device,
                )
                skipped_queries.extend(skipped)
            elif mode == "vlm_qa":
                query_results, vid_data = _evaluate_video_vlm_qa(
                    cfg, video_path, samples, tal,
                    chunk_model=chunk_model,
                    chunk_processor=chunk_processor,
                    emb_backend=emb_backend,
                    device=device,
                )
            elif mode == "gebd":
                video_result, vid_data = _evaluate_video_gebd(
                    cfg, video_path, samples,
                    chunk_model, chunk_processor, emb_backend, device,
                )

            per_video_data[video_path] = vid_data
            if mode == "gebd":
                per_query_results.append(video_result)
            elif mode == "vlm_qa":
                for qr in query_results:
                    per_query_results.append(qr)
            else:
                for qr in query_results:
                    all_predictions.append(qr["predictions"])
                    all_gt_windows.append(qr["gt_windows"])
                    per_query_results.append(qr)

        except WindowCacheMissError as exc:
            log.warning("Skipping video (window cache miss): %s — %s", video_path, exc)
            failed_videos.append({
                "video_path": video_path,
                "num_queries": len(samples),
                "error": f"WindowCacheMissError: {exc}",
                "skipped_reason": "window_cache_miss",
            })
            continue
        except Exception:
            log.error("Failed on video %s:\n%s", video_path, traceback.format_exc())
            failed_videos.append({
                "video_path": video_path,
                "num_queries": len(samples),
                "error": traceback.format_exc(),
            })
            continue

        vid_elapsed = time.time() - vid_start
        video_times.append(vid_elapsed)
        avg_time = sum(video_times) / len(video_times)
        remaining = total_videos - (vid_idx + 1)
        eta_sec = avg_time * remaining
        eta_min, eta_s = divmod(int(eta_sec), 60)
        pbar.set_postfix_str(
            f"avg {avg_time:.1f}s/vid | ETA {eta_min}m{eta_s:02d}s"
        )

        if (vid_idx + 1) % checkpoint_interval == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            run_dir_cp = os.path.dirname(checkpoint_path)
            _write_run_config_yaml(run_dir_cp, cfg)
            save_run_issues(run_dir_cp, failed_videos, skipped_queries)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "videos_processed": vid_idx + 1,
                    "queries_evaluated": len(per_query_results),
                    "failed_videos": len(failed_videos),
                    "per_query_results": per_query_results,
                }, f)
            log.info("Checkpoint saved (%d videos done)", vid_idx + 1)


# ---------------------------------------------------------------------------
# Concurrent VLM evaluation
# ---------------------------------------------------------------------------

def _collect_vlm_futures(
    futures, pending, mode,
    per_query_results, all_predictions, all_gt_windows,
    skipped_queries, failed_videos, pbar,
):
    """Collect completed VLM query futures and append results.

    For chunk-selection modes, ``all_predictions[i]``, ``all_gt_windows[i]``,
    and ``per_query_results[i]`` refer to the same query for all *i* at any
    time (append order follows completion order, not dataset order).
    """
    for future in futures:
        meta = pending.pop(future)
        try:
            if mode == "vlm_qa":
                qr = future.result()
                per_query_results.append(qr)
            else:
                qr, skipped_entry = future.result()
                per_query_results.append(qr)
                all_predictions.append(qr["predictions"])
                all_gt_windows.append(qr["gt_windows"])
                if skipped_entry is not None:
                    skipped_queries.append(skipped_entry)
        except Exception:
            log.error(
                "VLM query failed for %s:\n%s",
                meta.get("video_path", "unknown"),
                traceback.format_exc(),
            )
            failed_videos.append({
                "video_path": meta.get("video_path", "unknown"),
                "num_queries": 1,
                "error": traceback.format_exc(),
            })
        pbar.update(1)


def _run_vlm_concurrent(
    cfg, mode, video_to_samples, tal,
    chunk_model, chunk_processor, emb_backend, device,
    vlm_concurrency,
    per_query_results, all_predictions, all_gt_windows,
    per_video_data, failed_videos, skipped_queries,
    checkpoint_interval, checkpoint_path,
):
    """Concurrent VLM evaluation: GPU prep sequential, API calls in thread pool."""
    total_queries = sum(len(s) for s in video_to_samples.values())
    total_videos = len(video_to_samples)
    log.info(
        "VLM concurrent mode: %d workers, %d videos, %d queries",
        vlm_concurrency, total_videos, total_queries,
    )

    preselection = cfg.evaluation.get("chunk_preselection", "none")
    max_pred_windows = cfg.dataset.get("max_pred_windows", None)
    queries_at_last_checkpoint = len(per_query_results)
    pending = {}
    pbar = tqdm(total=total_queries, desc="Queries", unit="q")

    with ThreadPoolExecutor(max_workers=vlm_concurrency) as pool:
        for video_path, samples in video_to_samples.items():
            try:
                video_ctx = _prepare_vlm_video_context(
                    cfg, video_path, tal,
                    chunk_model=chunk_model,
                    chunk_processor=chunk_processor,
                    emb_backend=emb_backend,
                    device=device,
                )
                per_video_data[video_path] = video_ctx["vid_data"]

                iterative_config = OmegaConf.to_container(
                    cfg.temporal_abstraction.get("iterative", {}), resolve=True,
                ) if cfg.temporal_abstraction.get("iterative") else {}
                use_iterative = bool(iterative_config.get("n_queries"))

                for sample in samples:
                    while len(pending) >= vlm_concurrency:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        _collect_vlm_futures(
                            done, pending, mode,
                            per_query_results, all_predictions, all_gt_windows,
                            skipped_queries, failed_videos, pbar,
                        )

                    if use_iterative and mode == "vlm_qa":
                        future = pool.submit(
                            _run_vlm_iterative_qa_query,
                            tal, video_path, video_ctx["fps"], sample,
                            video_ctx["chunking_result"], emb_backend,
                            video_ctx["vid_data"],
                            len(video_ctx["chunks"]), iterative_config,
                        )
                    elif mode == "vlm_qa":
                        prompt_ctx = _prepare_vlm_prompt_context(
                            sample,
                            video_ctx["chunks"],
                            video_ctx["fps"],
                            video_ctx["chunk_source"],
                            video_ctx["allowed_chunks"],
                            video_ctx["use_best_window"],
                            chunking_result=video_ctx["chunking_result"],
                            emb_backend=emb_backend,
                            sample_interval=video_ctx["sample_interval"],
                            frames_per_chunk=tal.frames_per_chunk,
                            frame_method=tal.frame_method,
                            best_window_strategy=tal.best_window_strategy,
                            preselection=preselection,
                            query_merge_cfg=cfg.postprocessing.query_merge,
                            use_window_selection=video_ctx.get("use_window_selection", False),
                            window_method_kwargs=tal.window_method_kwargs,
                            n_frames=tal.n_frames,
                        )
                        future = pool.submit(
                            _run_vlm_qa_query,
                            tal, video_path, video_ctx["fps"], sample,
                            prompt_ctx, video_ctx["vid_data"],
                            len(video_ctx["chunks"]),
                        )
                    else:
                        prompt_ctx = _prepare_vlm_prompt_context(
                            sample,
                            video_ctx["chunks"],
                            video_ctx["fps"],
                            video_ctx["chunk_source"],
                            video_ctx["allowed_chunks"],
                            video_ctx["use_best_window"],
                            chunking_result=video_ctx["chunking_result"],
                            emb_backend=emb_backend,
                            sample_interval=video_ctx["sample_interval"],
                            frames_per_chunk=tal.frames_per_chunk,
                            frame_method=tal.frame_method,
                            best_window_strategy=tal.best_window_strategy,
                            preselection=preselection,
                            query_merge_cfg=cfg.postprocessing.query_merge,
                            use_window_selection=video_ctx.get("use_window_selection", False),
                            window_method_kwargs=tal.window_method_kwargs,
                            n_frames=tal.n_frames,
                        )
                        prompt_ctx["num_chunks_total"] = len(video_ctx["chunks"])
                        future = pool.submit(
                            _run_vlm_cs_query,
                            tal, video_path, video_ctx["fps"], sample,
                            prompt_ctx, max_pred_windows,
                        )
                    pending[future] = {
                        "video_path": video_path,
                        "sample_id": sample.sample_id,
                    }

            except WindowCacheMissError as exc:
                log.warning(
                    "Skipping video (window cache miss): %s — %s",
                    video_path, exc,
                )
                failed_videos.append({
                    "video_path": video_path,
                    "num_queries": len(samples),
                    "error": f"WindowCacheMissError: {exc}",
                    "skipped_reason": "window_cache_miss",
                })
                pbar.update(len(samples))
                continue
            except Exception:
                log.error(
                    "Failed preparing video %s:\n%s",
                    video_path, traceback.format_exc(),
                )
                failed_videos.append({
                    "video_path": video_path,
                    "num_queries": len(samples),
                    "error": traceback.format_exc(),
                })
                continue

            new_completed = len(per_query_results) - queries_at_last_checkpoint
            if new_completed >= checkpoint_interval * 3:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                run_dir_cp = os.path.dirname(checkpoint_path)
                _write_run_config_yaml(run_dir_cp, cfg)
                save_run_issues(run_dir_cp, failed_videos, skipped_queries)
                with open(checkpoint_path, "w") as f:
                    json.dump({
                        "videos_processed": len(per_video_data),
                        "queries_evaluated": len(per_query_results),
                        "failed_videos": len(failed_videos),
                        "per_query_results": per_query_results,
                    }, f)
                queries_at_last_checkpoint = len(per_query_results)
                log.info("Checkpoint saved (%d queries done)", len(per_query_results))

        if pending:
            done, _ = wait(pending)
            _collect_vlm_futures(
                done, pending, mode,
                per_query_results, all_predictions, all_gt_windows,
                skipped_queries, failed_videos, pbar,
            )

    pbar.close()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    start_time = time.time()
    logging.basicConfig(level=logging.INFO)

    resume_from = cfg.get("resume_from", None)
    resume_checkpoint = None
    if resume_from is not None:
        saved_cfg, resume_checkpoint = _load_resume_checkpoint(resume_from)
        cfg = saved_cfg
        log.info("Resuming from %s (%d videos, %d queries already done)",
                 resume_from, resume_checkpoint["videos_processed"],
                 len(resume_checkpoint["per_query_results"]))

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    mode = cfg.evaluation.mode
    if cfg.dataset.name == "lovr" and mode != "lovr_retrieval":
        raise ValueError(
            "dataset=lovr requires evaluation=lovr_retrieval."
        )
    if cfg.dataset.name == "kinetics_gebd" and mode != "gebd":
        raise ValueError(
            "dataset=kinetics_gebd requires evaluation=gebd."
        )
    _validate_vlm_config(cfg)

    # 1. Dataset -----------------------------------------------------------
    dataset = build_dataset(cfg)
    dataset.load()
    selected_subset_ids = None
    redo_failed_from = cfg.get("redo_failed_from", None)
    if redo_failed_from is not None:
        failed_video_paths = _load_failed_video_paths(redo_failed_from)
        dataset.filter_video_paths(failed_video_paths)
        log.info(
            "Filtered to %d samples across %d failed videos from %s",
            len(dataset),
            len(set(failed_video_paths)),
            redo_failed_from,
        )

    if cfg.subset_ids is not None:
        ids = list(cfg.subset_ids)
        dataset.filter_subset(ids)
        selected_subset_ids = [str(x) for x in ids]
        log.info("Filtered to %d samples (subset)", len(dataset))

    if cfg.sample_size is not None:
        sample_size = int(cfg.sample_size)
        assert sample_size > 0, "sample_size must be > 0"

        pool_ids = [s.sample_id for s in dataset]

        if sample_size > len(pool_ids):
            log.warning(
                "sample_size=%d is larger than available samples (%d). "
                "Using all available samples.",
                sample_size, len(pool_ids)
            )
            sample_size = len(pool_ids)

        rng = np.random.default_rng(int(cfg.sample_seed))
        sampled_ids = rng.choice(pool_ids, size=sample_size, replace=False).tolist()
        dataset.filter_subset(sampled_ids)
        selected_subset_ids = [str(x) for x in sampled_ids]
        log.info(
            "Sampled %d examples using seed=%d",
            len(selected_subset_ids), int(cfg.sample_seed)
        )

    log.info("Dataset: %s  samples: %d", cfg.dataset.name, len(dataset))

    # 2. Models (mode-specific) --------------------------------------------
    ret_backend = None
    chunk_model = chunk_processor = emb_backend = None
    tal = None
    score_method = None
    batch_size = 8

    if mode == "embedding":
        score_method = cfg.evaluation.chunk_score_method
        allowed_methods = (
            "recompute", "mean", "max_sim",
            "gem", "coherence_mean", "lse",
        )
        if score_method not in allowed_methods:
            raise ValueError(
                f"Unknown chunk_score_method: {score_method}. "
                f"Must be one of: {', '.join(allowed_methods)}"
            )
        if score_method != "recompute":
            emb_backend_type = cfg.chunking.get("embedding_backend", "internvideo2")
            if emb_backend_type != cfg.retrieval.backend:
                raise ValueError(
                    f"chunk_score_method={score_method} requires the chunking "
                    f"embedding backend ({emb_backend_type}) to match the "
                    f"retrieval backend ({cfg.retrieval.backend}), so embeddings "
                    f"are in the same space. "
                    f"Set chunking.embedding_backend={cfg.retrieval.backend}"
                )

        log.info("Loading retrieval backend (%s) ...", cfg.retrieval.backend)
        ret_backend = _load_retrieval_backend(cfg, device)
        log.info("Loading chunking model (%s) ...", cfg.chunking.type)
        chunk_model, chunk_processor, emb_backend = _load_chunking_models(
            cfg, device, ret_backend=ret_backend,
        )
        batch_size = int(cfg.evaluation.batch_size)
        log.info("Chunk score method: %s", score_method)
        log.info("Batch size: %d", batch_size)

    elif mode == "lovr_retrieval":
        log.info("Loading retrieval backend (%s) ...", cfg.retrieval.backend)
        ret_backend = _load_retrieval_backend(cfg, device)
        batch_size = int(cfg.evaluation.batch_size)
        log.info("LoVR retrieval batch size: %d", batch_size)

    elif mode == "vlm_chunk_selection":
        vlm_cfg = cfg.get("vlm")
        if vlm_cfg is None:
            raise ValueError(
                "evaluation=vlm_chunk_selection requires a vlm config "
                "(e.g. vlm=qwen3_vl_8b)"
            )
        log.info("Loading VLM client (%s) ...", vlm_cfg.model)
        vlm_client = VLMClient.from_config(vlm_cfg)
        tal = TemporalAbstractionLayer.from_config(
            vlm_client, cfg.temporal_abstraction,
        )
        allowed_vlm_chunks = compute_allowed_chunks(
            tal.max_chunks,
            tal.frames_per_chunk,
            tal.vlm.max_images_per_request,
        )
        chunk_source = cfg.evaluation.chunk_source
        if chunk_source == "stable_chunks":
            log.info(
                "Loading chunking model (%s) for stable_chunks ...",
                cfg.chunking.type,
            )
            chunk_model, chunk_processor, emb_backend = _load_chunking_models(
                cfg, device,
            )
        log.info("VLM chunk source: %s", chunk_source)
        log.info(
            "VLM prompt budget: %d chunks (%d frames/chunk, max_images=%s)",
            allowed_vlm_chunks,
            tal.frames_per_chunk,
            tal.vlm.max_images_per_request,
        )
        preselection = cfg.evaluation.get("chunk_preselection", "none")
        if preselection != "none" and chunk_source == "stable_chunks":
            log.info("VLM chunk preselection: %s", preselection)

    elif mode == "vlm_qa":
        vlm_cfg = cfg.get("vlm")
        if vlm_cfg is None:
            raise ValueError(
                "evaluation=vlm_qa requires a vlm config "
                "(e.g. vlm=qwen3_vl_8b)"
            )
        log.info("Loading VLM client (%s) ...", vlm_cfg.model)
        vlm_client = VLMClient.from_config(vlm_cfg)
        tal = TemporalAbstractionLayer.from_config(
            vlm_client, cfg.temporal_abstraction,
        )
        allowed_vlm_chunks = compute_allowed_chunks(
            tal.max_chunks,
            tal.frames_per_chunk,
            tal.vlm.max_images_per_request,
        )
        chunk_source = cfg.evaluation.chunk_source
        if chunk_source == "stable_chunks":
            embedding_device = cfg.evaluation.get("embedding_device") or None
            log.info(
                "Loading chunking model (%s) for stable_chunks (embedding_device=%s) ...",
                cfg.chunking.type, embedding_device or device,
            )
            chunk_model, chunk_processor, emb_backend = _load_chunking_models(
                cfg, device, embedding_device=embedding_device,
            )
            if cfg.evaluation.get("require_window_cache", False):
                log.info(
                    "require_window_cache=True: videos with cache misses will be skipped"
                )
        log.info("VLM QA: %s", vlm_cfg.model)
        log.info("VLM chunk source: %s", chunk_source)
        log.info(
            "VLM prompt budget: %d chunks (%d frames/chunk, max_images=%s)",
            allowed_vlm_chunks,
            tal.frames_per_chunk,
            tal.vlm.max_images_per_request,
        )
        preselection = cfg.evaluation.get("chunk_preselection", "none")
        if preselection != "none" and chunk_source == "stable_chunks":
            log.info("VLM chunk preselection: %s", preselection)
        if cfg.postprocessing.query_merge.enabled and chunk_source == "stable_chunks":
            log.info("VLM QA query merging: enabled")

    elif mode == "gebd":
        log.info("Loading chunking model (%s) for GEBD ...", cfg.chunking.type)
        chunk_model, chunk_processor, emb_backend = _load_chunking_models(
            cfg, device,
        )
        batch_size = int(cfg.evaluation.batch_size)
        log.info("GEBD batch size: %d", batch_size)

    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    # 3. Group samples by video --------------------------------------------
    if mode == "lovr_retrieval":
        video_to_samples = {}
    else:
        video_to_samples = defaultdict(list)
        for sample in dataset:
            video_to_samples[sample.video_path].append(sample)
        log.info("Unique videos: %d", len(video_to_samples))

    # 4. Evaluate ----------------------------------------------------------
    all_predictions = []
    all_gt_windows = []
    per_query_results = []
    per_video_data = {}
    failed_videos = []
    skipped_queries = []

    if resume_checkpoint is not None:
        if mode == "lovr_retrieval":
            raise ValueError("resume_from is not supported for lovr_retrieval mode")
        per_query_results = resume_checkpoint["per_query_results"]
        if mode not in ("vlm_qa", "gebd"):
            for qr in per_query_results:
                all_predictions.append(qr["predictions"])
                all_gt_windows.append(qr["gt_windows"])
        done_ids = {qr["sample_id"] for qr in per_query_results}
        video_to_samples = {
            vp: [s for s in samples if s.sample_id not in done_ids]
            for vp, samples in video_to_samples.items()
            if any(s.sample_id not in done_ids for s in samples)
        }
        run_dir = resume_from
        log.info(
            "Resume: %d queries in checkpoint, %d videos still have pending samples",
            len(per_query_results),
            len(video_to_samples),
        )
    else:
        run_dir = make_run_dir(
            cfg.output.dir,
            task_type_from_mode(mode),
            cfg.dataset.name,
            cfg.output.notes,
        )

    checkpoint_interval = 10
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")
    if mode != "lovr_retrieval":
        _write_run_config_yaml(run_dir, cfg)

    try:
        if mode == "lovr_retrieval":
            try:
                (
                    lovr_metrics,
                    per_query_results,
                    lovr_summary,
                    lovr_failed_items,
                    lovr_skipped_queries,
                ) = _evaluate_lovr_retrieval(cfg, dataset, ret_backend, batch_size)
                per_video_data = {"lovr_summary": lovr_summary}
                if lovr_failed_items["full_videos"] or lovr_failed_items["clips"]:
                    failed_videos = [
                        {
                            "video_path": entry["video_path"],
                            "num_queries": 0,
                            "error": entry["error"],
                        }
                        for entry in lovr_failed_items["full_videos"] + lovr_failed_items["clips"]
                    ]
                skipped_queries.extend(lovr_skipped_queries)
            except Exception:
                log.error("LoVR evaluation failed:\n%s", traceback.format_exc())
                raise
        else:
            vlm_concurrency = int(cfg.evaluation.get("vlm_concurrency", 1))
            use_concurrent = (
                vlm_concurrency > 1
                and mode in ("vlm_qa", "vlm_chunk_selection")
            )
    
            if use_concurrent:
                _run_vlm_concurrent(
                    cfg, mode, video_to_samples, tal,
                    chunk_model, chunk_processor, emb_backend, device,
                    vlm_concurrency,
                    per_query_results, all_predictions, all_gt_windows,
                    per_video_data, failed_videos, skipped_queries,
                    checkpoint_interval, checkpoint_path,
                )
            else:
                _run_sequential(
                    cfg, mode, video_to_samples,
                    ret_backend, chunk_model, chunk_processor, emb_backend,
                    device, tal, score_method, batch_size,
                    per_query_results, all_predictions, all_gt_windows,
                    per_video_data, failed_videos, skipped_queries,
                    checkpoint_interval, checkpoint_path,
                )
    
        evaluated = len(per_query_results)
        eval_unit = "videos" if mode == "gebd" else "queries"
        log.info(
            "Evaluated %d %s (%d videos failed, %d queries skipped)",
            evaluated, eval_unit, len(failed_videos), len(skipped_queries),
        )
        if evaluated == 0:
            log.error("No %s were evaluated — all videos failed.", eval_unit)
            save_run_issues(run_dir, failed_videos, skipped_queries)
            if failed_videos:
                with open(os.path.join(run_dir, "failed_videos.json"), "w") as f:
                    json.dump(failed_videos, f, indent=2)
            if skipped_queries:
                with open(os.path.join(run_dir, "skipped_queries.json"), "w") as f:
                    json.dump(skipped_queries, f, indent=2)
            return
    
        # 5. Metrics -----------------------------------------------------------
        mcfg = cfg.dataset.metrics
        metrics = {}
    
        metric_type = mcfg.get("type")
        if metric_type == "activitynet_qa_paper":
            qa_metrics = compute_activitynet_qa_metrics(per_query_results)
            for name, value in qa_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "lovr_pass_at_k":
            for name, value in lovr_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "longvideobench_mcq":
            qa_metrics = compute_longvideobench_metrics(per_query_results)
            for name, value in qa_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "videomme_mcq":
            qa_metrics = compute_videomme_mcq_metrics(per_query_results)
            for name, value in qa_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "lvbench_mcq":
            qa_metrics = compute_lvbench_metrics(per_query_results)
            for name, value in qa_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "mlvu_mcq":
            qa_metrics = compute_mlvu_metrics(per_query_results)
            for name, value in qa_metrics.items():
                metrics[name] = round(value, 6)
        elif metric_type == "gebd_f1":
            gebd_thresholds = list(mcfg.get("thresholds", [0.05]))
            gebd_metrics = compute_gebd_f1_metrics(per_query_results, thresholds=gebd_thresholds)
            for name, value in gebd_metrics.items():
                metrics[name] = round(value, 6)
        else:
            for k in mcfg.recall_at_k:
                for iou in mcfg.iou_thresholds:
                    val = compute_recall_at_k(all_predictions, all_gt_windows, k, iou)
                    metrics[f"R@{k}_IoU={iou}"] = round(val, 6)
    
            if mcfg.compute_map:
                for iou in mcfg.map_iou_thresholds:
                    val = compute_map_at_iou(all_predictions, all_gt_windows, iou)
                    metrics[f"mAP@{iou}"] = round(val, 6)
    
            if mcfg.get("compute_miou", False):
                val = compute_mean_iou(all_predictions, all_gt_windows)
                metrics["mIoU"] = round(val, 6)
    
            if cfg.dataset.name == "qvhighlights" and mcfg.get("compute_highlight", False):
                hl_thresholds = list(mcfg.get("highlight_min_scores", [2, 3, 4]))
                clip_length = float(mcfg.get("highlight_clip_length", 2.0))
                hl_metrics = compute_qvhighlight_highlight_metrics(
                    all_predictions,
                    [r["metadata"] for r in per_query_results],
                    clip_length=clip_length,
                    min_score_thresholds=hl_thresholds,
                )
                for name, value in hl_metrics.items():
                    metrics[name] = round(value, 6)
    
        # 6. Save --------------------------------------------------------------
        duration_sec = time.time() - start_time
        metrics["duration_sec"] = round(duration_sec, 2)
        metrics["failed_videos"] = len(failed_videos)
        metrics["skipped_queries"] = len(skipped_queries)
        log.info("Run duration: %.2f seconds", duration_sec)
    
        save_run(
            run_dir, cfg, metrics, per_query_results,
            subset_ids=selected_subset_ids,
            per_video_data=per_video_data,
        )
        save_run_issues(run_dir, failed_videos, skipped_queries)
        if failed_videos:
            with open(os.path.join(run_dir, "failed_videos.json"), "w") as f:
                json.dump(failed_videos, f, indent=2)
        if skipped_queries:
            with open(os.path.join(run_dir, "skipped_queries.json"), "w") as f:
                json.dump(skipped_queries, f, indent=2)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        log.info("Results saved to %s", run_dir)
    
    except Exception:
        if run_dir:
            try:
                _exc = sys.exc_info()
                save_run_crash(run_dir, *_exc)
                save_run_issues(run_dir, failed_videos, skipped_queries)
            except OSError as _issues_exc:
                log.warning("Could not write run_issues.json / run_crash.json: %s", _issues_exc)
        raise

    # 7. Plots -------------------------------------------------------------
    if mode == "lovr_retrieval":
        # TODO: Add dedicated gallery-retrieval visualizations for LoVR. The
        # current plotters target temporal grounding and QA tasks only.
        plots_dir = None
        log.info("Skipping plots for LoVR retrieval runs")
    elif not cfg.output.get("generate_plots", True):
        plots_dir = None
        log.info("Skipping plots (output.generate_plots=false)")
    else:
        plots_dir = generate_all_plots(
            run_dir,
            generate_videos=cfg.output.get("generate_videos", False),
        )
        log.info("Plots saved to %s", plots_dir)

    # 8. Print summary -----------------------------------------------------
    print("\n===== Evaluation Results =====")
    unit = "videos" if mode == "gebd" else "queries"
    print(f"Dataset:   {cfg.dataset.name} ({evaluated} {unit})")
    print(f"Mode:      {mode}")
    if mode == "embedding":
        print(f"Chunking:  {cfg.chunking.type}")
        print(f"Retrieval: {cfg.retrieval.backend}")
    elif mode == "lovr_retrieval":
        print(f"Retrieval: {cfg.retrieval.backend}")
        print(f"Split:     {cfg.dataset.get('split', 'test')}")
    elif mode == "vlm_chunk_selection":
        print(f"VLM:       {cfg.get('vlm', {}).get('model', 'N/A')}")
        print(f"Chunks:    {cfg.evaluation.chunk_source}")
        presel = cfg.evaluation.get("chunk_preselection", "none")
        if presel != "none":
            print(f"Preselect: {presel}")
    elif mode == "vlm_qa":
        print(f"VLM:       {cfg.get('vlm', {}).get('model', 'N/A')}")
        print(f"Chunks:    {cfg.evaluation.chunk_source}")
    elif mode == "gebd":
        print(f"Chunking:  {cfg.chunking.type}")
        print(f"Split:     {cfg.dataset.get('split', 'val')}")
    if cfg.output.notes:
        print(f"Notes:     {cfg.output.notes}")
    if failed_videos or skipped_queries:
        print(f"Issues:    see run_issues.json ({len(failed_videos)} failed videos, {len(skipped_queries)} skipped queries)")
    if failed_videos:
        print(f"Failed:    {len(failed_videos)} videos (see failed_videos.json)")
    if skipped_queries:
        print(f"Skipped:   {len(skipped_queries)} queries (see skipped_queries.json)")
    print()
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
