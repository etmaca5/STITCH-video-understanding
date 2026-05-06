"""Generate evaluation plots from saved run results.

Can be used standalone:
    python src/plots.py results/<run_dir>
    python src/plots.py results/<run_dir> --videos
    python src/plots.py results/<run_dir> --videos-only

Or called programmatically after evaluation.
"""

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

for _calibri in (
    "/mnt/c/Windows/Fonts/calibri.ttf",
    "/mnt/c/Windows/Fonts/calibrib.ttf",
    "/mnt/c/Windows/Fonts/calibrii.ttf",
):
    if os.path.isfile(_calibri):
        try:
            _fm.fontManager.addfont(_calibri)
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results import load_run
from temporal_abstraction import compute_allowed_chunks, select_frames

def _infer_duration(query_result, video_data):
    """Get video duration from video_data, metadata, or data extents."""
    if video_data and "duration" in video_data:
        return video_data["duration"]
    if query_result["metadata"].get("duration"):
        return query_result["metadata"]["duration"]
    # Fall back to the maximum time seen in GT / predictions
    max_t = 0
    for gt in query_result.get("gt_windows", []):
        max_t = max(max_t, gt[1])
    for p in query_result.get("predictions", []):
        if isinstance(p, dict):
            max_t = max(max_t, p.get("end", 0))
    return max_t * 1.05 if max_t > 0 else 100


def _truncate_text(text, max_len=100):
    """Return a shortened single-line label."""
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _resize_frame(frame, target_height):
    """Resize an RGB frame to a fixed height while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = target_height / max(height, 1)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height))


def _build_chunk_card(frames, target_height=180, gap=6):
    """Combine one or more frames for a single chunk into one RGB image."""
    resized = [_resize_frame(frame, target_height) for frame in frames]
    if len(resized) == 1:
        return resized[0]

    gap_strip = np.full((target_height, gap, 3), 255, dtype=np.uint8)
    pieces = [resized[0]]
    for frame in resized[1:]:
        pieces.append(gap_strip)
        pieces.append(frame)
    return np.concatenate(pieces, axis=1)


def _make_uniform_chunks(total_frames, num_chunks):
    """Split a video into equal-length frame ranges."""
    chunk_size = total_frames / num_chunks
    return [
        (int(i * chunk_size), int(min((i + 1) * chunk_size, total_frames)))
        for i in range(num_chunks)
    ]


def _compute_allowed_vlm_chunks(config):
    """Return the number of chunks the VLM prompt could show."""
    frames_per_chunk = int(
        config["temporal_abstraction"]["frame_selection"]["frames_per_chunk"]
    )
    max_chunks = int(config["temporal_abstraction"]["prompt"]["max_chunks_per_prompt"])
    max_images = config.get("vlm", {}).get("max_images_per_request")
    try:
        return compute_allowed_chunks(max_chunks, frames_per_chunk, max_images)
    except ValueError:
        return 0


def _extract_stable_chunks(video_data):
    """Recover the final stable chunk list from saved per-video stages."""
    if not video_data:
        return []
    stages = video_data.get("stages", {})
    for stage_name in ("after_merge", "after_transitions", "initial"):
        stage_data = stages.get(stage_name)
        if not stage_data:
            continue
        stable = [
            tuple(lc["chunk"])
            for lc in stage_data
            if lc.get("label") == "stable"
        ]
        if stable:
            return stable
    return []


def _infer_selected_chunk_index(query_result, used_chunks, fps):
    """Infer the chosen prompt chunk from the saved prediction window."""
    top1 = query_result.get("top1_pred")
    if top1 is None:
        preds = query_result.get("predictions", [])
        top1 = preds[0] if preds else None
    if top1 is None:
        return None

    for idx, (start_frame, end_frame) in enumerate(used_chunks):
        start_sec = start_frame / fps
        end_sec = end_frame / fps
        if (
            abs(start_sec - top1["start"]) < 1e-6
            and abs(end_sec - top1["end"]) < 1e-6
        ):
            return idx
    return None


def _get_vlm_prompt_metadata(query_result, video_data, config):
    """Return saved or reconstructed prompt metadata for a VLM query."""
    stored = query_result.get("vlm_prompt_metadata")
    if stored is not None:
        return stored

    if config is None:
        return None
    try:
        if config["evaluation"]["mode"] != "vlm_chunk_selection":
            return None
    except (KeyError, TypeError):
        return None

    try:
        chunk_source = config["evaluation"]["chunk_source"]
        frame_method = config["temporal_abstraction"]["frame_selection"]["method"]
        frames_per_chunk = int(
            config["temporal_abstraction"]["frame_selection"]["frames_per_chunk"]
        )
    except (KeyError, TypeError, ValueError):
        return None

    if frame_method == "best_window":
        return None

    allowed_chunks = _compute_allowed_vlm_chunks(config)

    if chunk_source == "uniform":
        total_frames = int(video_data.get("total_frames", 0)) if video_data else 0
        num_uniform_chunks = int(config["evaluation"].get("num_uniform_chunks", 4))
        if total_frames <= 0:
            return None
        query_chunks = _make_uniform_chunks(total_frames, num_uniform_chunks)
        original_chunk_indices = None
    elif chunk_source == "stable_chunks":
        base_chunks = _extract_stable_chunks(video_data)
        if not base_chunks:
            return None
        selector_meta = query_result.get("selector_metadata")
        prompt_indices = None
        if selector_meta and selector_meta.get("prompt_original_indices"):
            prompt_indices = [
                int(i) for i in selector_meta["prompt_original_indices"]
                if 0 <= int(i) < len(base_chunks)
            ]
        elif selector_meta and selector_meta.get("original_indices"):
            ranked_indices = [int(i) for i in selector_meta["original_indices"]]
            prompt_indices = sorted(
                i for i in ranked_indices[:allowed_chunks]
                if 0 <= i < len(base_chunks)
            )
        if prompt_indices is not None:
            query_chunks = [
                base_chunks[i] for i in prompt_indices
            ]
            original_chunk_indices = prompt_indices
        else:
            query_chunks = base_chunks
            original_chunk_indices = None
    else:
        return None

    used_chunks = query_chunks[:allowed_chunks]
    fps = video_data["fps"] if video_data else 30.0

    meta = {
        "chunk_source": chunk_source,
        "frame_method": frame_method,
        "frames_per_chunk": frames_per_chunk,
        "used_chunks": [[int(s), int(e)] for s, e in used_chunks],
        "selected_chunk_index": _infer_selected_chunk_index(
            query_result, used_chunks, fps,
        ),
    }
    if original_chunk_indices is not None:
        meta["original_chunk_indices"] = original_chunk_indices[: len(used_chunks)]
    return meta


def _get_effective_num_chunks(query_result, video_data, config):
    """Return the chunk count that the model actually saw for this query."""
    meta = _get_vlm_prompt_metadata(query_result, video_data, config)
    if meta is not None:
        used_chunks = meta.get("used_chunks", [])
        if used_chunks:
            return len(used_chunks)
    return query_result.get("num_chunks", 0)


def _plot_vlm_gallery_row(ax, query_result, video_data, config):
    """Draw the frames shown to the VLM for one query."""
    meta = _get_vlm_prompt_metadata(query_result, video_data, config)
    if not meta:
        ax.axis("off")
        ax.text(0.5, 0.5, "No VLM prompt metadata available",
                ha="center", va="center", fontsize=14)
        return

    used_chunks = [tuple(chunk) for chunk in meta.get("used_chunks", [])]
    if not used_chunks:
        ax.axis("off")
        ax.text(0.5, 0.5, "No chunks shown to the VLM",
                ha="center", va="center", fontsize=14)
        return

    fps = video_data["fps"] if video_data else 30.0
    selected_idx = meta.get("selected_chunk_index")
    frame_method = meta.get("frame_method", "middle")
    frames_per_chunk = int(meta.get("frames_per_chunk", 1))
    chunk_frame_selections = meta.get("chunk_frame_selections")
    query_text = _truncate_text(query_result["query"], 180)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_chunks = len(used_chunks)
    left_margin = 0.004
    right_margin = 0.004
    card_gap = 0.006
    usable_width = 1 - left_margin - right_margin - card_gap * (n_chunks - 1)
    card_width = usable_width / max(n_chunks, 1)
    y0 = 0.28
    height = 0.66

    for idx, chunk in enumerate(used_chunks):
        frame_kwargs = {"n": frames_per_chunk}
        if chunk_frame_selections is not None and idx < len(chunk_frame_selections):
            frame_kwargs["frame_indices"] = chunk_frame_selections[idx].get(
                "frame_indices"
            )
        frames = select_frames(
            query_result["video_path"],
            chunk,
            method=frame_method,
            **frame_kwargs,
        )
        card = _build_chunk_card(frames)
        x0 = left_margin + idx * (card_width + card_gap)
        card_ax = ax.inset_axes([x0, y0, card_width, height])
        card_ax.imshow(card)
        card_ax.set_xticks([])
        card_ax.set_yticks([])
        border_color = "#A23B72" if selected_idx == idx else "#4F5B66"
        border_width = 3.0 if selected_idx == idx else 1.2
        for spine in card_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)

        start_sec = chunk[0] / fps
        end_sec = chunk[1] / fps
        chunk_label = f"{start_sec:.1f}s - {end_sec:.1f}s"
        ax.text(
            x0 + card_width / 2, 0.19, chunk_label,
            transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="#333333",
        )

    ax.text(
        0.5, 0.045, f'"{query_text}"',
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=15, fontstyle="italic",
    )


def _plot_vlm_timeline_row(ax, query_result, video_data, config):
    """Draw the shown chunks, GT, and prediction context below the frame gallery."""
    meta = _get_vlm_prompt_metadata(query_result, video_data, config)
    if not meta:
        ax.axis("off")
        return

    used_chunks = [tuple(chunk) for chunk in meta.get("used_chunks", [])]
    if not used_chunks:
        ax.axis("off")
        return

    fps = video_data["fps"] if video_data else 30.0
    duration = _infer_duration(query_result, video_data)
    selected_idx = meta.get("selected_chunk_index")
    preds = query_result.get("predictions", [])
    top1 = preds[0] if preds else None

    row_specs = [("Prompt", 2.0), ("GT", 1.0), ("Pred", 0.0)]
    row_y = dict(row_specs)
    bar_height = 0.45

    for idx, chunk in enumerate(used_chunks):
        start_sec = chunk[0] / fps
        end_sec = chunk[1] / fps
        is_selected = selected_idx == idx
        color = "#A23B72" if is_selected else "#D9D9D9"
        edge = "#6b1e4a" if is_selected else "#7f7f7f"
        ax.barh(row_y["Prompt"], end_sec - start_sec, left=start_sec,
                height=bar_height, color=color, edgecolor=edge, linewidth=1.0)
        ax.text((start_sec + end_sec) / 2, row_y["Prompt"], str(idx + 1),
                ha="center", va="center", fontsize=11,
                color="white" if is_selected else "#333333",
                fontweight="bold")

    for gt in query_result.get("gt_windows", []):
        ax.barh(row_y["GT"], gt[1] - gt[0], left=gt[0],
                height=bar_height, color="#2E86AB", edgecolor="#1a5c7a",
                linewidth=1.0, alpha=0.65)

    if top1 is not None:
        ax.barh(row_y["Pred"], top1["end"] - top1["start"], left=top1["start"],
                height=bar_height, color="#A23B72", edgecolor="#6b1e4a",
                linewidth=1.0, alpha=0.9)

    for name, y in row_specs:
        ax.text(-duration * 0.01, y, name, ha="right", va="center",
                fontsize=11, fontweight="bold")

    ax.set_xlim(-duration * 0.02, duration * 1.02)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Time (seconds)", fontsize=11)


def plot_vlm_prompt_examples(
    query_results, per_video_data, out_path, config=None,
    title="VLM Prompt Examples",
):
    """Render VLM prompt examples showing frames and prompt-order chunks."""
    examples = [
        qr for qr in query_results
        if _get_vlm_prompt_metadata(
            qr,
            per_video_data.get(qr["video_path"]) if per_video_data else None,
            config,
        )
    ]
    if not examples:
        return

    n_examples = len(examples)
    fig = plt.figure(figsize=(24, n_examples * 4.6 + 0.8))
    gs = fig.add_gridspec(n_examples * 2, 1, height_ratios=[4.2, 1.0] * n_examples)

    for i, qr in enumerate(examples):
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        gallery_ax = fig.add_subplot(gs[i * 2, 0])
        timeline_ax = fig.add_subplot(gs[i * 2 + 1, 0])
        _plot_vlm_gallery_row(gallery_ax, qr, vdata, config)
        _plot_vlm_timeline_row(timeline_ax, qr, vdata, config)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.992], pad=0.35, h_pad=0.6)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared drawing helper
# ---------------------------------------------------------------------------

def _draw_example_row(ax, query_result, video_data, fps, duration, y_offset,
                      show_query=True):
    """Draw one example's GT + predicted chunks on the given axes.

    Returns the y_offset after drawing (for stacking).
    """
    gt_windows = query_result["gt_windows"]
    preds = query_result.get("predictions", [])
    if isinstance(preds, dict):
        preds = [preds]

    top1 = preds[0] if preds else None
    query = query_result["query"]

    bar_height = 0.35
    gap = 0.15
    y_gt = y_offset
    y_pred = y_offset - bar_height - gap

    if show_query:
        display_query = query if len(query) <= 80 else query[:77] + "..."
        ax.text(duration / 2, y_gt + bar_height + 0.15, f'"{display_query}"',
                ha="center", va="bottom", fontsize=8, fontstyle="italic")

    # Draw signal in background if available
    if video_data and "signal_times" in video_data:
        sig_t = np.array(video_data["signal_times"])
        sig_v = np.array(video_data["signal_values"])
        sig_min, sig_max = sig_v.min(), sig_v.max()
        if sig_max > sig_min:
            sig_norm = (sig_v - sig_min) / (sig_max - sig_min)
        else:
            sig_norm = np.zeros_like(sig_v)
        # Scale signal to fit in the row height
        row_bottom = y_pred - 0.05
        row_top = y_gt + bar_height + 0.05
        sig_scaled = row_bottom + sig_norm * (row_top - row_bottom)
        ax.fill_between(sig_t, row_bottom, sig_scaled, alpha=0.08,
                        color="gray")
        ax.plot(sig_t, sig_scaled, color="gray", alpha=0.15, linewidth=0.5)

    # Ground truth
    for gt in gt_windows:
        ax.barh(y_gt, gt[1] - gt[0], left=gt[0], height=bar_height,
                color="#2E86AB", edgecolor="#1a5c7a", linewidth=1.0,
                alpha=0.6)
    ax.text(-duration * 0.01, y_gt + bar_height / 2, "GT", ha="right",
            va="center", fontsize=8, fontweight="bold")

    # All chunks (non-top1 in light gray, top1 highlighted)
    for pred in preds:
        is_top1 = (pred is top1)
        color = "#A23B72" if is_top1 else "#B0B0B0"
        edgecolor = "#6b1e4a" if is_top1 else "#808080"
        ax.barh(y_pred, pred["end"] - pred["start"], left=pred["start"],
                height=bar_height, color=color, edgecolor=edgecolor,
                linewidth=1.0)
        score_str = f"{pred['score']:.3f}"
        chunk_mid = (pred["start"] + pred["end"]) / 2
        chunk_width = pred["end"] - pred["start"]
        if chunk_width > duration * 0.05:
            ax.text(chunk_mid, y_pred - 0.04, score_str,
                    ha="center", va="top", fontsize=6, color="black")

    ax.text(-duration * 0.01, y_pred + bar_height / 2, "Pred", ha="right",
            va="center", fontsize=8, fontweight="bold")

    return y_pred - gap


# ---------------------------------------------------------------------------
# Plot: Stacked examples (5 examples of Plot 2)
# ---------------------------------------------------------------------------

def plot_stacked_examples(query_results, per_video_data, out_path,
                          title="Predictions vs Ground Truth"):
    """Stack examples vertically showing GT and predicted chunks."""
    examples = list(query_results)
    if not examples:
        return

    row_height = 1.6
    fig_height = row_height * len(examples) + 0.8
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y = 0.5 + (len(examples) - 1) * row_height
    for qr in examples:
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        fps = vdata["fps"] if vdata else 30.0
        duration = _infer_duration(qr, vdata)
        _draw_example_row(ax, qr, vdata, fps, duration, y_offset=y)
        y -= row_height

    max_dur = max(
        (_infer_duration(qr, per_video_data.get(qr["video_path"]) if per_video_data else None)
         for qr in examples),
        default=100,
    )
    ax.set_xlim(-max_dur * 0.02, max_dur * 1.02)
    ax.set_ylim(y + row_height - 0.8, 0.5 + (len(examples) - 1) * row_height + 1.5)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor="#2E86AB", alpha=0.85, label="Ground Truth"),
        mpatches.Patch(facecolor="#A23B72", alpha=0.85, label="Top-1 Prediction"),
        mpatches.Patch(facecolor="#D9D9D9", alpha=0.5, label="Other Chunks"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative moment retrieval examples (frame strip + GT/Pred timeline)
# ---------------------------------------------------------------------------

_MR_LABEL_FONT = ["Calibri", "Carlito", "DejaVu Sans"]


def _mr_load_frame(cap, fps, t_sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _mr_resize(frame, h, w):
    return cv2.resize(frame, (w, h))


def _merge_consecutive_chunks(chunks, max_chunks):
    """Group adjacent chunks into ~equal-sized buckets so len <= max_chunks."""
    if len(chunks) <= max_chunks:
        return list(chunks)
    group_size = (len(chunks) + max_chunks - 1) // max_chunks
    merged = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        merged.append((group[0][0], group[-1][1]))
    return merged


def _mr_build_sample_times(vid_data, sec_per_frame, max_chunks):
    """Sampling times (seconds) for the moment-retrieval frame strip, or None."""
    if not vid_data or "stages" not in vid_data:
        return None
    chunks_frames = vid_data["stages"].get("after_merge", [])
    if not chunks_frames:
        return None
    fps = vid_data["fps"]
    duration = vid_data["duration"]
    raw_chunks = []
    for c in chunks_frames:
        s = c["chunk"][0] / fps
        e = c["chunk"][1] / fps
        if s >= duration:
            break
        raw_chunks.append((s, min(e, duration)))
    if not raw_chunks:
        return None
    chunks = _merge_consecutive_chunks(raw_chunks, max_chunks)
    all_sample_times = []
    for s, e in chunks:
        nframes = max(1, round((e - s) / sec_per_frame))
        all_sample_times.extend(np.linspace(s, e, nframes + 2)[1:-1].tolist())
    return all_sample_times


def _mr_subsample_times(all_sample_times, max_frames):
    """Evenly subsample *all_sample_times* to at most *max_frames* entries."""
    n = len(all_sample_times)
    if max_frames is None or n <= max_frames:
        return all_sample_times
    if max_frames == 1:
        return [all_sample_times[n // 2]]
    idx = [int(round(k * (n - 1) / (max_frames - 1))) for k in range(max_frames)]
    return [all_sample_times[i] for i in idx]


def _iou(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _query_iou(query_result):
    """Best IoU between top-1 prediction and any GT window."""
    top1 = query_result.get("top1_pred")
    if top1 is None:
        preds = query_result.get("predictions", [])
        top1 = preds[0] if preds else None
    if not top1:
        return 0.0
    return max(
        (_iou(top1["start"], top1["end"], gs, ge)
         for gs, ge in query_result.get("gt_windows", [])),
        default=0.0,
    )


def plot_moment_retrieval_example(
    query_result, vid_data, out_path,
    sec_per_frame=8.0,
    frame_height=140,
    max_chunks=12,
    chunk_gap_px=2,
    intra_gap_px=2,
    bar_height=16,
    pred_gap_px=18,
    timeline_gap_px=12,
    query_band_px=48,
    bottom_pad_px=16,
    query_fontsize=12,
    label_fontsize=9,
    gt_pred_fontsize=11,
    dpi=250,
    max_total_px=6000,
    max_frames=None,
):
    """One-row qualitative moment retrieval example.

    Top: full video as time-proportional chunks (merged if there are too
    many), with one or more equal-sized frames per chunk and white gaps as
    boundaries.
    Bottom: GT (blue) and top-1 prediction (pink) bars, vertically separated
    and aligned to the same time axis as the frame strip.

    If *max_frames* is set, sample times are evenly subsampled so the strip
    has at most that many frames (same nominal frame size and timeline width).
    """
    video_path = query_result["video_path"]
    if not os.path.isfile(video_path):
        return False

    all_sample_times = _mr_build_sample_times(
        vid_data, sec_per_frame, max_chunks,
    )
    if not all_sample_times:
        return False

    all_sample_times = _mr_subsample_times(all_sample_times, max_frames)

    fps = vid_data["fps"]
    duration = vid_data["duration"]

    cap = cv2.VideoCapture(video_path)
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    f0 = _mr_load_frame(cap, cap_fps, duration / 2)
    if f0 is None:
        cap.release()
        return False
    aspect = f0.shape[1] / f0.shape[0]

    nominal_w = max(1, int(round(frame_height * aspect)))

    n_frames = len(all_sample_times)
    # total canvas width driven by uniform frame layout
    total_w = n_frames * nominal_w + (n_frames - 1) * intra_gap_px
    total_w = max(2, min(total_w, max_total_px))

    if total_w == max_total_px:
        # scale frame size down to fit within the cap
        nominal_w = max(1, (total_w + intra_gap_px) // n_frames - intra_gap_px)
        frame_height = max(40, int(round(nominal_w / aspect)))
        total_w = n_frames * nominal_w + (n_frames - 1) * intra_gap_px

    # px_per_sec used only for GT/Pred bar alignment
    px_per_sec = total_w / duration

    canvas_top = np.full((frame_height, total_w, 3), 255, np.uint8)

    cur_x = 0
    for t in all_sample_times:
        f = _mr_load_frame(cap, cap_fps, t)
        if f is not None:
            canvas_top[:, cur_x:cur_x + nominal_w] = _mr_resize(f, frame_height, nominal_w)
        cur_x += nominal_w + intra_gap_px

    cap.release()

    timeline_height = bar_height * 2 + pred_gap_px
    canvas_bot = np.full((timeline_height, total_w, 3), 255, np.uint8)
    GT_COLOR = (46, 134, 171)
    PRED_COLOR = (162, 59, 114)
    for gt_s, gt_e in query_result.get("gt_windows", []):
        x0 = max(0, int(round(gt_s * px_per_sec)))
        x1 = min(total_w, int(round(gt_e * px_per_sec)))
        canvas_bot[:bar_height, x0:x1] = GT_COLOR
    top1 = query_result.get("top1_pred")
    if top1 is None:
        preds = query_result.get("predictions", [])
        top1 = preds[0] if preds else None
    if top1:
        x0 = max(0, int(round(top1["start"] * px_per_sec)))
        x1 = min(total_w, int(round(top1["end"] * px_per_sec)))
        canvas_bot[bar_height + pred_gap_px:, x0:x1] = PRED_COLOR

    total_h = (
        query_band_px + frame_height + timeline_gap_px
        + timeline_height + bottom_pad_px
    )
    fig_w = total_w / dpi
    fig_h = total_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(total_w / 2, query_band_px / 2, f'"{query_result.get("query", "")}"',
            ha="center", va="center",
            fontsize=query_fontsize,
            family=_MR_LABEL_FONT)

    ax.imshow(canvas_top, extent=[
        0, total_w, query_band_px + frame_height, query_band_px,
    ])

    bot_y = query_band_px + frame_height + timeline_gap_px
    ax.imshow(canvas_bot, extent=[
        0, total_w, bot_y + timeline_height, bot_y,
    ])

    ax.text(-10, bot_y + bar_height / 2, "GT",
            ha="right", va="center",
            fontsize=gt_pred_fontsize, color="#1a5c7a", fontweight="bold",
            family=_MR_LABEL_FONT)
    ax.text(-10, bot_y + bar_height + pred_gap_px + bar_height / 2, "Pred",
            ha="right", va="center",
            fontsize=gt_pred_fontsize, color="#6b1e4a", fontweight="bold",
            family=_MR_LABEL_FONT)

    tick_y = bot_y + timeline_height + 4
    ax.text(0, tick_y, "0s",
            ha="left", va="top",
            fontsize=label_fontsize, color="#444",
            family=_MR_LABEL_FONT)
    ax.text(total_w, tick_y, f"{duration:.1f}s",
            ha="right", va="top",
            fontsize=label_fontsize, color="#444",
            family=_MR_LABEL_FONT)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_moment_retrieval_examples(
    query_results, per_video_data, out_dir, n=5, seed=0,
    iou_min=None, iou_max=None, file_prefix="example",
    match_frame_count=False,
    sec_per_frame=8.0,
    max_chunks=12,
):
    """Pick *n* random queries with renderable videos and write one PNG each.

    Optionally restrict to queries whose top-1 IoU lies in [iou_min, iou_max].

    If *match_frame_count* is True, each PNG uses the same number of frames:
    the minimum count among the chosen examples (even subsampling, no rescaling).
    """
    renderable = [
        qr for qr in query_results
        if os.path.isfile(qr.get("video_path", ""))
        and per_video_data
        and per_video_data.get(qr.get("video_path"), {}).get("stages")
    ]
    if iou_min is not None or iou_max is not None:
        lo = -1.0 if iou_min is None else iou_min
        hi = 2.0 if iou_max is None else iou_max
        renderable = [qr for qr in renderable if lo <= _query_iou(qr) <= hi]
    if not renderable:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(renderable, min(n, len(renderable)))

    max_frames = None
    if match_frame_count:
        counts = []
        for qr in chosen:
            vd = per_video_data.get(qr["video_path"], {})
            times = _mr_build_sample_times(vd, sec_per_frame, max_chunks)
            if times:
                counts.append(len(times))
        if not counts:
            return []
        max_frames = min(counts)

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, qr in enumerate(chosen, start=1):
        vid_data = per_video_data.get(qr["video_path"], {})
        out_path = os.path.join(out_dir, f"{file_prefix}_{i}.png")
        if plot_moment_retrieval_example(
            qr, vid_data, out_path,
            sec_per_frame=sec_per_frame,
            max_chunks=max_chunks,
            max_frames=max_frames,
        ):
            written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Plot 4: Chunking stages (initial → transitions → merged)
# ---------------------------------------------------------------------------

def plot_chunking_stages(query_results, per_video_data, out_path):
    """Show chunking pipeline stages for the provided examples.

    Each video gets its own subplot with an independent x-axis scale.
    Deduplicates by video path so each video appears only once.
    """
    if not per_video_data:
        return

    seen_videos = set()
    examples = []
    for qr in query_results:
        vpath = qr["video_path"]
        if vpath in seen_videos:
            continue
        vdata = per_video_data.get(vpath)
        if vdata and "stages" in vdata:
            seen_videos.add(vpath)
            examples.append((qr, vdata))

    if not examples:
        return

    stage_names = ["initial", "after_transitions", "after_merge"]
    stage_labels = ["Initial Chunks", "After Transition Detection", "After Merging"]
    stage_colors = {"stable": "#4472C4", "transition": "#ED7D31"}

    n_examples = len(examples)
    n_stages = len(stage_names)
    n_rows = n_stages + 2  # +GT +Pred
    row_height = 0.6
    subplot_height = n_rows * row_height + 1.2
    fig_height = n_examples * subplot_height + 0.8

    fig, axes = plt.subplots(n_examples, 1, figsize=(13, fig_height),
                             squeeze=False)

    for ex_idx, (qr, vdata) in enumerate(examples):
        ax = axes[ex_idx, 0]
        fps = vdata["fps"]
        duration = vdata["duration"]
        stages = vdata["stages"]

        display_query = qr["query"]
        if len(display_query) > 70:
            display_query = display_query[:67] + "..."

        y = n_rows * row_height
        ax.text(duration / 2, y + 0.3, f'"{display_query}"',
                ha="center", va="bottom", fontsize=8, fontstyle="italic")

        for s_idx, (sname, slabel) in enumerate(zip(stage_names, stage_labels)):
            stage_data = stages.get(sname)
            if stage_data is None:
                continue
            for lc in stage_data:
                s_frame, e_frame = lc["chunk"]
                s_sec, e_sec = s_frame / fps, e_frame / fps
                color = stage_colors.get(lc["label"], "#4472C4")
                ax.barh(y, e_sec - s_sec, left=s_sec, height=row_height * 0.8,
                        color=color, alpha=0.75, edgecolor="white",
                        linewidth=0.5)
            ax.text(-duration * 0.01, y + row_height * 0.4, slabel,
                    ha="right", va="center", fontsize=7)
            y -= row_height

        # Ground truth
        for gt in qr["gt_windows"]:
            ax.barh(y, gt[1] - gt[0], left=gt[0], height=row_height * 0.8,
                    color="#2E86AB", alpha=0.6, edgecolor="#1a5c7a",
                    linewidth=0.5)
        ax.text(-duration * 0.01, y + row_height * 0.4, "Ground Truth",
                ha="right", va="center", fontsize=7)
        y -= row_height

        # Predictions
        preds = qr.get("predictions", [])
        top1 = qr.get("top1_pred")
        top1_score = top1["score"] if top1 else None
        for pred in preds:
            is_top1 = (top1_score is not None
                       and abs(pred["score"] - top1_score) < 1e-6)
            color = "#A23B72" if is_top1 else "#B0B0B0"
            ec = "#6b1e4a" if is_top1 else "#808080"
            ax.barh(y, pred["end"] - pred["start"], left=pred["start"],
                    height=row_height * 0.8, color=color, alpha=0.75,
                    edgecolor=ec, linewidth=0.5)
        ax.text(-duration * 0.01, y + row_height * 0.4, "Prediction",
                ha="right", va="center", fontsize=7)
        y -= row_height

        if "signal_times" in vdata:
            sig_t = np.array(vdata["signal_times"])
            sig_v = np.array(vdata["signal_values"])
            sig_min, sig_max = sig_v.min(), sig_v.max()
            if sig_max > sig_min:
                sig_norm = (sig_v - sig_min) / (sig_max - sig_min)
            else:
                sig_norm = np.zeros_like(sig_v)
            row_bottom = y + row_height
            row_top = row_bottom + n_rows * row_height
            sig_scaled = row_bottom + sig_norm * (row_top - row_bottom)
            ax.plot(sig_t, sig_scaled, color="gray", alpha=0.12, linewidth=0.5)

        ax.set_xlim(-duration * 0.02, duration * 1.02)
        ax.set_ylim(y + row_height - 0.3, n_rows * row_height + 0.8)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if ex_idx == n_examples - 1:
            ax.set_xlabel("Time (seconds)", fontsize=10)

    legend_elements = [
        mpatches.Patch(facecolor="#4472C4", alpha=0.75, label="Stable"),
        mpatches.Patch(facecolor="#ED7D31", alpha=0.75, label="Transition"),
        mpatches.Patch(facecolor="#2E86AB", alpha=0.6, label="Ground Truth"),
        mpatches.Patch(facecolor="#A23B72", alpha=0.75, label="Top-1 Prediction"),
        mpatches.Patch(facecolor="#B0B0B0", alpha=0.75, label="Other Chunks"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=8,
                      framealpha=0.8)
    fig.suptitle("Chunking Pipeline Stages", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Cosine similarity per chunk for 5 examples
# ---------------------------------------------------------------------------

def plot_chunk_similarities(query_results, per_video_data, out_path):
    """Bar chart of chunk similarities to query for each example."""
    examples = list(query_results)
    if not examples:
        return

    n_examples = len(examples)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2.2 * n_examples),
                             squeeze=False)

    for i, qr in enumerate(examples):
        ax = axes[i, 0]
        preds = qr.get("predictions", [])
        if isinstance(preds, dict):
            preds = [preds]

        # Sort by start time for display
        preds_sorted = sorted(preds, key=lambda p: p["start"])
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        duration = _infer_duration(qr, vdata)

        top1_score = max(p["score"] for p in preds_sorted) if preds_sorted else 0

        for pred in preds_sorted:
            is_top1 = abs(pred["score"] - top1_score) < 1e-6
            color = "#A23B72" if is_top1 else "#4472C4"
            edgecolor = "#6b1e4a" if is_top1 else "#2c4f8a"
            ax.barh(0, pred["end"] - pred["start"], left=pred["start"],
                    height=0.6, color=color, edgecolor=edgecolor,
                    linewidth=1.0)
            chunk_mid = (pred["start"] + pred["end"]) / 2
            chunk_width = pred["end"] - pred["start"]
            if chunk_width > duration * 0.04:
                ax.text(chunk_mid, 0, f"{pred['score']:.3f}",
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color="white")
            else:
                ax.annotate(f"{pred['score']:.3f}",
                            xy=(chunk_mid, 0.35), fontsize=6,
                            ha="center", va="bottom", color="#333333")

        display_query = qr["query"]
        if len(display_query) > 70:
            display_query = display_query[:67] + "..."
        ax.set_title(f'"{display_query}"', fontsize=8, fontstyle="italic",
                     pad=4)
        ax.set_xlim(-duration * 0.02, duration * 1.02)
        ax.set_ylim(-0.5, 0.8)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if i == n_examples - 1:
            ax.set_xlabel("Time (seconds)", fontsize=9)

    legend_elements = [
        mpatches.Patch(facecolor="#A23B72", edgecolor="#6b1e4a",
                       label="Top-1 (highest similarity)"),
        mpatches.Patch(facecolor="#4472C4", edgecolor="#2c4f8a",
                       label="Other chunks"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=7,
                      framealpha=0.8)

    fig.suptitle("Chunk–Query Cosine Similarity", fontsize=12,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared helper: IoU computation
# ---------------------------------------------------------------------------

def _compute_iou(pred, gt_windows):
    """Return the best IoU of a single prediction against a list of GT windows."""
    best = 0.0
    for gt in gt_windows:
        inter_start = max(pred["start"], gt[0])
        inter_end = min(pred["end"], gt[1])
        inter = max(0, inter_end - inter_start)
        union = (pred["end"] - pred["start"]) + (gt[1] - gt[0]) - inter
        if union > 0:
            best = max(best, inter / union)
    return best


def _compute_all_ious(query_results):
    """Return list of (iou, query_result) for every query using top-1 pred."""
    ious = []
    for qr in query_results:
        pred = qr.get("top1_pred")
        if pred is None:
            preds = qr.get("predictions", [])
            pred = preds[0] if preds else None
        iou = _compute_iou(pred, qr["gt_windows"]) if pred else 0.0
        ious.append((iou, qr))
    return ious


def _select_example_sets(query_results, n=5):
    """Pick three disjoint example groups for consistent use across all plots.

    Returns ``(best, worst, fixed)`` where each is a list of query_result
    dicts.  *best* has the highest IoU, *worst* the lowest, and *fixed* is the
    first *n* from *query_results* (stable across runs on the same dataset).
    Duplicates are removed so no query appears in more than one group.
    """
    scored = _compute_all_ious(query_results)
    scored_asc = sorted(scored, key=lambda x: x[0])
    scored_desc = sorted(scored, key=lambda x: -x[0])

    used_ids = set()

    def _take(ordered, k):
        out = []
        for _, qr in ordered:
            sid = qr["sample_id"]
            if sid in used_ids:
                continue
            used_ids.add(sid)
            out.append(qr)
            if len(out) >= k:
                break
        return out

    worst = _take(scored_asc, n)
    best = _take(scored_desc, n)
    fixed = _take([(0, qr) for qr in query_results], n)
    return best, worst, fixed


# ---------------------------------------------------------------------------
# Plot 6: IoU distribution histogram
# ---------------------------------------------------------------------------

def plot_iou_histogram(query_results, out_path):
    """Histogram of top-1 IoU values across all queries."""
    ious = [iou for iou, _ in _compute_all_ious(query_results)]
    if not ious:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ious, bins=20, range=(0, 1), color="#4472C4", edgecolor="white",
            linewidth=0.8)
    mean_iou = np.mean(ious)
    ax.axvline(mean_iou, color="#A23B72", linewidth=2, linestyle="--",
               label=f"Mean IoU = {mean_iou:.3f}")
    ax.set_xlabel("IoU (top-1 prediction vs ground truth)", fontsize=10)
    ax.set_ylabel("Number of queries", fontsize=10)
    ax.set_title("IoU Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Performance vs video duration
# ---------------------------------------------------------------------------

def plot_performance_vs_duration(query_results, per_video_data, out_path):
    """Scatter plot of IoU vs video duration with trend line."""
    points = []
    for iou, qr in _compute_all_ious(query_results):
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        dur = _infer_duration(qr, vdata)
        points.append((dur, iou))

    if not points:
        return

    durations, ious = zip(*points)
    durations = np.array(durations)
    ious = np.array(ious)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(durations, ious, alpha=0.6, s=30, color="#4472C4",
               edgecolor="#2c4f8a", linewidth=0.5)

    if len(durations) >= 3:
        coeffs = np.polyfit(durations, ious, 1)
        x_line = np.linspace(durations.min(), durations.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="#A23B72",
                linewidth=2, linestyle="--",
                label=f"Trend (slope={coeffs[0]:.4f})")
        ax.legend(fontsize=9)

    ax.set_xlabel("Video duration (seconds)", fontsize=10)
    ax.set_ylabel("IoU (top-1 prediction)", fontsize=10)
    ax.set_title("Performance vs Video Duration", fontsize=12,
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 8: Number of chunks vs performance
# ---------------------------------------------------------------------------

def plot_chunks_vs_performance(query_results, per_video_data, out_path, config=None):
    """Scatter plot of IoU vs number of chunks with trend line."""
    points = []
    for iou, qr in _compute_all_ious(query_results):
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        points.append((_get_effective_num_chunks(qr, vdata, config), iou))

    if not points:
        return

    n_chunks, ious = zip(*points)
    n_chunks = np.array(n_chunks, dtype=float)
    ious = np.array(ious)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Jitter x slightly so overlapping points are visible
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, size=len(n_chunks))
    ax.scatter(n_chunks + jitter, ious, alpha=0.6, s=30, color="#4472C4",
               edgecolor="#2c4f8a", linewidth=0.5)

    if len(n_chunks) >= 3:
        coeffs = np.polyfit(n_chunks, ious, 1)
        x_line = np.linspace(n_chunks.min(), n_chunks.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="#A23B72",
                linewidth=2, linestyle="--",
                label=f"Trend (slope={coeffs[0]:.4f})")
        ax.legend(fontsize=9)

    ax.set_xlabel("Number of chunks", fontsize=10)
    ax.set_ylabel("IoU (top-1 prediction)", fontsize=10)
    ax.set_title("Number of Chunks vs Performance", fontsize=12,
                 fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 9: Chunk duration distribution
# ---------------------------------------------------------------------------

def plot_chunk_duration_distribution(query_results, out_path):
    """Histogram of chunk durations across all queries."""
    durations = []
    seen = set()
    for qr in query_results:
        # Avoid double-counting chunks from the same video
        vid = qr["video_path"]
        if vid in seen:
            continue
        seen.add(vid)
        preds = qr.get("predictions", [])
        if isinstance(preds, dict):
            preds = [preds]
        for p in preds:
            if isinstance(p, dict):
                durations.append(p["end"] - p["start"])

    if not durations:
        return

    durations = np.array(durations)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations, bins=30, color="#4472C4", edgecolor="white",
            linewidth=0.8)
    mean_d = np.mean(durations)
    median_d = np.median(durations)
    ax.axvline(mean_d, color="#A23B72", linewidth=2, linestyle="--",
               label=f"Mean = {mean_d:.1f}s")
    ax.axvline(median_d, color="#ED7D31", linewidth=2, linestyle=":",
               label=f"Median = {median_d:.1f}s")
    ax.set_xlabel("Chunk duration (seconds)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Chunk Duration Distribution", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 10: Failure case analysis (worst N predictions)
# ---------------------------------------------------------------------------

def plot_failure_cases(query_results, per_video_data, out_path):
    """Show the worst predictions (lowest IoU) with chunks and scores."""
    scored = _compute_all_ious(query_results)
    scored.sort(key=lambda x: x[0])

    if not scored:
        return

    n_examples = len(scored)
    row_height = 1.6
    fig_height = row_height * n_examples + 1.0
    fig, ax = plt.subplots(figsize=(13, fig_height))

    y = 0.5 + (n_examples - 1) * row_height
    for iou_val, qr in scored:
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        fps = vdata["fps"] if vdata else 30.0
        duration = _infer_duration(qr, vdata)
        _draw_example_row(ax, qr, vdata, fps, duration, y_offset=y)
        ax.text(duration * 1.01, y + 0.175, f"IoU={iou_val:.3f}",
                fontsize=7, va="center", color="#A23B72", fontweight="bold")
        y -= row_height

    max_dur = max(
        (_infer_duration(qr, per_video_data.get(qr["video_path"]) if per_video_data else None)
         for _, qr in scored),
        default=100,
    )
    ax.set_xlim(-max_dur * 0.02, max_dur * 1.08)
    ax.set_ylim(y + row_height - 0.8,
                0.5 + (n_examples - 1) * row_height + 1.5)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    legend_elements = [
        mpatches.Patch(facecolor="#2E86AB", edgecolor="#1a5c7a",
                       label="Ground Truth"),
        mpatches.Patch(facecolor="#A23B72", edgecolor="#6b1e4a",
                       label="Top-1 Prediction"),
        mpatches.Patch(facecolor="#B0B0B0", edgecolor="#808080",
                       label="Other Chunks"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8)
    ax.set_title(f"Failure Cases (worst {n_examples} by IoU)", fontsize=12,
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Example video generation (MP4 with signal analysis overlay)
# ---------------------------------------------------------------------------

_CHUNKING_STYLE = {
    "content_detector": ("#E74C3C", "#C0392B"),
    "embedding":        ("#2ECC71", "#1E8449"),
    "surprise":         ("#3498DB", "#1F618D"),
}


def _render_signal_background(video_data, query_result, width, height,
                               line_color="#2ECC71", boundary_color="#1E8449"):
    """Render the static signal analysis plot as a BGR numpy array.

    Returns ``(image, plot_bounds)`` where *plot_bounds* is
    ``(data_left_px, data_right_px, data_top_px, data_bottom_px)``.
    """
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.08, 0.20, 0.87, 0.63])

    duration = video_data["duration"]
    fps = video_data["fps"]
    gt_windows = query_result["gt_windows"]
    predictions = query_result.get("predictions", [])
    top1_pred = query_result.get("top1_pred")

    # Ground-truth shading
    for i, gt in enumerate(gt_windows):
        ax.axvspan(gt[0], gt[1], alpha=0.15, color="#2E86AB",
                   label="Ground Truth" if i == 0 else None, zorder=1)

    # Signal curve + threshold
    y_lo, y_hi = 0.0, 1.0
    if "signal_times" in video_data and "signal_values" in video_data:
        sig_t = np.array(video_data["signal_times"])
        sig_v = np.array(video_data["signal_values"])
        ax.plot(sig_t, sig_v, color=line_color, linewidth=1.8,
                label="Chunking Signal", zorder=3)

        threshold = video_data.get("threshold")
        if threshold is not None:
            ax.axhline(threshold, color="#4f4f4f", linestyle="--", linewidth=1.2,
                        alpha=0.7, label="Threshold", zorder=2)
            y_lo = min(float(sig_v.min()), threshold)
            y_hi = max(float(sig_v.max()), threshold)
        else:
            y_lo = float(sig_v.min())
            y_hi = float(sig_v.max())

    ax.grid(True, alpha=0.12, linewidth=0.5)

    # Chunk boundaries derived from the actual prediction windows
    boundary_set = set()
    for pred in predictions:
        boundary_set.add(pred["start"])
        boundary_set.add(pred["end"])
    boundary_times = sorted(t for t in boundary_set if 0.5 < t < duration - 0.5)
    for i, bt in enumerate(boundary_times):
        ax.axvline(bt, color=boundary_color, linestyle="--", linewidth=1.5,
                   alpha=0.6, zorder=4,
                   label="Chunk Boundary" if i == 0 else None)
        if video_data.get("threshold") is not None:
            ax.plot(bt, video_data["threshold"], "d",
                    color=boundary_color, markersize=6, zorder=5)

    # GT / Prediction timeline bars below the signal
    y_range = (y_hi - y_lo) or 1.0
    pad = y_range * 0.1
    bar_h = y_range * 0.05
    gt_y = y_lo - pad * 1.8
    pred_y = gt_y - bar_h - pad * 0.5

    for gt in gt_windows:
        ax.barh(gt_y, gt[1] - gt[0], left=gt[0], height=bar_h,
                color="#2E86AB", alpha=0.7, edgecolor="#1a5c7a",
                linewidth=0.5, zorder=4)
    ax.text(-duration * 0.008, gt_y, "GT", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#2E86AB")

    top1_score = top1_pred["score"] if top1_pred else None
    top1_labeled = False
    for pred in predictions:
        is_top1 = (top1_score is not None
                   and abs(pred["score"] - top1_score) < 1e-6)
        color = "#A23B72" if is_top1 else "#B0B0B0"
        ec = "#6b1e4a" if is_top1 else "#888"
        lbl = "Top-1 Prediction" if is_top1 and not top1_labeled else None
        ax.barh(pred_y, pred["end"] - pred["start"], left=pred["start"],
                height=bar_h, color=color, alpha=0.7, edgecolor=ec,
                linewidth=0.5, zorder=4, label=lbl)
        if is_top1:
            top1_labeled = True
    ax.text(-duration * 0.008, pred_y, "Pred", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#A23B72")

    ax.set_xlim(0, duration)
    ax.set_ylim(pred_y - pad, y_hi + pad * 2)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, ncol=2)

    query = query_result["query"]
    display_q = query if len(query) <= 85 else query[:82] + "..."
    fig.suptitle(f'"{display_q}"', fontsize=11, fontstyle="italic", y=0.98)

    fig.canvas.draw()
    pos = ax.get_position()
    dl = int(pos.x0 * width)
    dr = int(pos.x1 * width)
    dt = int((1 - pos.y1) * height)
    db = int((1 - pos.y0) * height)

    w_px, h_px = fig.canvas.get_width_height()
    buf = np.frombuffer(
        fig.canvas.buffer_rgba(), dtype=np.uint8,
    ).reshape(h_px, w_px, 4)
    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return img_bgr.copy(), (dl, dr, dt, db)


def render_example_mp4(query_result, video_data, out_path,
                       target_width=640, output_fps=10,
                       line_color="#2ECC71", boundary_color="#1E8449"):
    """Render an MP4 compositing the source video with signal analysis below.

    The output frame stacks the resized video on top with the signal analysis
    plot (including GT windows, chunk boundaries, and prediction bars) beneath.
    A black cursor line tracks the current playback time.
    """
    video_path = query_result["video_path"]
    if not os.path.isfile(video_path):
        return False

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or orig_w <= 0:
        cap.release()
        return False

    duration = video_data["duration"]
    video_h = int(orig_h * (target_width / orig_w))
    signal_h = 280
    frame_w = target_width
    frame_h = video_h + signal_h

    signal_bg, (dl, dr, dt, db) = _render_signal_background(
        video_data, query_result, frame_w, signal_h,
        line_color=line_color, boundary_color=boundary_color,
    )
    if signal_bg.shape[1] != frame_w or signal_bg.shape[0] != signal_h:
        signal_bg = cv2.resize(signal_bg, (frame_w, signal_h))

    frame_step = max(1, round(video_fps / output_fps))
    actual_fps = video_fps / frame_step

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path, fourcc, actual_fps, (frame_w, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        return False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        t = frame_idx / video_fps
        vid_frame = cv2.resize(frame, (frame_w, video_h))

        sig_frame = signal_bg.copy()
        frac = min(1.0, t / max(0.001, duration))
        cx = dl + int(frac * (dr - dl))
        cv2.line(sig_frame, (cx, dt), (cx, db), (0, 0, 0), 2)

        writer.write(np.vstack([vid_frame, sig_frame]))
        frame_idx += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 so the file plays in browsers / VS Code
    tmp_path = out_path + ".tmp.mp4"
    os.rename(out_path, tmp_path)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
             out_path],
            check=True, capture_output=True,
        )
        os.remove(tmp_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.rename(tmp_path, out_path)

    return True


def generate_example_videos(query_results, per_video_data, plots_dir,
                            prefix="example", max_duration=150, config=None):
    """Render MP4s with signal analysis for the given examples.

    *query_results* should be a pre-selected list (e.g. best/worst).
    Videos that are too long, missing signal data, or not on disk are skipped.
    """
    if not per_video_data:
        return []

    chunking_type = None
    if config is not None:
        try:
            chunking_type = config["chunking"]["type"]
        except (KeyError, TypeError):
            pass
    line_color, boundary_color = _CHUNKING_STYLE.get(
        chunking_type, ("#3498DB", "#1F618D"),
    )

    seen = set()
    selected = []
    for qr in query_results:
        vpath = qr["video_path"]
        if vpath in seen:
            continue
        seen.add(vpath)
        vdata = per_video_data.get(vpath)
        if not vdata or "signal_times" not in vdata:
            continue
        dur = vdata.get("duration", 0)
        if dur > max_duration or dur < 10:
            continue
        if not os.path.isfile(vpath):
            continue
        selected.append((qr, vdata))

    paths = []
    for i, (qr, vdata) in enumerate(selected):
        out = os.path.join(plots_dir, f"{prefix}_{i + 1}.mp4")
        print(f"  Rendering {prefix} video {i + 1}/{len(selected)}: "
              f"{Path(qr['video_path']).name}")
        try:
            ok = render_example_mp4(
                qr, vdata, out,
                line_color=line_color, boundary_color=boundary_color,
            )
            if ok:
                paths.append(out)
        except Exception as e:
            print(f"    Warning: failed to render: {e}")
    return paths


# ---------------------------------------------------------------------------
# Main: generate all plots for a run
# ---------------------------------------------------------------------------

def generate_all_plots(run_dir, generate_videos=False, videos_only=False):
    """Generate plots for a saved run directory, optionally rendering MP4s."""
    run = load_run(run_dir)
    query_results = run["per_query_results"]
    per_video_data = run.get("per_video_data")
    config = run.get("config")
    dataset_name = None
    evaluation_mode = None
    if config is not None:
        try:
            dataset_name = config["dataset"]["name"]
        except (KeyError, TypeError):
            pass
        try:
            evaluation_mode = config["evaluation"]["mode"]
        except (KeyError, TypeError):
            pass

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if evaluation_mode == "vlm_qa":
        from plots_qa import generate_qa_plots
        return generate_qa_plots(run_dir, query_results, per_video_data, config)

    if evaluation_mode == "gebd":
        from plots_gebd import generate_gebd_plots
        return generate_gebd_plots(run_dir, query_results, per_video_data, config)

    # --- Centralized example sets (used by all per-example plots) ---
    best, worst, fixed = _select_example_sets(query_results)
    best_and_worst = best + worst
    all_examples = best + worst + fixed

    if not videos_only:
        # For the custom dataset, show every saved query instead of sampling.
        stacked_examples = query_results if dataset_name == "custom_dataset" else best_and_worst
        stacked_title = (
            f"All {len(stacked_examples)} Query Predictions"
            if dataset_name == "custom_dataset"
            else f"Best {len(best)} & Worst {len(worst)} Predictions"
        )
        if stacked_examples:
            plot_stacked_examples(
                stacked_examples, per_video_data,
                os.path.join(plots_dir, "stacked_examples.png"),
                title=stacked_title,
            )

        # Chunking stages
        if per_video_data and all_examples:
            plot_chunking_stages(
                all_examples, per_video_data,
                os.path.join(plots_dir, "chunking_stages.png"),
            )

        # Chunk similarities
        if all_examples:
            plot_chunk_similarities(
                all_examples, per_video_data,
                os.path.join(plots_dir, "chunk_similarities.png"),
            )

        # Failure cases (worst 5)
        if worst:
            plot_failure_cases(
                worst, per_video_data,
                os.path.join(plots_dir, "failure_cases.png"),
            )

        if evaluation_mode == "vlm_chunk_selection" and all_examples:
            plot_vlm_prompt_examples(
                all_examples, per_video_data,
                os.path.join(plots_dir, "vlm_prompt_examples.png"),
                config=config,
            )
        if evaluation_mode == "vlm_chunk_selection" and best:
            plot_vlm_prompt_examples(
                best[:1], per_video_data,
                os.path.join(plots_dir, "vlm_best_example.png"),
                config=config,
                title="VLM Best Example",
            )

        # --- Aggregate plots (use ALL query results) ---
        if query_results:
            plot_iou_histogram(
                query_results,
                os.path.join(plots_dir, "iou_distribution.png"),
            )
        if query_results:
            plot_performance_vs_duration(
                query_results, per_video_data,
                os.path.join(plots_dir, "performance_vs_duration.png"),
            )
        if query_results:
            plot_chunks_vs_performance(
                query_results,
                per_video_data,
                os.path.join(plots_dir, "chunks_vs_performance.png"),
                config=config,
            )
        if query_results:
            plot_chunk_duration_distribution(
                query_results,
                os.path.join(plots_dir, "chunk_duration_distribution.png"),
            )

    # --- Example videos: best + worst + random ---
    if generate_videos and per_video_data and best:
        generate_example_videos(
            best, per_video_data, plots_dir,
            prefix="best", config=config,
        )
    if generate_videos and per_video_data and worst:
        generate_example_videos(
            worst, per_video_data, plots_dir,
            prefix="worst", config=config,
        )
    if generate_videos and per_video_data and fixed:
        generate_example_videos(
            fixed[:3], per_video_data, plots_dir,
            prefix="random", max_duration=150, config=config,
        )

    print(f"Plots saved to {plots_dir}/")
    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Saved evaluation run directory")
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Also render example MP4 videos",
    )
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Render only example MP4 videos",
    )
    args = parser.parse_args()

    generate_all_plots(
        args.run_dir,
        generate_videos=args.videos or args.videos_only,
        videos_only=args.videos_only,
    )
