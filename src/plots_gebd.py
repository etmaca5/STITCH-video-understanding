"""GEBD-specific evaluation plots.

Visualizes predicted chunk boundaries against ground-truth event boundaries.
"""

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import _gebd_f1_single_rater


def _load_frame_at_time(video_path, time_sec):
    """Load a single RGB frame from a video at a given time in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_frame(frame, target_height):
    h, w = frame.shape[:2]
    scale = target_height / max(h, 1)
    tw = max(1, int(round(w * scale)))
    return cv2.resize(frame, (tw, target_height))


def _per_video_f1(video_result, threshold=0.05):
    """Compute per-video F1 at a given threshold, returning (f1, best_rater_idx)."""
    meta = video_result.get("metadata", {})
    gt_per_rater = meta.get("gt_boundaries_per_rater", [])
    det = video_result.get("predicted_boundaries", [])
    duration = float(meta.get("video_duration", 0.0))

    if not gt_per_rater:
        return 0.0, 0

    best_f1 = -1.0
    best_idx = 0
    for i, rater_gt in enumerate(gt_per_rater):
        tp, num_pos, num_det = _gebd_f1_single_rater(
            rater_gt, det, threshold, duration,
        )
        fn = num_pos - tp
        fp = (len(det) if det else 0) - tp
        rec = 1.0 if num_pos == 0 else tp / (tp + fn)
        prec = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        f1 = 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)
        if f1 > best_f1:
            best_f1 = f1
            best_idx = i

    return best_f1, best_idx


def plot_gebd_example(video_result, vid_data, out_path, num_frames=8):
    """Plot a single GEBD example: frame strip with GT/predicted boundary lines."""
    video_path = video_result["video_path"]
    meta = video_result.get("metadata", {})
    pred_bdys = video_result.get("predicted_boundaries", [])
    gt_per_rater = meta.get("gt_boundaries_per_rater", [])
    duration = float(meta.get("video_duration", vid_data.get("duration", 10.0)))

    f1, best_rater = _per_video_f1(video_result)
    best_gt = gt_per_rater[best_rater] if gt_per_rater else []

    sample_times = np.linspace(0, duration, num_frames + 2)[1:-1]
    frames = []
    for t in sample_times:
        frame = _load_frame_at_time(video_path, t)
        if frame is not None:
            frames.append(_resize_frame(frame, 120))
        else:
            frames.append(np.zeros((120, 160, 3), dtype=np.uint8))

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), height_ratios=[3, 1],
                             gridspec_kw={"hspace": 0.05})

    ax_frames = axes[0]
    strip = np.concatenate(frames, axis=1)
    ax_frames.imshow(strip, aspect="auto", extent=[0, duration, 0, 1])
    ax_frames.set_xlim(0, duration)
    ax_frames.set_yticks([])
    ax_frames.set_title(
        f"{Path(video_path).stem}  |  "
        f"F1@0.05={f1:.2f}  |  "
        f"pred={len(pred_bdys)} boundaries  |  "
        f"gt={len(best_gt)} boundaries (best rater)",
        fontsize=10,
    )

    for bdy in best_gt:
        ax_frames.axvline(bdy, color="green", linewidth=2, alpha=0.8, linestyle="-")
    for bdy in pred_bdys:
        ax_frames.axvline(bdy, color="red", linewidth=2, alpha=0.8, linestyle="--")

    ax_tl = axes[1]
    ax_tl.set_xlim(0, duration)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_xlabel("Time (s)", fontsize=9)
    ax_tl.set_yticks([0.25, 0.75])
    ax_tl.set_yticklabels(["Predicted", "GT"], fontsize=8)

    for bdy in best_gt:
        ax_tl.plot(bdy, 0.75, "^", color="green", markersize=8)
    for bdy in pred_bdys:
        ax_tl.plot(bdy, 0.25, "v", color="red", markersize=8)

    ax_tl.axhline(0.5, color="gray", linewidth=0.5, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gebd_stacked(
    video_results, per_video_data, out_path, max_rows=8,
    num_frames=6, strip_height=96,
):
    """Stacked GEBD examples sorted by per-video F1 (best and worst).

    Each row shows:
      - a horizontal strip of frames sampled uniformly in time (like ``plot_gebd_example``),
        with GT (green) and predicted (red, dashed) boundaries overlaid;
      - a compact timeline with the same boundaries for quick reading.
    """
    scored = []
    for vr in video_results:
        f1, _ = _per_video_f1(vr)
        scored.append((f1, vr))

    scored.sort(key=lambda x: x[0], reverse=True)
    n_best = min(max_rows // 2, len(scored))
    n_worst = min(max_rows - n_best, len(scored) - n_best)
    selected = scored[:n_best] + scored[-n_worst:] if n_worst > 0 else scored[:n_best]

    if not selected:
        return

    nrows = len(selected)
    fig = plt.figure(figsize=(14, 3.4 * nrows))
    outer = gridspec.GridSpec(nrows, 1, figure=fig, hspace=0.42)

    for row_idx, (f1, vr) in enumerate(selected):
        video_path = vr["video_path"]
        vid_data = (
            per_video_data.get(video_path, {}) if per_video_data else {}
        )
        meta = vr.get("metadata", {})
        pred_bdys = vr.get("predicted_boundaries", [])
        gt_per_rater = meta.get("gt_boundaries_per_rater", [])
        duration = float(
            meta.get("video_duration", vid_data.get("duration", 10.0)),
        )

        _, best_rater = _per_video_f1(vr)
        best_gt = gt_per_rater[best_rater] if gt_per_rater else []

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, outer[row_idx], height_ratios=[2.4, 1.0], hspace=0.06,
        )
        ax_strip = fig.add_subplot(inner[0])
        ax_tl = fig.add_subplot(inner[1], sharex=ax_strip)

        # Frame strip (uniform samples over [0, duration], matching timeline x)
        sample_times = np.linspace(0, duration, num_frames + 2)[1:-1]
        frames = []
        for t in sample_times:
            frame = _load_frame_at_time(video_path, t)
            if frame is not None:
                frames.append(_resize_frame(frame, strip_height))
            else:
                frames.append(
                    np.zeros((strip_height, 160, 3), dtype=np.uint8),
                )
        strip = np.concatenate(frames, axis=1)
        ax_strip.imshow(strip, aspect="auto", extent=[0, duration, 0, 1])
        ax_strip.set_xlim(0, duration)
        ax_strip.set_ylim(0, 1)
        ax_strip.set_yticks([])

        label = "BEST" if row_idx < n_best else "WORST"
        stem = Path(video_path).stem[:40]
        ax_strip.set_title(
            f"{label}  |  F1@0.05={f1:.2f}  |  {stem}  "
            f"(pred={len(pred_bdys)}, gt={len(best_gt)})",
            fontsize=9,
            loc="left",
        )

        for bdy in best_gt:
            ax_strip.axvline(
                bdy, color="green", linewidth=2, alpha=0.85, linestyle="-",
            )
        for bdy in pred_bdys:
            ax_strip.axvline(
                bdy, color="red", linewidth=2, alpha=0.85, linestyle="--",
            )

        # Timeline row (markers)
        ax_tl.set_xlim(0, duration)
        ax_tl.set_ylim(0, 1)
        ax_tl.set_ylabel(f"F1\n{f1:.2f}", fontsize=7, rotation=0, labelpad=8, va="center")
        for bdy in best_gt:
            ax_tl.plot(bdy, 0.72, "^", color="green", markersize=7)
        for bdy in pred_bdys:
            ax_tl.plot(bdy, 0.28, "v", color="red", markersize=7)
        ax_tl.axhline(0.5, color="gray", linewidth=0.5, alpha=0.35)
        ax_tl.set_yticks([0.28, 0.72])
        ax_tl.set_yticklabels(["Pred", "GT"], fontsize=7)
        plt.setp(ax_strip.get_xticklabels(), visible=False)
        if row_idx < nrows - 1:
            plt.setp(ax_tl.get_xticklabels(), visible=False)
        else:
            ax_tl.set_xlabel("Time (s)", fontsize=9)

    legend_elements = [
        plt.Line2D([0], [0], color="green", linewidth=2, label="GT boundary"),
        plt.Line2D(
            [0], [0], color="red", linewidth=2, linestyle="--",
            label="Predicted boundary",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    fig.suptitle(
        "GEBD: Best & Worst by F1@0.05 (frame strip + boundaries)",
        fontsize=12,
        y=1.002,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_LABEL_FONT = ["Calibri", "Carlito", "DejaVu Sans"]


def _build_chunk_strips(
    video_result, vid_data, num_frames_per_chunk, frame_height, intra_gap_px,
):
    """Return list of (strip_np, (start_sec, end_sec)) for predicted chunks."""
    video_path = video_result["video_path"]
    if not os.path.isfile(video_path):
        return []

    meta = video_result.get("metadata", {})
    pred_bdys = sorted(video_result.get("predicted_boundaries", []))
    duration = float(meta.get("video_duration", vid_data.get("duration", 0.0)))
    if duration <= 0:
        return []

    edges = [0.0] + [b for b in pred_bdys if 0 < b < duration] + [duration]
    chunks = list(zip(edges[:-1], edges[1:]))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    strips = []
    intra = np.full((frame_height, intra_gap_px, 3), 255, np.uint8)
    for s, e in chunks:
        sample_times = np.linspace(s, e, num_frames_per_chunk + 2)[1:-1]
        frames = []
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames.append(_resize_frame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_height,
            ))
        if not frames:
            continue
        pieces = [frames[0]]
        for f in frames[1:]:
            pieces.extend([intra, f])
        strips.append((np.concatenate(pieces, axis=1), (s, e)))
    cap.release()
    return strips


def _compose_row(chunk_strips, frame_height, chunk_gap_px):
    """Glue per-chunk strips into one row with white inter-chunk gaps."""
    chunk_gap = np.full((frame_height, chunk_gap_px, 3), 255, np.uint8)
    pieces = [chunk_strips[0][0]]
    for strip, _ in chunk_strips[1:]:
        pieces.extend([chunk_gap, strip])
    return np.concatenate(pieces, axis=1)


def _draw_row(ax, chunk_strips, full, y_top, frame_height, chunk_gap_px,
              label_offset=12, fontsize=7):
    """Render an already-composed row at vertical offset *y_top* on *ax*."""
    ax.imshow(full, extent=[0, full.shape[1], y_top + frame_height, y_top])
    x = 0
    for strip, (s, e) in chunk_strips:
        w = strip.shape[1]
        ax.text(x + w / 2, y_top + frame_height + label_offset,
                f"{s:.1f}–{e:.1f}s",
                ha="center", va="center", fontsize=fontsize, color="#444",
                family=_LABEL_FONT)
        x += w + chunk_gap_px


def plot_event_detection_example(
    video_result, vid_data, out_path,
    num_frames_per_chunk=3, frame_height=96,
    chunk_gap_px=24, intra_gap_px=2,
):
    """Render a single video as a horizontal strip of predicted chunks.

    Each chunk shows ``num_frames_per_chunk`` frames sampled uniformly inside
    it. Chunks are separated by white gaps, which act as the visual boundary.
    """
    chunk_strips = _build_chunk_strips(
        video_result, vid_data, num_frames_per_chunk, frame_height,
        intra_gap_px,
    )
    if not chunk_strips:
        return False

    full = _compose_row(chunk_strips, frame_height, chunk_gap_px)

    total_w = full.shape[1]
    fig_w = max(6.0, total_w / 150)
    fig_h = frame_height / 150 + 0.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, total_w)
    ax.set_ylim(frame_height + 22, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    _draw_row(ax, chunk_strips, full, 0, frame_height, chunk_gap_px)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_event_detection_combined(
    video_results, per_video_data, out_path,
    num_frames_per_chunk=3, frame_height=220,
    chunk_gap_px=48, intra_gap_px=4, row_gap_px=40, dpi=300,
    label_band_px=64, label_offset_px=44, label_fontsize=9,
):
    """Stack multiple event-detection rows into a single high-resolution plot."""
    rows = []
    for vr in video_results:
        vid_data = (
            per_video_data.get(vr["video_path"], {}) if per_video_data else {}
        )
        strips = _build_chunk_strips(
            vr, vid_data, num_frames_per_chunk, frame_height, intra_gap_px,
        )
        if strips:
            rows.append((strips, _compose_row(strips, frame_height, chunk_gap_px)))
    if not rows:
        return False

    row_height = frame_height + label_band_px
    total_h = row_height * len(rows) + row_gap_px * (len(rows) - 1)
    total_w = max(full.shape[1] for _, full in rows)

    fig_w = total_w / dpi
    fig_h = total_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    y = 0
    for strips, full in rows:
        x_off = (total_w - full.shape[1]) // 2
        ax.imshow(
            full,
            extent=[x_off, x_off + full.shape[1], y + frame_height, y],
        )
        x = x_off
        for strip, (s, e) in strips:
            w = strip.shape[1]
            ax.text(x + w / 2, y + frame_height + label_offset_px,
                    f"{s:.1f}–{e:.1f}s",
                    ha="center", va="top",
                    fontsize=label_fontsize, color="#444",
                    family=_LABEL_FONT)
            x += w + chunk_gap_px
        y += row_height + row_gap_px

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_event_detection_examples(
    video_results, per_video_data, out_dir, n=5, seed=0,
):
    """Pick *n* random videos and render one event-detection strip each."""
    renderable = [
        vr for vr in video_results
        if os.path.isfile(vr.get("video_path", ""))
    ]
    if not renderable:
        return []

    rng = random.Random(seed)
    chosen = rng.sample(renderable, min(n, len(renderable)))

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, vr in enumerate(chosen, start=1):
        vid_data = (
            per_video_data.get(vr["video_path"], {}) if per_video_data else {}
        )
        out_path = os.path.join(out_dir, f"event_detection_example_{i}.png")
        if plot_event_detection_example(vr, vid_data, out_path):
            written.append(out_path)
    return written


def generate_gebd_plots(run_dir, query_results, per_video_data, config=None):
    """Generate GEBD-specific plots for a run directory."""
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    renderable = [
        vr for vr in query_results
        if os.path.isfile(vr.get("video_path", ""))
    ]

    if renderable:
        plot_gebd_example(
            renderable[0],
            per_video_data.get(renderable[0]["video_path"], {}),
            os.path.join(plots_dir, "gebd_example.png"),
        )

    if len(renderable) >= 2:
        plot_gebd_stacked(
            renderable,
            per_video_data,
            os.path.join(plots_dir, "gebd_stacked_examples.png"),
        )

    plot_event_detection_examples(
        query_results, per_video_data,
        os.path.join(plots_dir, "event_detection_examples"),
    )

    return plots_dir
