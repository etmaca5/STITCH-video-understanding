"""QA-specific evaluation plots.

Separate from plots.py which handles moment retrieval visualizations.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import exact_match_accuracy, extract_mcq_letter, mcq_letter_match
from temporal_abstraction import select_frames


def _resize_frame(frame, target_height):
    """Resize an RGB frame to a fixed height while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = target_height / max(height, 1)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(frame, (target_width, target_height))


def _escape_matplotlib_text(text):
    """Escape text that matplotlib may otherwise parse as math."""
    return str(text).replace("$", r"\$")


def _load_frame_by_index(video_path, frame_idx):
    """Load a single frame from a video by frame index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _qa_probe_true_duration_sec(video_path, video_data, fps):
    """Best estimate of full-video duration in seconds (prefers saved ``duration``)."""
    fps = float(fps) if fps and float(fps) > 0 else 30.0
    if video_data:
        d = video_data.get("duration")
        if d is not None and float(d) > 0:
            return float(d)
        tf = video_data.get("total_frames")
        if tf is not None and int(tf) > 0:
            return float(int(tf)) / fps
    cap = cv2.VideoCapture(video_path)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    cap.release()
    if n > 0 and cap_fps > 0:
        return n / cap_fps
    return None


def _qa_format_duration_axis_label(sec):
    """Human-readable end time for the timeline (matches common mm:ss.s style)."""
    sec = max(0.0, float(sec))
    whole_min = int(sec // 60)
    rem = sec - 60 * whole_min
    if whole_min > 0:
        return f"{whole_min:d}:{rem:04.1f}"
    return f"{rem:.1f}s"


def _qa_collect_cells(query_result, video_data, config):
    """Build frame cells for one QA query. Returns None if nothing to draw."""
    video_path = query_result["video_path"]
    if not os.path.isfile(video_path):
        return None

    fps = video_data["fps"] if video_data else 30.0
    prompt_meta = query_result.get("vlm_prompt_metadata", {})
    chunks = [tuple(chunk) for chunk in prompt_meta.get("used_chunks", [])]
    frame_method = prompt_meta.get("frame_method")
    frames_per_chunk = int(prompt_meta.get("frames_per_chunk", 1))
    chunk_frame_selections = prompt_meta.get("chunk_frame_selections")

    if not frame_method and config is not None:
        try:
            frame_method = config["temporal_abstraction"]["frame_selection"]["method"]
        except (KeyError, TypeError):
            pass
    frame_method = frame_method or "middle"

    # Window-based methods store frame_indices directly in metadata;
    # iterative QA uses all_frame_indices / all_frame_times for the final prompt.
    window_frame_indices = prompt_meta.get("frame_indices")
    window_frame_times = prompt_meta.get("frame_times")
    if not window_frame_indices and prompt_meta.get("all_frame_indices"):
        window_frame_indices = prompt_meta.get("all_frame_indices")
        window_frame_times = prompt_meta.get("all_frame_times")

    total_duration_sec = None
    layout_window = False

    if window_frame_indices and window_frame_times:
        layout_window = True
        provisional = _qa_probe_true_duration_sec(video_path, video_data, fps)
        if provisional is None or provisional <= 0:
            provisional = max(float(t) for t in window_frame_times) + 1.0

        cells = []
        shown_frame_points = []
        for fi, ft in zip(window_frame_indices, window_frame_times):
            frame = _load_frame_by_index(video_path, int(fi))
            if frame is None:
                continue
            ts_sec = float(ft)
            shown_frame_points.append(ts_sec)
            cells.append((frame, ts_sec, 0.0, provisional))

        if not cells:
            return None

        frame_images = [c[0] for c in cells]
        timestamps_sec = [c[1] for c in cells]
        chunk_ranges = [(c[2], c[3]) for c in cells]
        unique_chunk_ranges = [(0.0, provisional)]
    else:
        if not chunks:
            total_frames = int(video_data.get("total_frames", 0)) if video_data else 0
            num_chunks = 8
            if config is not None:
                try:
                    num_chunks = int(config["evaluation"]["num_uniform_chunks"])
                except (KeyError, TypeError, ValueError):
                    pass
            if total_frames <= 0:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            chunk_size = total_frames / num_chunks
            chunks = [
                (int(i * chunk_size), int(min((i + 1) * chunk_size, total_frames)))
                for i in range(num_chunks)
            ]

        cells = []
        shown_frame_points = []
        for idx, chunk in enumerate(chunks):
            sf, ef = chunk
            chunk_t0 = sf / fps
            chunk_t1 = ef / fps
            try:
                frame_kwargs = {"n": frames_per_chunk}
                selection_meta = None
                if chunk_frame_selections is not None and idx < len(chunk_frame_selections):
                    selection_meta = chunk_frame_selections[idx]
                    frame_kwargs["frame_indices"] = selection_meta.get("frame_indices")
                frames = select_frames(video_path, chunk, method=frame_method, **frame_kwargs)

                selected_frames = []
                if selection_meta is not None:
                    selected_frames = selection_meta.get("selected_frames", [])

                for frame_idx, frame in enumerate(frames):
                    if frame_idx < len(selected_frames):
                        ts_sec = float(selected_frames[frame_idx]["frame_index"]) / fps
                    else:
                        ts_sec = ((sf + ef) // 2) / fps
                    shown_frame_points.append(ts_sec)
                    cells.append((frame, ts_sec, chunk_t0, chunk_t1))
            except Exception:
                pass

        if not cells:
            return None

        frame_images = [c[0] for c in cells]
        timestamps_sec = [c[1] for c in cells]
        chunk_ranges = [(c[2], c[3]) for c in cells]
        unique_chunk_ranges = [(sf / fps, ef / fps) for sf, ef in chunks]

    if total_duration_sec is None:
        if video_data:
            dur = video_data.get("duration")
            if dur is not None and float(dur) > 0:
                total_duration_sec = float(dur)
            else:
                tf = video_data.get("total_frames")
                if tf:
                    total_duration_sec = float(tf) / fps
        if total_duration_sec is None and unique_chunk_ranges:
            total_duration_sec = max(end for _, end in unique_chunk_ranges)

    extent = 0.0
    if shown_frame_points:
        extent = max(shown_frame_points)
    for rs, re in unique_chunk_ranges:
        extent = max(extent, float(rs), float(re))

    true_dur = _qa_probe_true_duration_sec(video_path, video_data, fps)
    if true_dur is None or true_dur <= 0:
        if total_duration_sec and total_duration_sec > 0:
            true_dur = float(total_duration_sec)
        elif extent > 0:
            true_dur = float(extent)
        else:
            true_dur = 1.0

    axis_end = max(float(true_dur), float(extent), 1e-6)
    qa_true_duration_sec = float(true_dur)
    total_duration_sec = axis_end

    if layout_window:
        unique_chunk_ranges = [(0.0, axis_end)]
        chunk_ranges = [(0.0, axis_end) for _ in timestamps_sec]

    question = query_result["query"]
    metadata = query_result.get("metadata", {})
    is_mcq = metadata.get("qa_format") == "mcq"
    predicted_raw = query_result.get("predicted_answer", "")
    gt_raw = query_result.get("gt_answer", "")
    if is_mcq:
        predicted = (
            query_result.get("predicted_answer_letter")
            or extract_mcq_letter(predicted_raw)
            or str(predicted_raw)
        )
        gt = (
            query_result.get("gt_answer_letter")
            or extract_mcq_letter(gt_raw)
            or str(gt_raw)
        )
        stored_pred_letter = query_result.get("predicted_answer_letter")
        stored_gt_letter = query_result.get("gt_answer_letter")
        if stored_pred_letter and stored_gt_letter:
            is_correct = bool(stored_pred_letter == stored_gt_letter)
        else:
            valid_letters = "".join(
                chr(ord("A") + idx)
                for idx in range(len(metadata.get("options", []) or []))
            ) or "ABCD"
            is_correct = bool(
                mcq_letter_match(predicted_raw, gt_raw, valid_letters=valid_letters)
            )
    else:
        predicted = predicted_raw
        gt = gt_raw
        is_correct = bool(exact_match_accuracy(predicted_raw, gt_raw))

    return {
        "frame_images": frame_images,
        "timestamps_sec": timestamps_sec,
        "chunk_ranges": chunk_ranges,
        "shown_chunk_ranges": unique_chunk_ranges,
        "shown_frame_points": shown_frame_points,
        "total_duration_sec": total_duration_sec,
        "qa_true_duration_sec": qa_true_duration_sec,
        "question": question,
        "predicted": predicted,
        "gt": gt,
        "is_correct": is_correct,
        "is_mcq": is_mcq,
    }


def _qa_render_coverage_axis(ax, data, *, fs_tick=11):
    """Draw a compact timeline showing coverage across the full video."""
    total_duration_sec = float(data.get("total_duration_sec") or 0.0)
    shown_chunk_ranges = data.get("shown_chunk_ranges") or []
    shown_frame_points = data.get("shown_frame_points") or []

    if total_duration_sec <= 0.0 and shown_chunk_ranges:
        total_duration_sec = max(end for _, end in shown_chunk_ranges)
    if total_duration_sec <= 0.0:
        total_duration_sec = 1.0

    ax.set_xlim(0.0, total_duration_sec)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    band_y = 0.42
    band_h = 0.18
    ax.add_patch(
        Rectangle(
            (0.0, band_y),
            total_duration_sec,
            band_h,
            facecolor="#E5E7EB",
            edgecolor="#CBD5E1",
            linewidth=1.0,
        )
    )

    for start_sec, end_sec in shown_chunk_ranges:
        width = max(end_sec - start_sec, total_duration_sec * 0.0015)
        ax.add_patch(
            Rectangle(
                (start_sec, band_y),
                width,
                band_h,
                facecolor="#90CAF9",
                edgecolor="#1E88E5",
                linewidth=1.1,
                alpha=0.95,
            )
        )

    for ts_sec in shown_frame_points:
        ax.plot(
            [ts_sec, ts_sec],
            [band_y - 0.12, band_y + band_h + 0.12],
            color="#C62828",
            linewidth=1.1,
            alpha=0.95,
            zorder=5,
        )
        ax.scatter(
            [ts_sec],
            [band_y + band_h / 2.0],
            s=18,
            color="#C62828",
            zorder=6,
        )

    label_y = 0.88
    ax.text(
        0.0,
        label_y,
        "Coverage in full video",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_tick,
        fontweight="bold",
        color="#374151",
    )
    ax.text(
        0.0,
        0.10,
        "0.0s",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_tick,
        color="#4B5563",
    )
    ax.text(
        1.0,
        0.10,
        f"{total_duration_sec:.1f}s",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=fs_tick,
        color="#4B5563",
    )


def _qa_display_subset(data, n_display=6):
    """Keep up to *n_display* evenly spaced frames for thumbnail strip.

    Timeline fields (``shown_frame_points``, ``shown_chunk_ranges``) are unchanged
    so markers reflect every frame shown to the model, not only the subset.
    """
    n = len(data["frame_images"])
    if n == 0:
        return None
    if n <= n_display:
        return dict(data)
    raw = np.linspace(0, n - 1, n_display)
    idxs = np.unique(np.clip(np.round(raw).astype(int), 0, n - 1)).tolist()
    for j in range(n):
        if len(idxs) >= n_display:
            break
        if j not in idxs:
            idxs.append(j)
    idxs = sorted(idxs)[:n_display]
    return {
        **data,
        "frame_images": [data["frame_images"][i] for i in idxs],
        "timestamps_sec": [data["timestamps_sec"][i] for i in idxs],
        "chunk_ranges": [data["chunk_ranges"][i] for i in idxs],
    }


def _qa_render_coverage_axis_high_quality(
    ax, data, *, fs_tick=14, marker_size=120, line_width=2.8,
):
    """Timeline like `_qa_render_coverage_axis` but larger markers and no chunk/frame count."""
    total_duration_sec = float(data.get("total_duration_sec") or 0.0)
    shown_chunk_ranges = data.get("shown_chunk_ranges") or []
    shown_frame_points = data.get("shown_frame_points") or []

    if total_duration_sec <= 0.0 and shown_chunk_ranges:
        total_duration_sec = max(end for _, end in shown_chunk_ranges)
    if total_duration_sec <= 0.0:
        total_duration_sec = 1.0

    ax.set_xlim(0.0, total_duration_sec)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    band_y = 0.40
    band_h = 0.22
    ax.add_patch(
        Rectangle(
            (0.0, band_y),
            total_duration_sec,
            band_h,
            facecolor="#E5E7EB",
            edgecolor="#94A3B8",
            linewidth=1.2,
        )
    )

    for start_sec, end_sec in shown_chunk_ranges:
        width = max(end_sec - start_sec, total_duration_sec * 0.0015)
        ax.add_patch(
            Rectangle(
                (start_sec, band_y),
                width,
                band_h,
                facecolor="#90CAF9",
                edgecolor="#1565C0",
                linewidth=1.3,
                alpha=0.95,
            )
        )

    stem_reach = 0.20
    for ts_sec in shown_frame_points:
        ax.plot(
            [ts_sec, ts_sec],
            [band_y - stem_reach, band_y + band_h + stem_reach],
            color="#B71C1C",
            linewidth=line_width,
            alpha=0.95,
            zorder=5,
            solid_capstyle="round",
        )
        ax.scatter(
            [ts_sec],
            [band_y + band_h / 2.0],
            s=marker_size,
            color="#C62828",
            edgecolors="#1A1A1A",
            linewidths=max(1.5, line_width * 0.45),
            zorder=6,
        )

    ax.text(
        0.0,
        0.90,
        "Timeline",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=fs_tick + 4,
        fontweight="bold",
        color="#1F2937",
    )
    time_lbl_y = 0.02
    ax.text(
        0.0,
        time_lbl_y,
        "0:00",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fs_tick,
        color="#374151",
    )
    label_sec = float(
        data.get("qa_true_duration_sec") or total_duration_sec or 0.0,
    )
    end_lbl = _qa_format_duration_axis_label(label_sec)
    ax.text(
        1.0,
        time_lbl_y,
        end_lbl,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fs_tick,
        color="#374151",
    )


def _qa_render_high_quality_frames_row(
    ax_frames,
    data,
    *,
    target_h,
    fs_timestamp,
    n_total_frames=None,
    ellipsis_fs=96,
    ellipsis_note_fs=26,
):
    """Single row of frames with timestamp labels only (no chunk span text).

    If *n_total_frames* is greater than the number of thumbnails, reserve space
    on the right for a large ellipsis and a short “more frames” note.
    """
    frame_images = data["frame_images"]
    timestamps_sec = data["timestamps_sec"]

    resized = [_resize_frame(f, target_h) for f in frame_images]
    n_frames = len(resized)
    n_full = int(n_total_frames) if n_total_frames is not None else n_frames
    show_ellipsis = n_full > n_frames

    ax_frames.set_xlim(0, 1)
    ax_frames.set_ylim(0, 1)
    ax_frames.axis("off")

    left_margin = 0.02
    right_margin = 0.02
    ellipsis_w = 0.11 if show_ellipsis else 0.0
    card_gap = 0.012
    usable = (
        1.0
        - left_margin
        - right_margin
        - ellipsis_w
        - card_gap * max(0, n_frames - 1)
    )
    card_w = usable / max(n_frames, 1)
    ts_band = 0.22
    img_y0 = ts_band
    img_h = 1.0 - ts_band - 0.02

    for idx, frame in enumerate(resized):
        x0 = left_margin + idx * (card_w + card_gap)
        card_ax = ax_frames.inset_axes([x0, img_y0, card_w, img_h])
        card_ax.imshow(frame)
        card_ax.set_xticks([])
        card_ax.set_yticks([])
        for spine in card_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#37474F")
            spine.set_linewidth(1.2)

        ts_ax = ax_frames.inset_axes([x0, 0.0, card_w, ts_band - 0.02])
        ts_ax.axis("off")
        t = timestamps_sec[idx]
        ts_ax.text(
            0.5,
            0.5,
            f"{t:.1f} s",
            transform=ts_ax.transAxes,
            ha="center",
            va="center",
            fontsize=fs_timestamp,
            fontweight="bold",
            color="#111827",
        )

    if show_ellipsis:
        x_center = left_margin + usable + card_gap * max(0, n_frames - 1) + ellipsis_w / 2
        more = n_full - n_frames
        ax_frames.text(
            x_center,
            img_y0 + img_h * 0.52,
            "\u2026",
            transform=ax_frames.transAxes,
            ha="center",
            va="center",
            fontsize=ellipsis_fs,
            fontweight="bold",
            color="#111827",
            clip_on=False,
        )
        ax_frames.text(
            x_center,
            0.08,
            f"+{more} more frame{'s' if more != 1 else ''}",
            transform=ax_frames.transAxes,
            ha="center",
            va="center",
            fontsize=ellipsis_note_fs,
            color="#374151",
            clip_on=False,
        )


def _qa_render_high_quality_text_panels(
    ax_q, ax_a, data, *, fs_question, fs_answers, question_wrap=72,
):
    """Question and verdict rows with larger type."""
    question = data["question"]
    predicted = data["predicted"]
    gt = data["gt"]
    is_correct = data["is_correct"]
    is_mcq = data.get("is_mcq", False)

    ax_q.axis("off")
    display_q = textwrap.fill(
        _escape_matplotlib_text(question),
        width=question_wrap,
        break_long_words=False,
        break_on_hyphens=False,
    )
    ax_q.text(
        0.5,
        0.5,
        display_q,
        transform=ax_q.transAxes,
        ha="center",
        va="center",
        fontsize=fs_question,
        fontweight="normal",
        color="#111827",
        linespacing=1.45,
    )

    ax_a.axis("off")
    verdict_text = "CORRECT" if is_correct else "INCORRECT"
    verdict_color = "#1B5E20" if is_correct else "#B71C1C"
    answer_body = (
        f'Predicted{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(predicted)}"   ·   '
        f'Ground truth{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(gt)}"   ·   '
        f"{verdict_text}"
    )
    ax_a.text(
        0.5,
        0.5,
        answer_body,
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=fs_answers,
        fontweight="bold",
        color=verdict_color,
        linespacing=1.35,
    )


def plot_qa_high_quality_example(
    query_result,
    video_data,
    out_path,
    config=None,
    n_display=6,
    dpi=200,
):
    """Publication-style QA figure: six thumbnails, clean timeline, large question text."""
    data_full = _qa_collect_cells(query_result, video_data, config)
    if not data_full:
        return

    data_disp = _qa_display_subset(data_full, n_display)
    if not data_disp:
        return

    n_full = len(data_full["frame_images"])
    target_h = 320
    n_show = len(data_disp["frame_images"])
    frame_w = max(_resize_frame(f, target_h).shape[1] for f in data_disp["frame_images"])
    total_w = n_show * frame_w + (n_show - 1) * 10
    fig_w = max(15.0, total_w / 75 + (0.9 if n_full > n_show else 0))

    fig = plt.figure(figsize=(fig_w, 11.5), facecolor="white")
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[6.5, 1.55, 2.8, 1.35],
        hspace=0.30,
    )
    ax_frames = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])
    ax_q = fig.add_subplot(gs[2])
    ax_a = fig.add_subplot(gs[3])

    _qa_render_high_quality_frames_row(
        ax_frames,
        data_disp,
        target_h=target_h,
        fs_timestamp=26,
        n_total_frames=n_full,
        ellipsis_fs=110,
        ellipsis_note_fs=22,
    )
    _qa_render_coverage_axis_high_quality(
        ax_timeline,
        data_full,
        fs_tick=22,
        marker_size=420,
        line_width=5.2,
    )
    _qa_render_high_quality_text_panels(
        ax_q,
        ax_a,
        data_full,
        fs_question=30,
        fs_answers=22,
        question_wrap=max(48, min(88, int(fig_w * 3.6))),
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _qa_render_to_axes(
    ax_frames, ax_timeline, ax_q, ax_a, data, *, target_h, fs_question,
    fs_answers, fs_timestamp, fs_chunk_range, fs_timeline=11, question_wrap=0,
    multiline_answer=False, frame_subcaptions=True, question_fontstyle="italic",
    panel_fontfamily=None,
):
    """Draw frames + question + answer rows onto the given axes.

    *question_wrap*: if > 0, wrap question text to this many characters per line.
    *multiline_answer*: if True, stack predicted / ground truth / verdict on separate lines.
    *frame_subcaptions*: if False, omit chunk/time labels under each thumbnail.
    If *ax_timeline* is None, the coverage strip is skipped.
    *panel_fontfamily*: if set (e.g. ``'sans-serif'``), applied to question and answer text.
    """
    frame_images = data["frame_images"]
    timestamps_sec = data["timestamps_sec"]
    chunk_ranges = data["chunk_ranges"]
    question = data["question"]
    predicted = data["predicted"]
    gt = data["gt"]
    is_correct = data["is_correct"]
    is_mcq = data.get("is_mcq", False)

    resized = [_resize_frame(f, target_h) for f in frame_images]
    n_frames = len(resized)

    ax_frames.set_xlim(0, 1)
    ax_frames.set_ylim(0, 1)
    ax_frames.axis("off")

    left_margin = 0.01
    right_margin = 0.01
    card_gap = 0.008
    usable = 1 - left_margin - right_margin - card_gap * (n_frames - 1)
    card_w = usable / n_frames
    ts_band = 0.24 if frame_subcaptions else 0.0
    img_y0 = ts_band
    img_h = 1.0 - ts_band - 0.02

    for idx, frame in enumerate(resized):
        x0 = left_margin + idx * (card_w + card_gap)
        card_ax = ax_frames.inset_axes([x0, img_y0, card_w, img_h])
        card_ax.imshow(frame)
        card_ax.set_xticks([])
        card_ax.set_yticks([])
        for spine in card_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#4F5B66")
            spine.set_linewidth(1.0)

        if frame_subcaptions:
            ts_ax = ax_frames.inset_axes([x0, 0.0, card_w, ts_band - 0.01])
            ts_ax.axis("off")
            t = timestamps_sec[idx]
            t0, t1 = chunk_ranges[idx]
            ts_ax.text(
                0.5, 0.78, f"{t0:.1f}–{t1:.1f} s",
                transform=ts_ax.transAxes, ha="center", va="bottom",
                fontsize=fs_chunk_range, color="#555555",
            )
            ts_ax.text(
                0.5, 0.22, f"{t:.1f} s",
                transform=ts_ax.transAxes, ha="center", va="top",
                fontsize=fs_timestamp, color="#333333",
            )

    if ax_timeline is not None:
        _qa_render_coverage_axis(ax_timeline, data, fs_tick=fs_timeline)

    ax_q.axis("off")
    if question_wrap > 0:
        display_q = textwrap.fill(
            _escape_matplotlib_text(question), width=question_wrap, break_long_words=False,
            break_on_hyphens=False,
        )
        q_label = f'Question:\n"{display_q}"'
    else:
        safe_question = _escape_matplotlib_text(question)
        display_q = safe_question if len(safe_question) <= 120 else safe_question[:117] + "..."
        q_label = f'Question: "{display_q}"'
    q_text_kw = dict(
        transform=ax_q.transAxes, ha="center", va="center",
        fontsize=fs_question, fontstyle=question_fontstyle,
        linespacing=1.35,
    )
    if panel_fontfamily is not None:
        q_text_kw["family"] = panel_fontfamily
    ax_q.text(0.5, 0.5, q_label, **q_text_kw)

    ax_a.axis("off")
    verdict_text = "CORRECT" if is_correct else "INCORRECT"
    verdict_color = "#2E7D32" if is_correct else "#C62828"
    if multiline_answer:
        answer_body = (
            f'Predicted{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(predicted)}"\n'
            f'Ground truth{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(gt)}"\n'
            f"[{verdict_text}]"
        )
    else:
        answer_body = (
            f'Predicted{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(predicted)}"    '
            f'Ground truth{" letter" if is_mcq else ""}: "{_escape_matplotlib_text(gt)}"    [{verdict_text}]'
        )
    a_text_kw = dict(
        transform=ax_a.transAxes, ha="center", va="center",
        fontsize=fs_answers, fontweight="bold", color=verdict_color,
        linespacing=1.4,
    )
    if panel_fontfamily is not None:
        a_text_kw["family"] = panel_fontfamily
    ax_a.text(0.5, 0.5, answer_body, **a_text_kw)


def plot_qa_example(query_result, video_data, out_path, config=None):
    """Render a single QA example: frames + question + predicted vs GT answer."""
    data = _qa_collect_cells(query_result, video_data, config)
    if not data:
        return

    n_frames = len(data["frame_images"])
    target_h = 220
    frame_w = max(_resize_frame(f, target_h).shape[1] for f in data["frame_images"])
    total_w = n_frames * frame_w + (n_frames - 1) * 6
    fig_w = max(12, total_w / 72)

    fig = plt.figure(figsize=(fig_w, 6.9))
    gs = fig.add_gridspec(4, 1, height_ratios=[7.2, 1.15, 1.0, 1.0], hspace=0.22)
    ax_frames = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])
    ax_q = fig.add_subplot(gs[2])
    ax_a = fig.add_subplot(gs[3])

    _qa_render_to_axes(
        ax_frames, ax_timeline, ax_q, ax_a, data,
        target_h=target_h,
        fs_question=18,
        fs_answers=18,
        fs_timestamp=13,
        fs_chunk_range=12,
        fs_timeline=12,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qa_example_with_prompt(query_result, video_data, out_path, config=None):
    """Like plot_qa_example but adds a panel showing the VLM prompt text."""
    data = _qa_collect_cells(query_result, video_data, config)
    if not data:
        return

    prompt_text = query_result.get("vlm_prompt_text", "")

    n_frames = len(data["frame_images"])
    target_h = 220
    frame_w = max(_resize_frame(f, target_h).shape[1] for f in data["frame_images"])
    total_w = n_frames * frame_w + (n_frames - 1) * 6
    fig_w = max(12, total_w / 72)

    has_prompt = bool(prompt_text.strip())
    if has_prompt:
        fig = plt.figure(figsize=(fig_w, 12.0))
        gs = fig.add_gridspec(
            5, 1, height_ratios=[7.2, 1.15, 1.0, 1.0, 6.0], hspace=0.25,
        )
    else:
        fig = plt.figure(figsize=(fig_w, 6.9))
        gs = fig.add_gridspec(4, 1, height_ratios=[7.2, 1.15, 1.0, 1.0], hspace=0.22)

    ax_frames = fig.add_subplot(gs[0])
    ax_timeline = fig.add_subplot(gs[1])
    ax_q = fig.add_subplot(gs[2])
    ax_a = fig.add_subplot(gs[3])

    _qa_render_to_axes(
        ax_frames, ax_timeline, ax_q, ax_a, data,
        target_h=target_h,
        fs_question=18,
        fs_answers=18,
        fs_timestamp=13,
        fs_chunk_range=12,
        fs_timeline=12,
    )

    if has_prompt:
        ax_prompt = fig.add_subplot(gs[4])
        ax_prompt.axis("off")
        wrapped = textwrap.fill(
            _escape_matplotlib_text(prompt_text),
            width=120,
            break_long_words=False,
            break_on_hyphens=False,
        )
        ax_prompt.text(
            0.02, 0.98, "VLM Prompt Text:",
            transform=ax_prompt.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color="#1A237E",
        )
        ax_prompt.text(
            0.02, 0.92, wrapped,
            transform=ax_prompt.transAxes, ha="left", va="top",
            fontsize=8, fontfamily="monospace", color="#333333",
            linespacing=1.3,
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qa_example_clean(query_result, video_data, out_path, config=None):
    """Publication-style QA strip: question above frames, sampling timeline, verdict.

    No per-frame captions under thumbnails; uses sans-serif stack (Calibri when installed).
    """
    data = _qa_collect_cells(query_result, video_data, config)
    if not data:
        return

    n_frames = len(data["frame_images"])
    target_h = 220
    frame_w = max(_resize_frame(f, target_h).shape[1] for f in data["frame_images"])
    total_w = n_frames * frame_w + (n_frames - 1) * 6
    fig_w = max(12, total_w / 72)

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "DejaVu Sans"],
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, 7.0))
        gs = fig.add_gridspec(
            4,
            1,
            height_ratios=[1.35, 7.2, 1.15, 1.1],
            hspace=0.22,
        )
        ax_q = fig.add_subplot(gs[0])
        ax_frames = fig.add_subplot(gs[1])
        ax_timeline = fig.add_subplot(gs[2])
        ax_a = fig.add_subplot(gs[3])
        q_wrap = max(48, min(96, int(fig_w * 4)))
        _qa_render_to_axes(
            ax_frames,
            ax_timeline,
            ax_q,
            ax_a,
            data,
            target_h=target_h,
            fs_question=26,
            fs_answers=18,
            fs_timestamp=13,
            fs_chunk_range=12,
            fs_timeline=12,
            question_wrap=q_wrap,
            multiline_answer=True,
            frame_subcaptions=False,
            question_fontstyle="normal",
            panel_fontfamily="sans-serif",
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_qa_stacked_correct_incorrect(
    correct_results,
    incorrect_results,
    per_video_data,
    out_path,
    config=None,
    title=None,
):
    """Stack the same QA layout as plot_qa_example: correct rows then incorrect rows.

    Callers should pass at most three correct and three incorrect query results.
    """
    correct_data = []
    for qr in correct_results:
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        if not vdata:
            continue
        d = _qa_collect_cells(qr, vdata, config)
        if d:
            correct_data.append(d)

    incorrect_data = []
    for qr in incorrect_results:
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        if not vdata:
            continue
        d = _qa_collect_cells(qr, vdata, config)
        if d:
            incorrect_data.append(d)

    rows_spec = []
    if correct_data:
        rows_spec.append(("header", "Correct examples", "#2E7D32"))
        for d in correct_data:
            rows_spec.append(("block", d))
    if incorrect_data:
        rows_spec.append(("header", "Incorrect examples", "#C62828"))
        for d in incorrect_data:
            rows_spec.append(("block", d))

    if not rows_spec:
        return

    # Tall figure: frame row weighted heavily vs question/answer text.
    inner_ratios = [8.0, 1.35, 2.5, 2.5]
    n_blocks = sum(1 for r in rows_spec if r[0] == "block")
    n_headers = sum(1 for r in rows_spec if r[0] == "header")
    block_h_in = 7.6
    header_h_in = 0.55
    fig_h = n_headers * header_h_in + n_blocks * block_h_in + 0.6

    all_block_data = [r[1] for r in rows_spec if r[0] == "block"]
    max_frames = max(len(d["frame_images"]) for d in all_block_data)
    fig_w = max(16.0, min(52.0, 15.0 + max_frames * 1.28))

    height_ratios = [
        0.14 if kind == "header" else 1.0 for kind, *_ in rows_spec
    ]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(rows_spec), 1, height_ratios=height_ratios, hspace=0.62)

    for row, spec in enumerate(rows_spec):
        kind = spec[0]
        if kind == "header":
            _, title, color = spec
            ax_h = fig.add_subplot(gs[row, 0])
            ax_h.axis("off")
            ax_h.text(
                0.02, 0.5, title,
                transform=ax_h.transAxes, ha="left", va="center",
                fontsize=16, fontweight="bold", color=color,
            )
        else:
            _, data = spec
            sub = gs[row, 0].subgridspec(4, 1, height_ratios=inner_ratios, hspace=0.48)
            ax_f = fig.add_subplot(sub[0, 0])
            ax_t = fig.add_subplot(sub[1, 0])
            ax_q = fig.add_subplot(sub[2, 0])
            ax_a = fig.add_subplot(sub[3, 0])
            _qa_render_to_axes(
                ax_f, ax_t, ax_q, ax_a, data,
                target_h=220,
                fs_question=16,
                fs_answers=16,
                fs_timestamp=13,
                fs_chunk_range=12,
                fs_timeline=11,
                question_wrap=96,
                multiline_answer=True,
            )

    fig.suptitle(
        title or "VLM QA: stacked examples (same layout as qa_example.png)",
        fontsize=14, fontweight="bold", y=0.998,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.45)
    plt.close(fig)


def _qa_is_correct(query_result):
    """Return whether a QA query result is correct."""
    pred = query_result.get("predicted_answer", "")
    gt = query_result.get("gt_answer", "")
    if query_result.get("metadata", {}).get("qa_format") == "mcq":
        pred_letter = query_result.get("predicted_answer_letter")
        gt_letter = query_result.get("gt_answer_letter")
        if pred_letter and gt_letter:
            return pred_letter == gt_letter
        valid_letters = "".join(
            chr(ord("A") + idx)
            for idx in range(len(query_result.get("metadata", {}).get("options", []) or []))
        ) or "ABCD"
        return mcq_letter_match(pred, gt, valid_letters=valid_letters)
    return exact_match_accuracy(pred, gt)


def _qa_example_renderable(query_result, video_data, config):
    """Return True if we can build cells for this query (video on disk, etc.)."""
    vpath = query_result["video_path"]
    if not os.path.isfile(vpath):
        return False
    if video_data is None and not query_result.get("vlm_prompt_metadata"):
        return False
    return bool(_qa_collect_cells(query_result, video_data, config))


def _pick_single_qa_example(
    query_results,
    per_video_data,
    config,
    run_dir,
    *,
    duration_label=None,
    exclude_sample_ids=None,
):
    """Pick one query for single-example QA figures.

    Order: optional env ``QA_HIGH_QUALITY_SAMPLE_ID``, optional
    ``output.qa_high_quality_sample_id``, then first ``sample_id`` in
    ``subset_ids.json`` (run order), else first row in ``query_results`` that
    renders. ``exclude_sample_ids`` skips rows (e.g. avoid duplicating the
    default example in the long-only figure).
    """
    dl = None
    if duration_label is not None:
        dl = str(duration_label).lower()

    excluded = {str(x) for x in (exclude_sample_ids or []) if x is not None}

    def _vdata_for(qr):
        vpath = qr["video_path"]
        if not per_video_data:
            return None
        return per_video_data.get(vpath)

    def _try_qr(qr):
        sid = str(qr.get("sample_id", ""))
        if sid and sid in excluded:
            return None, None
        vpath = qr["video_path"]
        if not os.path.isfile(vpath):
            return None, None
        if per_video_data and vpath not in per_video_data:
            return None, None
        if dl is not None:
            qdl = str(qr.get("metadata", {}).get("duration_label", "")).lower()
            if qdl != dl:
                return None, None
        vd = _vdata_for(qr)
        if per_video_data and vd is None:
            return None, None
        if not _qa_example_renderable(qr, vd, config):
            return None, None
        return qr, vd

    preferred = os.environ.get("QA_HIGH_QUALITY_SAMPLE_ID")
    if not preferred and config is not None:
        try:
            out = config.get("output")
            if out is not None:
                preferred = out.get("qa_high_quality_sample_id")
        except (KeyError, TypeError, AttributeError):
            preferred = None
    if preferred:
        for qr in query_results:
            if str(qr.get("sample_id")) != str(preferred):
                continue
            got = _try_qr(qr)
            if got[0] is not None:
                return got

    subset_path = os.path.join(run_dir, "subset_ids.json")
    if os.path.isfile(subset_path):
        with open(subset_path) as f:
            subset_ids = json.load(f)
        for sid in subset_ids:
            for qr in query_results:
                if str(qr.get("sample_id")) != str(sid):
                    continue
                got = _try_qr(qr)
                if got[0] is not None:
                    return got

    for qr in query_results:
        got = _try_qr(qr)
        if got[0] is not None:
            return got

    return None, None


def _select_qa_examples(query_results, per_video_data, *, duration_label=None):
    """Select up to three correct and three incorrect examples."""
    correct = []
    incorrect = []
    for qr in query_results:
        vpath = qr["video_path"]
        if not os.path.isfile(vpath):
            continue
        if not per_video_data or vpath not in per_video_data:
            continue
        if duration_label is not None:
            qr_duration = str(qr.get("metadata", {}).get("duration_label", "")).lower()
            if qr_duration != str(duration_label).lower():
                continue
        if _qa_is_correct(qr):
            if len(correct) < 3:
                correct.append(qr)
        else:
            if len(incorrect) < 3:
                incorrect.append(qr)
        if len(correct) >= 3 and len(incorrect) >= 3:
            break
    return correct, incorrect


def _select_qa_examples_n(query_results, per_video_data, config=None, n=10):
    """Select up to *n* distinct renderable examples, alternating correct/incorrect."""
    correct = []
    incorrect = []
    seen = set()
    for qr in query_results:
        vpath = qr["video_path"]
        if not os.path.isfile(vpath):
            continue
        if not per_video_data or vpath not in per_video_data:
            continue
        sid = qr.get("sample_id")
        if sid is not None:
            dedupe = ("sid", str(sid))
        else:
            dedupe = ("vq", vpath, str(qr.get("query", "")))
        if dedupe in seen:
            continue
        vd = per_video_data[vpath]
        if not _qa_example_renderable(qr, vd, config):
            continue
        seen.add(dedupe)
        if _qa_is_correct(qr):
            correct.append(qr)
        else:
            incorrect.append(qr)

    selected = []
    ic = ii = 0
    take_correct = True
    while len(selected) < n:
        if take_correct and ic < len(correct):
            selected.append(correct[ic])
            ic += 1
        elif not take_correct and ii < len(incorrect):
            selected.append(incorrect[ii])
            ii += 1
        elif ic < len(correct):
            selected.append(correct[ic])
            ic += 1
        elif ii < len(incorrect):
            selected.append(incorrect[ii])
            ii += 1
        else:
            break
        take_correct = not take_correct
    return selected


def generate_qa_plots(run_dir, query_results, per_video_data, config=None):
    """Generate QA-specific plots for a run directory.

    Writes ``qa_example.png``, ``qa_high_quality_example.png``,
    ``qa_high_quality_example_long.png`` (when a long-duration row exists),
    ``qa_example_clean_01.png`` … ``qa_example_clean_10.png``, and stacked figures
    under ``<run_dir>/plots/``.

    Single-example plots use :func:`_pick_single_qa_example`: optional
    ``output.qa_high_quality_sample_id``, else the first renderable row whose
    ``sample_id`` appears in ``subset_ids.json`` (subset order), else the first
    renderable row in ``per_query_results``.
    """
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    example_qr, example_vdata = _pick_single_qa_example(
        query_results, per_video_data, config, run_dir,
    )
    if example_qr is not None:
        plot_qa_example(
            example_qr,
            example_vdata,
            os.path.join(plots_dir, "qa_example.png"),
            config=config,
        )
        if example_qr.get("vlm_prompt_text"):
            plot_qa_example_with_prompt(
                example_qr,
                example_vdata,
                os.path.join(plots_dir, "qa_example_with_prompt.png"),
                config=config,
            )
        plot_qa_high_quality_example(
            example_qr,
            example_vdata,
            os.path.join(plots_dir, "qa_high_quality_example.png"),
            config=config,
        )

    long_exclude = []
    if example_qr is not None and example_qr.get("sample_id") is not None:
        long_exclude.append(example_qr["sample_id"])
    long_qr, long_vdata = _pick_single_qa_example(
        query_results, per_video_data, config, run_dir,
        duration_label="long",
        exclude_sample_ids=long_exclude,
    )
    if long_qr is None:
        long_qr, long_vdata = _pick_single_qa_example(
            query_results, per_video_data, config, run_dir,
            duration_label="long",
        )
    if long_qr is not None:
        plot_qa_high_quality_example(
            long_qr,
            long_vdata,
            os.path.join(plots_dir, "qa_high_quality_example_long.png"),
            config=config,
        )

    correct, incorrect = _select_qa_examples(query_results, per_video_data)

    if correct or incorrect:
        plot_qa_stacked_correct_incorrect(
            correct,
            incorrect,
            per_video_data,
            os.path.join(plots_dir, "qa_stacked_examples.png"),
            config=config,
            title="VLM QA: stacked examples (same layout as qa_example.png)",
        )

    for duration_label in ("short", "medium", "long"):
        correct, incorrect = _select_qa_examples(
            query_results,
            per_video_data,
            duration_label=duration_label,
        )
        if not correct and not incorrect:
            continue
        plot_qa_stacked_correct_incorrect(
            correct,
            incorrect,
            per_video_data,
            os.path.join(plots_dir, f"qa_stacked_examples_{duration_label}.png"),
            config=config,
            title=(
                f"VLM QA: {duration_label.capitalize()} stacked examples "
                "(same layout as qa_example.png)"
            ),
        )

    clean_examples = _select_qa_examples_n(
        query_results, per_video_data, config=config, n=10,
    )
    for i, qr in enumerate(clean_examples, start=1):
        vdata = per_video_data.get(qr["video_path"]) if per_video_data else None
        if not vdata:
            continue
        plot_qa_example_clean(
            qr,
            vdata,
            os.path.join(plots_dir, f"qa_example_clean_{i:02d}.png"),
            config=config,
        )

    # --- Iterative QA progression plots ---
    has_iterative = any(
        qr.get("vlm_prompt_metadata", {}).get("iterative_rounds")
        for qr in query_results
    )
    if has_iterative:
        iterative_qrs = [
            qr for qr in query_results
            if qr.get("vlm_prompt_metadata", {}).get("iterative_rounds")
            and os.path.isfile(qr["video_path"])
        ]
        if iterative_qrs:
            first_qr = iterative_qrs[0]
            first_vdata = (
                per_video_data.get(first_qr["video_path"])
                if per_video_data else None
            )
            plot_iterative_progression(
                first_qr, first_vdata,
                os.path.join(plots_dir, "iterative_progression.png"),
                config=config,
            )

            plot_iterative_stacked(
                iterative_qrs, per_video_data,
                os.path.join(plots_dir, "iterative_stacked.png"),
                config=config,
                title="Iterative QA: query progression and frame retrieval",
            )

    return plots_dir


def generate_disagreement_plots(
    run_a_dir,
    run_b_dir,
    out_dir,
    label_a="Run A",
    label_b="Run B",
    max_examples=3,
    duration_label=None,
):
    """Generate plots for samples where one run is correct and the other is wrong.

    Produces two files:
      - ``{label_a}_wins.png``  — samples where A is correct, B is wrong
      - ``{label_b}_wins.png``  — samples where B is correct, A is wrong
    Each file shows the *winning* run's frames for up to ``max_examples`` cases.
    """
    import json, yaml

    def _load_run(run_dir):
        with open(os.path.join(run_dir, "per_query_results.json")) as f:
            qr = json.load(f)
        pvd_path = os.path.join(run_dir, "per_video_data.json")
        pvd = {}
        if os.path.isfile(pvd_path):
            with open(pvd_path) as f:
                pvd = json.load(f)
        cfg_path = os.path.join(run_dir, "config.yaml")
        cfg = None
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        return qr, pvd, cfg

    qr_a, pvd_a, cfg_a = _load_run(run_a_dir)
    qr_b, pvd_b, cfg_b = _load_run(run_b_dir)

    map_a = {r["sample_id"]: r for r in qr_a}
    map_b = {r["sample_id"]: r for r in qr_b}

    a_wins = []
    b_wins = []
    for sid in map_a:
        ra, rb = map_a[sid], map_b.get(sid)
        if rb is None:
            continue
        if duration_label:
            dl = str(ra.get("metadata", {}).get("duration_label", "")).lower()
            if dl != duration_label.lower():
                continue
        a_ok = _qa_is_correct(ra)
        b_ok = _qa_is_correct(rb)
        if a_ok and not b_ok:
            a_wins.append(sid)
        elif b_ok and not a_ok:
            b_wins.append(sid)

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{duration_label}" if duration_label else ""

    for winners, win_label, win_map, win_pvd, win_cfg, lose_label in [
        (a_wins, label_a, map_a, pvd_a, cfg_a, label_b),
        (b_wins, label_b, map_b, pvd_b, cfg_b, label_a),
    ]:
        if not winners:
            continue
        correct_qrs = [win_map[sid] for sid in winners[:max_examples]]
        title = (
            f"{win_label} correct, {lose_label} wrong "
            f"({len(winners)} total disagreements"
            f"{', ' + duration_label if duration_label else ''})"
        )
        safe_name = win_label.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}_wins{suffix}.png")
        plot_qa_stacked_correct_incorrect(
            correct_qrs, [], win_pvd, out_path, config=win_cfg, title=title,
        )
        print(f"  Wrote {out_path} ({len(correct_qrs)} examples)")

    return out_dir


# ---------------------------------------------------------------------------
# Iterative QA progression plot
# ---------------------------------------------------------------------------

_ROUND_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4"]


def _load_frames_at_times(video_path, frame_times, fps, target_height=120):
    """Load and resize frames at given timestamps."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    for t in frame_times:
        idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize_frame(frame, target_height))
        else:
            frames.append(None)
    cap.release()
    return frames


def plot_iterative_progression(query_result, video_data, out_path, config=None):
    """Plot the iterative query progression for a single sample.

    Shows one row per round (initial + each query), with frame thumbnails
    and a timeline at the bottom showing all frame positions color-coded
    by round.
    """
    meta = query_result.get("vlm_prompt_metadata", {})
    rounds = meta.get("iterative_rounds", [])
    if not rounds:
        return

    video_path = query_result["video_path"]
    if not os.path.isfile(video_path):
        return

    fps = video_data.get("fps", 30.0) if video_data else 30.0
    duration = video_data.get("duration", 0) if video_data else 0
    if duration <= 0:
        all_times = meta.get("all_frame_times", [])
        duration = max(all_times) + 1.0 if all_times else 100.0

    question = query_result.get("query", "")
    predicted = query_result.get("predicted_answer", "")
    gt = query_result.get("gt_answer", "")
    is_correct = _qa_is_correct(query_result)

    n_rounds = len(rounds)
    thumb_h = 100
    max_thumbs_per_row = 8

    fig_width = 16
    row_height = 1.6
    header_height = 0.8
    timeline_height = 1.0
    fig_height = header_height + n_rounds * row_height + timeline_height + 0.4

    fig, axes = plt.subplots(
        n_rounds + 2, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={
            "height_ratios": [header_height] + [row_height] * n_rounds + [timeline_height],
            "hspace": 0.15,
        },
    )

    # --- Header ---
    ax_header = axes[0]
    ax_header.axis("off")
    q_short = question[:120] + ("..." if len(question) > 120 else "")
    color = "#2E7D32" if is_correct else "#C62828"
    status = "CORRECT" if is_correct else "INCORRECT"
    ax_header.text(
        0.0, 0.6, f"Q: {q_short}",
        fontsize=9, va="center", ha="left", wrap=True,
        transform=ax_header.transAxes,
    )
    ax_header.text(
        0.0, 0.1,
        f"Predicted: {predicted[:60]}  |  GT: {gt[:60]}  [{status}]",
        fontsize=8, va="center", ha="left", color=color,
        fontweight="bold", transform=ax_header.transAxes,
    )

    # --- Per-round rows ---
    for r_idx, rd in enumerate(rounds):
        ax = axes[r_idx + 1]
        ax.axis("off")

        round_num = rd["round"]
        round_color = _ROUND_COLORS[round_num % len(_ROUND_COLORS)]

        if rd["type"] == "initial":
            label = f"Round 0 — Initial selection"
            frame_times = rd.get("frame_times", [])
        else:
            query_text = rd.get("query", "?")
            label = f'Round {round_num} — "{query_text}"'
            frame_times = rd.get("new_frame_times", [])

        n_new = len(frame_times)
        show_times = frame_times[:max_thumbs_per_row]
        frames = _load_frames_at_times(video_path, show_times, fps, target_height=thumb_h)

        ax.text(
            0.0, 0.95,
            label + f"  ({n_new} frames)",
            fontsize=8, va="top", ha="left",
            fontweight="bold", color=round_color,
            transform=ax.transAxes,
        )

        if frames:
            valid = [(f, t) for f, t in zip(frames, show_times) if f is not None]
            if valid:
                total_w = sum(f.shape[1] for f, _ in valid) + 4 * (len(valid) - 1)
                x_start = 0.0
                ax_pixel_w = fig_width * fig.dpi
                for frame, t in valid:
                    fw = frame.shape[1]
                    x_frac = x_start / max(total_w, 1)
                    w_frac = fw / max(total_w, 1)
                    x_frac = min(x_frac * 0.95, 0.95)
                    w_frac = min(w_frac * 0.95, 0.95 - x_frac)

                    inset = ax.inset_axes([x_frac, 0.0, w_frac, 0.75])
                    inset.imshow(frame)
                    inset.set_xticks([])
                    inset.set_yticks([])
                    for spine in inset.spines.values():
                        spine.set_color(round_color)
                        spine.set_linewidth(1.5)
                    inset.set_title(f"{t:.0f}s", fontsize=6, pad=1)
                    x_start += fw + 4

    # --- Timeline ---
    ax_tl = axes[-1]
    ax_tl.set_xlim(0, duration)
    ax_tl.set_ylim(-0.5, n_rounds - 0.5)
    ax_tl.set_xlabel("Video time (s)", fontsize=8)
    ax_tl.set_yticks(range(n_rounds))
    round_labels = []
    for rd in rounds:
        if rd["type"] == "initial":
            round_labels.append("Initial")
        else:
            q = rd.get("query", "?")
            round_labels.append(f'Q{rd["round"]}: {q[:25]}')
    ax_tl.set_yticklabels(round_labels, fontsize=7)
    ax_tl.tick_params(axis="x", labelsize=7)

    for r_idx, rd in enumerate(rounds):
        round_color = _ROUND_COLORS[rd["round"] % len(_ROUND_COLORS)]
        if rd["type"] == "initial":
            times = rd.get("frame_times", [])
        else:
            times = rd.get("new_frame_times", [])
        for t in times:
            ax_tl.plot(
                t, r_idx, "|", color=round_color,
                markersize=12, markeredgewidth=2,
            )

    ax_tl.axhline(y=-0.5, color="gray", linewidth=0.5)
    for r_idx in range(n_rounds):
        ax_tl.axhline(y=r_idx + 0.5, color="#E0E0E0", linewidth=0.5)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_iterative_stacked(query_results, per_video_data, out_path, config=None,
                           max_examples=4, title=None):
    """Plot iterative progression for multiple examples stacked vertically."""
    candidates = []
    for qr in query_results:
        meta = qr.get("vlm_prompt_metadata", {})
        if not meta.get("iterative_rounds"):
            continue
        vpath = qr["video_path"]
        if not os.path.isfile(vpath):
            continue
        candidates.append(qr)
        if len(candidates) >= max_examples * 3:
            break

    correct = [c for c in candidates if _qa_is_correct(c)]
    incorrect = [c for c in candidates if not _qa_is_correct(c)]
    selected = correct[:max_examples // 2] + incorrect[:max_examples // 2]
    if len(selected) < max_examples:
        remaining = [c for c in candidates if c not in selected]
        selected.extend(remaining[:max_examples - len(selected)])
    if not selected:
        return

    n_examples = len(selected)
    fig_height_per = 6.0
    fig, all_axes = plt.subplots(
        n_examples, 1,
        figsize=(16, fig_height_per * n_examples),
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)

    for ex_idx, qr in enumerate(selected):
        ax = all_axes[ex_idx, 0]
        ax.axis("off")

        meta = qr.get("vlm_prompt_metadata", {})
        rounds = meta.get("iterative_rounds", [])
        vpath = qr["video_path"]
        vdata = per_video_data.get(vpath, {}) if per_video_data else {}
        fps = vdata.get("fps", 30.0)
        duration = vdata.get("duration", 0)
        if duration <= 0:
            all_times = meta.get("all_frame_times", [])
            duration = max(all_times) + 1.0 if all_times else 100.0

        question = qr.get("query", "")[:100]
        predicted = qr.get("predicted_answer", "")[:40]
        gt = qr.get("gt_answer", "")[:40]
        is_correct = _qa_is_correct(qr)
        color = "#2E7D32" if is_correct else "#C62828"
        status = "CORRECT" if is_correct else "WRONG"

        n_rounds = len(rounds)
        y_top = 0.95
        ax.text(0.0, y_top, f"Q: {question}", fontsize=8, va="top",
                transform=ax.transAxes, fontweight="bold")
        ax.text(0.0, y_top - 0.06,
                f"Pred: {predicted}  |  GT: {gt}  [{status}]",
                fontsize=7, va="top", color=color, transform=ax.transAxes)

        tl_bottom = 0.02
        tl_height = 0.25
        tl_ax = ax.inset_axes([0.05, tl_bottom, 0.9, tl_height])
        tl_ax.set_xlim(0, duration)
        tl_ax.set_ylim(-0.5, n_rounds - 0.5)
        tl_ax.set_xlabel("time (s)", fontsize=6)
        tl_ax.tick_params(labelsize=6)

        round_labels = []
        for rd in rounds:
            if rd["type"] == "initial":
                round_labels.append("Initial")
            else:
                q = rd.get("query", "?")
                round_labels.append(f'Q{rd["round"]}: {q[:20]}')
        tl_ax.set_yticks(range(n_rounds))
        tl_ax.set_yticklabels(round_labels, fontsize=6)

        for r_idx, rd in enumerate(rounds):
            rc = _ROUND_COLORS[rd["round"] % len(_ROUND_COLORS)]
            times = rd.get("frame_times", []) if rd["type"] == "initial" else rd.get("new_frame_times", [])
            for t in times:
                tl_ax.plot(t, r_idx, "|", color=rc, markersize=10, markeredgewidth=2)

        for r_idx in range(n_rounds):
            tl_ax.axhline(y=r_idx + 0.5, color="#E0E0E0", linewidth=0.5)

        queries_text = []
        for rd in rounds:
            if rd["type"] == "initial":
                queries_text.append(f"R0: Initial ({rd.get('n_frames', '?')} frames)")
            else:
                queries_text.append(f"R{rd['round']}: \"{rd.get('query', '?')}\" ({rd.get('n_new_frames', '?')} frames)")

        text_y = y_top - 0.14
        for qt in queries_text:
            ax.text(0.0, text_y, qt, fontsize=7, va="top",
                    transform=ax.transAxes, family="monospace")
            text_y -= 0.05

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
