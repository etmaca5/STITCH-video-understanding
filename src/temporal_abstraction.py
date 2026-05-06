"""Temporal abstraction layer: chunks -> selected frames + metadata -> VLM prompt."""

from functools import lru_cache
import json
import logging
import re

import cv2
import numpy as np
import torch
from torch.nn.functional import cosine_similarity as torch_cos_sim

from vlm_client import VIDEO_INPUT_MODE, VLMClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame selection strategies
# ---------------------------------------------------------------------------

def _select_middle_frame(video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
    """Return the middle frame (RGB) of a chunk."""
    mid = (start_frame + end_frame) // 2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            f"Could not read frame {mid} from {video_path}"
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_frames_by_index(video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
    """Load explicit frame indices from a video."""
    if not frame_indices:
        return []
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(
            f"Could not read any requested frames from {video_path}: {frame_indices}"
        )
    return frames


def _sample_interior_frame_indices(
    start_frame: int, end_frame: int, n: int
) -> np.ndarray:
    """Return ``n`` chunk-relative frame indices excluding the outer bounds."""
    n = max(int(n), 1)
    max_frame = max(start_frame, end_frame - 1)
    positions = np.arange(1, n + 1, dtype=float) / (n + 1)
    indices = start_frame + positions * max(end_frame - start_frame, 0)
    return np.clip(indices.astype(int), start_frame, max_frame)


def _select_multi_frames(video_path: str, start_frame: int, end_frame: int,
                         n: int) -> list[np.ndarray]:
    """Return *n* interior uniformly-spaced frames (RGB) from a chunk."""
    indices = _sample_interior_frame_indices(start_frame, end_frame, n)
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
            f"Could not read any frames from {video_path} [{start_frame}-{end_frame}]"
        )
    return frames


def _select_best_window_frames(video_path: str, start_frame: int, end_frame: int,
                               frame_indices: list[int] | None = None,
                               n: int = 1) -> list[np.ndarray]:
    """Return the precomputed best-window frames for a chunk."""
    if not frame_indices:
        raise ValueError(
            "best_window frame selection requires explicit frame_indices metadata"
        )
    return _load_frames_by_index(video_path, frame_indices[: max(int(n), 1)])


FRAME_SELECTORS = {
    "middle": lambda vp, s, e, **kw: [_select_middle_frame(vp, s, e)],
    "multi":  lambda vp, s, e, **kw: _select_multi_frames(vp, s, e, kw.get("n", 3)),
    "best_window": lambda vp, s, e, **kw: _select_best_window_frames(
        vp, s, e, frame_indices=kw.get("frame_indices"), n=kw.get("n", 1)
    ),
}


def select_frames(video_path: str, chunk: tuple[int, int],
                   method: str = "middle", **kwargs) -> list[np.ndarray]:
    """Select representative frame(s) from a chunk.

    Args:
        video_path: Path to the video file.
        chunk: (start_frame, end_frame) tuple.
        method: Selection strategy name (see FRAME_SELECTORS).

    Returns:
        List of RGB numpy arrays.
    """
    if method not in FRAME_SELECTORS:
        raise ValueError(
            f"Unknown frame selection method: {method}. "
            f"Available: {list(FRAME_SELECTORS)}"
        )
    return FRAME_SELECTORS[method](video_path, chunk[0], chunk[1], **kwargs)


def compute_allowed_chunks(
    max_chunks: int,
    frames_per_chunk: int,
    max_images_per_request: int | None,
) -> int:
    """Return the maximum number of chunks that fit in one VLM prompt."""
    max_chunks = int(max_chunks)
    frames_per_chunk = int(frames_per_chunk)
    if max_chunks < 0:
        raise ValueError("max_chunks must be >= 0")
    if max_chunks == 0:
        return 0
    if frames_per_chunk < 1:
        raise ValueError("frames_per_chunk must be >= 1")
    if max_images_per_request is None:
        return max_chunks

    max_images_per_request = int(max_images_per_request)
    if max_images_per_request < 1:
        raise ValueError("max_images_per_request must be >= 1 when provided")
    if frames_per_chunk > max_images_per_request:
        raise ValueError(
            "frames_per_chunk exceeds the provider image budget. "
            f"Need {frames_per_chunk} images per chunk but the VLM limit is "
            f"{max_images_per_request}. Reduce frames_per_chunk or change VLM config."
        )

    return min(max_chunks, max_images_per_request // frames_per_chunk)


# ---------------------------------------------------------------------------
# Metadata config helpers
# ---------------------------------------------------------------------------

def _resolve_metadata_flag(metadata_config: dict, key: str) -> bool:
    """Resolve a metadata flag considering the ``include_metadata`` master toggle.

    When *key* is explicitly set in *metadata_config* it wins.  Otherwise
    the value of ``include_metadata`` is used (defaulting to ``True``).
    """
    explicit = metadata_config.get(key)
    if explicit is not None:
        return bool(explicit)
    return bool(metadata_config.get("include_metadata", True))


def _collect_per_chunk_subtitles(
    subtitle_path: str | None,
    chunks: list[tuple[int, int]],
    fps: float,
    start_offset_sec: float = 0.0,
) -> list[str]:
    """Return one combined subtitle string per chunk."""
    if not subtitle_path:
        return [""] * len(chunks)

    try:
        entries = _load_subtitle_entries(subtitle_path)
    except OSError:
        return [""] * len(chunks)

    result: list[str] = []
    for sf, ef in chunks:
        chunk_start = sf / fps
        chunk_end = ef / fps
        texts: list[str] = []
        for start_sec, end_sec, text in entries:
            adj_start = start_sec - start_offset_sec
            adj_end = end_sec - start_offset_sec
            if adj_start <= chunk_end and adj_end >= chunk_start:
                if text not in texts:
                    texts.append(text)
        result.append(" ".join(texts))
    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_chunk_selection_prompt(
    chunks: list[tuple[int, int]],
    fps: float,
    query: str,
    video_metadata: dict | None = None,
    chunk_metadata: list[dict] | None = None,
    metadata_config: dict | None = None,
    frames_per_chunk: int = 1,
    chunk_scores: list[float] | None = None,
) -> str:
    """Build the text portion of a chunk-selection prompt.

    The text lists every chunk with its index and time range, then poses
    the query.  Images are attached separately (text must come first).
    """
    metadata_config = metadata_config or {}
    include_timestamps = _resolve_metadata_flag(metadata_config, "timestamps")
    include_chunk_index = _resolve_metadata_flag(metadata_config, "chunk_index")
    include_video_duration = _resolve_metadata_flag(metadata_config, "video_duration")
    include_cosine_sim = _resolve_metadata_flag(metadata_config, "cosine_similarity")
    include_subtitles = _resolve_metadata_flag(metadata_config, "subtitles")

    subtitle_path = None
    subtitle_offset = 0.0
    if include_subtitles and video_metadata:
        subtitle_path = video_metadata.get("subtitle_path")
        subtitle_offset = float(
            video_metadata.get("starting_timestamp_for_subtitles", 0.0)
        )
    per_chunk_subs = _collect_per_chunk_subtitles(
        subtitle_path, chunks, fps, start_offset_sec=subtitle_offset,
    ) if include_subtitles and subtitle_path else [""] * len(chunks)

    lines = [
        f"Below are {len(chunks)} sequential, non-overlapping chunks extracted "
        "from a video. The images are in the same order as the chunks listed. "
        "If a chunk has multiple images, those images all belong to that chunk "
        "and are shown in chronological order.\n"
    ]
    for i, (sf, ef) in enumerate(chunks, 1):
        start_sec = sf / fps
        end_sec = ef / fps
        parts = []
        if include_chunk_index:
            parts.append(f"Chunk {i}")
        if frames_per_chunk > 1:
            image_start = (i - 1) * frames_per_chunk + 1
            image_end = i * frames_per_chunk
            parts.append(f"images {image_start}-{image_end}")
        if include_timestamps:
            parts.append(f"[{start_sec:.1f}s - {end_sec:.1f}s]")
        if include_cosine_sim and chunk_scores and i - 1 < len(chunk_scores):
            parts.append(f"cos relevance: {chunk_scores[i - 1]:.2f}")

        if chunk_metadata and i - 1 < len(chunk_metadata):
            included_fields = list(metadata_config.get("chunk_fields", []))
            for field in included_fields:
                value = chunk_metadata[i - 1].get(field)
                if value is None:
                    continue
                parts.append(f"{field}: {value}")

        chunk_line = " | ".join(parts)
        sub_text = per_chunk_subs[i - 1] if i - 1 < len(per_chunk_subs) else ""
        if sub_text:
            chunk_line += f'\n  Subtitles: "{sub_text}"'
        lines.append(chunk_line)

    if include_video_duration and video_metadata and video_metadata.get("duration"):
        lines.append(f"\nVideo duration: {video_metadata['duration']:.1f}s")

    lines.append(f"\nQuery: \"{query}\"")
    lines.append(
        "\nChoose the single chunk that best matches the query. "
        "Compare the chunks before deciding and focus on the visual evidence "
        "that is most relevant to the query. If several chunks are partially "
        "relevant, choose the best overall match. Respond with only the chunk "
        "number."
    )
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a careful video moment retrieval assistant. You are given frames "
    "from sequential, non-overlapping chunks of a single video along with "
    "their time ranges. Select the chunk that best matches the query and "
    "respond with only the chunk number."
)

QA_SYSTEM_PROMPT = (
    "You are a video question answering assistant. Answer questions about "
    "videos based on the visual content shown in the provided frames. "
    "Give short, direct answers."
)

QA_SYSTEM_PROMPT_MCQ = (
    "You are a video multiple-choice question answering assistant. "
    "Answer based on the provided video frames and metadata. "
    "Respond with only the single correct letter corresponding to one of the "
    "provided options."
)

# WFS-SB / lmms-eval LLaVA-OneVision uses the qwen_1_5 conv template, whose
# default system message is this exact string.
WFS_SB_SYSTEM_PROMPT = "You are a helpful assistant."


# VideoMME-style MCQ wording used by WFS-SB/lmms-eval.
VIDEOMME_STYLE_MCQ_PROMPT_STYLES = {"videomme_mcq"}
LEGACY_REPO_MCQ_PROMPT_STYLES = {"legacy_repo_mcq"}
OFFICIAL_MCQ_PROMPT_STYLES = {
    "mlvu_lmms_eval_mcq",
    "longvideobench_official_mcq",
}
WFS_SB_MCQ_PROMPT_STYLES = VIDEOMME_STYLE_MCQ_PROMPT_STYLES | OFFICIAL_MCQ_PROMPT_STYLES
MCQ_PROMPT_STYLES = WFS_SB_MCQ_PROMPT_STYLES | LEGACY_REPO_MCQ_PROMPT_STYLES


def _format_mcq_question_block(
    question: str,
    options: list[str],
    prompt_style: str = "videomme_mcq",
) -> str:
    """Format a multiple-choice question and options without duplicating letters."""
    cleaned_options = [str(option).strip() for option in options if str(option).strip()]
    q_text = str(question).strip()
    if cleaned_options:
        letters = [chr(ord("A") + idx) for idx in range(len(cleaned_options))]
        if prompt_style == "mlvu_lmms_eval_mcq":
            normalized_options = []
            for letter, option in zip(letters, cleaned_options):
                option_text = re.sub(
                    rf"^{letter}\s*[\.\):]\s*",
                    "",
                    option,
                    flags=re.IGNORECASE,
                ).strip()
                normalized_options.append(f"({letter}) {option_text}")
            options_block = "\n".join(normalized_options)
        elif all(
            re.match(rf"^{letter}\s*[\.\):]", option, flags=re.IGNORECASE)
            for letter, option in zip(letters, cleaned_options)
        ):
            options_block = "\n".join(cleaned_options)
        else:
            options_block = "\n".join(
                f"{letter}. {option}"
                for letter, option in zip(letters, cleaned_options)
            )
    else:
        options_block = ""
    return "\n".join(part for part in [q_text, options_block] if part)


def _resolve_qa_prompt_style(
    prompt_config: dict | None,
    question_metadata: dict | None,
) -> str:
    """Resolve prompt style names. Omitted ``style`` defaults to official_dataset_mcq; ``auto`` still means VideoMME-style MCQ for compatibility."""
    prompt_config = prompt_config or {}
    qa_format = None if question_metadata is None else question_metadata.get("qa_format")
    requested = str(prompt_config.get("style", "official_dataset_mcq")).lower()
    if requested == "auto":
        return "legacy_repo_mcq" if qa_format == "mcq" else "short_answer"
    if requested == "videomme_mcq":
        return "videomme_mcq"
    if requested == "legacy_repo_mcq":
        return "legacy_repo_mcq"
    if requested == "official_dataset_mcq":
        if qa_format != "mcq":
            return "short_answer"
        prompt_dataset = ""
        if question_metadata is not None:
            prompt_dataset = str(
                question_metadata.get("prompt_dataset")
                or question_metadata.get("dataset_name")
                or ""
            ).lower()
        if prompt_dataset == "mlvu":
            return "mlvu_lmms_eval_mcq"
        if prompt_dataset == "longvideobench":
            return "longvideobench_official_mcq"
        if prompt_dataset == "videomme":
            return "videomme_mcq"
        return "videomme_mcq"
    return requested


def _uses_wfs_sb_mcq_prompt(prompt_style: str) -> bool:
    """Return whether a style should match WFS-SB's task-only MCQ prompt."""
    return prompt_style in WFS_SB_MCQ_PROMPT_STYLES


def _parse_subtitle_timestamp(timestamp) -> float:
    """Convert common subtitle timestamps into seconds."""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)

    hours, minutes, rest = str(timestamp).split(":")
    if "," in rest:
        seconds, millis = rest.split(",")
    elif "." in rest:
        seconds, millis = rest.split(".")
    else:
        seconds, millis = rest, "0"
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def _format_seconds(sec: float) -> str:
    """Render seconds as ``HH:MM:SS`` for prompt metadata."""
    total = max(int(sec), 0)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _clean_subtitle_text(text: str) -> str:
    """Strip simple subtitle markup and collapse whitespace."""
    text = re.sub(r"<[^>]+>", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=256)
def _load_subtitle_entries(subtitle_path: str) -> list[tuple[float, float, str]]:
    """Parse SRT or JSON subtitles into ``(start_sec, end_sec, text)`` tuples."""
    if subtitle_path.lower().endswith(".json"):
        return _load_json_subtitle_entries(subtitle_path)
    return _load_srt_entries(subtitle_path)


def _load_srt_entries(subtitle_path: str) -> list[tuple[float, float, str]]:
    """Parse an SRT subtitle file into ``(start_sec, end_sec, text)`` tuples."""
    with open(subtitle_path, encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    entries = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_idx = 1 if "-->" in lines[1] else 0
        if "-->" not in lines[time_idx]:
            continue

        start_raw, end_raw = [part.strip() for part in lines[time_idx].split("-->")]
        text = _clean_subtitle_text(" ".join(lines[time_idx + 1:]))
        if not text:
            continue

        entries.append((
            _parse_subtitle_timestamp(start_raw),
            _parse_subtitle_timestamp(end_raw),
            text,
        ))
    return entries


def _load_json_subtitle_entries(subtitle_path: str) -> list[tuple[float, float, str]]:
    """Parse LongVideoBench-style JSON subtitles."""
    with open(subtitle_path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data:
        if "timestamp" in item:
            start_raw, end_raw = item["timestamp"]
            text_raw = item.get("text", "")
        else:
            start_raw = item.get("start")
            end_raw = item.get("end")
            text_raw = item.get("line", item.get("text", ""))

        text = _clean_subtitle_text(text_raw)
        if not text or start_raw is None or end_raw is None:
            continue

        entries.append((
            _parse_subtitle_timestamp(start_raw),
            _parse_subtitle_timestamp(end_raw),
            text,
        ))
    return entries


def _collect_aligned_subtitles(
    subtitle_path: str | None,
    chunks: list[tuple[int, int]],
    fps: float,
    include_timestamps: bool = False,
    start_offset_sec: float = 0.0,
) -> list[str]:
    """Return subtitle lines overlapping the displayed chunks."""
    if not subtitle_path:
        return []

    try:
        entries = _load_subtitle_entries(subtitle_path)
    except OSError:
        return []

    chunk_ranges = [(start / fps, end / fps) for start, end in chunks]
    lines = []
    seen = set()
    for start_sec, end_sec, text in entries:
        start_sec -= float(start_offset_sec)
        end_sec -= float(start_offset_sec)
        overlaps = any(
            start_sec <= chunk_end and end_sec >= chunk_start
            for chunk_start, chunk_end in chunk_ranges
        )
        if not overlaps:
            continue

        key = (round(start_sec, 3), round(end_sec, 3), text)
        if key in seen:
            continue
        seen.add(key)

        if include_timestamps:
            lines.append(
                f"[{_format_seconds(start_sec)}-{_format_seconds(end_sec)}] {text}"
            )
        else:
            lines.append(text)
    return lines


def _resolve_subtitle_source(
    question_metadata: dict | None = None,
    video_metadata: dict | None = None,
) -> tuple[str | None, float]:
    """Return subtitle path and timestamp offset for QA prompting."""
    source = question_metadata or video_metadata or {}
    subtitle_path = source.get("subtitle_path")
    subtitle_offset = float(source.get("starting_timestamp_for_subtitles", 0.0))
    return subtitle_path, subtitle_offset


def _collect_full_subtitle_text(
    subtitle_path: str | None,
    include_timestamps: bool = False,
    start_offset_sec: float = 0.0,
) -> str:
    """Return the full subtitle transcript in chronological order."""
    if not subtitle_path:
        return ""
    try:
        entries = _load_subtitle_entries(subtitle_path)
    except OSError:
        return ""

    lines = []
    for start_sec, end_sec, text in entries:
        start_sec -= float(start_offset_sec)
        end_sec -= float(start_offset_sec)
        if include_timestamps:
            lines.append(
                f"[{_format_seconds(start_sec)}-{_format_seconds(end_sec)}] {text}"
            )
        else:
            lines.append(text)
    return "\n".join(lines).strip()


def _build_frame_metadata_block(
    frame_times: list[float] | None = None,
    frame_scores: list[float] | None = None,
    video_metadata: dict | None = None,
    question_metadata: dict | None = None,
    metadata_config: dict | None = None,
    frame_labels: list[str] | None = None,
    force_show: bool = False,
) -> str:
    """Build metadata block for window-selected frames (per-frame timestamps)."""
    metadata_config = metadata_config or {}
    include_timestamps = _resolve_metadata_flag(metadata_config, "timestamps")
    include_video_duration = _resolve_metadata_flag(metadata_config, "video_duration")
    include_subtitles = _resolve_metadata_flag(metadata_config, "subtitles")
    include_cosine_sim = _resolve_metadata_flag(metadata_config, "cosine_similarity")
    lines: list[str] = []

    subtitle_path = None
    subtitle_offset = 0.0
    if question_metadata:
        subtitle_path = question_metadata.get("subtitle_path")
        subtitle_offset = float(
            question_metadata.get("starting_timestamp_for_subtitles", 0.0)
        )

    need_subtitles = include_subtitles

    show_frame_block = (
        force_show
        or include_timestamps
        or include_cosine_sim
        or need_subtitles
    ) and frame_times

    if show_frame_block:
        lines.append("Frame timestamps:")
        for idx, t in enumerate(frame_times):
            parts: list[str] = [f"Frame {idx + 1}:"]
            if include_timestamps:
                parts.append(f"{t:.1f}s")
            if frame_labels and idx < len(frame_labels):
                label = str(frame_labels[idx]).strip()
                if label:
                    parts.append(label)
            if include_cosine_sim and frame_scores and idx < len(frame_scores):
                parts.append(f"(query-match relevance score: {frame_scores[idx]:.2f})")
            seg_line = "- " + " ".join(parts)
            if need_subtitles and subtitle_path:
                sub_text = _collect_subtitle_around_time(
                    subtitle_path, t, window_sec=3.0,
                    start_offset_sec=subtitle_offset,
                )
                if sub_text:
                    seg_line += f'\n  Subtitles: "{sub_text}"'
            lines.append(seg_line)

    if include_video_duration and video_metadata:
        duration = video_metadata.get("duration")
        if duration is not None:
            lines.append(f"Video duration: {float(duration):.1f}s")

    return "\n".join(lines).strip()


# TODO: Improve subtitle retrieval for QA prompts. The current fixed local
# window is useful for parity across methods, but it is a blunt heuristic and
# likely not the best way to surface subtitle context for the displayed frames.
def _collect_subtitle_around_time(
    subtitle_path: str, time_sec: float, window_sec: float = 3.0,
    start_offset_sec: float = 0.0,
) -> str:
    """Collect subtitle text within a time window around a frame timestamp."""
    try:
        entries = _load_subtitle_entries(subtitle_path)
    except OSError:
        return ""
    texts: list[str] = []
    for start_sec, end_sec, text in entries:
        adj_start = start_sec - start_offset_sec
        adj_end = end_sec - start_offset_sec
        if adj_start <= time_sec + window_sec and adj_end >= time_sec - window_sec:
            if text not in texts:
                texts.append(text)
    return " ".join(texts)


def build_videomme_mcq_prompt_frames(
    question: str,
    options: list[str],
    num_frames: int,
    frame_times: list[float] | None = None,
    frame_scores: list[float] | None = None,
    video_metadata: dict | None = None,
    question_metadata: dict | None = None,
    metadata_config: dict | None = None,
    prompt_config: dict | None = None,
) -> str:
    """Build a multiple-choice QA prompt for window-selected frames."""
    metadata_block = _build_frame_metadata_block(
        frame_times=frame_times,
        frame_scores=frame_scores,
        video_metadata=video_metadata,
        question_metadata=question_metadata,
        metadata_config=metadata_config,
    )
    prompt_config = prompt_config or {}
    prompt_style = _resolve_qa_prompt_style(prompt_config, question_metadata)
    question_block = _format_mcq_question_block(question, options, prompt_style)

    custom_template = prompt_config.get("custom_template")
    if custom_template:
        options_block = "\n".join(question_block.splitlines()[1:])
        return str(custom_template).format(
            num_frames=int(num_frames),
            metadata_block=metadata_block,
            question=str(question).strip(),
            options_block=options_block,
            question_block=question_block,
        ).strip()

    if prompt_style == "mlvu_lmms_eval_mcq":
        return "\n".join([
            question_block,
            "",
            "Only give the best option.",
            "Best option: (",
        ])

    if prompt_style == "longvideobench_official_mcq":
        return "\n".join([
            question_block,
            "Answer with the option's letter from the given choices directly.",
        ])

    if prompt_style == "videomme_mcq":
        return "\n".join([
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter (A, B, C, or D) "
            "of the correct option.",
            question_block,
            "",
            "Answer with the option's letter from the given choices directly.",
        ])

    lines = (
        [f"Below are {num_frames} frames sampled from a video, shown in chronological order."]
        if num_frames > 0 else []
    )
    if metadata_block:
        lines.extend(["", metadata_block])

    lines.extend([
        "",
        "Select the best answer to the following multiple-choice question "
        "based on the video. Respond with only the letter of the correct option.",
        question_block,
        "The best answer is:",
    ])
    return "\n".join(lines)


def build_qa_prompt(
    question: str,
    num_frames: int,
    metadata_block: str = "",
    prompt_config: dict | None = None,
) -> str:
    """Build the text portion of a video QA prompt."""
    prompt_config = prompt_config or {}
    custom_template = prompt_config.get("custom_template")
    if custom_template:
        return str(custom_template).format(
            num_frames=int(num_frames),
            metadata_block=metadata_block,
            question=str(question).strip(),
            question_block=str(question).strip(),
        ).strip()

    lines = (
        [f"Below are {num_frames} frames sampled from a video, shown in chronological order.\n"]
        if num_frames > 0 else []
    )
    if metadata_block:
        lines.extend([metadata_block, ""])
    lines.extend([
        f"Question: {question}",
        "\nAnswer with only a single word or short phrase. "
        "Do not explain or elaborate.",
    ])
    return "\n".join(lines)


def build_subtitle_only_prompt(
    question: str,
    subtitle_text: str,
    options: list[str] | None = None,
    prompt_style: str = "short_answer",
) -> str:
    """Build a QA prompt that uses only the full subtitle transcript."""
    lines = [
        "No video frames are provided for this question.",
        "Use only the full subtitle transcript below.",
        "",
        "Full subtitle transcript:",
        subtitle_text,
        "",
    ]

    if prompt_style in MCQ_PROMPT_STYLES:
        question_block = _format_mcq_question_block(question, options or [], prompt_style)
        if prompt_style == "mlvu_lmms_eval_mcq":
            lines.extend([
                question_block,
                "Only give the best option.",
                "Best option: (",
            ])
        elif prompt_style == "longvideobench_official_mcq":
            lines.extend([
                question_block,
                "Answer with the option's letter from the given choices directly.",
            ])
        else:
            lines.extend([
                "Select the best answer to the following multiple-choice question "
                "using only the subtitle transcript. Respond with only the letter "
                "of the correct option.",
                question_block,
                "The best answer is:",
            ])
    else:
        lines.extend([
            f"Question: {question}",
            "",
            "Answer using only a single word or short phrase. "
            "Do not explain or elaborate.",
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# End-to-end: chunks -> VLM answer
# ---------------------------------------------------------------------------

class TemporalAbstractionLayer:
    """Selects frames from chunks and queries a VLM."""

    WINDOW_METHODS = {
        "weighted", "mmr", "temporal", "coverage", "rdmv",
        "nudge", "stretch", "constrained", "focused", "segment_adaptive",
        "uniform_topk", "random",
        "mmr_chunk_penalty", "mmr_chunk_constrained",
        "intra_chunk_greedy",
    }

    def __init__(
        self,
        vlm_client: VLMClient,
        frame_method: str = "middle",
        best_window_strategy: str = "ranked_round_robin",
        frames_per_chunk: int = 1,
        max_chunks: int = 20,
        metadata_config: dict | None = None,
        qa_prompt_config: dict | None = None,
        window_method_kwargs: dict | None = None,
        n_frames: int | None = None,
    ):
        self.vlm = vlm_client
        self.frame_method = frame_method
        self.best_window_strategy = best_window_strategy
        self.frames_per_chunk = frames_per_chunk
        self.max_chunks = max_chunks
        self.metadata_config = metadata_config or {}
        self.qa_prompt_config = qa_prompt_config or {}
        self.window_method_kwargs = window_method_kwargs or {}
        self.is_window_method = frame_method in self.WINDOW_METHODS
        if n_frames is not None:
            self.n_frames = n_frames
        elif self.is_window_method:
            max_img = vlm_client.max_images_per_request or 30
            self.n_frames = min(max_chunks, max_img)
        else:
            self.n_frames = max_chunks * frames_per_chunk

    @classmethod
    def from_config(cls, vlm_client: VLMClient, cfg):
        """Build a temporal abstraction layer from config."""
        method = cfg.frame_selection.method
        window_kwargs = {}
        if hasattr(cfg.frame_selection, method):
            window_kwargs = dict(cfg.frame_selection[method])
        return cls(
            vlm_client=vlm_client,
            frame_method=method,
            best_window_strategy=cfg.frame_selection.get(
                "best_window_strategy", "ranked_round_robin"
            ),
            frames_per_chunk=cfg.frame_selection.frames_per_chunk,
            max_chunks=cfg.prompt.max_chunks_per_prompt,
            metadata_config=dict(cfg.metadata),
            qa_prompt_config=dict(cfg.prompt),
            window_method_kwargs=window_kwargs,
            n_frames=cfg.frame_selection.get("n_frames", None),
        )

    def select_best_chunk(
        self,
        video_path: str,
        chunks: list[tuple[int, int]],
        fps: float,
        query: str,
        video_metadata: dict | None = None,
        chunk_metadata: list[dict] | None = None,
        chunk_frame_selections: list[dict] | None = None,
        chunk_scores: list[float] | None = None,
    ) -> dict:
        """Ask the VLM which chunk best matches *query*.

        Returns:
            dict with keys: ``chunk_index`` (0-based), ``raw_response``,
            ``chunk`` (start_frame, end_frame).
        """
        allowed_chunks = compute_allowed_chunks(
            self.max_chunks,
            self.frames_per_chunk,
            self.vlm.max_images_per_request,
        )

        used_chunks = chunks[:allowed_chunks]
        if len(chunks) > allowed_chunks:
            log.warning(
                "Truncated %d chunks to %d due to prompt/image budget",
                len(chunks), allowed_chunks,
            )

        all_frames = []
        used_chunk_frame_selections = None
        if chunk_frame_selections is not None:
            used_chunk_frame_selections = chunk_frame_selections[: len(used_chunks)]

        for idx, chunk in enumerate(used_chunks):
            frame_kwargs = {"n": self.frames_per_chunk}
            if used_chunk_frame_selections is not None and idx < len(used_chunk_frame_selections):
                frame_kwargs["frame_indices"] = used_chunk_frame_selections[idx].get(
                    "frame_indices"
                )
            frames = select_frames(
                video_path, chunk,
                method=self.frame_method,
                **frame_kwargs,
            )
            all_frames.extend(frames)

        used_chunk_metadata = None
        if chunk_metadata is not None:
            used_chunk_metadata = chunk_metadata[: len(used_chunks)]

        used_chunk_scores = None
        if chunk_scores is not None:
            used_chunk_scores = chunk_scores[: len(used_chunks)]

        text = build_chunk_selection_prompt(
            used_chunks,
            fps,
            query,
            video_metadata=video_metadata,
            chunk_metadata=used_chunk_metadata,
            metadata_config=self.metadata_config,
            frames_per_chunk=self.frames_per_chunk,
            chunk_scores=used_chunk_scores,
        )
        max_retries = 3
        for attempt in range(max_retries):
            raw = self.vlm.query(text, images=all_frames, system_prompt=SYSTEM_PROMPT)
            parsed = _parse_chunk_selection_response(raw, len(used_chunks))
            if parsed["decision"] == "chunk":
                chunk_idx = parsed["chunk_index"]
                return {
                    "decision": "chunk",
                    "chunk_index": chunk_idx,
                    "chunk": used_chunks[chunk_idx],
                    "raw_response": raw,
                }
            if parsed["decision"] == "no_match":
                return {
                    "decision": "no_match",
                    "chunk_index": None,
                    "chunk": None,
                    "raw_response": raw,
                }
            if attempt < max_retries - 1:
                log.warning(
                    "Retry %d/%d: could not parse chunk from VLM response: %s",
                    attempt + 1, max_retries, raw,
                )

        log.warning("All %d attempts failed to parse a chunk number", max_retries)
        return {
            "decision": "parse_failure",
            "chunk_index": None,
            "chunk": None,
            "raw_response": raw,
        }

    def answer_question(
        self,
        video_path: str,
        chunks: list[tuple[int, int]],
        fps: float,
        question: str,
        video_metadata: dict | None = None,
        question_metadata: dict | None = None,
        window_frame_indices: list[int] | None = None,
        window_frame_times: list[float] | None = None,
        window_frame_scores: list[float] | None = None,
    ) -> dict:
        """Ask the VLM to answer *question* given frames from *chunks*.

        Returns:
            dict with keys ``answer`` (extracted text) and ``raw_response``.
        """
        subtitle_only = bool(self.qa_prompt_config.get("subtitle_only", False))
        if not subtitle_only and (window_frame_indices is None or window_frame_times is None):
            raise ValueError(
                "Non-iterative QA requires explicit prompt frame indices and times."
            )

        prompt_style = _resolve_qa_prompt_style(
            self.qa_prompt_config,
            question_metadata,
        )
        use_video_input = False

        if subtitle_only:
            subtitle_path, subtitle_offset = _resolve_subtitle_source(
                question_metadata=question_metadata,
                video_metadata=video_metadata,
            )
            subtitle_text = _collect_full_subtitle_text(
                subtitle_path,
                start_offset_sec=subtitle_offset,
            )
            if not subtitle_text:
                raise ValueError(
                    "subtitle_only QA requires a non-empty subtitle transcript."
                )
            all_frames = None
            text = build_subtitle_only_prompt(
                question=question_metadata.get("question", question)
                if question_metadata else question,
                subtitle_text=subtitle_text,
                options=question_metadata.get("options", []) if question_metadata else [],
                prompt_style=prompt_style,
            )
            system_prompt = (
                WFS_SB_SYSTEM_PROMPT
                if _uses_wfs_sb_mcq_prompt(prompt_style)
                else
                QA_SYSTEM_PROMPT_MCQ
                if prompt_style in MCQ_PROMPT_STYLES
                else QA_SYSTEM_PROMPT
            )
        else:
            use_video_input = getattr(self.vlm, "input_mode", None) == VIDEO_INPUT_MODE
            all_frames = _load_frames_by_index(video_path, window_frame_indices)
            if prompt_style in MCQ_PROMPT_STYLES:
                text = build_videomme_mcq_prompt_frames(
                    question=question_metadata.get("question", question)
                    if question_metadata else question,
                    options=question_metadata.get("options", []) if question_metadata else [],
                    num_frames=len(all_frames),
                    frame_times=window_frame_times,
                    frame_scores=window_frame_scores,
                    video_metadata=video_metadata,
                    question_metadata=question_metadata,
                    metadata_config=self.metadata_config,
                    prompt_config=self.qa_prompt_config,
                )
                system_prompt = (
                    WFS_SB_SYSTEM_PROMPT
                    if _uses_wfs_sb_mcq_prompt(prompt_style)
                    else QA_SYSTEM_PROMPT_MCQ
                )
            else:
                metadata_block = _build_frame_metadata_block(
                    frame_times=window_frame_times,
                    frame_scores=window_frame_scores,
                    video_metadata=video_metadata,
                    question_metadata=question_metadata,
                    metadata_config=self.metadata_config,
                )
                text = build_qa_prompt(
                    question,
                    num_frames=len(all_frames),
                    metadata_block=metadata_block,
                    prompt_config=self.qa_prompt_config,
                )
                system_prompt = QA_SYSTEM_PROMPT

        query_kwargs = {
            "text": text,
            "images": all_frames,
            "system_prompt": system_prompt,
        }
        if not subtitle_only and use_video_input:
            query_kwargs["video_metadata"] = video_metadata
            query_kwargs["video_frame_indices"] = window_frame_indices
        raw = self.vlm.query(**query_kwargs)
        answer = _response_to_text(raw)
        return {"answer": answer, "raw_response": raw, "prompt_text": text}

    def answer_question_iterative(
        self,
        video_path: str,
        fps: float,
        question: str,
        window_embeddings: np.ndarray,
        window_times: np.ndarray,
        query_embedding: np.ndarray,
        emb_backend,
        iterative_config: dict,
        video_metadata: dict | None = None,
        question_metadata: dict | None = None,
        duration_sec: float | None = None,
    ) -> dict:
        """Run iterative search-and-answer loop.

        1. Select initial frames via select_frames_from_windows
        2. For each query round: prompt VLM for a search query, embed it,
           find new frames via cosine similarity, accumulate
        3. Final round: prompt VLM for answer with all accumulated frames
        """
        from frame_selection import select_frames_from_windows

        n_queries = int(iterative_config.get("n_queries", 3))
        min_time_gap = float(iterative_config.get("min_time_gap", 1.0))
        frames_per_round = _resolve_frames_per_round(
            iterative_config.get("frames_per_round", 6), n_queries,
        )
        if duration_sec is None:
            duration_sec = float(window_times[-1]) + 1.0

        total_frames = sum(frames_per_round)
        if (
            self.vlm.max_images_per_request is not None
            and total_frames > self.vlm.max_images_per_request
        ):
            raise ValueError(
                f"Iterative QA needs {total_frames} total frames "
                f"(sum of frames_per_round={frames_per_round}) but "
                f"max_images_per_request={self.vlm.max_images_per_request}. "
                f"Reduce frames_per_round or increase the VLM image budget."
            )

        # --- Round 0: initial frame selection ---
        k_initial = frames_per_round[0]
        initial_method = iterative_config.get(
            "initial_method",
            self.frame_method if self.is_window_method else "constrained",
        )

        if initial_method == "uniform":
            n_windows = len(window_embeddings)
            k_init = min(k_initial, n_windows)
            indices = np.linspace(0, n_windows - 1, k_init, dtype=int).tolist()
            query_t = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
            emb_t = torch.tensor(
                window_embeddings[indices], dtype=torch.float32,
            )
            scores = torch.nn.functional.cosine_similarity(
                query_t.expand(len(indices), -1), emb_t,
            ).tolist()
            all_window_indices = indices
            all_frame_times = [float(window_times[i]) for i in indices]
            all_frame_scores = scores
        else:
            init_result = select_frames_from_windows(
                method=initial_method,
                window_embeddings=window_embeddings,
                window_times=window_times,
                query_embedding=query_embedding,
                n_frames=k_initial,
                duration_sec=duration_sec,
                fps=fps,
                **self.window_method_kwargs,
            )
            all_window_indices = list(init_result["window_indices"])
            all_frame_times = [float(window_times[i]) for i in all_window_indices]
            all_frame_scores = [float(s) for s in init_result["scores"]]

        all_frame_sources = ["initial selection"] * len(all_window_indices)

        round_details = [{
            "round": 0,
            "type": "initial",
            "n_frames": len(all_window_indices),
            "window_indices": [int(i) for i in all_window_indices],
            "frame_times": list(all_frame_times),
            "frame_scores": list(all_frame_scores),
        }]
        queries_so_far: list[str] = []
        raw_responses: list[str] = []

        # --- Query rounds ---
        for q_round in range(n_queries):
            remaining = n_queries - q_round
            k_round = frames_per_round[q_round + 1]

            all_frame_indices = [int(window_times[i] * fps) for i in all_window_indices]
            all_frames = _load_frames_by_index(video_path, all_frame_indices)

            metadata_block = _build_iterative_frame_metadata_block(
                frame_times=all_frame_times,
                frame_scores=all_frame_scores,
                frame_sources=all_frame_sources,
                video_metadata=video_metadata,
                question_metadata=question_metadata,
                metadata_config=self.metadata_config,
            )

            query_prompt = build_iterative_query_prompt(
                question=question,
                num_frames=len(all_frames),
                metadata_block=metadata_block,
                queries_so_far=queries_so_far,
                remaining_queries=remaining,
                question_metadata=question_metadata,
                prompt_config=self.qa_prompt_config,
            )

            raw = self.vlm.query(
                query_prompt,
                images=all_frames,
                system_prompt=ITERATIVE_QUERY_SYSTEM_PROMPT,
            )
            raw_text = _response_to_text(raw)
            query_text = raw_text.strip().strip('"').strip("'")
            if len(query_text) > 200:
                query_text = query_text[:200]
            queries_so_far.append(query_text)
            raw_responses.append(raw)

            log.info(
                "Iterative round %d/%d query: '%s'",
                q_round + 1, n_queries, query_text,
            )

            query_emb = emb_backend.embed_text(query_text)
            new_indices, new_scores = search_frames_by_query(
                query_embedding=query_emb,
                window_embeddings=window_embeddings,
                window_times=window_times,
                already_selected_times=all_frame_times,
                k=k_round,
                min_time_gap=min_time_gap,
            )

            new_times = [float(window_times[i]) for i in new_indices]
            new_sources = [f'search: "{query_text}"'] * len(new_indices)

            combined = list(zip(
                all_window_indices + new_indices,
                all_frame_times + new_times,
                all_frame_scores + new_scores,
                all_frame_sources + new_sources,
            ))
            combined.sort(key=lambda x: x[1])

            all_window_indices = [c[0] for c in combined]
            all_frame_times = [c[1] for c in combined]
            all_frame_scores = [c[2] for c in combined]
            all_frame_sources = [c[3] for c in combined]

            round_details.append({
                "round": q_round + 1,
                "type": "query",
                "query": query_text,
                "raw_response": raw,
                "n_new_frames": len(new_indices),
                "new_window_indices": [int(i) for i in new_indices],
                "new_frame_times": new_times,
                "new_frame_scores": new_scores,
            })

        # --- Final answer round ---
        all_frame_indices = [int(window_times[i] * fps) for i in all_window_indices]
        all_frames = _load_frames_by_index(video_path, all_frame_indices)

        metadata_block = _build_frame_metadata_block(
            frame_times=all_frame_times,
            frame_scores=all_frame_scores,
            video_metadata=video_metadata,
            question_metadata=question_metadata,
            metadata_config=self.metadata_config,
        )

        final_text, final_system = build_iterative_final_prompt(
            question=question,
            num_frames=len(all_frames),
            metadata_block=metadata_block,
            queries_so_far=queries_so_far,
            question_metadata=question_metadata,
            prompt_config=self.qa_prompt_config,
        )

        raw = self.vlm.query(
            final_text, images=all_frames, system_prompt=final_system,
        )
        answer = _response_to_text(raw)

        return {
            "answer": answer,
            "raw_response": raw,
            "prompt_text": final_text,
            "iterative_queries": queries_so_far,
            "iterative_rounds": round_details,
            "total_frames_used": len(all_frames),
            "total_api_calls": n_queries + 1,
            "all_frame_times": all_frame_times,
            "all_frame_indices": all_frame_indices,
            "all_frame_scores": all_frame_scores,
            "all_frame_sources": all_frame_sources,
        }


def _response_to_text(response) -> str:
    """Normalize common OpenAI-compatible response payloads to plain text."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, list):
        parts = []
        for item in response:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = _response_to_text(item)
                if text:
                    parts.append(text)
        return " ".join(parts).strip()
    return str(response).strip()


def _is_no_match_response(response_text: str) -> bool:
    """Return True when the model explicitly says no chunk matches."""
    if not response_text:
        return False
    normalized = response_text.strip().lower()
    patterns = (
        r"^none[.!]?$",
        r"^no match[.!]?$",
        r"\bnone of (the )?(provided )?(chunks|chunk|frames|images) match\b",
        r"\bnone of (the )?(provided )?(chunks|chunk|frames|images) matches\b",
        r"\bno (provided )?(chunk|chunks) match\b",
        r"\bno (provided )?(chunk|chunks) matches\b",
        r"\bno suitable chunk\b",
        r"\bno matching chunk\b",
        r"\bthere is no suitable chunk\b",
        r"\bthere is no matching chunk\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _parse_chunk_selection_response(response, n_chunks: int) -> dict:
    """Parse a VLM response into a selected chunk, explicit no-match, or failure."""
    response_text = _response_to_text(response)
    numbers = re.findall(r"\d+", response_text)
    for num_str in numbers:
        num = int(num_str)
        if 1 <= num <= n_chunks:
            return {"decision": "chunk", "chunk_index": num - 1}
    if _is_no_match_response(response_text):
        return {"decision": "no_match", "chunk_index": None}
    return {"decision": "parse_failure", "chunk_index": None}


# ---------------------------------------------------------------------------
# Iterative VLM QA: search-and-refine loop
# ---------------------------------------------------------------------------

ITERATIVE_QUERY_SYSTEM_PROMPT = (
    "You are a video question answering assistant. The user can run a text "
    "search over the video to retrieve more frames with extra visual detail. "
    "When asked for a search, respond with ONLY a short query (a few words: "
    "what to look for in the video). Use this to ask the video for more "
    "information before you are allowed to answer. Do not answer the main "
    "question until instructed."
)


def _resolve_frames_per_round(
    frames_per_round,
    n_queries: int,
) -> list[int]:
    """Normalize frames_per_round to a list of length n_queries + 1."""
    n_rounds = n_queries + 1
    if isinstance(frames_per_round, (int, float)):
        return [int(frames_per_round)] * n_rounds
    frames_per_round = [int(x) for x in frames_per_round]
    if len(frames_per_round) == 1:
        return frames_per_round * n_rounds
    if len(frames_per_round) != n_rounds:
        raise ValueError(
            f"frames_per_round has {len(frames_per_round)} entries but "
            f"n_queries={n_queries} requires {n_rounds} "
            f"(1 initial + {n_queries} query rounds)"
        )
    return frames_per_round


def search_frames_by_query(
    query_embedding: np.ndarray,
    window_embeddings: np.ndarray,
    window_times: np.ndarray,
    already_selected_times: list[float],
    k: int,
    min_time_gap: float = 1.0,
) -> tuple[list[int], list[float]]:
    """Find k windows matching a query, avoiding already-selected times.

    Returns (window_indices, cosine_scores) sorted by time.
    """
    query_t = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)
    emb_t = torch.tensor(window_embeddings, dtype=torch.float32)
    n_windows = len(window_embeddings)

    cos_sims = torch_cos_sim(
        query_t.expand(n_windows, -1), emb_t,
    ).numpy()

    ranked = np.argsort(-cos_sims)
    selected: list[int] = []
    for idx in ranked:
        t = float(window_times[idx])
        too_close = any(
            abs(t - t2) < min_time_gap
            for t2 in already_selected_times
        )
        if too_close:
            continue
        also_too_close_to_new = any(
            abs(t - float(window_times[s])) < min_time_gap
            for s in selected
        )
        if also_too_close_to_new:
            continue
        selected.append(int(idx))
        if len(selected) >= k:
            break

    time_order = sorted(selected, key=lambda i: window_times[i])
    scores = [float(cos_sims[i]) for i in time_order]
    return time_order, scores


def _build_iterative_frame_metadata_block(
    frame_times: list[float],
    frame_scores: list[float],
    frame_sources: list[str],
    video_metadata: dict | None = None,
    question_metadata: dict | None = None,
    metadata_config: dict | None = None,
) -> str:
    """Build iterative-query metadata using the shared frame metadata format."""
    frame_labels = [
        f"({source})" if str(source).strip() else ""
        for source in frame_sources
    ]
    return _build_frame_metadata_block(
        frame_times=frame_times,
        frame_scores=frame_scores,
        video_metadata=video_metadata,
        question_metadata=question_metadata,
        metadata_config=metadata_config,
        frame_labels=frame_labels,
        force_show=bool(frame_times),
    )


def build_iterative_query_prompt(
    question: str,
    num_frames: int,
    metadata_block: str,
    queries_so_far: list[str],
    remaining_queries: int,
    question_metadata: dict | None = None,
    prompt_config: dict | None = None,
) -> str:
    """Build a prompt asking the VLM to generate a search query."""
    prompt_style = _resolve_qa_prompt_style(prompt_config, question_metadata)

    lines = [
        f"You need to answer a question about a video. Below are {num_frames} "
        "frames from the video, shown in chronological order. "
        "Between now and the final answer step, you can query the video "
        "(short text search) to retrieve more frames with more information.",
    ]
    if metadata_block:
        lines.extend(["", metadata_block])

    lines.append("")
    if prompt_style in MCQ_PROMPT_STYLES and question_metadata:
        q_text = question_metadata.get("question", question)
        options = question_metadata.get("options", [])
        lines.append(_format_mcq_question_block(q_text, options, prompt_style))
    else:
        lines.append(f"Question: {question}")

    if queries_so_far:
        lines.append("")
        lines.append("Search history (queries you already sent, in order):")
        for i, q in enumerate(queries_so_far, 1):
            lines.append(f'  {i}. "{q}"')
        lines.append(
            "You may repeat a previous query if you still need more evidence "
            "on that topic; the system will return different times when possible."
        )

    lines.extend([
        "",
        f"You have {remaining_queries} more chance{'s' if remaining_queries != 1 else ''} "
        "to query the video for additional frames.",
        "If anything is unclear from the frames above, issue a short video search "
        "query so more relevant moments can be retrieved.",
        "",
        "Respond with ONLY a short search query (a few words describing what "
        "visual content to look for in the video). Do not answer the question yet.",
    ])
    return "\n".join(lines)


def build_iterative_final_prompt(
    question: str,
    num_frames: int,
    metadata_block: str,
    queries_so_far: list[str],
    question_metadata: dict | None = None,
    prompt_config: dict | None = None,
) -> tuple[str, str]:
    """Build the final answer prompt for iterative QA.

    Returns (prompt_text, system_prompt).
    """
    prompt_style = _resolve_qa_prompt_style(prompt_config, question_metadata)

    lines = [
        f"Below are {num_frames} frames sampled from a video, shown in chronological order.",
    ]
    if metadata_block:
        lines.extend(["", metadata_block])

    if queries_so_far:
        lines.append("")
        lines.append("Search history (video queries you issued, in order):")
        for i, q in enumerate(queries_so_far, 1):
            lines.append(f'  {i}. "{q}"')

    lines.append("")

    if prompt_style in MCQ_PROMPT_STYLES:
        q_text = question if not question_metadata else question_metadata.get("question", question)
        options = [] if not question_metadata else question_metadata.get("options", [])
        question_block = _format_mcq_question_block(q_text, options, prompt_style)

        if prompt_style == "mlvu_lmms_eval_mcq":
            lines.extend([
                question_block,
                "Only give the best option.",
                "Best option: (",
            ])
            return "\n".join(lines), QA_SYSTEM_PROMPT_MCQ

        if prompt_style == "longvideobench_official_mcq":
            lines.extend([
                question_block,
                "Answer with the option's letter from the given choices directly.",
            ])
            return "\n".join(lines), QA_SYSTEM_PROMPT_MCQ

        lines.extend([
            "Select the best answer to the following multiple-choice question "
            "based on the video. Respond with only the letter of the correct option.",
            question_block,
            "The best answer is:",
        ])
        return "\n".join(lines), QA_SYSTEM_PROMPT_MCQ
    else:
        lines.extend([
            f"Question: {question}",
            "",
            "Answer with only a single word or short phrase. "
            "Do not explain or elaborate.",
        ])
        return "\n".join(lines), QA_SYSTEM_PROMPT
