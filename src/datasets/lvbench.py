"""LVBench dataset loader for multiple-choice long-video QA evaluation."""

import json
import logging
import re
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)

_OPTION_RE = re.compile(r"\n\(([A-Z])\)\s*")


def _parse_question_and_options(raw: str) -> tuple[str, list[str]]:
    """Split an LVBench question string into the question body and options.

    LVBench questions embed options as ``\\n(A) ...\\n(B) ...`` etc.
    Returns ``(question_text, ["A. ...", "B. ...", ...])``.
    """
    parts = _OPTION_RE.split(raw)
    question = parts[0].strip()
    options = []
    for i in range(1, len(parts), 2):
        letter = parts[i]
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        options.append(f"{letter}. {text}")
    return question, options


def _build_youtube_id_to_mp4(video_dir: Path) -> dict[str, str]:
    """Scan sidecar JSONs to map YouTube video IDs to local mp4 paths."""
    mapping: dict[str, str] = {}
    if not video_dir.exists():
        return mapping
    for json_path in sorted(video_dir.glob("*.json")):
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)
        url = sidecar.get("url", "")
        if "watch?v=" in url:
            yt_id = url.split("watch?v=")[1].split("&")[0]
            mapping[yt_id] = str(mp4_path)
    return mapping


class LVBenchDataset(EvalDataset):
    """Load LVBench JSONL annotations into ``EvalSample`` objects."""

    def load(self):
        dcfg = self.cfg.dataset
        annotation_path = Path(dcfg.annotation_path)
        video_dir = Path(dcfg.video_dir)

        yt_to_mp4 = _build_youtube_id_to_mp4(video_dir)
        log.info("LVBench: mapped %d YouTube IDs to local mp4 files", len(yt_to_mp4))

        with open(annotation_path, encoding="utf-8") as f:
            video_entries = [json.loads(line) for line in f if line.strip()]

        skipped = 0
        for entry in video_entries:
            yt_key = entry["key"]
            mp4_path = yt_to_mp4.get(yt_key)
            if mp4_path is None:
                skipped += len(entry.get("qa", []))
                continue

            video_type = str(entry.get("type", ""))
            video_info = entry.get("video_info", {})
            duration_minutes = float(video_info.get("duration_minutes", 0.0))

            for qa in entry.get("qa", []):
                uid = str(qa["uid"])
                question, options = _parse_question_and_options(qa["question"])
                gt_answer = str(qa["answer"]).strip().upper()
                question_types = list(qa.get("question_type", []))

                self._samples.append(
                    EvalSample(
                        sample_id=uid,
                        video_path=mp4_path,
                        query=question,
                        gt_windows=[],
                        metadata={
                            "qa_format": "mcq",
                            "display_query": "\n".join([question, *options]),
                            "question": question,
                            "options": options,
                            "gt_answer": gt_answer,
                            "uid": uid,
                            "youtube_id": yt_key,
                            "video_type": video_type,
                            "duration_minutes": duration_minutes,
                            "question_type": question_types,
                        },
                    )
                )

        if skipped:
            log.warning(
                "LVBench: skipped %d QA items whose videos are not available locally",
                skipped,
            )
