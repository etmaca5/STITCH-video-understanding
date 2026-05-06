"""LongVideoBench dataset loader for multiple-choice video QA evaluation."""

import json
from pathlib import Path

from .base import EvalDataset, EvalSample


def _format_options(candidates: list[str]) -> list[str]:
    """Return answer options prefixed with their choice letters."""
    formatted = []
    for idx, candidate in enumerate(candidates):
        letter = chr(ord("A") + idx)
        formatted.append(f"{letter}. {str(candidate).strip()}")
    return formatted


class LongVideoBenchDataset(EvalDataset):
    """Load LongVideoBench JSON annotations into ``EvalSample`` objects."""

    def load(self):
        dcfg = self.cfg.dataset
        split = str(dcfg.get("split", "val")).lower()
        allowed_splits = {"val", "test"}
        if split not in allowed_splits:
            raise ValueError(
                f"dataset.split must be one of {sorted(allowed_splits)}; got {split!r}"
            )

        raw_annotation_path = dcfg.get("annotation_path")
        if raw_annotation_path:
            annotation_path = Path(raw_annotation_path)
        else:
            dataset_root = dcfg.get("dataset_root")
            if dataset_root is None:
                raise ValueError(
                    "LongVideoBench requires either dataset.annotation_path or "
                    "dataset.dataset_root"
                )
            annotation_name = {
                "val": "lvb_val.json",
                "test": "lvb_test_wo_gt.json",
            }[split]
            annotation_path = Path(dataset_root) / annotation_name

        video_dir = Path(dcfg.video_dir)
        subtitle_dir = Path(dcfg.subtitle_dir)

        with open(annotation_path, encoding="utf-8") as f:
            rows = json.load(f)

        for row in rows:
            candidates = list(row.get("candidates") or [])
            if not candidates:
                continue

            correct_choice = row.get("correct_choice")
            gt_answer = None
            if correct_choice is not None:
                gt_answer = chr(ord("A") + int(correct_choice))

            video_path = self._resolve_video_path(
                video_dir,
                file_name=row["video_path"],
            )
            subtitle_path = subtitle_dir / str(row["subtitle_path"])
            question = str(row["question"]).strip()
            options = _format_options(candidates)

            metadata = {
                "qa_format": "mcq",
                "mcq_eval": "longvideobench",
                "prompt_dataset": "longvideobench",
                "display_query": "\n".join([question, *options]),
                "question": question,
                "options": options,
                "question_id": str(row["id"]),
                "video_id": str(row["video_id"]),
                "split": split,
                "duration": float(row["duration"]),
                "duration_group": str(row["duration_group"]),
                "question_category": str(row["question_category"]),
                "topic_category": str(row.get("topic_category", "")),
                "level": str(row.get("level", "")),
                "position": str(row.get("position", "")),
                "question_wo_referring_query": str(
                    row.get("question_wo_referring_query", "")
                ),
                "view_count": row.get("view_count"),
                "subtitle_path": str(subtitle_path),
                "starting_timestamp_for_subtitles": float(
                    row.get("starting_timestamp_for_subtitles", 0.0)
                ),
                "include_subtitles_in_prompt": True,
            }
            if gt_answer is not None:
                metadata["gt_answer"] = gt_answer
                metadata["correct_choice"] = int(correct_choice)

            self._samples.append(
                EvalSample(
                    sample_id=str(row["id"]),
                    video_path=video_path,
                    query=question,
                    gt_windows=[],
                    metadata=metadata,
                )
            )
