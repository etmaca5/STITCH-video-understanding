"""Video-MME dataset loader for multiple-choice video QA evaluation."""

from pathlib import Path

import pyarrow.parquet as pq

from .base import EvalDataset, EvalSample


def _format_mcq_query(question: str, options: list[str]) -> str:
    lines = [str(question).strip()]
    lines.extend(str(option).strip() for option in options if str(option).strip())
    return "\n".join(lines)


class VideoMMEDataset(EvalDataset):
    """Load Video-MME parquet annotations into ``EvalSample`` objects."""

    def load(self):
        dcfg = self.cfg.dataset
        annotation_path = Path(dcfg.annotation_path)
        video_dir = Path(dcfg.video_dir)
        subtitle_dir = Path(dcfg.subtitle_dir)
        duration_split = str(dcfg.get("duration_split", "all")).lower()

        allowed_splits = {"all", "short", "medium", "long"}
        if duration_split not in allowed_splits:
            raise ValueError(
                f"dataset.duration_split must be one of {sorted(allowed_splits)}; "
                f"got {duration_split!r}"
            )

        rows = pq.read_table(annotation_path).to_pylist()
        for row in rows:
            duration_label = str(row.get("duration", "")).lower()
            if duration_split != "all" and duration_label != duration_split:
                continue

            video_stem = str(row["videoID"])
            video_path = self._resolve_video_path(video_dir, video_id=video_stem)
            subtitle_path = subtitle_dir / f"{video_stem}.srt"
            options = list(row.get("options") or [])
            question = str(row["question"]).strip()

            self._samples.append(
                EvalSample(
                    sample_id=str(row["question_id"]),
                    video_path=video_path,
                    query=question,
                    gt_windows=[],
                    metadata={
                        "qa_format": "mcq",
                        "prompt_dataset": "videomme",
                        "display_query": _format_mcq_query(question, options),
                        "question": question,
                        "options": options,
                        "gt_answer": str(row["answer"]).strip(),
                        "question_id": str(row["question_id"]),
                        "video_id": str(row["video_id"]),
                        "video_stem": video_stem,
                        "duration_label": duration_label,
                        "domain": str(row.get("domain", "")),
                        "sub_category": str(row.get("sub_category", "")),
                        "task_type": str(row.get("task_type", "")),
                        "subtitle_path": str(subtitle_path),
                    },
                )
            )
