"""MLVU dataset loader for multiple-choice long-video QA evaluation.

Supports two splits, both MCQ-only:

* ``split=dev``  — 7 MCQ tasks across ``1_plotQA.json`` ... ``7_topic_reasoning.json``
  (2,174 questions).  Videos live under ``video/<task>/``.
* ``split=test`` — 502 MCQ questions scored locally against
  ``test-ground-truth/test_mcq_gt.json``.  Videos live in a flat ``video/`` dir.
  Test items have 6 candidates and 9 question_types (adds ``sportsQA``,
  ``tutorialQA``; uses ``needleQA`` instead of dev's ``findNeedle``).

Open-ended generation tasks (dev ``8_sub_scene`` SSC, ``9_summary`` VS; test
``test_generation_tasks.json``) are excluded — the official MLVU eval scores
them with a GPT-4 judge, which is out of scope here.

Reference: https://github.com/JUNJIE99/MLVU
"""

import json
import logging
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)


# (json_filename, video_subdir, question_type) for the 7 MCQ dev tasks, in
# the canonical MLVU order so per-task metric output is stable.
_MLVU_DEV_MCQ_TASKS: list[tuple[str, str, str]] = [
    ("1_plotQA.json",          "1_plotQA",          "plotQA"),
    ("2_needle.json",          "2_needle",          "findNeedle"),
    ("3_ego.json",             "3_ego",             "ego"),
    ("4_count.json",           "4_count",           "count"),
    ("5_order.json",           "5_order",           "order"),
    ("6_anomaly_reco.json",    "6_anomaly_reco",    "anomaly_reco"),
    ("7_topic_reasoning.json", "7_topic_reasoning", "topic_reasoning"),
]


def _answer_letter(candidates: list[str], answer_text: str) -> str:
    """Map MLVU's free-text ``answer`` field to its option letter."""
    answer_norm = str(answer_text).strip()
    for idx, cand in enumerate(candidates):
        if str(cand).strip() == answer_norm:
            return chr(ord("A") + idx)
    raise ValueError(
        f"MLVU answer {answer_text!r} not found among candidates {candidates!r}"
    )


def _build_mcq_sample(
    self_dataset,
    item: dict,
    sample_id: str,
    video_dir: Path,
    task: str,
    question_type: str,
) -> EvalSample:
    """Shared EvalSample construction for both MLVU splits."""
    candidates = list(item["candidates"])
    gt_letter = _answer_letter(candidates, item["answer"])
    options = [
        f"{chr(ord('A') + i)}. {str(c).strip()}"
        for i, c in enumerate(candidates)
    ]
    question = str(item["question"]).strip()
    video_path = self_dataset._resolve_video_path(
        video_dir, file_name=item["video"],
    )
    return EvalSample(
        sample_id=sample_id,
        video_path=video_path,
        query=question,
        gt_windows=[],
        metadata={
            "qa_format": "mcq",
            "prompt_dataset": "mlvu",
            "display_query": "\n".join([question, *options]),
            "question": question,
            "options": options,
            "gt_answer": gt_letter,
            "question_type": question_type,
            "task": task,
            "video_file": str(item["video"]),
            "duration": float(item.get("duration", 0.0)),
        },
    )


class MLVUDataset(EvalDataset):
    """Load MLVU MCQ annotations into ``EvalSample`` objects."""

    def load(self):
        dcfg = self.cfg.dataset
        split = str(dcfg.get("split", "dev")).lower()

        task_filter = dcfg.get("tasks")
        if task_filter is not None:
            task_filter = {str(t) for t in task_filter}

        if split == "dev":
            self._load_dev(dcfg, task_filter)
        elif split == "test":
            self._load_test(dcfg, task_filter)
        else:
            raise ValueError(
                f"dataset.split={split!r} is not supported; expected 'dev' or 'test'."
            )

        log.info("MLVU: loaded %d MCQ samples (split=%s)", len(self._samples), split)

    def _load_dev(self, dcfg, task_filter):
        json_dir = Path(dcfg.json_dir)
        video_dir = Path(dcfg.video_dir)

        for json_name, video_subdir, qtype in _MLVU_DEV_MCQ_TASKS:
            if task_filter is not None and qtype not in task_filter:
                continue

            json_path = json_dir / json_name
            with open(json_path, encoding="utf-8") as f:
                items = json.load(f)

            task_video_dir = video_dir / video_subdir
            for idx, item in enumerate(items):
                self._samples.append(
                    _build_mcq_sample(
                        self,
                        item,
                        sample_id=f"{qtype}_{idx}",
                        video_dir=task_video_dir,
                        task=video_subdir,
                        question_type=qtype,
                    )
                )

    def _load_test(self, dcfg, task_filter):
        annotation_path = Path(dcfg.test_annotation_path)
        video_dir = Path(dcfg.test_video_dir)

        with open(annotation_path, encoding="utf-8") as f:
            items = json.load(f)

        for item in items:
            qtype = str(item["question_type"]).strip()
            if task_filter is not None and qtype not in task_filter:
                continue
            self._samples.append(
                _build_mcq_sample(
                    self,
                    item,
                    sample_id=str(item["question_id"]),
                    video_dir=video_dir,
                    task="test",
                    question_type=qtype,
                )
            )
