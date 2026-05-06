"""ActivityNet-QA dataset loader for video question answering evaluation."""

import json
from pathlib import Path

from .base import EvalDataset, EvalSample


class ActivityNetQADataset(EvalDataset):

    def load(self):
        dcfg = self.cfg.dataset
        qa_dir = Path(dcfg.qa_dir)
        video_dir = Path(dcfg.video_dir)

        with open(qa_dir / f"{dcfg.split}_q.json") as f:
            questions = json.load(f)
        with open(qa_dir / f"{dcfg.split}_a.json") as f:
            answers = json.load(f)

        answer_map = {a["question_id"]: a for a in answers}

        for q in questions:
            qid = q["question_id"]
            a = answer_map[qid]
            video_path = self._resolve_video_path(
                video_dir, video_id=f"v_{q['video_name']}"
            )
            self._samples.append(EvalSample(
                sample_id=qid,
                video_path=video_path,
                query=q["question"],
                gt_windows=[],
                metadata={
                    "video_name": q["video_name"],
                    "gt_answer": a["answer"],
                    "answer_type": a["type"],
                },
            ))
