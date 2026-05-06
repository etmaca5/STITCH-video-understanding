"""QVHighlights dataset loader for moment retrieval evaluation."""

import json
from pathlib import Path

from .base import EvalDataset, EvalSample


class QVHighlightsDataset(EvalDataset):

    def load(self):
        dcfg = self.cfg.dataset
        data_root = Path(dcfg.data_root)
        annot_path = data_root / "annotations" / f"highlight_{dcfg.split}_release.jsonl"

        with open(annot_path) as f:
            entries = [json.loads(line) for line in f]

        for entry in entries:
            vid = entry["vid"]
            video_path = self._resolve_video_path(
                data_root / "videos", video_id=vid
            )
            gt_windows = [tuple(w) for w in entry["relevant_windows"]]

            self._samples.append(EvalSample(
                sample_id=str(entry["qid"]),
                video_path=video_path,
                query=entry["query"],
                gt_windows=gt_windows,
                metadata={
                    "vid": vid,
                    "duration": entry["duration"],
                    "relevant_clip_ids": entry.get("relevant_clip_ids", []),
                    "saliency_scores": entry.get("saliency_scores", []),
                },
            ))
