"""Ego4D Natural Language Queries (NLQ) dataset loader."""

import json
import logging
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)


class Ego4dNlqDataset(EvalDataset):

    def load(self):
        dcfg = self.cfg.dataset
        annot_path = Path(dcfg.annotations_dir) / f"nlq_{dcfg.split}.json"
        clips_dir = Path(dcfg.clips_dir)

        with open(annot_path) as f:
            data = json.load(f)

        idx = 0
        for video in data["videos"]:
            for clip in video["clips"]:
                clip_uid = clip["clip_uid"]
                video_path = self._resolve_video_path(clips_dir, video_id=clip_uid)

                for annotation in clip["annotations"]:
                    for lang_query in annotation["language_queries"]:
                        if "query" not in lang_query:
                            log.warning(
                                "Skipping language_query without 'query' key "
                                "in clip %s", clip_uid,
                            )
                            continue
                        gt_start = lang_query["clip_start_sec"]
                        gt_end = lang_query["clip_end_sec"]

                        self._samples.append(EvalSample(
                            sample_id=f"{clip_uid}_{idx}",
                            video_path=video_path,
                            query=lang_query["query"],
                            gt_windows=[(gt_start, gt_end)],
                            metadata={
                                "clip_uid": clip_uid,
                                "video_uid": video["video_uid"],
                            },
                        ))
                        idx += 1
