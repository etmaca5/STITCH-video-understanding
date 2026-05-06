"""Ego4D Moments Queries (MQ) dataset loader."""

import json
import logging
from collections import defaultdict
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)


class Ego4dMqDataset(EvalDataset):

    def load(self):
        dcfg = self.cfg.dataset
        annot_path = Path(dcfg.annotations_dir) / f"moments_{dcfg.split}.json"
        clips_dir = Path(dcfg.clips_dir)

        with open(annot_path) as f:
            data = json.load(f)

        idx = 0
        for video in data["videos"]:
            for clip in video["clips"]:
                clip_uid = clip["clip_uid"]
                video_path = self._resolve_video_path(clips_dir, video_id=clip_uid)

                label_windows = defaultdict(list)
                for annotation in clip["annotations"]:
                    for label_entry in annotation["labels"]:
                        lbl = label_entry["label"]
                        start = label_entry["start_time"]
                        end = label_entry["end_time"]
                        label_windows[lbl].append((start, end))

                for label, windows in label_windows.items():
                    query = label.strip('"').replace("_", " ")
                    self._samples.append(EvalSample(
                        sample_id=f"{clip_uid}_{idx}",
                        video_path=video_path,
                        query=query,
                        gt_windows=windows,
                        metadata={
                            "clip_uid": clip_uid,
                            "video_uid": video["video_uid"],
                            "raw_label": label,
                        },
                    ))
                    idx += 1
