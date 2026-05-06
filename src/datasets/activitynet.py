"""ActivityNet Captions dataset loader for temporal grounding evaluation."""

import json
from pathlib import Path

from .base import EvalDataset, EvalSample


class ActivityNetCaptionsDataset(EvalDataset):

    def load(self):
        dcfg = self.cfg.dataset
        captions_path = Path(dcfg.captions_dir) / f"{dcfg.split}.json"
        video_dir = Path(dcfg.video_dir)

        with open(captions_path) as f:
            data = json.load(f)

        for video_id, info in data.items():
            video_path = self._resolve_video_path(video_dir, video_id=video_id)

            for i, (timestamp, sentence) in enumerate(
                zip(info["timestamps"], info["sentences"])
            ):
                self._samples.append(EvalSample(
                    sample_id=f"{video_id}_{i}",
                    video_path=video_path,
                    query=sentence.strip(),
                    gt_windows=[tuple(timestamp)],
                    metadata={
                        "video_id": video_id,
                        "duration": info["duration"],
                    },
                ))
