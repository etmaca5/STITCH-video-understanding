"""TAPOS dataset loader for generic event boundary detection evaluation."""

import json
import logging
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)


class TaposGEBDDataset(EvalDataset):
    """Load processed TAPOS annotations into ``EvalSample`` objects.

    Expects ``tapos_val_gebd.json`` produced by the TAPOS preprocessing
    script, which contains per-clip boundary annotations from one or more
    raters, along with fps, duration, and frame counts.
    """

    def load(self):
        dcfg = self.cfg.dataset
        annotation_path = Path(dcfg.annotation_path)
        video_dir = Path(dcfg.video_dir)

        with open(annotation_path) as f:
            data = json.load(f)

        clips = data["clips"]
        n_missing = 0

        for clip in clips:
            video_path = video_dir / Path(clip["video_path"]).name

            if not video_path.exists():
                n_missing += 1
                continue

            gt_boundaries_per_rater = [
                [float(t) for t in rater]
                for rater in clip["gt_boundaries_per_rater"]
            ]

            self._samples.append(
                EvalSample(
                    sample_id=clip["sample_id"],
                    video_path=str(video_path),
                    query="",
                    gt_windows=[],
                    metadata={
                        "gt_boundaries_per_rater": gt_boundaries_per_rater,
                        "fps": float(clip["fps"]),
                        "video_duration": float(clip["video_duration"]),
                        "num_frames": int(clip["num_frames"]),
                        "action": clip.get("action"),
                    },
                )
            )

        log.info(
            "TAPOS GEBD: %d clips total, %d missing on disk, %d usable samples",
            len(clips), n_missing, len(self._samples),
        )
