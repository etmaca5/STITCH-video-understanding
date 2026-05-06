"""Kinetics-GEBD dataset loader for event boundary detection evaluation."""

import logging
import pickle
from pathlib import Path

from .base import EvalDataset, EvalSample

log = logging.getLogger(__name__)


class KineticsGEBDDataset(EvalDataset):
    """Load processed Kinetics-GEBD annotations into ``EvalSample`` objects.

    The processed pkl (``k400_mr345_{split}_min_change_duration0.3.pkl``)
    contains per-video boundary annotations from multiple raters, along with
    fps, duration, and inter-rater consistency scores.

    Only videos that exist on disk are included. This is important because
    Kinetics-400 downloads are often incomplete (some YouTube videos become
    unavailable).
    """

    def load(self):
        dcfg = self.cfg.dataset
        annotation_path = Path(dcfg.annotation_path)
        video_dir = Path(dcfg.video_dir)
        split = str(dcfg.get("split", "val"))
        min_f1 = float(dcfg.get("min_f1_consistency", 0.3))

        with open(annotation_path, "rb") as f:
            gt_dict = pickle.load(f, encoding="latin1")

        n_low_f1 = 0
        n_missing = 0

        for vid_id, info in gt_dict.items():
            f1_avg = float(info.get("f1_consis_avg", 0.0))
            if f1_avg < min_f1:
                n_low_f1 += 1
                continue

            rel_path = info.get("path_video", "")
            video_path = video_dir / split / rel_path

            if not video_path.exists():
                n_missing += 1
                continue

            gt_boundaries_per_rater = []
            for rater_timestamps in info.get("substages_timestamps", []):
                gt_boundaries_per_rater.append(
                    [float(t) for t in rater_timestamps]
                )

            self._samples.append(
                EvalSample(
                    sample_id=str(vid_id),
                    video_path=str(video_path),
                    query="",
                    gt_windows=[],
                    metadata={
                        "gt_boundaries_per_rater": gt_boundaries_per_rater,
                        "fps": float(info["fps"]),
                        "video_duration": float(info["video_duration"]),
                        "num_frames": int(info["num_frames"]),
                        "f1_consis_avg": f1_avg,
                    },
                )
            )

        log.info(
            "Kinetics-GEBD %s: %d annotations total, %d low f1 filtered, "
            "%d missing on disk, %d usable samples",
            split, len(gt_dict), n_low_f1, n_missing, len(self._samples),
        )
