"""Custom dataset loader for user-provided videos, queries, and windows."""

import json
from pathlib import Path

from .base import EvalDataset, EvalSample


class CustomDataset(EvalDataset):

    @staticmethod
    def _normalize_window(window, sample_id, field_name):
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(
                f"Sample {sample_id} has invalid {field_name}: {window!r}. "
                "Expected [start_sec, end_sec]."
            )
        return (float(window[0]), float(window[1]))

    def _make_gt_windows(self, entry, sample_id):
        if "gt_windows" in entry:
            gt_windows = [
                self._normalize_window(window, sample_id, "gt_windows")
                for window in entry["gt_windows"]
            ]
        elif "timestamp" in entry:
            timestamp = entry["timestamp"]
            if (
                isinstance(timestamp, (list, tuple))
                and len(timestamp) == 2
                and not isinstance(timestamp[0], (list, tuple))
            ):
                gt_windows = [
                    self._normalize_window(timestamp, sample_id, "timestamp")
                ]
            else:
                gt_windows = [
                    self._normalize_window(window, sample_id, "timestamp")
                    for window in timestamp
                ]
        else:
            raise ValueError(
                f"Sample {sample_id} must define 'timestamp' or 'gt_windows'"
            )

        if not gt_windows:
            raise ValueError(f"Sample {sample_id} has no gt_windows")
        return gt_windows

    def load(self):
        dcfg = self.cfg.dataset
        annotation_path = Path(dcfg.annotation_path)
        video_dir = Path(dcfg.video_dir)
        target_split = str(dcfg.get("split", "all"))

        with open(annotation_path) as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            if "videos" in payload:
                entries = []
                for video_entry in payload["videos"]:
                    base = {
                        "video_id": video_entry.get("video_id"),
                        "file_name": video_entry.get("file_name"),
                        "video_path": video_entry.get("video_path"),
                        "split": video_entry.get("split", "all"),
                        "metadata": dict(video_entry.get("metadata", {})),
                    }
                    for query_idx, query_entry in enumerate(
                        video_entry.get("queries", [])
                    ):
                        merged = dict(base)
                        merged["metadata"] = dict(base["metadata"])
                        merged.update(query_entry)
                        merged["metadata"].update(query_entry.get("metadata", {}))
                        merged.setdefault(
                            "sample_id",
                            f"{base.get('video_id', 'sample')}_{query_idx}",
                        )
                        entries.append(merged)
            else:
                entries = payload.get("samples", [])
        else:
            entries = payload

        for idx, entry in enumerate(entries):
            metadata = dict(entry.get("metadata", {}))
            entry_split = str(entry.get("split", metadata.get("split", "all")))
            if target_split not in ("all", "*") and entry_split not in (
                target_split, "all"
            ):
                continue

            sample_id = str(
                entry.get("sample_id")
                or f"{entry.get('video_id', 'sample')}_{idx}"
            )
            video_path = entry.get("video_path")
            if video_path is None:
                video_path = self._resolve_video_path(
                    video_dir,
                    video_id=entry.get("video_id"),
                    file_name=entry.get("file_name"),
                )

            gt_windows = self._make_gt_windows(entry, sample_id)

            metadata.setdefault("video_id", entry.get("video_id"))
            metadata.setdefault("file_name", entry.get("file_name"))

            self._samples.append(EvalSample(
                sample_id=sample_id,
                video_path=str(video_path),
                query=entry["query"].strip(),
                gt_windows=gt_windows,
                metadata=metadata,
            ))
