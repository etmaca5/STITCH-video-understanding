"""LoVR dataset loader for long video-text retrieval evaluation."""

from pathlib import Path

import pyarrow.parquet as pq

from .base import EvalDataset, EvalSample


class LoVRDataset(EvalDataset):
    """Load LoVR clip/video retrieval annotations."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.video_items: list[dict] = []
        self.clip_items: list[dict] = []
        self.video_text_queries: list[EvalSample] = []
        self.clip_text_queries: list[EvalSample] = []

    def load(self):
        dcfg = self.cfg.dataset
        split = str(dcfg.get("split", "test")).lower()
        allowed_splits = {"train", "test"}
        if split not in allowed_splits:
            raise ValueError(
                f"dataset.split must be one of {sorted(allowed_splits)}; got {split!r}"
            )

        dataset_root = Path(dcfg.dataset_root)
        full_video_dir = Path(dcfg.full_video_dir)
        clip_video_dir = Path(dcfg.clip_video_dir)

        raw_video_annotation_path = dcfg.get("video_annotation_path")
        if raw_video_annotation_path:
            video_annotation_path = Path(raw_video_annotation_path)
        else:
            video_annotation_path = dataset_root / "caption_data" / f"video_{split}.parquet"

        raw_clip_annotation_path = dcfg.get("clip_annotation_path")
        if raw_clip_annotation_path:
            clip_annotation_path = Path(raw_clip_annotation_path)
        else:
            clip_annotation_path = dataset_root / "caption_data" / f"clip_{split}.parquet"

        video_rows = pq.read_table(video_annotation_path).to_pylist()
        clip_rows = pq.read_table(clip_annotation_path).to_pylist()

        self._samples = []
        self.video_items = []
        self.clip_items = []
        self.video_text_queries = []
        self.clip_text_queries = []

        for row in video_rows:
            video_id = str(row["vid"])
            video_path = self._resolve_video_path(
                full_video_dir,
                file_name=f"{video_id}.mp4",
            )
            item = {
                "item_id": video_id,
                "video_path": video_path,
                "caption": str(row["cap"]).strip(),
                "theme_info": str(row.get("theme_info", "")).strip(),
                "start_slice_num": int(row.get("start_slice_num", 0)),
                "end_slice_num": int(row.get("end_slice_num", 0)),
                "split": split,
            }
            self.video_items.append(item)

            sample = EvalSample(
                sample_id=f"video_text::{video_id}",
                video_path=video_path,
                query=item["caption"],
                gt_windows=[],
                metadata={
                    "lovr_query_type": "text_to_video",
                    "target_id": video_id,
                    "item_id": video_id,
                    "theme_info": item["theme_info"],
                    "start_slice_num": item["start_slice_num"],
                    "end_slice_num": item["end_slice_num"],
                    "split": split,
                },
            )
            self.video_text_queries.append(sample)
            self._samples.append(sample)

        for row in clip_rows:
            rel_path = str(row["path"])
            clip_id = Path(rel_path).stem
            clip_path = self._resolve_video_path(
                clip_video_dir,
                file_name=rel_path,
            )
            item = {
                "item_id": clip_id,
                "video_id": str(row["vid"]),
                "video_path": clip_path,
                "relative_path": rel_path,
                "caption": str(row["cap"]).strip(),
                "theme_info": str(row.get("theme_info", "")).strip(),
                "slice_num": int(row.get("slice_num", 0)),
                "split": split,
            }
            self.clip_items.append(item)

            sample = EvalSample(
                sample_id=f"clip_text::{clip_id}",
                video_path=clip_path,
                query=item["caption"],
                gt_windows=[],
                metadata={
                    "lovr_query_type": "text_to_clip",
                    "target_id": clip_id,
                    "item_id": clip_id,
                    "video_id": item["video_id"],
                    "relative_path": rel_path,
                    "theme_info": item["theme_info"],
                    "slice_num": item["slice_num"],
                    "split": split,
                },
            )
            self.clip_text_queries.append(sample)
            self._samples.append(sample)

    def filter_subset(self, subset_ids):
        super().filter_subset(subset_ids)
        kept_ids = {sample.sample_id for sample in self._samples}
        self.video_text_queries = [
            sample for sample in self.video_text_queries if sample.sample_id in kept_ids
        ]
        self.clip_text_queries = [
            sample for sample in self.clip_text_queries if sample.sample_id in kept_ids
        ]

    def filter_video_paths(self, video_paths):
        super().filter_video_paths(video_paths)
        kept_paths = {sample.video_path for sample in self._samples}
        self.video_text_queries = [
            sample for sample in self.video_text_queries if sample.video_path in kept_paths
        ]
        self.clip_text_queries = [
            sample for sample in self.clip_text_queries if sample.video_path in kept_paths
        ]
