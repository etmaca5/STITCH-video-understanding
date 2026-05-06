"""Precompute cached window embeddings for dataset videos.

Run from the project root:
    python src/precompute_window_embeddings.py
    python src/precompute_window_embeddings.py dataset=activitynet
    python src/precompute_window_embeddings.py sample_size=50
"""

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import Chunking, VideoMAEv2EmbeddingWrapper, load_videomaev2_model
from datasets import build_dataset
from retrieval import InternVideo2Backend

log = logging.getLogger(__name__)


def _load_failed_video_paths(run_dir):
    failed_path = Path(run_dir) / "failed_videos.json"
    if not failed_path.exists():
        raise FileNotFoundError(f"No failed_videos.json found in {run_dir}")

    with open(failed_path, "r", encoding="utf-8") as f:
        failed = json.load(f)

    video_paths = []
    for entry in failed:
        video_path = entry.get("video_path")
        if video_path:
            video_paths.append(str(video_path))

    if not video_paths:
        raise ValueError(f"No failed videos listed in {failed_path}")

    return video_paths


def _load_embedding_backend(cfg, device):
    if cfg.chunking.type != "embedding":
        raise ValueError(
            "precompute_window_embeddings.py currently supports only "
            "chunking.type=embedding"
        )

    backend = cfg.chunking.get("embedding_backend", "internvideo2")
    if backend == "internvideo2":
        return InternVideo2Backend(
            model_dir=cfg.chunking.get("embedding_model_dir", cfg.retrieval.model_dir),
            device=device,
            num_frames=cfg.chunking.get("embedding_num_frames", cfg.retrieval.num_frames),
            orig_num_frames=cfg.chunking.get(
                "embedding_orig_num_frames",
                cfg.retrieval.get("orig_num_frames"),
            ),
            image_size=cfg.chunking.get(
                "embedding_image_size", cfg.retrieval.image_size
            ),
            use_flash_attn=cfg.chunking.get(
                "embedding_use_flash_attn",
                cfg.retrieval.get("use_flash_attn", False),
            ),
        )

    raw_model, _ = load_videomaev2_model(cfg.chunking.model_dir, device=device)
    return VideoMAEv2EmbeddingWrapper(raw_model, device)


def _load_dataset(cfg):
    dataset = build_dataset(cfg)
    dataset.load()

    redo_failed_from = cfg.get("redo_failed_from", None)
    if redo_failed_from is not None:
        failed_video_paths = _load_failed_video_paths(redo_failed_from)
        dataset.filter_video_paths(failed_video_paths)
        log.info(
            "Filtered to %d samples across %d failed videos from %s",
            len(dataset),
            len(set(failed_video_paths)),
            redo_failed_from,
        )

    if cfg.subset_ids is not None:
        ids = list(cfg.subset_ids)
        dataset.filter_subset(ids)
        log.info("Filtered to %d samples (subset)", len(dataset))

    if cfg.sample_size is not None:
        sample_size = int(cfg.sample_size)
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        pool_ids = [s.sample_id for s in dataset]
        if sample_size > len(pool_ids):
            log.warning(
                "sample_size=%d is larger than available samples (%d). Using all available samples.",
                sample_size,
                len(pool_ids),
            )
            sample_size = len(pool_ids)

        rng = np.random.default_rng(int(cfg.sample_seed))
        sampled_ids = rng.choice(pool_ids, size=sample_size, replace=False).tolist()
        dataset.filter_subset(sampled_ids)
        log.info(
            "Sampled %d examples using seed=%d",
            len(sampled_ids),
            int(cfg.sample_seed),
        )

    return dataset


def _cleanup_hydra_log():
    hydra_cfg = HydraConfig.get()
    log_path = Path(f"{hydra_cfg.job.name}.log")
    if log_path.exists():
        log_path.unlink()


def _write_failure_artifacts(failed_videos):
    """Persist precompute failures to a rerunnable directory."""
    failure_dir = Path("precompute_window_embeddings_failures")
    failure_dir.mkdir(exist_ok=True)

    failed_path = failure_dir / "failed_videos.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_videos, f, indent=2)

    summary_path = failure_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "failed_videos": len(failed_videos),
                "failed_videos_path": str(failed_path),
            },
            f,
            indent=2,
        )
    return failure_dir


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    try:
        logging.basicConfig(level=logging.INFO)

        if cfg.chunking.type != "embedding":
            raise ValueError(
                "This cache warmer is meant for the standard embedding window cache. "
                "Use chunking=embedding."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", device)

        dataset = _load_dataset(cfg)
        video_paths = list(dict.fromkeys(str(sample.video_path) for sample in dataset))
        log.info("Dataset: %s  samples: %d", cfg.dataset.name, len(dataset))
        log.info("Unique videos: %d", len(video_paths))

        embedding_backend = _load_embedding_backend(cfg, device)
        chunker = Chunking(chunking_type="embedding")
        sample_interval = float(cfg.chunking.sample_interval)

        hits = 0
        written = 0
        failed_videos = []
        for video_path in tqdm(video_paths, desc="Videos", unit="vid"):
            try:
                window_data = chunker.get_window_embeddings(
                    video_path,
                    embedding_backend,
                    sample_interval=sample_interval,
                )
                if window_data["cache_hit"]:
                    hits += 1
                    log.info("Cache hit: %s", video_path)
                else:
                    written += 1
                    log.info("Cache miss -> wrote: %s", video_path)
            except Exception as exc:
                failed_videos.append({
                    "video_path": str(video_path),
                    "error": str(exc),
                })
                log.exception("Failed to precompute window embeddings for %s", video_path)

        log.info(
            "Finished warming cache: %d videos scanned, %d hits, %d newly written, %d failed",
            len(video_paths),
            hits,
            written,
            len(failed_videos),
        )
        if failed_videos:
            failure_dir = _write_failure_artifacts(failed_videos)
            log.warning(
                "Saved %d failed videos to %s. Re-run only failures with "
                "`python src/precompute_window_embeddings.py redo_failed_from=%s ...`",
                len(failed_videos),
                failure_dir / "failed_videos.json",
                failure_dir,
            )
    finally:
        _cleanup_hydra_log()


if __name__ == "__main__":
    main()
