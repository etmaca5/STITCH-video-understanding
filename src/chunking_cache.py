import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_VERSION = "window_embeddings_v1"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache" / "window_embeddings"


class WindowCacheMissError(Exception):
    """Raised when window embeddings are required from cache but not present."""


def _normalize_video_path(video_path):
    return str(Path(video_path).expanduser().resolve())


def _video_fingerprint(video_path):
    normalized_path = _normalize_video_path(video_path)
    stat_result = os.stat(normalized_path)
    return {
        "path": normalized_path,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _sampling_fingerprint(sample_interval):
    return {"sample_interval": float(sample_interval)}


def _build_request_metadata(video_path, embedding_backend, sample_interval):
    if not hasattr(embedding_backend, "cache_identity"):
        raise ValueError(
            "Embedding backend must define cache_identity() to support caching"
        )
    return {
        "cache_version": CACHE_VERSION,
        "video": _video_fingerprint(video_path),
        "backend": embedding_backend.cache_identity(),
        "sampling": _sampling_fingerprint(sample_interval),
    }


def _request_hash(request_metadata):
    payload = json.dumps(request_metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_cached_payload(payload, request_metadata):
    if not isinstance(payload, dict):
        return False, "cache payload is not a dict"
    if payload.get("request") != request_metadata:
        return False, "cache request metadata mismatch"

    fps = float(payload.get("fps", 0.0))
    total_frames = int(payload.get("total_frames", 0))
    if fps <= 0:
        return False, "cached fps must be > 0"
    if total_frames <= 0:
        return False, "cached total_frames must be > 0"

    window_times = payload.get("window_times")
    window_embeddings = payload.get("window_embeddings")
    if not isinstance(window_times, np.ndarray):
        return False, "cached window_times must be a numpy array"
    if not isinstance(window_embeddings, np.ndarray):
        return False, "cached window_embeddings must be a numpy array"
    if window_embeddings.ndim != 2:
        return False, "cached window_embeddings must be 2-D"
    if len(window_times) != window_embeddings.shape[0]:
        return False, "cached window arrays have inconsistent lengths"
    return True, None


class WindowEmbeddingCache:
    """Persistent cache for per-video window embeddings."""

    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)

    def _entry_paths(self, request_hash):
        shard_dir = self.cache_dir / request_hash[:2]
        stem = shard_dir / request_hash
        return {
            "dir": shard_dir,
            "meta": stem.with_suffix(".json"),
            "arrays": stem.with_suffix(".npz"),
        }

    def load(self, video_path, embedding_backend, sample_interval):
        request_metadata = _build_request_metadata(
            video_path, embedding_backend, sample_interval
        )
        request_hash = _request_hash(request_metadata)
        paths = self._entry_paths(request_hash)
        if not paths["meta"].exists() or not paths["arrays"].exists():
            return None

        try:
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            with np.load(paths["arrays"]) as arrays:
                payload = {
                    "request": meta.get("request"),
                    "fps": meta.get("fps"),
                    "total_frames": meta.get("total_frames"),
                    "window_times": np.asarray(arrays["window_times"], dtype=np.float32),
                    "window_embeddings": np.asarray(
                        arrays["window_embeddings"], dtype=np.float32
                    ),
                }
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Ignoring unreadable window embedding cache: %s", exc)
            return None

        is_valid, reason = _validate_cached_payload(payload, request_metadata)
        if not is_valid:
            log.warning("Ignoring invalid window embedding cache: %s", reason)
            return None

        return payload

    def save(
        self,
        video_path,
        embedding_backend,
        sample_interval,
        fps,
        total_frames,
        window_times,
        window_embeddings,
    ):
        request_metadata = _build_request_metadata(
            video_path, embedding_backend, sample_interval
        )
        request_hash = _request_hash(request_metadata)
        paths = self._entry_paths(request_hash)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        payload = {
            "request": request_metadata,
            "fps": float(fps),
            "total_frames": int(total_frames),
            "window_times": np.asarray(window_times, dtype=np.float32),
            "window_embeddings": np.asarray(window_embeddings, dtype=np.float32),
        }
        is_valid, reason = _validate_cached_payload(payload, request_metadata)
        if not is_valid:
            raise ValueError(f"Refusing to save invalid window embedding cache: {reason}")

        tmp_meta = paths["meta"].with_suffix(".json.tmp")
        tmp_arrays = paths["arrays"].with_suffix(".npz.tmp")
        try:
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "request": request_metadata,
                        "fps": float(fps),
                        "total_frames": int(total_frames),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            with open(tmp_arrays, "wb") as f:
                np.savez_compressed(
                    f,
                    window_times=payload["window_times"],
                    window_embeddings=payload["window_embeddings"],
                )
            os.replace(tmp_meta, paths["meta"])
            os.replace(tmp_arrays, paths["arrays"])
        finally:
            for tmp_path in (tmp_meta, tmp_arrays):
                if tmp_path.exists():
                    tmp_path.unlink()
