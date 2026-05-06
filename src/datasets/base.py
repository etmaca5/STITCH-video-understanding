"""Abstract base class for evaluation datasets.

Every dataset loader must produce a flat list of :class:`EvalSample`
objects, each representing one (video, query, ground-truth) triplet.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    """A single evaluation sample.

    Attributes:
        sample_id:  Unique identifier (used for subset filtering / results).
        video_path: Absolute or project-relative path to the video file.
        query:      Natural-language query string.
        gt_windows: Ground-truth temporal windows as ``[(start_sec, end_sec), ...]``.
        metadata:   Optional extra info (dataset-specific).
    """
    sample_id: str
    video_path: str
    query: str
    gt_windows: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EvalDataset(ABC):
    """Interface that every evaluation dataset loader implements."""

    DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mov")

    def __init__(self, cfg):
        self.cfg = cfg
        self._samples: list[EvalSample] = []

    @abstractmethod
    def load(self):
        """Parse annotations and populate ``self._samples``."""

    @property
    def samples(self) -> list[EvalSample]:
        return self._samples

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def filter_subset(self, subset_ids):
        """Keep only samples whose ``sample_id`` is in *subset_ids*."""
        id_set = {str(x) for x in subset_ids}
        self._samples = [s for s in self._samples if s.sample_id in id_set]

    def filter_video_paths(self, video_paths):
        """Keep only samples whose ``video_path`` is in *video_paths*."""
        path_set = {str(x) for x in video_paths}
        self._samples = [s for s in self._samples if s.video_path in path_set]

    def _allowed_video_extensions(self):
        raw = self.cfg.dataset.get(
            "allowed_extensions", list(self.DEFAULT_VIDEO_EXTENSIONS)
        )
        return tuple(str(ext) for ext in raw)

    def _resolve_video_path(self, video_dir, video_id=None, file_name=None):
        """Resolve a video path, preferring existing `.mp4` / `.mov` files."""
        root = Path(video_dir)
        exts = self._allowed_video_extensions()
        exts_lower = {ext.lower() for ext in exts}
        candidates = []

        def _normalize(candidate):
            candidate = Path(candidate)
            if not candidate.is_absolute():
                candidate = root / candidate
            return candidate

        def _append_candidate(candidate):
            candidate = _normalize(candidate)
            if candidate not in candidates:
                candidates.append(candidate)

        def _match_existing(candidate):
            if candidate.exists():
                return candidate

            parent = candidate.parent
            if not parent.exists():
                return None

            pattern = f"{candidate.stem}.*" if candidate.suffix else f"{candidate.name}.*"
            for match in parent.glob(pattern):
                if match.suffix.lower() in exts_lower:
                    return match
            return None

        if file_name:
            file_path = Path(file_name)
            if file_path.suffix:
                _append_candidate(file_path)
            else:
                for ext in exts:
                    _append_candidate(file_path.with_suffix(ext))

        if video_id is not None:
            for ext in exts:
                _append_candidate(f"{video_id}{ext}")

        if not candidates:
            raise ValueError("Must provide either video_id or file_name")

        for candidate in candidates:
            match = _match_existing(candidate)
            if match is not None:
                return str(match)

        return str(candidates[0])
