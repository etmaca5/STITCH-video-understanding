"""Save and load evaluation run results."""

import json
import os
import re
import traceback
import uuid

from omegaconf import OmegaConf


def save_run_issues(run_dir, failed_videos, skipped_queries):
    """Write a single JSON record of per-video failures and skipped queries.

    Updated on each checkpoint (partial) and at run end. See also
    failed_videos.json / skipped_queries.json when non-empty (redo_failed_from).
    """
    path = os.path.join(run_dir, "run_issues.json")
    payload = {
        "failed_videos": list(failed_videos),
        "skipped_queries": list(skipped_queries),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_run_crash(run_dir, exc_type, exc_value, tb):
    """Write traceback for an uncaught exception (run_dir must exist)."""
    path = os.path.join(run_dir, "run_crash.json")
    payload = {
        "error": f"{exc_type.__name__}: {exc_value}",
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, tb)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


_MODE_TO_TASK_TYPE = {
    "vlm_qa": "vlm_qa",
    "gebd": "event_detection",
    "embedding": "moment_retrieval",
    "vlm_chunk_selection": "moment_retrieval",
    "lovr_retrieval": "moment_retrieval",
}


def task_type_from_mode(mode):
    """Map an evaluation mode to its task-type folder name.

    Used to organize runs under clean_results/{task_type}/{dataset}/.
    """
    try:
        return _MODE_TO_TASK_TYPE[mode]
    except KeyError:
        raise ValueError(
            f"Unknown evaluation mode {mode!r}; expected one of "
            f"{sorted(_MODE_TO_TASK_TYPE)}"
        )


def _next_run_number(output_dir):
    """Return the next available run number by scanning existing directories."""
    max_num = 0
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            m = re.match(r"^(\d+)-", name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def make_run_dir(output_dir, task_type, dataset_name, notes=""):
    """Create a numbered directory for this evaluation run.

    Layout: ``{output_dir}/{task_type}/{dataset_name}/{run_number}-{notes}-{uniq}``
    (e.g. ``clean_results/moment_retrieval/qvhighlights/3-query_merging-a1b2c3d4``).

    The run number is scoped to the dataset folder, so each dataset's runs
    number from 1 independently. A short random suffix is always appended
    so multiple processes starting together never share the same directory
    (``_next_run_number`` is not atomic across processes).
    """
    dataset_dir = os.path.join(output_dir, task_type, dataset_name)
    num = _next_run_number(dataset_dir)
    slug = notes.replace(" ", "_")[:40] if notes else ""
    parts = [str(num)]
    if slug:
        parts.append(slug)
    parts.append(uuid.uuid4().hex[:8])
    name = "-".join(parts)
    run_dir = os.path.join(dataset_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run(run_dir, cfg, aggregate_metrics, per_query_results,
             subset_ids=None, per_video_data=None):
    """Persist a complete evaluation run.

    Args:
        run_dir: directory to write into (from :func:`make_run_dir`).
        cfg: the Hydra DictConfig for this run.
        aggregate_metrics: dict of metric name -> value.
        per_query_results: list of dicts, one per evaluated sample.
        subset_ids: list of sample IDs if a subset was used, else None.
        per_video_data: dict mapping video_path to chunking data (signal,
            chunk stages, fps, etc.) for visualization.
    """
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(aggregate_metrics, f, indent=2)

    with open(os.path.join(run_dir, "per_query_results.json"), "w") as f:
        json.dump(per_query_results, f, indent=2)

    if subset_ids is not None:
        with open(os.path.join(run_dir, "subset_ids.json"), "w") as f:
            json.dump(list(subset_ids), f, indent=2)

    if per_video_data is not None:
        with open(os.path.join(run_dir, "per_video_data.json"), "w") as f:
            json.dump(per_video_data, f, indent=2)


def load_run(run_dir):
    """Load a previously saved evaluation run.

    Returns:
        dict with keys ``config``, ``metrics``, ``per_query_results``,
        and optionally ``subset_ids``.
    """
    with open(os.path.join(run_dir, "config.yaml")) as f:
        config = OmegaConf.load(f)

    with open(os.path.join(run_dir, "metrics.json")) as f:
        metrics = json.load(f)

    with open(os.path.join(run_dir, "per_query_results.json")) as f:
        per_query = json.load(f)

    result = {
        "config": config,
        "metrics": metrics,
        "per_query_results": per_query,
    }

    subset_path = os.path.join(run_dir, "subset_ids.json")
    if os.path.exists(subset_path):
        with open(subset_path) as f:
            result["subset_ids"] = json.load(f)

    video_data_path = os.path.join(run_dir, "per_video_data.json")
    if os.path.exists(video_data_path):
        with open(video_data_path) as f:
            result["per_video_data"] = json.load(f)

    return result
