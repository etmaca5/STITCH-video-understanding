"""Focused tests for the MLVU dataset loader and metric."""

import json
import os
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.mlvu import MLVUDataset
from metrics import compute_mlvu_metrics


def _write_dev_fixture(root):
    """Lay out a tiny MLVU-dev tree under *root* with two MCQ tasks."""
    json_dir = root / "json"
    video_dir = root / "video"
    json_dir.mkdir(parents=True)
    (video_dir / "1_plotQA").mkdir(parents=True)
    (video_dir / "2_needle").mkdir(parents=True)

    plot_items = [
        {
            "video": "vid1.mp4",
            "duration": 100.0,
            "question": "What color is the hat?",
            "candidates": ["Red", "Blue", "Green", "Yellow"],
            "answer": "Blue",
            "question_type": "plotQA",
        },
        {
            "video": "vid2.mp4",
            "duration": 200.0,
            "question": "Who enters the room first?",
            "candidates": ["Alice", "Bob", "Carol", "Dan"],
            "answer": "Dan",
            "question_type": "plotQA",
        },
    ]
    needle_items = [
        {
            "video": "needle1.mp4",
            "duration": 300.0,
            "question": "What number appears at minute 5?",
            "candidates": ["7", "12", "33", "99"],
            "answer": "33",
            "question_type": "findNeedle",
        },
    ]
    (json_dir / "1_plotQA.json").write_text(json.dumps(plot_items))
    (json_dir / "2_needle.json").write_text(json.dumps(needle_items))

    for name in ("vid1.mp4", "vid2.mp4"):
        (video_dir / "1_plotQA" / name).touch()
    (video_dir / "2_needle" / "needle1.mp4").touch()

    return json_dir, video_dir


def _build_cfg(root, **overrides):
    base = {
        "dataset": {
            "name": "mlvu",
            "dataset_root": str(root),
            "split": "dev",
            "json_dir": str(root / "json"),
            "video_dir": str(root / "video"),
            "test_dataset_root": str(root / "test_root"),
            "test_annotation_path": str(root / "test_root" / "test_mcq_gt.json"),
            "test_video_dir": str(root / "test_root" / "video"),
            "tasks": None,
            "metrics": {"type": "mlvu_mcq"},
        }
    }
    base["dataset"].update(overrides)
    return OmegaConf.create(base)


def _write_test_fixture(root):
    """Lay out a tiny MLVU-test tree under *root/test_root*."""
    test_root = root / "test_root"
    video_dir = test_root / "video"
    video_dir.mkdir(parents=True)

    items = [
        {
            "video": "test_surveil_27.mp4",
            "duration": 297.57,
            "question": "Is there anything unusual?",
            "candidates": ["Fighting", "Shoplifting", "Arrest",
                           "Normal", "Shooting", "Vandalism"],
            "answer": "Shoplifting",
            "question_type": "anomaly_reco",
            "question_id": "AR_0",
        },
        {
            "video": "test_tutorial_1.mp4",
            "duration": 120.0,
            "question": "What comes next?",
            "candidates": ["Step1", "Step2", "Step3", "Step4", "Step5", "Step6"],
            "answer": "Step5",
            "question_type": "tutorialQA",
            "question_id": "TQ_7",
        },
    ]
    (test_root / "test_mcq_gt.json").write_text(json.dumps(items))
    for it in items:
        (video_dir / it["video"]).touch()
    return test_root


def test_loader_builds_expected_samples(tmp_path):
    _write_dev_fixture(tmp_path)
    ds = MLVUDataset(_build_cfg(tmp_path, tasks=["plotQA", "findNeedle"]))
    ds.load()

    assert len(ds) == 3

    plot_first = ds.samples[0]
    assert plot_first.sample_id == "plotQA_0"
    assert plot_first.metadata["question_type"] == "plotQA"
    assert plot_first.metadata["gt_answer"] == "B"
    assert plot_first.metadata["options"][0] == "A. Red"
    assert plot_first.metadata["display_query"].startswith("What color is the hat?")
    assert plot_first.video_path.endswith("/1_plotQA/vid1.mp4")
    assert plot_first.gt_windows == []

    plot_second = ds.samples[1]
    assert plot_second.sample_id == "plotQA_1"
    assert plot_second.metadata["gt_answer"] == "D"

    needle = ds.samples[2]
    assert needle.sample_id == "findNeedle_0"
    assert needle.metadata["question_type"] == "findNeedle"
    assert needle.metadata["gt_answer"] == "C"


def test_loader_rejects_unknown_split(tmp_path):
    _write_dev_fixture(tmp_path)
    ds = MLVUDataset(_build_cfg(tmp_path, split="train", tasks=["plotQA"]))
    with pytest.raises(ValueError):
        ds.load()


def test_loader_test_split_builds_expected_samples(tmp_path):
    _write_test_fixture(tmp_path)
    ds = MLVUDataset(_build_cfg(tmp_path, split="test"))
    ds.load()

    assert len(ds) == 2

    first = ds.samples[0]
    assert first.sample_id == "AR_0"                  # uses question_id
    assert first.metadata["question_type"] == "anomaly_reco"
    assert first.metadata["gt_answer"] == "B"         # Shoplifting is index 1
    assert len(first.metadata["options"]) == 6        # test has 6 candidates
    assert first.metadata["options"][-1] == "F. Vandalism"
    assert first.metadata["task"] == "test"
    assert first.video_path.endswith("/test_root/video/test_surveil_27.mp4")

    second = ds.samples[1]
    assert second.sample_id == "TQ_7"
    assert second.metadata["gt_answer"] == "E"        # Step5 is index 4
    assert second.metadata["question_type"] == "tutorialQA"


def test_loader_test_split_tasks_filter(tmp_path):
    _write_test_fixture(tmp_path)
    ds = MLVUDataset(_build_cfg(tmp_path, split="test", tasks=["tutorialQA"]))
    ds.load()
    assert len(ds) == 1
    assert ds.samples[0].sample_id == "TQ_7"


def test_loader_tasks_filter(tmp_path):
    _write_dev_fixture(tmp_path)
    ds = MLVUDataset(_build_cfg(tmp_path, tasks=["plotQA"]))
    ds.load()
    assert len(ds) == 2
    assert {s.metadata["question_type"] for s in ds.samples} == {"plotQA"}


def test_loader_raises_on_unmatched_answer(tmp_path):
    json_dir, video_dir = _write_dev_fixture(tmp_path)
    bad = [
        {
            "video": "vid1.mp4",
            "duration": 1.0,
            "question": "Q?",
            "candidates": ["X", "Y"],
            "answer": "Z",
            "question_type": "plotQA",
        }
    ]
    (json_dir / "1_plotQA.json").write_text(json.dumps(bad))
    ds = MLVUDataset(_build_cfg(tmp_path, tasks=["plotQA"]))
    with pytest.raises(ValueError):
        ds.load()


def test_compute_mlvu_metrics_macro_average():
    # Task A: 1/2 correct (50%); Task B: 3/3 correct (100%).
    # Macro avg = 75%; micro avg = 4/5 = 80%.
    per_query = [
        {
            "predicted_answer": "A",
            "gt_answer": "A",
            "metadata": {"question_type": "plotQA",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
        {
            "predicted_answer": "B",
            "gt_answer": "A",
            "metadata": {"question_type": "plotQA",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
        {
            "predicted_answer": "C",
            "gt_answer": "C",
            "metadata": {"question_type": "findNeedle",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
        {
            "predicted_answer": "D",
            "gt_answer": "D",
            "metadata": {"question_type": "findNeedle",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
        {
            "predicted_answer": "B",
            "gt_answer": "B",
            "metadata": {"question_type": "findNeedle",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
    ]
    m = compute_mlvu_metrics(per_query)
    assert m["Accuracy"] == pytest.approx(0.75)
    assert m["Accuracy_micro"] == pytest.approx(0.8)
    assert m["Accuracy_plotqa"] == pytest.approx(0.5)
    assert m["Accuracy_findneedle"] == pytest.approx(1.0)


def test_compute_mlvu_metrics_parses_freeform_answer():
    per_query = [
        {
            "predicted_answer": "The best answer is (B) blue",
            "gt_answer": "B",
            "metadata": {"question_type": "plotQA",
                         "options": ["A. x", "B. y", "C. z", "D. w"]},
        },
    ]
    m = compute_mlvu_metrics(per_query)
    assert m["Accuracy"] == pytest.approx(1.0)
