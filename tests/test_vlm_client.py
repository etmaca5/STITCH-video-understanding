"""Tests for OpenAI-compatible VLM request construction."""

from types import SimpleNamespace
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import temporal_abstraction
from temporal_abstraction import TemporalAbstractionLayer
from vlm_client import VIDEO_INPUT_MODE, VLMClient


class _CaptureCreate:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A"),
                )
            ],
        )


def _client(monkeypatch, input_mode="image_url", send_video_metadata=False):
    monkeypatch.setenv("DUMMY_KEY", "test")
    client = VLMClient(
        model="dummy-model",
        base_url="http://example.test/v1",
        api_key_env="DUMMY_KEY",
        input_mode=input_mode,
        send_video_metadata=send_video_metadata,
        max_images_per_request=8,
    )
    capture = _CaptureCreate()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=capture),
        )
    )
    return client, capture


def test_image_mode_sends_separate_image_urls(monkeypatch):
    client, capture = _client(monkeypatch)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert client.query("question", images=[frame]) == "A"

    content = capture.kwargs["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "question"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "extra_body" not in capture.kwargs


def test_video_mode_sends_single_video_url_with_metadata(monkeypatch):
    client, capture = _client(
        monkeypatch,
        input_mode=VIDEO_INPUT_MODE,
        send_video_metadata=True,
    )
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

    assert client.query(
        "question",
        images=frames,
        video_metadata={"fps": 30.0, "total_frames": 300, "duration": 10.0},
        video_frame_indices=[0, 150],
    ) == "A"

    content = capture.kwargs["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["text", "video_url"]
    assert content[1]["video_url"]["url"].startswith("data:video/jpeg;base64,")
    video_kwargs = capture.kwargs["extra_body"]["media_io_kwargs"]["video"]
    assert video_kwargs["do_sample_frames"] is False
    assert video_kwargs["fps"] == 30.0
    assert video_kwargs["total_num_frames"] == 300
    assert video_kwargs["duration"] == 10.0
    assert video_kwargs["frames_indices"] == [0, 150]


def test_video_mode_metadata_body_is_optional(monkeypatch):
    client, capture = _client(monkeypatch, input_mode=VIDEO_INPUT_MODE)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert client.query(
        "question",
        images=[frame],
        video_metadata={"fps": 30.0, "total_frames": 300, "duration": 10.0},
        video_frame_indices=[0],
    ) == "A"

    content = capture.kwargs["messages"][-1]["content"]
    assert content[1]["type"] == "video_url"
    assert "extra_body" not in capture.kwargs


def test_invalid_input_mode_rejected(monkeypatch):
    monkeypatch.setenv("DUMMY_KEY", "test")
    with pytest.raises(ValueError, match="Unsupported VLM input_mode"):
        VLMClient(
            model="dummy-model",
            base_url="http://example.test/v1",
            api_key_env="DUMMY_KEY",
            input_mode="bad",
        )


class _FakeVideoVLM:
    input_mode = VIDEO_INPUT_MODE
    max_images_per_request = 8

    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return "A"


def test_answer_question_passes_video_metadata(monkeypatch):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    monkeypatch.setattr(temporal_abstraction, "_load_frames_by_index", lambda *_: frames)
    vlm = _FakeVideoVLM()
    tal = TemporalAbstractionLayer(
        vlm_client=vlm,
        metadata_config={"include_metadata": False},
        qa_prompt_config={"style": "videomme_mcq"},
    )

    result = tal.answer_question(
        video_path="video.mp4",
        chunks=[(0, 100), (100, 200)],
        fps=30.0,
        question="What happened?",
        video_metadata={"fps": 30.0, "total_frames": 300, "duration": 10.0},
        question_metadata={
            "qa_format": "mcq",
            "question": "What happened?",
            "options": ["A. yes", "B. no"],
        },
        window_frame_indices=[10, 200],
        window_frame_times=[0.33, 6.67],
    )

    assert result["answer"] == "A"
    assert vlm.calls[0]["images"] == frames
    assert vlm.calls[0]["system_prompt"] is None
    assert vlm.calls[0]["video_metadata"]["total_frames"] == 300
    assert vlm.calls[0]["video_frame_indices"] == [10, 200]
