"""Tests for VLM prompt builders and metadata helpers in temporal_abstraction.py."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from temporal_abstraction import (
    _resolve_metadata_flag,
    _collect_per_chunk_subtitles,
    _build_frame_metadata_block,
    _build_iterative_frame_metadata_block,
    build_chunk_selection_prompt,
    build_qa_prompt,
    build_subtitle_only_prompt,
    build_videomme_mcq_prompt_frames,
    build_iterative_final_prompt,
    TemporalAbstractionLayer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

CHUNKS = [(0, 300), (300, 750), (750, 1200)]
FPS = 30.0

SRT_CONTENT = """\
1
00:00:01,000 --> 00:00:08,000
Hello and welcome to the show.

2
00:00:12,000 --> 00:00:22,000
Today we are going to cook something special.

3
00:00:28,000 --> 00:00:38,000
Let me grab the ingredients from the shelf.
"""


@pytest.fixture
def srt_file(tmp_path):
    p = tmp_path / "test.srt"
    p.write_text(SRT_CONTENT)
    return str(p)


# ── _resolve_metadata_flag ────────────────────────────────────────────────

class TestResolveMetadataFlag:
    def test_master_true_no_override(self):
        cfg = {"include_metadata": True}
        assert _resolve_metadata_flag(cfg, "timestamps") is True
        assert _resolve_metadata_flag(cfg, "subtitles") is True

    def test_master_false_no_override(self):
        cfg = {"include_metadata": False}
        assert _resolve_metadata_flag(cfg, "timestamps") is False
        assert _resolve_metadata_flag(cfg, "cosine_similarity") is False

    def test_override_true_wins(self):
        cfg = {"include_metadata": False, "subtitles": True}
        assert _resolve_metadata_flag(cfg, "subtitles") is True

    def test_override_false_wins(self):
        cfg = {"include_metadata": True, "subtitles": False}
        assert _resolve_metadata_flag(cfg, "subtitles") is False

    def test_empty_config_defaults_true(self):
        assert _resolve_metadata_flag({}, "timestamps") is True

    def test_none_config_key_falls_through(self):
        cfg = {"include_metadata": True, "timestamps": None}
        assert _resolve_metadata_flag(cfg, "timestamps") is True


# ── _collect_per_chunk_subtitles ──────────────────────────────────────────

class TestCollectPerChunkSubtitles:
    def test_no_subtitle_path(self):
        result = _collect_per_chunk_subtitles(None, CHUNKS, FPS)
        assert result == ["", "", ""]

    def test_missing_file(self):
        result = _collect_per_chunk_subtitles("/nonexistent.srt", CHUNKS, FPS)
        assert result == ["", "", ""]

    def test_aligned_subtitles(self, srt_file):
        # Chunk 0: 0-10s -> matches sub 1 (1-8s)
        # Chunk 1: 10-25s -> matches sub 2 (12-22s)
        # Chunk 2: 25-40s -> matches sub 3 (28-38s)
        result = _collect_per_chunk_subtitles(srt_file, CHUNKS, FPS)
        assert "Hello and welcome" in result[0]
        assert "cook something special" in result[1]
        assert "grab the ingredients" in result[2]

    def test_empty_chunks(self, srt_file):
        result = _collect_per_chunk_subtitles(srt_file, [], FPS)
        assert result == []


# ── build_chunk_selection_prompt ──────────────────────────────────────────

class TestBuildChunkSelectionPrompt:
    def test_all_metadata_enabled(self, srt_file):
        cfg = {"include_metadata": True}
        video_meta = {"duration": 40.0, "subtitle_path": srt_file}
        scores = [0.42, 0.78, 0.35]
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "How to cook?",
            video_metadata=video_meta,
            metadata_config=cfg,
            chunk_scores=scores,
        )
        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "[0.0s - 10.0s]" in prompt
        assert "cos relevance: 0.42" in prompt
        assert "cos relevance: 0.78" in prompt
        assert 'Subtitles: "Hello and welcome' in prompt
        assert "Video duration: 40.0s" in prompt
        assert 'Query: "How to cook?"' in prompt

    def test_metadata_disabled(self):
        cfg = {"include_metadata": False}
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            metadata_config=cfg,
        )
        assert "Chunk" not in prompt.split("\n")[1]
        assert "cos relevance" not in prompt
        assert "Video duration" not in prompt
        assert "Subtitles" not in prompt

    def test_selective_override(self, srt_file):
        cfg = {"include_metadata": True, "cosine_similarity": False}
        video_meta = {"duration": 40.0, "subtitle_path": srt_file}
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            video_metadata=video_meta,
            metadata_config=cfg,
            chunk_scores=[0.5, 0.6, 0.7],
        )
        assert "cos relevance" not in prompt
        assert "[0.0s - 10.0s]" in prompt
        assert "Subtitles" in prompt

    def test_no_scores_available(self):
        cfg = {"include_metadata": True}
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            metadata_config=cfg,
        )
        assert "cos relevance" not in prompt
        assert "Chunk 1" in prompt

    def test_no_subtitle_path(self):
        cfg = {"include_metadata": True}
        video_meta = {"duration": 40.0}
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            video_metadata=video_meta,
            metadata_config=cfg,
        )
        assert "Subtitles" not in prompt

    def test_multi_frames_per_chunk(self):
        cfg = {"include_metadata": True}
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            metadata_config=cfg,
            frames_per_chunk=2,
        )
        assert "images 1-2" in prompt
        assert "images 3-4" in prompt

    def test_chunk_fields_still_work(self):
        cfg = {"include_metadata": True, "chunk_fields": ["caption"]}
        chunk_meta = [
            {"caption": "person cooking"},
            {"caption": "plating food"},
            {},
        ]
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "query",
            chunk_metadata=chunk_meta,
            metadata_config=cfg,
        )
        assert "caption: person cooking" in prompt
        assert "caption: plating food" in prompt

    def test_instruction_text(self):
        prompt = build_chunk_selection_prompt(CHUNKS, FPS, "query")
        assert "Choose the single chunk" in prompt
        assert "Respond with only the chunk number" in prompt


# ── _build_frame_metadata_block ───────────────────────────────────────────

class TestBuildFrameMetadataBlock:
    def test_all_metadata_enabled(self, srt_file):
        cfg = {"include_metadata": True}
        q_meta = {"subtitle_path": srt_file}
        v_meta = {"duration": 40.0}
        scores = [0.78, 0.65, 0.50]
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0, 30.0],
            frame_scores=scores,
            video_metadata=v_meta,
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "Frame timestamps:" in block
        assert "Frame 1: 5.0s" in block
        assert "(query-match relevance score: 0.78)" in block
        assert "Video duration: 40.0s" in block
        assert 'Subtitles: "Hello and welcome' in block

    def test_metadata_disabled(self):
        cfg = {"include_metadata": False}
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0],
            metadata_config=cfg,
        )
        assert block == ""

    def test_cosine_sim_without_timestamps(self):
        cfg = {
            "include_metadata": True,
            "timestamps": False,
            "subtitles": False,
            "video_duration": False,
        }
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0, 30.0],
            frame_scores=[0.9, 0.5, 0.2],
            metadata_config=cfg,
        )
        assert "Frame timestamps:" in block
        assert "(query-match relevance score: 0.90)" in block
        assert "5.0s" not in block

    def test_longvideobench_subtitles_respect_explicit_off(self, srt_file):
        cfg = {"include_metadata": True, "subtitles": False}
        q_meta = {
            "subtitle_path": srt_file,
            "include_subtitles_in_prompt": True,
        }
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0],
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "Subtitles:" not in block

    def test_longvideobench_subtitles_show_when_enabled(self, srt_file):
        cfg = {"include_metadata": True, "subtitles": True}
        q_meta = {
            "subtitle_path": srt_file,
            "include_subtitles_in_prompt": True,
        }
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0],
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "Subtitles:" in block

    def test_subtitle_offset(self, srt_file):
        cfg = {"include_metadata": True}
        q_meta = {
            "subtitle_path": srt_file,
            "starting_timestamp_for_subtitles": 5.0,
        }
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0],
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "Subtitles:" in block


# ── _build_iterative_frame_metadata_block ─────────────────────────────────

class TestBuildIterativeFrameMetadataBlock:
    def test_uses_same_header_and_adds_sources(self, srt_file):
        cfg = {"include_metadata": True, "cosine_similarity": True}
        q_meta = {"subtitle_path": srt_file}
        block = _build_iterative_frame_metadata_block(
            frame_times=[5.0, 15.0],
            frame_scores=[0.8, 0.6],
            frame_sources=["initial selection", 'search: "shelf"'],
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "Frame timestamps:" in block
        assert "(initial selection)" in block
        assert '(search: "shelf")' in block
        assert "(query-match relevance score: 0.80)" in block


# ── build_qa_prompt ──────────────────────────────────────────────────────

class TestBuildQAPrompt:
    def test_with_metadata_block(self):
        block = "Frame timestamps:\n- Frame 1: 0.0s"
        prompt = build_qa_prompt("What color?", 5, metadata_block=block)
        assert "5 frames sampled" in prompt
        assert "Frame timestamps:" in prompt
        assert "Question: What color?" in prompt
        assert "Answer with only a single word" in prompt

    def test_no_metadata_block(self):
        prompt = build_qa_prompt("What color?", 3)
        assert "Frame timestamps" not in prompt
        assert "Question: What color?" in prompt

    def test_custom_template(self):
        tmpl = "Frames: {num_frames}. {metadata_block} Q: {question}"
        prompt = build_qa_prompt(
            "What?", 2,
            metadata_block="meta here",
            prompt_config={"custom_template": tmpl},
        )
        assert "Frames: 2" in prompt
        assert "meta here" in prompt
        assert "Q: What?" in prompt


# ── build_subtitle_only_prompt ───────────────────────────────────────────

class TestBuildSubtitleOnlyPrompt:
    def test_short_answer_prompt(self):
        prompt = build_subtitle_only_prompt(
            question="What is cooking?",
            subtitle_text="Line one.\nLine two.",
        )
        assert "No video frames are provided" in prompt
        assert "Full subtitle transcript:" in prompt
        assert "Line one." in prompt
        assert "Question: What is cooking?" in prompt

    def test_mcq_prompt(self):
        prompt = build_subtitle_only_prompt(
            question="What is on the shelf?",
            subtitle_text="Shelf subtitle.",
            options=["Plates", "Books"],
            prompt_style="videomme_mcq",
        )
        assert "using only the subtitle transcript" in prompt
        assert "A. Plates" in prompt
        assert "B. Books" in prompt
        assert "The best answer is:" in prompt


# ── build_videomme_mcq_prompt_frames ─────────────────────────────────────

class TestBuildVideoMMEMCQPromptFrames:
    def test_with_all_metadata(self, srt_file):
        cfg = {"include_metadata": True}
        q_meta = {"subtitle_path": srt_file}
        v_meta = {"duration": 40.0}
        scores = [0.8, 0.6, 0.4]
        prompt = build_videomme_mcq_prompt_frames(
            question="What color is the shirt?",
            options=["Red", "Blue", "Green", "Yellow"],
            num_frames=3,
            frame_times=[5.0, 15.0, 30.0],
            frame_scores=scores,
            video_metadata=v_meta,
            question_metadata=q_meta,
            metadata_config=cfg,
        )
        assert "3 frames sampled" in prompt
        assert "Frame timestamps:" in prompt
        assert "(query-match relevance score: 0.80)" in prompt
        assert 'Subtitles: "Hello and welcome' in prompt
        assert "A. Red" in prompt
        assert "D. Yellow" in prompt
        assert "The best answer is:" in prompt

    def test_no_metadata(self):
        cfg = {"include_metadata": False}
        prompt = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=3,
            frame_times=[5.0, 15.0, 30.0],
            metadata_config=cfg,
        )
        assert "query-match relevance score" not in prompt
        assert "Subtitles:" not in prompt
        assert "A. Yes" in prompt
        assert "The best answer is:" in prompt

    def test_pre_lettered_options(self):
        cfg = {"include_metadata": False}
        prompt = build_videomme_mcq_prompt_frames(
            question="Color?",
            options=["A. Red", "B. Blue"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config=cfg,
        )
        assert "A. Red" in prompt
        assert "B. Blue" in prompt
        assert prompt.count("A.") == 1

    def test_custom_template(self, srt_file):
        cfg = {"include_metadata": True}
        q_meta = {"subtitle_path": srt_file}
        tmpl = "{metadata_block}\n{question_block}\nAnswer:"
        prompt = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["Yes", "No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            frame_scores=[0.9, 0.7],
            question_metadata=q_meta,
            metadata_config=cfg,
            prompt_config={"custom_template": tmpl},
        )
        assert "(query-match relevance score: 0.90)" in prompt
        assert "Answer:" in prompt

    def test_videomme_mcq_prompt_style(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config={"include_metadata": False},
            prompt_config={"style": "videomme_mcq"},
        )
        assert "Select the best answer" in prompt
        assert "Below are" not in prompt
        assert "Frame timestamps:" not in prompt
        assert "The best answer is:" not in prompt
        assert prompt.endswith("Answer with the option's letter from the given choices directly.")

    def test_legacy_repo_mcq_keeps_old_videomme_prompt(self):
        v = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config={"include_metadata": False},
            prompt_config={"style": "videomme_mcq"},
        )
        legacy = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config={"include_metadata": False},
            prompt_config={"style": "legacy_repo_mcq"},
        )
        assert v != legacy
        assert "The best answer is:" in legacy

    def test_mlvu_lmms_eval_prompt_style(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What color is the hat?",
            options=["A. Red", "B. Blue", "C. Green", "D. Yellow"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config={"include_metadata": False},
            prompt_config={"style": "mlvu_lmms_eval_mcq"},
        )
        assert prompt.startswith("What color is the hat?")
        assert "Question:" not in prompt
        assert "Options:" not in prompt
        assert "Below are" not in prompt
        assert "(A) Red" in prompt
        assert "Only give the best option." in prompt
        assert prompt.endswith("Best option: (")
        assert "The best answer is:" not in prompt

    def test_longvideobench_official_prompt_style(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What happens next?",
            options=["A. Running", "B. Eating", "C. Sleeping", "D. Driving", "E. Dancing"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            metadata_config={"include_metadata": False},
            prompt_config={"style": "longvideobench_official_mcq"},
        )
        assert prompt.startswith("What happens next?")
        assert "Question:" not in prompt
        assert "Below are" not in prompt
        assert "E. Dancing" in prompt
        assert prompt.endswith("Answer with the option's letter from the given choices directly.")
        assert "The best answer is:" not in prompt

    def test_official_dataset_mcq_resolves_from_metadata(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What color is the hat?",
            options=["A. Red", "B. Blue"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            question_metadata={"qa_format": "mcq", "prompt_dataset": "mlvu"},
            metadata_config={"include_metadata": False},
            prompt_config={"style": "official_dataset_mcq"},
        )
        assert "Options:" not in prompt
        assert "Below are" not in prompt
        assert "(A) Red" in prompt
        assert prompt.endswith("Best option: (")

    def test_official_dataset_mcq_videomme_resolves_to_videomme_prompt(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            question_metadata={"qa_format": "mcq", "prompt_dataset": "videomme"},
            metadata_config={"include_metadata": False},
            prompt_config={"style": "official_dataset_mcq"},
        )
        assert "Select the best answer" in prompt
        assert "Below are" not in prompt
        assert "The best answer is:" not in prompt
        assert prompt.endswith("Answer with the option's letter from the given choices directly.")
        assert "Best option: (" not in prompt

    def test_auto_mcq_uses_videomme_style_even_with_mlvu_metadata(self):
        prompt = build_videomme_mcq_prompt_frames(
            question="What?",
            options=["A. Yes", "B. No"],
            num_frames=2,
            frame_times=[5.0, 15.0],
            question_metadata={"qa_format": "mcq", "prompt_dataset": "mlvu"},
            metadata_config={"include_metadata": False},
            prompt_config={"style": "auto"},
        )
        assert "The best answer is:" in prompt
        assert "Best option: (" not in prompt


# ── build_iterative_final_prompt ─────────────────────────────────────────

class TestBuildIterativeFinalPrompt:
    def test_mcq_prompt_includes_search_history(self):
        metadata_block = "Frame timestamps:\n- Frame 1: 5.0s"
        prompt, system = build_iterative_final_prompt(
            question="ignored raw question",
            num_frames=2,
            metadata_block=metadata_block,
            queries_so_far=["open drawer", "pick up plate"],
            question_metadata={
                "qa_format": "mcq",
                "question": "What is on the shelf?",
                "options": ["Plates", "Books", "Ingredients", "Nothing"],
            },
            prompt_config={"style": "auto"},
        )
        assert system
        assert "Frame timestamps:" in prompt
        assert "Search history" in prompt
        assert '"open drawer"' in prompt
        assert "What is on the shelf?" in prompt
        assert "A. Plates" in prompt
        assert "The best answer is:" in prompt

    def test_short_answer_prompt(self):
        prompt, _ = build_iterative_final_prompt(
            question="What is cooking?",
            num_frames=3,
            metadata_block="Frame timestamps:\n- Frame 1: 5.0s",
            queries_so_far=[],
            prompt_config={"style": "short_answer"},
        )
        assert "Question: What is cooking?" in prompt
        assert "Answer with only a single word or short phrase" in prompt


class _FakeVLM:
    def __init__(self):
        self.max_images_per_request = 30
        self.calls = []

    def query(self, text, images=None, system_prompt=None):
        self.calls.append({
            "text": text,
            "images": images,
            "system_prompt": system_prompt,
        })
        return "A"


class TestSubtitleOnlyIntegration:
    def test_answer_question_uses_no_images(self, srt_file):
        vlm = _FakeVLM()
        tal = TemporalAbstractionLayer(
            vlm_client=vlm,
            metadata_config={"include_metadata": True, "subtitles": True},
            qa_prompt_config={"style": "videomme_mcq", "subtitle_only": True},
        )
        result = tal.answer_question(
            video_path="unused.mp4",
            chunks=[],
            fps=FPS,
            question="What is on the shelf?",
            question_metadata={
                "qa_format": "mcq",
                "question": "What is on the shelf?",
                "options": ["Plates", "Books"],
                "subtitle_path": srt_file,
            },
        )
        assert result["answer"] == "A"
        assert len(vlm.calls) == 1
        assert vlm.calls[0]["images"] is None
        assert "Full subtitle transcript:" in vlm.calls[0]["text"]


# ── Integration: full prompt pipelines ────────────────────────────────────

class TestPromptIntegration:
    def test_chunk_selection_full_pipeline(self, srt_file):
        metadata_config = {"include_metadata": True}
        video_meta = {
            "duration": 40.0,
            "subtitle_path": srt_file,
        }
        chunk_scores = [0.42, 0.78, 0.35]
        prompt = build_chunk_selection_prompt(
            CHUNKS, FPS, "How does the person prepare the sauce?",
            video_metadata=video_meta,
            metadata_config=metadata_config,
            chunk_scores=chunk_scores,
        )
        lines = prompt.split("\n")
        assert "3 sequential" in lines[0]
        assert "Chunk 1" in prompt
        assert "Chunk 3" in prompt
        assert "cos relevance: 0.78" in prompt
        assert "Subtitles:" in prompt
        assert "Video duration: 40.0s" in prompt
        assert "prepare the sauce" in prompt

    def test_qa_mcq_full_pipeline(self, srt_file):
        metadata_config = {"include_metadata": True}
        question_metadata = {"subtitle_path": srt_file}
        video_metadata = {"duration": 40.0}
        prompt = build_videomme_mcq_prompt_frames(
            question="What is on the shelf?",
            options=["Plates", "Books", "Ingredients", "Nothing"],
            num_frames=3,
            frame_times=[5.0, 15.0, 30.0],
            frame_scores=[0.78, 0.65, 0.50],
            video_metadata=video_metadata,
            question_metadata=question_metadata,
            metadata_config=metadata_config,
        )
        assert "Frame timestamps:" in prompt
        assert "(query-match relevance score: 0.78)" in prompt
        assert 'Subtitles: "Hello and welcome' in prompt
        assert "Ingredients" in prompt
        assert "The best answer is:" in prompt

    def test_qa_short_answer_full_pipeline(self, srt_file):
        metadata_config = {"include_metadata": True}
        question_metadata = {"subtitle_path": srt_file}
        video_metadata = {"duration": 40.0}
        block = _build_frame_metadata_block(
            frame_times=[5.0, 15.0, 30.0],
            frame_scores=[0.9, 0.5, 0.3],
            video_metadata=video_metadata,
            question_metadata=question_metadata,
            metadata_config=metadata_config,
        )
        prompt = build_qa_prompt("What is cooking?", 3, metadata_block=block)
        assert "Frame timestamps:" in prompt
        assert "(query-match relevance score: 0.90)" in prompt
        assert 'Subtitles: "Hello and welcome' in prompt
        assert "Question: What is cooking?" in prompt
        assert "Answer with only a single word" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
