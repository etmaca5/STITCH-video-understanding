# Evaluation Guide

This repository uses Hydra config groups. The default config is
`configs/eval.yaml`; you can inspect the fully resolved settings with:

```bash
python src/evaluate.py --cfg job
```

All examples below assume that benchmark data and model checkpoints are stored
outside the repository and passed in through config overrides.

## 1. Semantic Chunking + Moment Retrieval

QVHighlights:

```bash
python src/evaluate.py \
  dataset=qvhighlights \
  dataset.data_root=/path/to/qvhighlights_data \
  retrieval.model_dir=/path/to/InternVideo2-Stage2_1B-224p-f4
```

ActivityNet Captions:

```bash
python src/evaluate.py \
  dataset=activitynet \
  dataset.video_dir=/path/to/activitynet/videos \
  dataset.captions_dir=/path/to/activitynet/captions \
  retrieval.model_dir=/path/to/InternVideo2-Stage2_1B-224p-f4
```

Useful paper-style overrides:

```bash
python src/evaluate.py \
  dataset=activitynet \
  chunking=embedding \
  chunking.sample_interval=2.0 \
  chunking.threshold_method=kernel_cpd \
  evaluation=embedding \
  evaluation.chunk_score_method=coherence_mean \
  evaluation.chunk_aggregation.coherence_tau=5.0 \
  postprocessing.query_merge.enabled=false \
  postprocessing.moment_selection.enabled=true \
  postprocessing.moment_selection.method=top_gap \
  postprocessing.moment_selection.gap=0.06
```

## 2. Generic Event Boundary Detection

Kinetics-GEBD:

```bash
python src/evaluate.py \
  dataset=kinetics_gebd \
  evaluation=gebd \
  chunking=embedding \
  chunking.sample_interval=0.5 \
  chunking.penalty=0.03 \
  postprocessing.gebd_boundary_refinement.enabled=true \
  postprocessing.merge.min_chunk_sec=0.5 \
  dataset.video_dir=/path/to/kinetics_gebd/clips \
  dataset.annotation_path=/path/to/k400_mr345_val_min_change_duration0.3.pkl
```

This mode treats STITCH chunk boundaries as event-boundary predictions and
reports GEBD F1 over the configured thresholds.

## 3. Long-Video VLM QA Frame Selection

MLVU:

```bash
python src/evaluate.py \
  dataset=mlvu \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  vlm=llava_onevision_hf \
  dataset.dataset_root=/path/to/MLVU \
  vlm.base_url=https://your-hf-endpoint.example.com/v1 \
  vlm.api_key_env=HF_TOKEN \
  temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
  temporal_abstraction.frame_selection.n_frames=8
```

LongVideoBench:

```bash
python src/evaluate.py \
  dataset=longvideobench \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  dataset.dataset_root=/path/to/LongVideoBench \
  temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
  temporal_abstraction.frame_selection.n_frames=8
```

Video-MME:

```bash
python src/evaluate.py \
  dataset=videomme \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  dataset.video_dir=/path/to/Video-MME/data \
  dataset.subtitle_dir=/path/to/Video-MME/subtitle \
  dataset.annotation_path=/path/to/Video-MME/videomme/test-00000-of-00001.parquet \
  temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
  temporal_abstraction.frame_selection.n_frames=8
```

Uniform-frame baseline:

```bash
python src/evaluate.py \
  dataset=mlvu \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  evaluation.chunk_source=uniform \
  evaluation.num_uniform_chunks=8 \
  temporal_abstraction.frame_selection.method=middle \
  temporal_abstraction.prompt.max_chunks_per_prompt=8 \
  postprocessing.query_merge.enabled=false
```

## Outputs

Runs write resolved configs, metrics, and optional plots/videos to
`clean_results/` by default:

```text
clean_results/<run-id>/config.yaml
clean_results/<run-id>/metrics.json
```

Result folders are ignored by git. Copy selected metrics into your paper tables
or archival artifact separately rather than committing full result directories.

## Reproducibility Notes

- The paper uses frozen InternVideo2 window embeddings for STITCH chunking.
- VLM QA results can vary across hosted model providers even at temperature
  zero. Record endpoint, model, prompt style, and generation settings with each
  run.
- For large datasets, cache warming with `src/precompute_window_embeddings.py`
  can avoid repeated video encoding.
