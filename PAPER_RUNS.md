# Paper Run Configurations

Exact Hydra overrides for every table and figure in the paper. Each entry lists
only the parameters that differ from the project defaults in `configs/eval.yaml`.

Shared settings across all runs (unless overridden below):

- Frozen InternVideo2 Stage2 1B backbone (`retrieval=internvideo2`)
- Kernel-CPD chunking (`chunking=embedding`, `chunking.threshold_method=kernel_cpd`)
- 4-frame windows, 2-second sample interval (`chunking.sample_interval=2.0`)
- `postprocessing.query_merge.enabled=false`

---

## Application 1 — Event Detection (Table 1)

### Kinetics-GEBD (val)

```bash
python src/evaluate.py \
  dataset=kinetics_gebd \
  evaluation=gebd \
  chunking.sample_interval=0.5 \
  chunking.penalty=0.03 \
  postprocessing.merge.min_chunk_sec=0.5 \
  dataset.video_dir=/path/to/kinetics_gebd/clips \
  dataset.annotation_path=/path/to/k400_mr345_val_min_change_duration0.3.pkl
```

Expected: F1@.05 69.90, F1@.10 79.42, F1@.20 84.56, F1@.30 86.55, Avg 83.89

### TAPOS (val)

```bash
python src/evaluate.py \
  dataset=tapos_gebd \
  evaluation=gebd \
  chunking.sample_interval=0.5 \
  chunking.k=3.0 \
  chunking.penalty=0.07 \
  postprocessing.detect_transitions=true \
  postprocessing.merge.min_chunk_sec=1.1 \
  postprocessing.query_merge.enabled=false \
  dataset.video_dir=/path/to/tapos/clips \
  dataset.annotation_path=/path/to/tapos_annotation.pkl
```

Expected: F1@.05 31.15, F1@.10 39.56, F1@.15 43.18, F1@.30 47.20, Avg 44.78

---

## Application 2 — Moment Retrieval (Tables 2 & 3)

### ActivityNet Captions (val) — Table 2

```bash
python src/evaluate.py \
  dataset=activitynet \
  evaluation=embedding \
  evaluation.chunk_score_method=max_sim \
  postprocessing.query_merge.enabled=false \
  postprocessing.moment_selection.enabled=true \
  postprocessing.moment_selection.method=top_gap \
  postprocessing.moment_selection.gap=0.06 \
  dataset.video_dir=/path/to/activitynet/videos \
  dataset.captions_dir=/path/to/activitynet/captions
```

Expected: R@1 IoU=.3 52.25, R@1 IoU=.5 32.85, R@1 IoU=.7 17.87, mIoU 37.40

### QVHighlights (val) — Table 3

```bash
python src/evaluate.py \
  dataset=qvhighlights \
  evaluation=embedding \
  evaluation.chunk_score_method=max_sim \
  postprocessing.query_merge.enabled=false \
  postprocessing.moment_selection.enabled=true \
  postprocessing.moment_selection.method=top_gap \
  postprocessing.moment_selection.gap=0.06 \
  dataset.data_root=/path/to/qvhighlights_data
```

Expected: R@1 IoU=.5 64.58, R@1 IoU=.7 49.10, mAP avg 41.76

---

## Application 3 — VLM QA (Table 4)

All runs use `evaluation=vlm_qa`, `temporal_abstraction=qa`,
`temporal_abstraction.prompt.style=official_dataset_mcq`, 8 input frames, and
full dataset splits (MLVU dev, LongVideoBench val, VideoMME test).

### Qwen3-VL-8B

**Uniform baseline** (all three datasets):

```bash
python src/evaluate.py \
  dataset=<dataset> \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  vlm=qwen3_vl_8b \
  evaluation.chunk_source=uniform \
  evaluation.num_uniform_chunks=8 \
  temporal_abstraction.frame_selection.method=middle \
  temporal_abstraction.prompt.max_chunks_per_prompt=8 \
  postprocessing.query_merge.enabled=false
```

**Intra-chunk greedy** (all three datasets):

```bash
python src/evaluate.py \
  dataset=<dataset> \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  vlm=qwen3_vl_8b \
  temporal_abstraction.frame_selection.method=intra_chunk_greedy \
  temporal_abstraction.frame_selection.n_frames=8
```

**MMR chunk penalty** (all three datasets):

```bash
python src/evaluate.py \
  dataset=<dataset> \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  vlm=qwen3_vl_8b \
  temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
  temporal_abstraction.frame_selection.n_frames=8 \
  temporal_abstraction.frame_selection.mmr_chunk_penalty.relevance_weight=0.7 \
  temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.1
```

Where `<dataset>` is one of:
- `mlvu` with `dataset.dataset_root=/path/to/MLVU`
- `longvideobench` with `dataset.dataset_root=/path/to/LongVideoBench`
- `videomme` with `dataset.video_dir=... dataset.subtitle_dir=... dataset.annotation_path=...`

### LLaVA-OneVision Qwen2 7B

Same configurations as Qwen3-VL-8B above, replacing `vlm=qwen3_vl_8b` with
`vlm=llava_onevision_hf_video`.

Note: LongVideoBench uses `temporal_abstraction.prompt.max_chunks_per_prompt=8`
for the LLaVA-OV paper runs.

---

## Summary Table Ablations (STITCH-PELT, STITCH-STD, STITCH-2F)

These ablations appear in the main results table. All use LLaVA-OneVision 7B
(`vlm=llava_onevision_hf_video`) for VLM QA columns.

### STITCH-PELT

Replace `chunking.threshold_method=kernel_cpd` with `chunking.threshold_method=pelt`.
All other settings match the corresponding main table row.

### STITCH-STD

Replace `chunking.threshold_method=kernel_cpd` with `chunking.threshold_method=std`.
All other settings match the corresponding main table row.

### STITCH-2F

Use `retrieval=internvideo2_stage2_1b_f2` (2-frame window encoding from the f4
checkpoint). The window-embedding cache must be warmed separately:

```bash
python src/precompute_window_embeddings.py \
  dataset=<dataset> \
  retrieval=internvideo2_stage2_1b_f2
```

All other settings match the corresponding main table row.

---

## Figure 3 — Frame-Budget Sweep (Qwen3-VL-8B, MLVU dev)

### Uniform sweep

Vary `evaluation.num_uniform_chunks` over {0, 1, 2, 4, 8, 16, 24} with
`evaluation.chunk_source=uniform`, `temporal_abstraction.frame_selection.method=middle`.

### MMR-chunk-penalty sweep

Vary `temporal_abstraction.frame_selection.n_frames` over {1, 2, 4, 8, 16, 24}
with `temporal_abstraction.frame_selection.method=mmr_chunk_penalty`,
`relevance_weight=0.7`, `chunk_weight=0.1`.

---

## Reproducibility Notes

- VLM QA results can vary across hosted model providers even at temperature
  zero. Record endpoint, model, prompt style, and generation settings with each
  run.
- For large datasets, cache warming with `src/precompute_window_embeddings.py`
  avoids repeated video encoding.
- All paper runs use `postprocessing.query_merge.enabled=false`.
