# Evaluation Framework

**Project note:** VideoMAEv2 is no longer needed for new runs. Use InternVideo2 only (`retrieval=internvideo2`, default `chunking.embedding_backend: internvideo2`). Do not spend cycles testing or validating the VideoMAEv2 embedding path unless you are debugging legacy configs.

## Quick Start

```bash
# Default: QVHighlights + embedding + internvideo2
python src/evaluate.py

# Override any combination
python src/evaluate.py dataset=ego4d_nlq chunking=embedding retrieval=internvideo2

# Run on a small subset first
python src/evaluate.py subset_ids=[2579,5071,5342]

# Or sample a random subset (reproducible with sample_seed)
python src/evaluate.py dataset=activitynet sample_size=100 sample_seed=0

# Rerun only the videos that failed in a previous run
python src/evaluate.py \
    --config-path "$(pwd)/results/144-activitynet_vlm_gemini25pro_bestparams_s-activitynet" \
    --config-name config \
    +redo_failed_from="$(pwd)/results/144-activitynet_vlm_gemini25pro_bestparams_s-activitynet"

# Add notes to tag a run
python src/evaluate.py output.notes="baseline content_detector"

# Warm the window-embedding cache without running evaluation
python src/precompute_window_embeddings.py dataset=activitynet

# Example MP4 videos are off by default
python src/evaluate.py output.generate_videos=true

# Embedding-specific evaluation settings now live under evaluation.*
python src/evaluate.py evaluation=embedding evaluation.chunk_score_method=max_sim

# VLM chunk selection for ActivityNet
python src/evaluate.py dataset=activitynet evaluation=vlm_chunk_selection \
    evaluation.chunk_source=uniform

# Show resolved config without running
python src/evaluate.py --cfg job
```

## Evaluation Modes

The evaluation path is now selected explicitly with `evaluation=...`.

- `evaluation=embedding`
  - Existing retrieval-based path
  - Uses `evaluation.chunk_score_method` and `evaluation.batch_size`
- `evaluation=vlm_chunk_selection`
  - Uses `VLMClient.from_config()` and `TemporalAbstractionLayer.from_config()`
  - Supports `evaluation.chunk_source=uniform` and `evaluation.chunk_source=stable_chunks`
  - Uses the default VLM config unless you override it with `vlm=...`

Examples:

```bash
# Embedding path
python src/evaluate.py dataset=activitynet evaluation=embedding \
    evaluation.chunk_score_method=max_sim

# VLM path with uniform chunks
python src/evaluate.py dataset=activitynet evaluation=vlm_chunk_selection \
    evaluation.chunk_source=uniform

# VLM path with existing stable chunks
python src/evaluate.py dataset=activitynet evaluation=vlm_chunk_selection \
    evaluation.chunk_source=stable_chunks
```

Current limitation:

- `postprocessing.query_merge` and `postprocessing.moment_selection` are only used in `evaluation=embedding`
- They are currently ignored in `evaluation=vlm_chunk_selection`
- TODO: design a VLM-compatible postprocessing path later instead of pretending the embedding-only logic applies there

## Best Results So Far

These are the best saved runs currently under `results/`. For datasets with a
full validation run, the table below uses that. For datasets that only have
subset experiments so far, the best subset run is listed explicitly.

### Full Validation Runs

| Dataset | Best run dir | Primary metric | Other metrics | Key params |
|---------|---------------|----------------|---------------|------------|
| `activitynet` | `results/95-activitynet_val-activitynet` | `R@1_IoU=0.5 = 0.3153` | `R@1_IoU=0.3 = 0.5036`, `R@5_IoU=0.5 = 0.4310`, `mIoU = 0.3603` | `evaluation=embedding`, `chunking=embedding`, `sample_interval=2.0`, `k=2.0`, `evaluation.chunk_score_method=max_sim`, `query_merge=true`, `moment_selection.enabled=false` |
| `ego4d_mq` | `results/96-ego4d_mq_val-ego4d_mq` | `R@1_IoU=0.3 = 0.1410` | `R@1_IoU=0.5 = 0.0782`, `R@5_IoU=0.3 = 0.3186`, `mIoU = 0.1222` | `evaluation=embedding`, `chunking=embedding`, `sample_interval=2.0`, `k=2.0`, `evaluation.chunk_score_method=max_sim`, `query_merge=true`, `moment_selection.enabled=false` |

### Best Subset Runs So Far

| Dataset | Subset | Best run dir | Primary metric | Other metrics | Key params |
|---------|--------|--------------|----------------|---------------|------------|
| `qvhighlights` | `sample_size=50` | `results/86-qvh_50_top_gap_0p05_noqm-qvhighlights` | `mAP@0.5 = 0.6432` | `R@1_IoU=0.5 = 0.7000`, `R@1_IoU=0.7 = 0.6000`, `HL-mAP_min2 = 0.7957`, `mIoU = 0.6648` | `evaluation=embedding`, `chunking=embedding`, `sample_interval=2.0`, `k=2.0`, `evaluation.chunk_score_method=max_sim`, `query_merge=false`, `moment_selection.enabled=true`, `moment_selection.method=top_gap`, `moment_selection.gap=0.05` |
| `ego4d_nlq` | `sample_size=10` | `results/1-20260303_002931_ego4d_nlq_first_test_run` | `R@1_IoU=0.5 = 0.0000` | `R@1_IoU=0.3 = 0.0000`, `R@5_IoU=0.5 = 0.0000` | `chunking=embedding`, `sample_interval=1.0`, `k=2.0` |

If you run a stronger config later, update this section and keep the run
directory here so the exact saved `config.yaml` and `metrics.json` remain easy
to find.

## Datasets

### QVHighlights (val, 1550 queries)
Moment retrieval on short video clips. Each query has one or more relevant
temporal windows annotated by humans.

**Metrics**: R@1 at IoU {0.5, 0.7}; MR-mAP at IoU {0.50:0.05:0.95};
mIoU; highlight metrics HL-Hit1 / HL-mAP at saliency thresholds {2, 3, 4}.

```bash
python src/evaluate.py dataset=qvhighlights
```

Config overrides:
- `dataset.data_root` — path containing `videos/` and `annotations/`.
- `dataset.split` — `val` (default) or `train`.
- `dataset.max_pred_windows` — cap predictions per query (default `10`).

### Ego4D NLQ (val, 4552 queries)
Natural language queries on long egocentric video clips. Each query has a
single ground-truth temporal window.

**Metrics**: R@{1,5} at IoU {0.3, 0.5}; mIoU.

```bash
python src/evaluate.py dataset=ego4d_nlq
```

Config overrides:
- `dataset.annotations_dir` — directory with `nlq_val.json`, etc.
- `dataset.clips_dir` — directory with clip mp4 files.

### Ego4D MQ (val, ~4932 queries)
Moment queries on long egocentric video clips. Each query is an activity
label (e.g. "stir mix ingredients in a bowl or pan") and can have multiple
ground-truth temporal windows per clip.

**Metrics**: R@{1,5} at IoU {0.1, 0.3, 0.5}; mIoU.

```bash
python src/evaluate.py dataset=ego4d_mq
```

Config overrides:
- `dataset.annotations_dir` — directory with `moments_val.json`, etc.
- `dataset.clips_dir` — directory with clip mp4 files.

### ActivityNet Captions (val, 17505 queries)
Temporal grounding of natural language descriptions in untrimmed videos.
Each query maps to one temporal segment.

**Metrics**: R@{1,5} at IoU {0.3, 0.5, 0.7}; mIoU.

```bash
python src/evaluate.py dataset=activitynet
```

Config overrides:
- `dataset.video_dir` — directory with `v_*.mp4` files.
- `dataset.captions_dir` — directory with `val.json` / `train.json`.

## Chunking Methods

| Config name        | Type              | Key parameters                              |
|--------------------|-------------------|---------------------------------------------|
| `content_detector` | Scene detection   | `threshold`, `min_scene_len`                |
| `embedding`        | Embedding similarity | `sample_interval`, `k`, `embedding_backend` (default `internvideo2`; VideoMAEv2 legacy only) |
| `surprise`         | V-JEPA2           | `window_frames`, `stride_seconds`, `k`, `model_path` |

Override parameters:
```bash
python src/evaluate.py chunking=embedding chunking.k=3.0
```

## Retrieval Backends

| Config name                       | Model                     | Frames | Key parameters                                              | Notes       |
|-----------------------------------|---------------------------|--------|--------------------------------------------------------------|-------------|
| `internvideo2`                    | InternVideo2 Stage2 1B    | 4      | `model_dir`, `num_frames`, `image_size`                      | **default** |
| `internvideo2_stage2_1b_f4`      | InternVideo2 Stage2 1B    | 4      | (same as above, explicit name)                               |             |
| `internvideo2_stage2_1b_f8`      | InternVideo2 Stage2 1B    | 8      | `model_dir`, `num_frames`, `orig_num_frames`, `image_size`   | interpolated pos-embed |
| `xclip`                          | X-CLIP                    | 32     | `model_path`                                                 | deprecated  |

The f8 variants use the same Stage2 checkpoints as the f4 variants. They
interpolate the temporal position embeddings from 4→8 frames at load time,
allowing denser temporal sampling without a separate checkpoint.

> **Note:** Microsoft's X-CLIP is a video *classification* model repurposed for
> retrieval. It underperforms InternVideo2 on all retrieval benchmarks. It is
> kept only for legacy comparison; use `internvideo2` for new experiments.

## Post-processing

Post-processing always runs after chunking. It can optionally detect
transitions (to mark and exclude "between scene" segments) and then merges
small/large chunks using embedding similarity.

Important:

- `postprocessing.query_merge` and `postprocessing.moment_selection` currently apply only to `evaluation=embedding`
- They are not used by `evaluation=vlm_chunk_selection`
- TODO: add a VLM-compatible design for postprocessing later

Key overrides:
```bash
python src/evaluate.py \
    postprocessing.detect_transitions=true \
    postprocessing.transition.smooth_window=5 \
    postprocessing.merge.min_chunk_sec=3.0 \
    postprocessing.merge.max_chunk_sec=120.0
```

## Pipeline

For each evaluation run:

1. **Load dataset** — parse annotations, optionally filter to `subset_ids`, then optionally sample `sample_size` examples.
2. **Group by video** — so chunk embeddings are computed once per video.
3. **Chunk** — run the configured chunking method.
4. **Post-process** — transition detection (optional) + chunk merging.
5. **Embed chunks** — embed all stable chunks with the retrieval backend (cached per video).
6. **Score queries** — embed each query, rank chunks by cosine similarity.
7. **Compute metrics** — R@K, optional mAP, optional mIoU, and (for QVHighlights) highlight metrics.
8. **Save results** — config, metrics, and per-query predictions.
9. **Generate plots** — PNG plots are generated automatically; MP4 example videos are optional.

For `evaluation=vlm_chunk_selection`, step 5 and step 6 are replaced by:

5. **Choose chunk candidates** — either uniform chunks or existing stable chunks.
6. **Ask the VLM** — run `TemporalAbstractionLayer.select_best_chunk(...)` and convert the chosen chunk back to seconds.

## Output Structure

Each run creates a numbered directory under `results/`:

```
results/
  17-query_merging-activitynet/
    config.yaml            # full resolved config
    metrics.json           # aggregate metrics
    per_query_results.json # per-sample predictions
    per_video_data.json    # visualization data for plots / videos
    failed_videos.json     # only written if one or more videos crashed
    subset_ids.json        # (if subset_ids or sample_size was used)
    plots/
      *.png
      *.mp4                # only if video rendering was enabled
```

- **config.yaml** — exactly reproduces the run.
- **metrics.json** — e.g. `{"R@1_IoU=0.5": 0.42, "mAP@0.5": 0.35}`.
- **per_query_results.json** — list of query result dicts. Core fields include `{sample_id, query, video_path, num_chunks, predictions, top1_pred, gt_windows, metadata}`. Some modes may add extra fields such as `num_chunks_before_query_merge`, `selector_metadata`, or `vlm_raw_response`.
- **per_video_data.json** — cached visualization data used to regenerate plots and example videos later.
- **TODO for all runs** — if a run writes `failed_videos.json`, rerun those videos with `+redo_failed_from=...` so final reported results are based on completed videos rather than silently keeping partial failures.

## Example Videos

Example MP4 rendering is disabled by default during evaluation:

```bash
python src/evaluate.py
```

Enable it for a run with:

```bash
python src/evaluate.py output.generate_videos=true
```

You can also render the videos later from an existing run directory without
rerunning evaluation:

```bash
# Regenerate plots and render videos
python src/plots.py results/<run_dir> --videos

# Or only render the videos
python src/plots.py results/<run_dir> --videos-only
```

## Loading Past Results

```python
from src.results import load_run

run = load_run("results/20250302_143022_qvhighlights_baseline")
print(run["metrics"])
print(run["config"].chunking.type)
```
