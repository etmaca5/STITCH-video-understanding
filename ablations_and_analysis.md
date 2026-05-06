# Ablations and analysis

Reference runs under `clean_results/`. Each directory has `config.yaml` (full resolved settings) and `metrics.json` (aggregate numbers).

Baselines below are the best **full official split** runs found in `clean_results` as of this snapshot. Future ablations should fork from these dirs unless you intentionally compare to a subset sweep.

| Dataset | Task folder | Best run directory | Primary metrics |
|---------|-------------|-------------------|-----------------|
| Kinetics GEBD (val) | `clean_results/event_detection/kinetics_gebd/` | `1-gebd_val_full_si05_mc0.5_pen0.03_refine-3787e505` | Avg_F1 **0.8367**, F1@0.5 **0.8793** |
| ActivityNet Captions (val) | `clean_results/moment_retrieval/activitynet/` | `5-anet_coh_tau5-1ab9c083` | R@1 IoU=0.5 **0.3256**, mIoU **0.3714** |
| QVHighlights (val, 1550 q) | `clean_results/moment_retrieval/qvhighlights/` | `8-qvh_max_final-e2f225f1` | mAP@0.5 **0.6092**, R@1 IoU=0.5 **0.6458** |
| MLVU (dev MCQ, 2174 q) | `clean_results/vlm_qa/mlvu/` | `47-llava-ov_mlvu-dev_mmr-chunk-penalty_l07_-c7370e54` | Accuracy **0.6677** |
| LongVideoBench (val, 1337 q) | `clean_results/vlm_qa/longvideobench/` | `57-llava-ov_lvb-val_mmr-chunk-penalty_l085_-cada4981` | Accuracy **0.5677** |
| VideoMME (test, 2700 q) | `clean_results/vlm_qa/videomme/` | `36-llava-ov_videomme-test_mmr-chunk-penalty-a699a83b` | Accuracy **0.5552** |

All of the following ablations should follow these basic results.

## Baseline commands

Each command below reproduces the corresponding baseline run. Overrides are only the parameters that differ from defaults (`configs/eval.yaml` + the loaded config groups). `evaluation.chunk_preselection=none` is needed for all VLM QA runs because `vlm_qa.yaml` now defaults to `query_similarity`.

### GEBD — Kinetics-GEBD val

```bash
python src/evaluate.py \
    dataset=kinetics_gebd evaluation=gebd \
    chunking.sample_interval=0.5 chunking.penalty=0.03 \
    postprocessing.gebd_boundary_refinement.enabled=true \
    postprocessing.merge.min_chunk_sec=0.5 \
    output.notes=gebd_val_full_si05_mc0.5_pen0.03_refine
```

### Moment retrieval — ActivityNet val

```bash
python src/evaluate.py \
    dataset=activitynet \
    evaluation.chunk_score_method=coherence_mean \
    evaluation.chunk_aggregation.coherence_tau=5.0 \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=anet_coh_tau5
```

### Moment retrieval — QVHighlights val

`dataset=qvhighlights` is the default so it is omitted.

```bash
python src/evaluate.py \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=qvh_max
```

### VLM QA — MLVU dev

All three VLM QA baselines use LLaVA-OneVision Qwen2 7B (HF endpoint, video URL input). Each dataset has two runs: the best frame-selection method and a uniform baseline for comparison.

**Best (mmr_chunk_penalty, chunk_weight=0.1, n_frames=8):**

```bash
python src/evaluate.py \
    dataset=mlvu temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.1 \
    evaluation.chunk_preselection=none \
    output.notes=llava-ov_mlvu-dev_mmr-chunk-penalty_l07_n8
```

**Uniform (8 frames):**

```bash
python src/evaluate.py \
    dataset=mlvu temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=middle \
    evaluation.chunk_source=uniform evaluation.num_uniform_chunks=8 \
    temporal_abstraction.prompt.max_chunks_per_prompt=8 \
    evaluation.chunk_preselection=none \
    postprocessing.query_merge.enabled=false \
    output.notes=llava-ov_mlvu-dev_uniform8
```

### VLM QA — LongVideoBench val

**Best (mmr_chunk_penalty, relevance_weight=0.85, chunk_weight=0.2, n_frames=8):**

```bash
python src/evaluate.py \
    dataset=longvideobench temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.relevance_weight=0.85 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.2 \
    temporal_abstraction.prompt.max_chunks_per_prompt=8 \
    evaluation.chunk_preselection=none \
    output.notes=llava-ov_lvb-val_mmr-chunk-penalty_l085_g02_n8
```

**Uniform (8 frames):**

```bash
python src/evaluate.py \
    dataset=longvideobench temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=middle \
    evaluation.chunk_source=uniform evaluation.num_uniform_chunks=8 \
    temporal_abstraction.prompt.max_chunks_per_prompt=8 \
    evaluation.chunk_preselection=none \
    postprocessing.query_merge.enabled=false \
    output.notes=llava-ov_lvb-val_uniform8
```

### VLM QA — VideoMME test

**Best (mmr_chunk_penalty, relevance_weight=0.85, chunk_weight=0.4, n_frames=8):**

```bash
python src/evaluate.py \
    dataset=videomme temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.relevance_weight=0.85 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.4 \
    evaluation.chunk_preselection=none \
    output.notes=llava-ov_videomme-test_mmr-chunk-penalty_l085_g04_n8
```

**Uniform (8 frames):**

```bash
python src/evaluate.py \
    dataset=videomme temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=middle \
    evaluation.chunk_source=uniform evaluation.num_uniform_chunks=8 \
    temporal_abstraction.prompt.max_chunks_per_prompt=8 \
    evaluation.chunk_preselection=none \
    postprocessing.query_merge.enabled=false \
    output.notes=llava-ov_videomme-test_uniform8
```

## Planned ablations

The following ablations are run in this order. Each later stage adopts the winner of the previous stage (if it beats the current baseline) before its sweep.

### 1. Chunk aggregation: GEM and mean

Scope: ANet val and QVHighlights val only. `chunk_score_method` is wired only in `mode=embedding`; VLM QA hardcodes `max_sim` in `_score_chunks`, and GEBD has no chunk scoring.

Prior `*_gemstudy_*` runs were done without the post-processing tuning the current baselines use (`moment_selection.top_gap`, `query_merge.enabled=false`), so they aren't directly comparable. We re-run on the current baseline post-processing.

GEM exponent is fixed per dataset to the best value from the prior gem study:
- ANet: `gem_p=3` (best of {2,3,5,8,12} — runs 12–16, R@1 IoU=0.5 = 0.3051)
- QVH: `gem_p=12` (monotonic in p across the prior sweep — run 7, R@1 IoU=0.5 = 0.6529)

**ANet — mean:**

```bash
python src/evaluate.py \
    dataset=activitynet \
    evaluation.chunk_score_method=mean \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=anet_agg_mean
```

**ANet — gem (p=3):**

```bash
python src/evaluate.py \
    dataset=activitynet \
    evaluation.chunk_score_method=gem \
    evaluation.chunk_aggregation.gem_p=3.0 \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=anet_agg_gem_p3
```

**QVH — mean:**

```bash
python src/evaluate.py \
    evaluation.chunk_score_method=mean \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=qvh_agg_mean
```

**QVH — gem (p=12):**

```bash
python src/evaluate.py \
    evaluation.chunk_score_method=gem \
    evaluation.chunk_aggregation.gem_p=12.0 \
    postprocessing.query_merge.enabled=false \
    postprocessing.moment_selection.enabled=true \
    postprocessing.moment_selection.method=top_gap \
    postprocessing.moment_selection.gap=0.06 \
    output.notes=qvh_agg_gem_p12
```

If either method beats the current best on its dataset, switch the baseline before stages 2 and 3.

### 2. Boundary detection: kernel CPD vs PELT vs Std

All six baselines use `chunking=embedding` with `threshold_method=kernel_cpd`. Kernel CPD has already been tuned in our current baselines, so we only run **one PELT and one Std configuration** here, each at its config default, and compare against the existing kernel-CPD baseline.

Representative dataset: **MLVU dev** (fast, exercises the full VLM QA pipeline shared by LVB / VideoMME). Held fixed: the current MLVU best — `mmr_chunk_penalty`, `chunk_weight=0.1`, `n_frames=8`. Kernel-CPD reference is the existing run `47-llava-ov_mlvu-dev_mmr-chunk-penalty_l07_n8` (penalty=null / BIC, `min_segment_windows=2`).

If either method looks competitive but sensitive to its hyperparameter, optionally do a small subset sweep first (e.g., `sample_size=300`) to pick that hyperparameter — this is not part of the official plan, just a fallback before adopting a winner.

**PELT** (penalty=null / BIC, `min_segment_windows=2` — config defaults):

```bash
python src/evaluate.py \
    dataset=mlvu temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.1 \
    chunking.threshold_method=pelt \
    evaluation.chunk_preselection=none \
    output.notes=mlvu_boundary_pelt
```

**Std** (`k=2.0` — config default):

```bash
python src/evaluate.py \
    dataset=mlvu temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.1 \
    chunking.threshold_method=std \
    chunking.k=2.0 \
    evaluation.chunk_preselection=none \
    output.notes=mlvu_boundary_std
```

If PELT or Std beats kernel CPD on MLVU, switch the baseline before stage 3.

### 3. InternVideo sample size (frames per window)

The chunking window encoder defaults to `cfg.retrieval.num_frames=4` (the f4 InternVideo2 checkpoint). The hypothesis is that 2 frames per window is sufficient (and faster). 8 frames is the upper option we'd consider keeping.

`retrieval` switches the visual encoder used for both query embedding and window embeddings; the window-embedding cache is keyed on the encoder, so a new sample size requires re-warming the cache for every dataset.

**Step A — re-warm window-embedding cache (all six datasets, 2 frames per window):**

```bash
for DS in kinetics_gebd activitynet qvhighlights mlvu longvideobench videomme; do
python src/precompute_window_embeddings.py \
    dataset=$DS \
    retrieval=internvideo2_stage2_1b_f2
done
```

(Use `dataset.split=test` overrides where the baseline uses test, e.g., MLVU/VideoMME, matching `DATASETS.md`.)

**Step B — re-run each baseline with `retrieval=internvideo2_stage2_1b_f2`.** Take the current baseline command for each dataset (above) and append `retrieval=internvideo2_stage2_1b_f2` plus an `_f2` suffix on `output.notes`. Example for MLVU:

```bash
python src/evaluate.py \
    dataset=mlvu temporal_abstraction=qa evaluation=vlm_qa vlm=llava_onevision_hf \
    retrieval=internvideo2_stage2_1b_f2 \
    temporal_abstraction.frame_selection.method=mmr_chunk_penalty \
    temporal_abstraction.frame_selection.n_frames=8 \
    temporal_abstraction.frame_selection.mmr_chunk_penalty.chunk_weight=0.1 \
    evaluation.chunk_preselection=none \
    output.notes=llava-ov_mlvu-dev_mmr-chunk-penalty_l07_n8_f2
```

If 2 frames per window matches or beats f4 on most datasets, adopt it as the new default. Otherwise keep f4 (and only then consider an f8 sweep).