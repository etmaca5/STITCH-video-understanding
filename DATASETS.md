# Dataset Setup

Datasets are not included in this repository. Download each benchmark from its
official source, store it outside the git checkout, and pass paths through Hydra
overrides or by editing the corresponding file in `configs/dataset/`.

The config files use `data/...` placeholders so the repository remains portable.

## Moment Retrieval

### QVHighlights

Config: `configs/dataset/qvhighlights.yaml`

Expected structure:

```text
qvhighlights_data/
  videos/
  annotations/
    highlight_val_release.jsonl
```

Run:

```bash
python src/evaluate.py \
  dataset=qvhighlights \
  dataset.data_root=/path/to/qvhighlights_data
```

Metrics: R@1 at IoU 0.5 and 0.7, MR-mAP over IoU 0.50:0.05:0.95, mIoU, and
highlight metrics when saliency labels are present.

### ActivityNet Captions

Config: `configs/dataset/activitynet.yaml`

Expected inputs:

- ActivityNet videos, usually named `v_*.mp4`
- ActivityNet Captions annotations containing `train.json` / `val.json`

Run:

```bash
python src/evaluate.py \
  dataset=activitynet \
  dataset.video_dir=/path/to/activitynet/videos \
  dataset.captions_dir=/path/to/activitynet/captions
```

Metrics: R@1/R@5 at IoU 0.3, 0.5, and 0.7, plus mIoU.

## Event Boundary Detection

### Kinetics-GEBD

Config: `configs/dataset/kinetics_gebd.yaml`

Expected inputs:

- clipped Kinetics videos
- GEBD annotation pickle, e.g. `k400_mr345_val_min_change_duration0.3.pkl`

Run:

```bash
python src/evaluate.py \
  dataset=kinetics_gebd \
  evaluation=gebd \
  dataset.video_dir=/path/to/kinetics_gebd/clips \
  dataset.annotation_path=/path/to/k400_mr345_val_min_change_duration0.3.pkl
```

Metrics: GEBD F1 over the configured relative-distance thresholds.

### TAPOS

Config: `configs/dataset/tapos_gebd.yaml`

Expected inputs:

- processed TAPOS video clips
- TAPOS boundary annotation JSON, e.g. `tapos_val_gebd.json`

Run:

```bash
python src/evaluate.py \
  dataset=tapos_gebd \
  evaluation=gebd \
  dataset.video_dir=/path/to/tapos/clips/val \
  dataset.annotation_path=/path/to/tapos_val_gebd.json
```

Metrics: GEBD F1 over the configured relative-distance thresholds.

## Long-Video QA

These datasets use multiple-choice answer metrics and the `evaluation=vlm_qa`
path. VLM endpoint credentials are not stored in the repo; set the relevant
environment variable, such as `HF_TOKEN` or `OPENROUTER_API_KEY`.

### MLVU

Config: `configs/dataset/mlvu.yaml`

Run:

```bash
python src/evaluate.py \
  dataset=mlvu \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  dataset.dataset_root=/path/to/MLVU
```

### LongVideoBench

Config: `configs/dataset/longvideobench.yaml`

Expected structure after extracting the official archives:

```text
LongVideoBench/
  lvb_val.json
  videos/
  subtitles/
```

Run:

```bash
python src/evaluate.py \
  dataset=longvideobench \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  dataset.dataset_root=/path/to/LongVideoBench
```

### Video-MME

Config: `configs/dataset/videomme.yaml`

Run:

```bash
python src/evaluate.py \
  dataset=videomme \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  dataset.video_dir=/path/to/Video-MME/data \
  dataset.subtitle_dir=/path/to/Video-MME/subtitle \
  dataset.annotation_path=/path/to/Video-MME/videomme/test-00000-of-00001.parquet
```

## Optional / Experimental Datasets

The repository also includes loaders for Ego4D NLQ/MQ, ActivityNet-QA, LVBench,
LoVR, and a small custom dataset format. These were useful during development
but are not required for the minimal paper example.

Use the matching config names and override their paths:

```bash
python src/evaluate.py dataset=ego4d_nlq \
  dataset.annotations_dir=/path/to/ego4d/annotations \
  dataset.clips_dir=/path/to/ego4d/clips
```

## Custom Dataset Format

Config: `configs/dataset/custom_dataset.yaml`

The custom loader accepts JSON with either a flat list of samples or a
`videos -> queries` layout. Each query should provide a natural-language
`query` and either:

- `timestamp: [start, end]`
- `gt_windows: [[start, end], ...]`

Video paths can be specified with `video_path`, `file_name`, or `video_id`.
