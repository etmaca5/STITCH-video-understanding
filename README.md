# STITCH: Training-Free Temporal Abstraction for General Video Understanding

This repository contains the example implementation for the paper
**Training-Free Temporal Abstraction for General Video Understanding**.

STITCH converts a video into reusable semantic chunks without task-specific
training. It samples short temporal windows, embeds them with a frozen
video-text model, detects semantic change points in the embedding sequence, and
then reuses the resulting chunks for:

- generic event boundary detection
- language-based moment retrieval
- frame selection for long-video VLM question answering

The code is intended as a compact reference artifact rather than a full dump of
all experiment outputs. Large datasets, model checkpoints, cached embeddings,
analysis notebooks, and result folders are intentionally excluded.

## Repository Layout

```text
configs/                 Hydra configs for datasets, chunking, retrieval, and evaluation
models/InternVideo2-...   lightweight InternVideo2 model-definition files only
src/                     STITCH implementation and evaluation code
DATASETS.md              dataset preparation notes and expected config fields
EVALUATION.md            commands for reproducing the main evaluation modes
requirements.txt         minimal Python dependencies
```

## Installation

Use Python 3.10 or newer. A CUDA-enabled PyTorch install is recommended for the
InternVideo2 embedding backend.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install the PyTorch build that matches your CUDA/CPU environment first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Download the InternVideo2 Stage2 1B 224p-f4 checkpoint (`.pt` file) from the
[OpenGVLab Hugging Face repository](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4)
and place it under:

```text
models/InternVideo2-Stage2_1B-224p-f4/
```

This repository includes the small model-definition/config files needed by the
loader, but not the checkpoint weights.

For VLM-based evaluation modes, copy `.env.example` to `.env` and fill in API
keys for your chosen provider (see `configs/vlm/` for available options).

## Quick Start

Show the resolved default config:

```bash
python src/evaluate.py --cfg job
```

Run the default embedding-based moment retrieval pipeline:

```bash
python src/evaluate.py \
  dataset=qvhighlights \
  dataset.data_root=/path/to/qvhighlights_data \
  retrieval.model_dir=/path/to/InternVideo2-Stage2_1B-224p-f4
```

Run STITCH for generic event boundary detection:

```bash
python src/evaluate.py \
  dataset=kinetics_gebd \
  evaluation=gebd \
  dataset.video_dir=/path/to/kinetics_gebd/clips \
  dataset.annotation_path=/path/to/k400_mr345_val_min_change_duration0.3.pkl
```

Run long-video VLM QA frame selection:

```bash
python src/evaluate.py \
  dataset=mlvu \
  evaluation=vlm_qa \
  temporal_abstraction=qa \
  vlm=llava_onevision_hf \
  dataset.dataset_root=/path/to/MLVU \
  vlm.base_url=https://your-endpoint.example.com/v1 \
  vlm.api_key_env=HF_TOKEN
```

Outputs are written to `clean_results/` by default and are ignored by git.

## Main Config Knobs

- `chunking=embedding`: STITCH semantic chunking with frozen video embeddings.
- `chunking.sample_interval`: temporal stride for window embeddings.
- `chunking.threshold_method=kernel_cpd`: change-point detection method used by
  the paper runs.
- `evaluation=embedding`: moment retrieval with text-query scoring.
- `evaluation=gebd`: event boundary detection from chunk edges.
- `evaluation=vlm_qa`: frame selection for multiple-choice video QA.
- `temporal_abstraction=qa`: VLM QA prompt/frame-selection settings.
- `retrieval=internvideo2`: default frozen video-text encoder.

See [EVALUATION.md](EVALUATION.md) for paper-oriented commands and
[DATASETS.md](DATASETS.md) for dataset preparation.

## Artifact Scope

The public repository intentionally excludes:

- benchmark videos and annotations
- model checkpoint weights
- generated result folders
- window-embedding caches
- notebooks and exploratory analysis files
- local API endpoints, tokens, and machine-specific paths

This keeps the example repository small and cloneable while preserving the code
needed to run STITCH once datasets and checkpoints are supplied locally.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{casanova2026stitch,
  title={Training-Free Temporal Abstraction for General Video Understanding},
  author={Casanova, Etienne and Brodjian, Sevan and Perona, Pietro},
  year={2026}
}
```
