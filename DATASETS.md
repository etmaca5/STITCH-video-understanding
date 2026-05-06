# Dataset Notes

This repo currently uses or stores these datasets:

## QVHighlights

Path used in this repo:

- Data root: `/mnt/c/qvhighlights_data`
- Config: `configs/dataset/qvhighlights.yaml`

What it is:

- QVHighlights is a query-based video highlight and moment retrieval dataset
- Each example starts from a free-form natural language query
- The dataset contains both temporal moment labels and highlight saliency labels

What the labels look like:

- Annotation file used here: `/mnt/c/qvhighlights_data/annotations/highlight_val_release.jsonl`
- Each JSONL row contains:
- `qid`: query id
- `query`: free-form text query
- `vid`: video id
- `duration`: clip duration
- `relevant_windows`: one or more ground-truth temporal windows
- `relevant_clip_ids`: relevant clip indices
- `saliency_scores`: per-clip saliency score triplets

How it is labeled:

- Temporal labels are the `relevant_windows`
- Highlight labels are the saliency scores for the relevant clips
- In the local file, saliency is stored as score triplets such as `[1, 1, 2]`
- In this repo, highlight evaluation uses 2-second clips as configured in `configs/dataset/qvhighlights.yaml`

How this repo uses it:

- The loader uses `query` as the text query
- It uses `relevant_windows` as the ground-truth windows
- It also keeps `duration`, `relevant_clip_ids`, and `saliency_scores` in metadata

Current local note:

- The local validation file has `1550` query examples

## ActivityNet Captions

Paths used in this repo:

- Videos: `activitynetdata/Anet_videos_15fps_short256`
- Captions: `third_party/VideoX/MS-2D-TAN/data/ActivityNet`
- Config: `configs/dataset/activitynet.yaml`

What it is:

- ActivityNet Captions is a dense video captioning dataset
- Each video has multiple temporally localized natural-language sentences
- It is commonly used both for dense captioning and for temporal sentence grounding

What the labels look like:

- Annotation file used here: `third_party/VideoX/MS-2D-TAN/data/ActivityNet/val.json`
- The JSON maps `video_id -> info`
- Each `info` object contains:
- `duration`
- `timestamps`: a list of `[start, end]` windows
- `sentences`: a list of captions aligned with those windows

How it is labeled:

- The labels are not class names
- They are natural-language event descriptions aligned to temporal windows
- Example sentence: `A weight lifting tutorial is given.`

How this repo uses it:

- The loader turns each sentence into a separate query
- The matching timestamp becomes the single ground-truth window for that query
- Metadata keeps the `video_id` and `duration`

Current local note:

- The local validation file has `4917` videos

## ActivityNet-QA

Paths used locally:

- Videos: `activitynetdata/Anet_videos_15fps_short256`
- QA annotations: `activitynetdata/activitynet-qa/dataset`
- Official eval script: `activitynetdata/activitynet-qa/evaluation`

What it is:

- ActivityNet-QA is a Video Question Answering benchmark built on ActivityNet videos
- The task is not temporal grounding
- Each example is a natural-language question paired with an answer

What the labels look like:

- The local clone contains split-specific question and answer files:
- `train_q.json`, `train_a.json`
- `val_q.json`, `val_a.json`
- `test_q.json`, `test_a.json`
- The upstream README describes the dataset as `58,000` human-annotated QA pairs on `5,800` videos

How it is labeled:

- Supervision is answer-based rather than `query -> gt_windows`
- This makes it a VideoQA dataset, not a moment retrieval dataset
- The official repository provides an accuracy-style evaluation script over predicted answers

How it would fit this repo:

- The current evaluation pipeline expects each sample to provide `video_path`, `query`, and `gt_windows`
- That matches `ActivityNet Captions`, but not `ActivityNet-QA`
- Proper integration here will need a dedicated loader and a QA-style evaluation path rather than the current IoU / recall metrics

Current local note:

- The official annotation repo is cloned locally under `activitynetdata/activitynet-qa`
- Cloning the repo provides the annotation json files and eval script, but not the raw videos

## Custom Dataset

Paths used in this repo:

- Annotation file: `customhighlightsdata/custom_dataset.json`
- Video directory: `customhighlightsdata`
- Config: `configs/dataset/custom_dataset.yaml`

What it is:

- This is the repo's user-defined dataset format for small custom evaluations
- It is not tied to an external benchmark taxonomy
- You define the videos, the text queries, and the correct temporal windows

What the labels look like:

- The file is organized as `videos`
- Each video entry can contain:
- `video_id`
- `file_name`
- `split`
- `queries`
- Each query entry contains:
- `query`
- either `timestamp: [start, end]`
- or `gt_windows: [[start, end], ...]`

How it is labeled:

- There are no class labels or captions unless you write them yourself as queries
- The supervision is purely temporal: one or more correct windows per query
- This format supports multiple valid windows for the same query

How this repo uses it:

- The loader accepts either a flat sample list or the nested `videos -> queries` layout
- It resolves the video from `video_path`, `file_name`, or `video_id`
- It normalizes `timestamp` / `gt_windows` into the common `gt_windows` format

Current local note:

- The local custom file has `21` videos and `38` query annotations

## LongVideoBench

Path currently used locally:

- Dataset root: `/mnt/c/longvideobench_data/LongVideoBench`

What it is:

- LongVideoBench is a long-context video-language benchmark
- It contains long web videos, subtitles, and multiple-choice questions
- The benchmark is designed for long-video understanding rather than simple moment retrieval

What the labels look like:

- Main files include `lvb_val.json` and `lvb_test_wo_gt.json`
- Video content is shipped as `videos.tar.part.*`
- Subtitle files are shipped in `subtitles.tar`
- The test file is explicitly `wo_gt`, so it does not include ground-truth answers

How it is labeled:

- The supervision is question answering, not temporal grounding labels
- Inputs are interleaved video frames and subtitle text
- The dataset card describes `3763` videos and `6678` human-annotated multiple-choice questions

How to prepare it locally:

```bash
cd "/mnt/c/longvideobench_data/LongVideoBench"
cat videos.tar.part.* | tar -xvf -
tar -xvf subtitles.tar
```

Repo note:

- `LongVideoBench` is now integrated in the repo as a first-class dataset
- The repo now supports `split: val | test`
- The implemented evaluation matches the public answer parsing and accuracy-style metrics used by the benchmark ecosystem:
  - overall multiple-choice accuracy
  - subset accuracy by `duration_group`
  - subset accuracy by `question_category`
- Important caveat: the current repo path is metric-faithful, but not protocol-identical to the official end-to-end benchmark input construction
- The official LongVideoBench loader interleaves subtitle text with sampled frames before the question/options
- This repo currently uses its own chunk/frame QA prompt path and injects subtitle text as prompt metadata rather than reproducing the exact official interleaved frame+subtitle format

## Video-MME

Path currently used locally:

- Dataset root: `/mnt/c/videomme_data/Video-MME`

What it is:

- `Video-MME` is a long-video understanding benchmark for multimodal video QA
- It contains long videos, subtitle files, and multiple-choice question annotations
- The official benchmark describes `900` videos and `2700` question-answer pairs

What the data looks like:

- Extracted videos live under `data/` as `.mp4` files
- Subtitle files live under `subtitle/`
- The annotation table is `videomme/test-00000-of-00001.parquet`
- The parquet uses `videoID` for the actual video / subtitle file stem

How it is labeled:

- Supervision is multiple-choice question answering rather than temporal grounding
- The parquet file includes fields such as `video_id`, `duration`, `domain`, `sub_category`, `question_id`, `task_type`, `question`, `options`, and `answer`
- The `duration` field is the paper split label (`short`, `medium`, or `long`), which makes it suitable for a clean `duration_split` loader option
- This is a long-video QA benchmark, not a `query -> gt_windows` retrieval dataset

How to prepare it locally:

```bash
mkdir -p "/mnt/c/videomme_data/Video-MME"
hf download lmms-lab/Video-MME \
  --repo-type dataset \
  --local-dir "/mnt/c/videomme_data/Video-MME"
cd "/mnt/c/videomme_data/Video-MME"
for f in videos_chunked_*.zip; do bsdtar -xf "$f"; done
bsdtar -xf subtitle.zip
```

Current local note:

- The local extraction currently has `900` `.mp4` files under `data/`
- The local `subtitle/` directory exists
- The local annotation parquet exists at `videomme/test-00000-of-00001.parquet`

Repo note:

- The repo now has a dedicated `videomme` dataset config with `duration_split: all | short | medium | long`
- The QA integration uses multiple-choice accuracy rather than retrieval metrics
- QA prompt metadata is controlled from `configs/temporal_abstraction/qa.yaml`
- The current QA metadata toggles are:
  - `metadata.timestamps`
  - `metadata.subtitles`
  - `metadata.video_duration`
- Subtitle inclusion is aligned to the selected chunks shown to the VLM, not the full transcript

## LVBench

Paths currently used locally:

- Dataset root: `/mnt/c/lvbench_data/LVBench`
- Metadata file: `/mnt/c/lvbench_data/LVBench/data/video_info.meta.jsonl`
- Downloaded videos: `/mnt/c/lvbench_data/LVBench/scripts/videos`

What it is:

- LVBench is a long-video understanding benchmark
- It uses long public-source web videos together with multiple-choice questions
- The task is long-video QA rather than temporal grounding or retrieval

What the data looks like:

- The official workflow uses `video_info.meta.jsonl` from the LVBench Hugging Face dataset
- Each metadata row contains a `key` field that maps to a YouTube video id
- The official download script materializes the video files under `scripts/videos`

How it is labeled:

- Supervision is multiple-choice question answering, not `query -> gt_windows`
- The videos are reconstructed from public source ids, so some raw videos can be unavailable over time
- The benchmark is intended for academic research use

How to prepare it locally:

```bash
cd /mnt/c/lvbench_data
git clone https://github.com/zai-org/LVBench.git
cd /mnt/c/lvbench_data/LVBench
pip install video2dataset
pip uninstall -y transformer-engine
pip install -e .
# then place video_info.meta.jsonl in data/
cd /mnt/c/lvbench_data/LVBench/scripts
bash download.sh
```

Current local note:

- A recent local run requested `103` videos and downloaded `85` successfully
- The corresponding summary file is `scripts/videos/00000_stats.json`

Repo note:

- `LVBench` is now integrated in the repo as a first-class dataset
- The loader reads `video_info.meta.jsonl` and maps YouTube IDs to locally downloaded mp4 files via sidecar JSONs
- The evaluation uses multiple-choice accuracy (overall + per question-type category), replicating the official LVBench `compute_accuracy` metric exactly
- QA items whose videos are not available locally are skipped at load time

## LoVR

> **OUT OF CONSIDERATION.** LoVR is currently dropped from the representation-technique
> sweep. The `lovr_retrieval` evaluation mode encodes each full video by sampling
> only `num_frames` frames uniformly across the whole (~26 min) video and does not
> go through the chunking + aggregation pipeline (`chunk_score_method` /
> `chunk_aggregation`) used by the moment-retrieval datasets. As a result,
> GEM / CoH / LSE / max_sim cannot be meaningfully compared on LoVR, and any
> numbers produced here are not comparable to the ActivityNet / QVHighlights
> sweeps. Revisit once chunking + aggregation are wired into
> `_evaluate_lovr_retrieval`.

Paths currently used locally:

- Dataset root: `/mnt/c/lovr_data`
- Captions: `/mnt/c/lovr_data/caption_data`
- Clip folders: `/mnt/c/lovr_data/video_data/long_video_clip`
- Merged full videos: `/mnt/c/lovr_data/video_data/full_videos`
- Merge script: `/mnt/c/lovr_data/scripts/merge_clips.py`

What it is:

- LoVR is a benchmark for long video retrieval in multimodal contexts
- It builds on `LongVideoBench` video content, but adds retrieval-oriented clip-level and video-level captions
- The benchmark supports both clip-level retrieval and full-video retrieval

What the data includes:

- `caption_data/clip_train.parquet`
- `caption_data/clip_test.parquet`
- `caption_data/video_train.parquet`
- `caption_data/video_test.parquet`
- `video_data/part_aa`, `part_ab`, `part_ac`
- `scripts/merge_clips.py`

What the labels look like:

- The clip-level parquet files reference individual segmented clips
- The video-level parquet files reference whole videos
- The Hugging Face dataset card describes `41,271` rows in total
- The project page describes `467` long videos and `40,804` fine-grained clips

How it is labeled:

- Clip-level supervision is natural-language caption text paired with a clip path
- Video-level supervision is natural-language caption text paired with a whole-video id
- This is a retrieval benchmark, not a temporal grounding benchmark
- Unlike `QVHighlights`, `Ego4D MQ`, or `ActivityNet Captions`, the core label is caption relevance rather than `query -> gt_windows`

How to prepare it locally:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='debugger123/LoVR-benchmark', repo_type='dataset', local_dir='/mnt/c/lovr_data')"
cat /mnt/c/lovr_data/video_data/part_* > /mnt/c/lovr_data/video_data/lovr_video_data.tar.gz
tar -xzf /mnt/c/lovr_data/video_data/lovr_video_data.tar.gz -C /mnt/c/lovr_data/video_data
python /mnt/c/lovr_data/scripts/merge_clips.py --input /mnt/c/lovr_data/video_data/long_video_clip --output /mnt/c/lovr_data/video_data/full_videos
```

Current local note:

- A clean re-extraction produced `468` clip folders locally
- A clean merge produced `467` merged full videos locally
- The only known missing merged id is `fR9dhkJyNNo`
- The corresponding source folder `long_video_clip/fR9dhkJyNNo` exists but is empty
- Earlier string checks against the parquet files did not show `fR9dhkJyNNo` in the visible annotation strings, so it may be an unused stray folder rather than an evaluation blocker

Repo note:

- `LoVR` is now integrated in the repo as a first-class dataset
- The repo uses a dedicated `lovr_retrieval` evaluation mode rather than forcing LoVR into the temporal grounding pipeline
- The implemented retrieval metrics match the public core LoVR evaluation outputs:
  - `clip_pass@k`
  - `full_pass@k`
  - `v2t_clip_pass@k`
  - `v2t_full_pass@k`
- Important caveat: this implementation is faithful to the core public retrieval metrics, but it is not a byte-for-byte clone of every optional official evaluation output
- In particular, the optional theme retrieval metrics mentioned in the LoVR evaluation code are not implemented here

## VLM / Temporal Abstraction Notes

These notes are here instead of a separate top-level notes file because they
describe how the long-video datasets should plug into the existing retrieval
repo.

Current implementation notes:

- `OpenRouter` is the provider entry point and the model id should stay
  configurable via Hydra (`configs/vlm/*.yaml`)
- The current default VLM is `qwen/qwen3-vl-8b-instruct`
- The temporal abstraction settings should stay separate from the model config
  in `configs/temporal_abstraction/*.yaml`
- That split is important because frame selection, metadata formatting, prompt
  shape, and chunk budgeting are task settings, not model settings
- The tested OpenRouter routes are model-specific: `qwen/qwen-2.5-vl-7b-instruct`
  only accepted `4` images in one request on the current provider route, while
  `qwen/qwen3-vl-8b-instruct` accepted at least `24`
- The image budget should stay configurable in `configs/vlm/*.yaml`

Critical multi-modal formatting rule:

- For `OpenRouter` and `Qwen2.5-VL`, put the text block first and the
  `image_url` entries after it in the same user message
- We should not switch this ordering casually: incorrect image-first ordering
  is known to produce weak or irrelevant outputs
- Frames should be sent as base64 JPEG data URLs
- The practical image budget must be enforced before sending a request because
  provider-specific limits can be lower than what the model can theoretically
  support

Prompt ordering note for chunk retrieval:

- Preserve the chunk order explicitly in the text prompt:
  `Chunk 1 | [start-end] | metadata ...`, then `Chunk 2`, etc.
- Attach images in that same chunk order
- If we add per-chunk metadata later, it should be configurable fields such as
  `caption`, `summary`, `score`, or dataset-specific labels

Hydra structure going forward:

- `configs/vlm/*.yaml`: provider/model/API settings
- `configs/temporal_abstraction/*.yaml`: frame selection, metadata inclusion,
  prompt budget, and response format
- Dataset/task configs should decide which metadata fields are available, but
  temporal abstraction configs should decide which of those fields are exposed
  to the VLM

Future task mapping:

- `ActivityNet Captions`: caption -> best chunk selection, then compare against
  uniform sampling
- `LongVideoBench`, `Video-MME`, `LVBench`: multiple-choice QA over ordered
  chunk frames plus metadata, with a different prompt template and accuracy
  evaluation
- `LoVR`: retrieval-style prompting where clip/video-level context is shown to
  the VLM without forcing it into the current `query -> gt_windows` API

## Kinetics-GEBD

Paths currently used locally:

- Dataset root: `/mnt/c/kinetics_gebd_data`
- Videos: `/mnt/c/kinetics_gebd_data/clips/{split}/{class_name}/{clip_stem}.mp4`
- Raw annotations: `/mnt/c/kinetics_gebd_data/downloads/gebd_gdrive/k400_{split}_raw_annotation.pkl`
- Processed annotations: `/mnt/c/kinetics_gebd_data/gebd_repo/data/export/k400_mr345_{split}_min_change_duration0.3.pkl`
- Official repo clone: `/mnt/c/kinetics_gebd_data/gebd_repo`
- Download manifest: `/mnt/c/kinetics_gebd_data/metadata/gebd_manifest.csv`
- Config: `configs/dataset/kinetics_gebd.yaml`

What it is:

- Kinetics-GEBD (Generic Event Boundary Detection) is an event segmentation benchmark built on Kinetics-400 videos
- The task is: given a ~10-second video clip, detect the timestamps of generic event boundaries (scene changes, action changes, shot changes)
- Multiple human raters annotate each video independently; the evaluation picks the best-matching rater per video
- Official reference: ICCV 2021, https://github.com/StanLei52/GEBD

What the labels look like:

- Processed annotation pkl maps `video_id -> info`
- Each info dict contains:
  - `fps`, `video_duration`, `num_frames`, `path_video`
  - `substages_timestamps`: list of raters, each rater has a list of boundary timestamps in seconds
  - `substages_myframeidx`: same but as frame indices
  - `f1_consis_avg`: inter-rater consistency score (videos below 0.3 are excluded)
- The processed pkl is generated from raw annotations using `prepare_k400_release.ipynb`
- Raw annotations include typed boundary labels (EventChange, ShotChangeGradualRange, ShotChangeImmediateTimestamp)

How this repo uses it:

- The loader creates one `EvalSample` per video (no text queries; GEBD is query-free)
- Evaluation mode `gebd` runs chunking on each video and extracts chunk boundaries as predicted event boundaries
- The metric is F1@threshold (default threshold=0.05, meaning 5% of video duration)
- This matches the official LOVEU Challenge evaluation code exactly
- Dedicated GEBD plots show GT vs predicted boundaries on a timeline

How to prepare it locally:

```bash
mkdir -p /mnt/c/kinetics_gebd_data
cd /mnt/c/kinetics_gebd_data

# 1. Clone official repo
git clone https://github.com/StanLei52/GEBD.git gebd_repo

# 2. Download raw annotations from Google Drive
# https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo
# Place files in downloads/gebd_gdrive/

# 3. Generate processed annotations (run in gebd_repo/data/export/)
# Use prepare_k400_release.ipynb with split='val' and split='train'

# 4. Download Kinetics-400 videos using the manifest CSV
# Videos go under clips/{split}/{class_name}/{clip_stem}.mp4
```

Current local note:

- Train: 14,849 of ~18,808 videos downloaded
- Val: download planned (~18,815 videos needed)
- Processed annotations: 18,808 train, 18,815 val videos
- After f1_consis_avg >= 0.3 filter: ~18,615 train, ~18,465 val evaluable videos

Video statistics (f1_consis >= 0.3):

| Statistic | Train | Val |
|---|---|---|
| Videos | 18,615 | 18,465 |
| Duration mean | 9.55s | 9.57s |
| Duration median | 10.00s | 10.00s |
| Duration std | 1.36s | 1.32s |
| Duration range | [0.63s, 10.12s] | [0.50s, 10.15s] |
| p5 / p25 / p75 / p95 | 6.07 / 10.00 / 10.01 / 10.02 | 6.23 / 10.00 / 10.01 / 10.02 |
| FPS range | 6–30 (mean 27.5) | 6–30 (mean 27.5) |
| GT boundaries per rater | mean 4.9, median 5, range [0, 14] | mean 4.4, median 4, range [0, 15] |

Embedding window counts at different `sample_interval` values:

| `sample_interval` | Mean windows | Min | Median |
|---|---|---|---|
| 0.5s | 19.0 | 1 | 20 |
| 1.0s | 9.5 | 1 | 10 |
| 1.5s | 5.7 | 1 | 6 |
| 2.0s | 4.7 | 1 | 5 |

Tuning considerations:

- The default `sample_interval=2.0s` gives only ~5 windows per video, while GT has ~5 boundaries on average. The change point detector can produce at most 4 boundaries from 5 windows, and `postprocessing.merge.min_chunk_sec=3.0` aggressively merges these further. This produces high precision but low recall.
- `sample_interval=1.0s` gives ~10 windows, providing much better resolution for detecting boundaries. This is likely the right starting point for GEBD tuning.
- `sample_interval=0.5s` gives ~20 windows, which provides the highest resolution but increases computation proportionally.
- `postprocessing.merge.min_chunk_sec` should also be reduced for GEBD (e.g., 1.0s or lower) since event boundaries in short clips can be very close together.
- The `kernel_cpd` penalty parameter controls how many boundaries are detected. A lower penalty yields more boundaries (higher recall, potentially lower precision).
- `postprocessing.gebd_boundary_refinement.enabled=true` adds a GEBD-only post-step that keeps the detected boundary count fixed, then snaps each boundary to the frame with the largest content change between the centers of the neighboring windows. This is intended to improve temporal precision without changing other evaluation modes.

Known issues:

- `cv2.CAP_PROP_FRAME_COUNT` over-reports for many Kinetics clips (e.g., 445 frames reported for a 300-frame / 10s video). The chunking system validates the actual readable frame count via binary search before generating windows.
- Some videos fail with `ruptures.BadSegmentationParameters` when the embedding signal is too uniform for change point detection. These are logged as failed videos and do not crash the run.

## TODO

- Add proper dataset integration for all locally stored datasets that are not yet first-class evaluation tasks in this repo
- Keep temporal grounding / retrieval tasks separate from long-video understanding tasks
- Add task-specific loaders and evaluation code instead of forcing all datasets into the current `query -> gt_windows` interface
- For `LongVideoBench`, add a new long-video QA task path with support for subtitles, interleaved video-text inputs, multiple-choice questions, and accuracy-style evaluation
- For `LVBench`, add a long-video QA task path with support for public-source video reconstruction, multiple-choice questions, and accuracy-style evaluation
- Consider whether future long-video datasets should share a common task abstraction that is different from moment retrieval

## Current Space

Current free space on `/mnt/c`:

- `963G` available at the time this note was written
