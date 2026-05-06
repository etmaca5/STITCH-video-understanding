# Paper runs

Pointers to the runs behind every table cell and figure in the main paper
(`STITCH_paper/main.tex`). Each entry lists the run directory and only the
parameters that differ from project defaults / are relevant to the comparison.
Shared settings (frozen InternVideo2 backbone, kernel-CPD chunking, MaxSim
chunk score for VLM QA, etc.) are not repeated.

## Discrepancies between paper text and runs (please review)

1. **TAPOS row in Table 1 (`5_app1_gebd.tex`)**: paper reports STITCH numbers
   31.2 / 39.6 / 43.2 / 47.2 / 44.8. These values are from Sevan and his local runs of the dataset

---

## Application 1 — Event Detection (Table 1, `5_app1_gebd.tex`)

### Kinetics-GEBD (val)

- Run: `clean_results/event_detection/kinetics_gebd/1-gebd_val_full_si05_mc0.5_pen0.03_refine-3787e505`
- Differing params: `chunking.sample_interval=0.5`, `chunking.penalty=0.03`,
  `postprocessing.merge.min_chunk_sec=0.5`,
  `postprocessing.gebd_boundary_refinement.enabled=true`
- Numbers reproduced: F1@.05 68.35, F1@.10 79.01, F1@.20 84.45, F1@.30 86.53, Avg 83.67

### TAPOS (val)

- See discrepancy #1 above. Ask Sevan.

---

## Application 2 — Moment Retrieval (Tables 2 & 3, `6_app2_mr.tex`)

### ActivityNet Captions (val) — Table 2

- Run: `clean_results/moment_retrieval/activitynet/5-anet_coh_tau5-1ab9c083`
- Differing params:
  `evaluation.chunk_score_method=coherence_mean`,
  `evaluation.chunk_aggregation.coherence_tau=5.0`,
  `postprocessing.query_merge.enabled=false`,
  `postprocessing.moment_selection.method=top_gap`,
  `postprocessing.moment_selection.gap=0.06`
- Numbers: R@1 IoU=.3 51.57, R@1 IoU=.5 32.56, R@1 IoU=.7 18.01, mIoU 37.14

### QVHighlights (val) — Table 3

- Run: `clean_results/moment_retrieval/qvhighlights/8-qvh_max_final-e2f225f1`
- Differing params:
  `evaluation.chunk_score_method=max_sim` (default),
  `postprocessing.query_merge.enabled=false`,
  `postprocessing.moment_selection.method=top_gap`,
  `postprocessing.moment_selection.gap=0.06`
- Numbers: R@1 IoU=.5 64.58, R@1 IoU=.7 49.10, mAP avg 41.76

---

## Application 3 — VLM QA (Table 4, `7_app3_vlm.tex`)

All runs use `prompt_style=official_dataset_mcq`, 8 input frames, full splits
(MLVU dev, LongVideoBench val, VideoMME test). Only the differing
frame-selection parameters are listed.

### Qwen3-VL-8B

| Cell | Run | Method params |
|---|---|---|
| MLVU uniform | `clean_results/vlm_qa/mlvu/48-qwen3-vl-8b_mlvu-dev_uniform8_official-m-9dd8ce01` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| MLVU greedy | `clean_results/vlm_qa/mlvu/49-qwen3-vl-8b_mlvu-dev_intra-chunk-greedy_-ce9195a2` | `method=intra_chunk_greedy`, `n_frames=8` |
| MLVU MMR penalty | `clean_results/vlm_qa/mlvu/50-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-356f1847` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.7`, `chunk_weight=0.1` |
| LVB uniform | `clean_results/vlm_qa/longvideobench/59-qwen3-vl-8b_lvb-val_uniform8_official-da-0d4988f9` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| LVB greedy | `clean_results/vlm_qa/longvideobench/60-qwen3-vl-8b_lvb-val_intra-chunk-greedy_n-5e6adfbf` | `method=intra_chunk_greedy`, `n_frames=8` |
| LVB MMR penalty | `clean_results/vlm_qa/longvideobench/58-qwen3-vl-8b_lvb-val_mmr-chunk-penalty_l0-3e9fbf82` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.7`, `chunk_weight=0.2` |
| VMME uniform | `clean_results/vlm_qa/videomme/37-qwen3-vl-8b_videomme-test_uniform8_offic-93dc8f3c` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| VMME greedy | `clean_results/vlm_qa/videomme/38-qwen3-vl-8b_videomme-test_intra-chunk-gr-9095ed86` | `method=intra_chunk_greedy`, `n_frames=8` |
| VMME MMR penalty | `clean_results/vlm_qa/videomme/40-qwen3-vl-8b_videomme-test_ablation-mlvu--db90b6fd` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.7`, `chunk_weight=0.1` (matches MLVU params; outperforms cw=0.2 run 39 by +0.66) |

### LLaVA-OneVision Qwen2 7B

| Cell | Run | Method params |
|---|---|---|
| MLVU uniform | `clean_results/vlm_qa/mlvu/46-llava-ov_mlvu-dev_uniform8_official-mcq_-5e625254` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| MLVU greedy | `clean_results/vlm_qa/mlvu/44-llava-ov_mlvu-dev_intra-chunk-greedy_n8_-e8196e60` | `method=intra_chunk_greedy`, `n_frames=8` |
| MLVU MMR penalty | `clean_results/vlm_qa/mlvu/47-llava-ov_mlvu-dev_mmr-chunk-penalty_l07_-c7370e54` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.7`, `chunk_weight=0.1` |
| LVB uniform | `clean_results/vlm_qa/longvideobench/54-llava-ov_lvb-val_uniform8_official-datas-d53e08eb` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| LVB greedy | `clean_results/vlm_qa/longvideobench/56-llava-ov_lvb-val_intra-chunk-greedy_n8_o-5e55376a` | `method=intra_chunk_greedy`, `n_frames=8` |
| LVB MMR penalty | `clean_results/vlm_qa/longvideobench/57-llava-ov_lvb-val_mmr-chunk-penalty_l085_-cada4981` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.85`, `chunk_weight=0.2` |
| VMME uniform | `clean_results/vlm_qa/videomme/34-llava-ov_videomme-test_uniform8_official-d26ae22c` | `chunk_source=uniform`, `num_uniform_chunks=8` |
| VMME greedy | `clean_results/vlm_qa/videomme/35-llava-ov_videomme-test_intra-chunk-greed-16426f98` | `method=intra_chunk_greedy`, `n_frames=8` |
| VMME MMR penalty | `clean_results/vlm_qa/videomme/36-llava-ov_videomme-test_mmr-chunk-penalty-a699a83b` | `method=mmr_chunk_penalty`, `n_frames=8`, `relevance_weight=0.85`, `chunk_weight=0.4` |

All LLaVA-OV numbers in Table 4 match these runs exactly.

External baselines in Table 4 (Frame-Voyager, KFC, BOLT, AKS, Frame-Oracle,
WFS-SB) are not our runs; numbers come from the cited papers (or from
WFS-SB's reported numbers).

---

## Figure 3 — Qwen3-VL-8B diagnostic plots (`7_app3_vlm.tex`)

Built by `analysis/generate_vlm_qa_plots.sh`.

### Fig 3(a) — `main1_aggregated_video_length_vs_error`

Aggregated over MLVU dev, LongVideoBench val, VideoMME test, 8-frame budget,
Qwen3-VL-8B. Same nine runs as the Qwen3-VL-8B rows of Table 4 above
(`vmme/{37,38,39}`, `lvb/{58,59,60}`, `mlvu/{48,49,50}`).

### Fig 3(b)/(c) — `main2_..._budget_vs_error_overall`, `main3_..._budget_per_duration_vs_error`

**Caption says LongVideoBench, but the plot script uses MLVU dev** — see
discrepancy #3. The runs actually consumed are a Qwen3-VL-8B frame-budget
sweep on MLVU dev.

Uniform sweep (`method=middle`, `chunk_source=uniform`, varying `num_uniform_chunks`):

| n | Run |
|---|---|
| 0  | `clean_results/vlm_qa/mlvu/64-qwen3-vl-8b_mlvu-dev_uniform_n0_official-b004e096` |
| 1  | `clean_results/vlm_qa/mlvu/65-qwen3-vl-8b_mlvu-dev_uniform_n1_official-1315080f` |
| 2  | `clean_results/vlm_qa/mlvu/66-qwen3-vl-8b_mlvu-dev_uniform_n2_official-050c64a4` |
| 4  | `clean_results/vlm_qa/mlvu/67-qwen3-vl-8b_mlvu-dev_uniform_n4_official-0465490d` |
| 8  | `clean_results/vlm_qa/mlvu/48-qwen3-vl-8b_mlvu-dev_uniform8_official-m-9dd8ce01` |
| 16 | `clean_results/vlm_qa/mlvu/68-qwen3-vl-8b_mlvu-dev_uniform_n16_officia-4a2b6071` |
| 24 | `clean_results/vlm_qa/mlvu/69-qwen3-vl-8b_mlvu-dev_uniform_n24_officia-57ece7cb` |

MMR-chunk-penalty sweep (`method=mmr_chunk_penalty`, `relevance_weight=0.7`,
`chunk_weight=0.1`, varying `n_frames`):

| n | Run |
|---|---|
| 1  | `clean_results/vlm_qa/mlvu/70-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-5d50a9f8` |
| 2  | `clean_results/vlm_qa/mlvu/71-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-d9915cf7` |
| 4  | `clean_results/vlm_qa/mlvu/72-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-22aa9d25` |
| 8  | `clean_results/vlm_qa/mlvu/50-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-356f1847` |
| 16 | `clean_results/vlm_qa/mlvu/73-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-a732d255` |
| 24 | `clean_results/vlm_qa/mlvu/74-qwen3-vl-8b_mlvu-dev_mmr-chunk-penalty_l-7a36b358` |
