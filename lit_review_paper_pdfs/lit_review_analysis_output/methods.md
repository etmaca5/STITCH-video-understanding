# Methods

**48 method papers total** (8 core, 33 supporting, 7 peripheral)

Sorted within each tier by relevance proximity to STITCH.

---

## Core Methods

These papers are directly cited as baselines, competitors, or foundational algorithms in STITCH.

| Title | Venue | TF | ZS | UN | QA | TU | LV | Role in STITCH |
|-------|-------|----|----|----|----|----|----|----------------|
| KTS Revisited (Afham et al.) | arXiv 2023 | ✓ | ✓ | ✓ | NQ | adaptive | ✓ | **Most similar prior work**: KTS as adaptive tokenizer; STITCH extends with cosine kernel + multi-task |
| TFVTG (Zheng et al.) | arXiv 2024 | ✓ | ✓ | ✓ | QD | proposal | ✗ | **Training-free moment retrieval baseline** (Table 2) |
| ZS Moment from Frozen VLMs | 2023 | ✓ | ✓ | ✗ | QD | proposal | ✗ | **Zero-shot moment retrieval baseline** (Table 2) |
| ZS Moment via MLLMs | 2024 | ✓ | ✓ | ✗ | QD | proposal | ✗ | **Zero-shot MLLM baseline** (Table 2) |
| GEBD via Diffusion | 2023/24 | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Supervised GEBD competitor** (Table 1) |
| Efficient GEBD | 2023 | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Supervised GEBD competitor** (Table 1) |
| Online GEBD | 2023 | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Supervised GEBD competitor** (Table 1) |
| Moment-DETR (with QVHighlights) | NeurIPS 2021 | ✗ | ✗ | ✗ | QD | proposal | ✗ | **Supervised moment retrieval baseline** (Table 2) |

---

## Supporting Methods — Training-free / Zero-shot (high relevance)

| Title | Venue | TF | ZS | UN | QA | TU | LV | Notes |
|-------|-------|----|----|----|----|----|----|-------|
| Wavelet Frame Selection | 2024 | ✓ | ✗ | ✓ | NQ | adaptive | ✗ | Unsupervised boundary-based frame selection; very close to STITCH App 3 |
| Moment Sampling VLM | 2024 | ✓ | ✗ | ✗ | NQ | adaptive | ✓ | Training-free frame selection for VLMs (App 3 adjacent) |
| VideoINSTA | EMNLP 2024 | ✓ | ✓ | ✗ | QD | window | ✓ | Zero-shot LLM-based long video reasoning |
| VideoTree | 2024 | ✓ | ✗ | ✗ | QD | hierarchical | ✓ | Adaptive tree representation for long video LLM |
| Prompts to Summaries | 2024 | ✓ | ✓ | ✗ | NQ | window | ✗ | Zero-shot VLM summarization |
| Point to Span | 2024 | ✓ | ✓ | ✗ | QD | proposal | ✓ | Zero-shot long video moment retrieval |
| AdaReTaKe | 2024 | ✓ | ✗ | ✗ | NQ | adaptive | ✓ | Adaptive redundancy reduction for VLMs |
| TAG | 2024 | ✗ | ✓ | ✗ | QD | proposal | ✗ | Zero-shot audio-visual temporal grounding |
| Vid-Morp | 2024 | ✗ | ✗ | ✓ | QD | proposal | ✗ | Unsupervised moment retrieval pretraining |

---

## Supporting Methods — Supervised moment retrieval & temporal grounding

| Title | Venue | QA | TU | Notes |
|-------|-------|----|----|-------|
| QD-DETR | CVPR 2023 | QD | proposal | Query-dependent DETR; Table 2 competitor |
| UniVTG | ICCV 2023 | QD | proposal | Unified VLM grounding; Table 2 competitor |
| UnLoc | ICCV 2023 | QP | proposal | Unified localization (moment + highlight + TAD) |
| SnAG | CVPR 2024 | QD | proposal | Scalable grounding; Table 2 competitor |
| BAM-DETR | 2024 | QD | proposal | Boundary-aligned DETR; Table 2 competitor |
| SimBase | 2024 | QD | proposal | Simple strong baseline for grounding |
| Self-Chained ILM | NeurIPS 2023 | QD | proposal | Multi-task VLM localization + QA |
| Aligning Moments | 2023 | QD | proposal | Moment retrieval baseline |
| Towards Balanced Alignment | 2024 | QD | proposal | Multi-modal alignment for grounding |
| TEMPURA | 2023 | QD | proposal | Object-level temporal grounding |
| TimeLoc | 2024 | QD | proposal | Long-horizon temporal localization |
| Measure Twice | 2024 | QD | proposal | Visual evidence for grounding |
| When One Moment Isn't Enough | 2024 | QD | hierarchical | Multi-scale multi-moment retrieval |
| 2D-TAN | AAAI 2020 | QD | proposal | 2D temporal adjacent network (classic) |
| LGVI | CVPR 2020 | QD | window | Local-global video-text interaction |
| TALL | ICCV 2017 | QD | proposal | Foundational moment retrieval paper |
| SCDM | NeurIPS 2019 | QD | window | Semantic conditioned dynamic modulation |

---

## Supporting Methods — Long video / VLM frame selection

| Title | Venue | TF | QA | TU | Notes |
|-------|-------|----|----|----|----|
| LongVU | NeurIPS 2024 | ✗ | NQ | adaptive | Spatiotemporal compression for long video VLMs |
| MovieChat | CVPR 2024 | ✗ | QD | window | Sliding window + cosine similarity memory |
| AdaVideoRAG | 2024 | ✗ | QD | window | RAG-based long video QA |
| LVC | 2024 | ✗ | QD | adaptive | Lightweight query-adaptive video compression |
| STORM | 2024 | ✗ | NQ | adaptive | Token-efficient long video MLLM |
| LVLR | 2024 | ✗ | QD | window | Long video with learnable retrieval |
| Chapter-Llama | 2024 | ✗ | NQ | adaptive | LLM-based video chaptering |

---

## Supporting Methods — Temporal action detection & scene segmentation

| Title | Venue | QA | TU | Notes |
|-------|-------|----|----|-------|
| TriDet | CVPR 2023 | NQ | proposal | Temporal action detection with boundary |
| Boundary-Aware SSL | 2023 | NQ | window | SSL scene boundary detection |
| Shot Relating | 2024 | QD | window | Shot-level video QA |

---

## Peripheral Methods

| Title | Venue | Notes |
|-------|-------|-------|
| ChatVTG | 2024 | LLM-chat temporal grounding; peripheral |
| Deep Video Discovery | 2024 | Agentic video search |
| DrVideo | 2024 | Document retrieval for long video |
| Hierarchical Long Video | 2024 | Multi-agent hierarchical video |
| Video-RAG | 2024 | Visual RAG for long video |
| Video-XL | 2024 | Extra-long VLM |
| Your Interest Your Summaries | 2024 | Query-aware summarization |
