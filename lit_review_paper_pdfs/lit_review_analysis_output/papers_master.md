# STITCH Literature Review — Master Table

**68 papers retained** (15 core, 40 supporting, 10 peripheral, 3 other) | **7 dropped**

Columns: `TF` = training-free | `ZS` = zero-shot | `UN` = unsupervised | `QA` = query_awareness (QD=query-dependent, QP=query-potential, NQ=no-query) | `TU` = temporal_unit | `LV` = long_video_framing | `RC` = reusable_compute_style

---

## CORE (15 papers)

| Title | Authors | Venue | Cat | TF | ZS | UN | QA | TU | LV | Closest Relation |
|-------|---------|-------|-----|----|----|----|----|----|----|-----------------|
| QVHighlights + Moment-DETR | Lei et al. | NeurIPS 2021 | dataset_benchmark | ✗ | ✗ | ✗ | QD | proposal | ✗ | **Our Table 2 benchmark** |
| Kinetics-GEBD Benchmark | Shou et al. | ICCV 2021 | dataset_benchmark | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Our Table 1 benchmark** |
| InternVideo2 | Wang et al. | ECCV 2024 | other | ✗ | ✗ | ✗ | NQ | window | ✓ | **Our frozen backbone** |
| MLVU | Zhou et al. | NeurIPS 2024 | dataset_benchmark | ✗ | ✗ | ✗ | QD | frame | ✓ | **Our Table 3 benchmark** |
| Video-MME | Fu et al. | CVPR 2024 | dataset_benchmark | ✗ | ✗ | ✗ | QD | frame | ✓ | **Our Table 3 benchmark** |
| LongVideoBench | Wu et al. | NeurIPS 2024 | dataset_benchmark | ✗ | ✗ | ✗ | QD | frame | ✓ | **Our Table 3 benchmark** |
| CPD (ruptures) | Truong et al. | Signal Proc. 2020 | other | ✓ | ✓ | ✓ | NQ | adaptive | ✗ | **Our CPD algorithm** |
| PELT | Killick et al. | JASA 2012 | other | ✓ | ✓ | ✓ | NQ | adaptive | ✗ | **CPD efficiency foundation** |
| GEBD via Diffusion | — | 2023/24 | method | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Table 1 GEBD competitor** |
| Efficient GEBD | — | 2023 | method | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Table 1 GEBD competitor** |
| Online GEBD | — | 2023 | method | ✗ | ✗ | ✗ | NQ | frame | ✗ | **Table 1 GEBD competitor** |
| KTS Revisited | Afham et al. (Meta AI) | arXiv 2023 | method | ✓ | ✓ | ✓ | NQ | adaptive | ✓ | **Most similar prior work** |
| TFVTG | Zheng et al. | arXiv 2024 | method | ✓ | ✓ | ✓ | QD | proposal | ✗ | **Training-free grounding baseline** |
| ZS Moment from Frozen VLMs | — | 2023 | method | ✓ | ✓ | ✗ | QD | proposal | ✗ | **Zero-shot grounding baseline** |
| ZS Moment via MLLMs | — | 2024 | method | ✓ | ✓ | ✗ | QD | proposal | ✗ | **Zero-shot grounding baseline** |

---

## SUPPORTING — Methods (28 papers)

| Title | Venue | TF | ZS | UN | QA | TU | LV | Closest Relation |
|-------|-------|----|----|----|----|----|----|-----------------|
| QD-DETR | CVPR 2023 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Table 2 supervised competitor |
| MovieChat | CVPR 2024 | ✗ | ✗ | ✗ | QD | window | ✓ | Long video chunking via cosine similarity |
| AdaReTaKe | 2024 | ✓ | ✗ | ✗ | NQ | adaptive | ✓ | Adaptive token reduction, VLM frame selection |
| Aligning Moments (2D-TAN+) | 2023 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Table 2 moment retrieval baseline |
| BAM-DETR | 2024 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Table 2 boundary-aligned competitor |
| Boundary-Aware SSL | 2023 | ✗ | ✗ | ✗ | NQ | window | ✗ | SSL boundary detection, adjacent to GEBD |
| Chapter-Llama | 2024 | ✗ | ✗ | ✗ | NQ | adaptive | ✓ | Video chaptering with LLM |
| 2D-TAN | AAAI 2020 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Classic moment retrieval baseline |
| LGVI | CVPR 2020 | ✗ | ✗ | ✗ | QD | window | ✗ | Supervised temporal grounding |
| LVLR | 2024 | ✗ | ✗ | ✗ | QD | window | ✓ | Long video retrieval |
| LongVU | NeurIPS 2024 | ✗ | ✗ | ✗ | NQ | adaptive | ✓ | Frame selection baseline for VLMs |
| Measure Twice | 2024 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Visual evidence temporal localization |
| Shot Relating | 2024 | ✗ | ✗ | ✗ | QD | window | ✗ | Shot-level video QA |
| Moment Sampling VLM | 2024 | ✓ | ✗ | ✗ | NQ | adaptive | ✓ | Frame selection for VLMs (App 3) |
| SCDM | NeurIPS 2019 | ✗ | ✗ | ✗ | QD | window | ✗ | Classic temporal grounding |
| Self-Chained ILM | NeurIPS 2023 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Multi-task frozen VLM localization |
| SimBase | 2024 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Simple supervised grounding baseline |
| SnAG | CVPR 2024 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Scalable supervised grounding |
| TAG | 2024 | ✗ | ✓ | ✗ | QD | proposal | ✗ | Zero-shot temporal grounding |
| TALL | ICCV 2017 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Foundational moment retrieval paper |
| TEMPURA | 2023 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Object-level temporal grounding |
| TimeLoc | 2024 | ✗ | ✗ | ✗ | QD | proposal | ✓ | Long-horizon temporal localization |
| Towards Balanced Alignment | 2024 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Multi-modal temporal grounding |
| TriDet | CVPR 2023 | ✗ | ✗ | ✗ | NQ | proposal | ✗ | Temporal action detection, boundary |
| UnLoc | ICCV 2023 | ✗ | ✗ | ✗ | QP | proposal | ✗ | Unified localization framework |
| UniVTG | ICCV 2023 | ✗ | ✗ | ✗ | QD | proposal | ✗ | Unified VLM temporal grounding |
| Vid-Morp | 2024 | ✗ | ✗ | ✓ | QD | proposal | ✗ | Unsupervised moment retrieval pretraining |
| VideoINSTA | EMNLP 2024 | ✓ | ✓ | ✗ | QD | window | ✓ | Zero-shot long video with LLM |
| VideoTree | 2024 | ✓ | ✗ | ✗ | QD | hierarchical | ✓ | Adaptive tree for VLM long video |
| Wavelet Frame Selection | 2024 | ✓ | ✗ | ✓ | NQ | adaptive | ✗ | Unsupervised frame selection via boundary |
| When One Moment Isn't Enough | 2024 | ✗ | ✗ | ✗ | QD | hierarchical | ✗ | Multi-scale moment retrieval |
| AdaVideoRAG | 2024 | ✗ | ✗ | ✗ | QD | window | ✓ | RAG-based long video QA |
| LVC | 2024 | ✗ | ✗ | ✗ | QD | adaptive | ✓ | Query-adaptive video compression |
| STORM | 2024 | ✗ | ✗ | ✗ | NQ | adaptive | ✓ | Token-efficient long video MLLM |
| Prompts to Summaries | 2024 | ✓ | ✓ | ✗ | NQ | window | ✗ | Zero-shot VLM summarization |
| Point to Span | 2024 | ✓ | ✓ | ✗ | QD | proposal | ✓ | Zero-shot long video moment retrieval |

---

## SUPPORTING — Datasets/Benchmarks (5 papers)

| Title | Venue | LV | Closest Relation |
|-------|-------|----|-----------------|
| ActivityNet Captions (Dense-Captioning) | ICCV 2017 | ✗ | Our Table 2 moment retrieval dataset |
| EgoSchema | NeurIPS 2023 | ✓ | Long-form VQA benchmark context |
| LoVR | 2024 | ✓ | Long video retrieval benchmark |
| MAD | CVPR 2022 | ✓ | Movie-scale moment retrieval dataset |
| MomentSeeker | 2024 | ✓ | Long video moment retrieval benchmark |
| NExT-QA | CVPR 2021 | ✗ | Video QA benchmark context |

---

## SUPPORTING — Other (1 paper)

| Title | Venue | Closest Relation |
|-------|-------|-----------------|
| CLIP (Radford et al.) | ICML 2021 | Foundation model underlying many methods |
| Uncovering Hidden Challenges | BMVC 2020 | Analysis of moment retrieval benchmark biases |

---

## PERIPHERAL (10 papers)

| Title | Venue | Closest Relation |
|-------|-------|-----------------|
| ChatVTG | 2024 | LLM-chat temporal grounding |
| Deep Video Discovery | 2024 | Agentic video search |
| DrVideo | 2024 | Document RAG for long video |
| Ego4D | CVPR 2022 | Large egocentric dataset |
| Hierarchical Long Video | 2024 | Multi-agent hierarchical understanding |
| V-JEPA 2 | 2025 | Self-supervised video backbone |
| VideoMAE V2 | CVPR 2023 | Video MAE pre-training backbone |
| Video-RAG | 2024 | Visual RAG for long video |
| Video-XL | 2024 | Extra-long VLM |
| Your Interest Your Summaries | 2024 | Query-aware summarization |

---

## DROPPED (7 papers)

| Title | Drop Reason |
|-------|------------|
| CLIP4Clip | Video-text retrieval, no task overlap |
| DreamerV3 | RL world models, completely unrelated |
| Video Summarization Survey | Too broad, no specific overlap |
| Video-ColBERT | Video QA without temporal segmentation |
| X-CLIP | Video-text retrieval only |
| Penguin-VL | VLM compression, not temporal segmentation |
| Video Re-localization | Outdated, peripheral task |
