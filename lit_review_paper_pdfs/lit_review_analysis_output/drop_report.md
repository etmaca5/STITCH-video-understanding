# Drop Report

**7 papers dropped** from active literature review. Dropped papers are not cited in STITCH and have no benchmark overlap or methodological connection to the paper's contributions.

---

## Dropped Papers

| Filename | Title | Reason |
|----------|-------|--------|
| 2104.08860v2.pdf | CLIP4Clip | Video-text clip retrieval (text→video), not temporal grounding or GEBD. No task overlap. |
| Mastering Diverse Domains through World Models.pdf | DreamerV3 | Reinforcement learning world model. Completely unrelated domain. |
| Video Summarization Techniques A Comprehensive Review.pdf | Video Summarization Survey | Overly broad survey; no specific method or benchmark overlap. |
| Video-ColBERT Visual Collaborative BERT for Video Question Answering.pdf | Video-ColBERT | Video QA without temporal segmentation focus; no STITCH benchmark overlap. |
| X-CLIP End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval.pdf | X-CLIP | Video-text retrieval only; no temporal grounding or GEBD tasks. |
| Penguin-VL Efficient Vision-Language Model for Extreme Compression.pdf | Penguin-VL | VLM architecture compression; not temporal segmentation or STITCH tasks. |
| Video Re-localization.pdf | Video Re-localization | Outdated (2018); peripheral task; no benchmark or method connection to STITCH. |

---

## Drop Criteria Used

Papers were dropped if ALL of the following applied:
1. **No benchmark overlap**: Not evaluated on GEBD, moment retrieval (QVHighlights, ActivityNet), or VLM reasoning (VideoMME, MLVU, LongVideoBench) tasks
2. **No method overlap**: No connection to temporal segmentation, change-point detection, training-free approaches, or frame selection
3. **Not a baseline or comparison**: Not cited in STITCH paper as a competitor, ablation, or background
4. **Not a cited foundation**: Not used as a building block (backbone, dataset, algorithm) in STITCH

Papers with any plausible connection were retained as peripheral rather than dropped.
