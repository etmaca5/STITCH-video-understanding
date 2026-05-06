# Core Related Work Shortlist (20 papers)

The 20 papers most likely to need substantive treatment in STITCH's NeurIPS related work, method, or experiments sections. Each entry below includes the verbatim audit fields from `core_related_work_shortlist.json`.

---

## Cluster A — Closest prior work (must differentiate)

### 1. KTS Revisited (Afham et al., 2023, Meta AI)
- **Axis:** temporal segmentation/chunking context (also: reusable multi-task, training-free)
- **Placement:** related work prose
- **Why it's #1:** Same recipe at a high level — KTS-based unsupervised adaptive tokenization on frozen features.
- **Key difference:** STITCH (a) cosine kernel, (b) adaptive partition step, (c) GEBD + moment retrieval + VLM frame selection scope, (d) InternVideo2 backbone. KTS Revisited evaluates only on classification + TAL.
- **Omission risk:** **High.** Reviewers will read STITCH as "KTS Revisited++" if not differentiated.
- **Confidence:** high.

### 2. TimeLoc (Zhang et al., 2025)
- **Axis:** reusable multi-task framing
- **Placement:** related work prose
- **Why:** Closest published multi-task temporal localization framework. Explicitly unifies TAL + moment retrieval + VTG + **GEBD**.
- **Key difference:** Supervised + per-task training; STITCH is training-free and query-agnostic at the segmentation step.
- **Omission risk:** **High.** Anchors STITCH's multi-task framing claim.

---

## Cluster B — Training-free moment retrieval baselines (Table 2)

### 3. TAG (Lee et al.)
- **Axis:** moment retrieval comparison
- **Why:** Methodologically closest training-free MR — uses **temporal coherence clustering** to fight semantic fragmentation, the same problem STITCH chunks address.
- **Key difference:** TAG clusters per query at inference; STITCH segments query-agnostically and reuses chunks.
- **Omission risk:** **High.**

### 4. TFVTG (Zheng et al., 2024, Peking U)
- **Axis:** moment retrieval comparison
- **Why:** Canonical training-free VTG baseline.
- **Key difference:** TFVTG enumerates candidate proposals + heavy LLM/VLM scoring per query. STITCH is lightweight and reusable.
- **Omission risk:** **High.**

### 5. Moment-GPT (Xu et al., 2024)
- **Axis:** moment retrieval comparison
- **Why:** Tuning-free pipeline using LLaMA-3 + frozen MLLMs for zero-shot VMR.
- **Key difference:** Heavy MLLM at inference per query vs. STITCH's lightweight reuse.
- **Omission risk:** Medium.

### 6. ZS Moment from Frozen VLMs (Luo et al.)
- **Axis:** moment retrieval comparison
- **Why:** Foundational frozen-VLM moment retrieval baseline.
- **Key difference:** Score fixed proposals; STITCH uses adaptive chunks.
- **Omission risk:** Medium-high.

### 7. Point to Span (Jeon et al., 2024)
- **Axis:** moment retrieval comparison + long-video VLM/frame selection
- **Why:** Training-free + long-video + moment retrieval — same intersection as STITCH.
- **Key difference:** Search-then-Refine (query-driven) vs. STITCH's segment-then-score.
- **Omission risk:** **High.**

---

## Cluster C — Training-free long-video VLM frame selection (Table 3)

### 8. WFS-SB / Wavelet Frame Selection (Chen et al.)
- **Axis:** long-video VLM/frame selection comparison
- **Why:** Closest published method to STITCH App 3 — training-free, query-agnostic, semantic-boundary-driven frame selection.
- **Key difference:** Wavelet-domain analysis vs. STITCH's kernel CPD on InternVideo2 embeddings; STITCH covers more downstream tasks.
- **Omission risk:** **High.**

### 9. VideoTree (Wang/Yu/Bansal et al., 2024, UNC)
- **Axis:** long-video VLM/frame selection comparison
- **Why:** Training-free hierarchical long-video LLM reasoning.
- **Key difference:** Builds query-adaptive tree per query; STITCH builds query-agnostic chunks once per video.
- **Omission risk:** **High.**

### 10. Moment Sampling in Video LLMs (Chasmai et al., 2025, UMass + Dolby)
- **Axis:** long-video VLM/frame selection comparison
- **Why:** Training-free, query-agnostic moment-based sampling for long-form Video LLM QA.
- **Key difference:** Different sampling policy; same axis.
- **Omission risk:** **High.**

---

## Cluster D — Benchmarks (must cite)

### 11. Kinetics-GEBD (Shou et al., 2021)
- **Axis:** benchmark/task setup
- **Why:** Defines GEBD task and Table 1 benchmark.
- **Omission risk:** **Critical.**

### 12. QVHighlights + Moment-DETR (Lei, Berg, Bansal, 2021)
- **Axis:** benchmark/task setup + moment retrieval comparison
- **Why:** Defines Table 2 dataset + introduces Moment-DETR baseline.
- **Omission risk:** **Critical.**

### 13. ActivityNet Captions / Dense-Captioning Events (Krishna et al., 2017)
- **Axis:** benchmark/task setup
- **Why:** Source for the second Table 2 dataset.
- **Omission risk:** **Critical.**

### 14. Video-MME (Fu et al., 2024)
- **Axis:** benchmark/task setup
- **Why:** Table 3 benchmark.
- **Omission risk:** **Critical.**

### 15. MLVU (Zhou et al., 2024)
- **Axis:** benchmark/task setup
- **Why:** Table 3 benchmark.
- **Omission risk:** **Critical.**

---

## Cluster E — Backbone & algorithm

### 16. InternVideo2 (Wang et al., 2024)
- **Axis:** backbone/foundation context
- **Why:** STITCH's frozen feature extractor.
- **Omission risk:** **Critical.**

### 17. PELT (Killick, Fearnhead, Eckley, 2012)
- **Axis:** temporal segmentation/chunking context
- **Why:** O(n) change-point detection algorithm; STITCH's efficiency claim depends on it.
- **Omission risk:** **High.**

---

## Cluster F — GEBD competitors

### 18. DiffGEBD (Hwang et al., POSTECH)
- **Axis:** GEBD comparison
- **Why:** Strong supervised GEBD baseline; methodologically interesting (also uses temporal self-similarity).
- **Omission risk:** **High** (Table 1 baseline).

---

## Cluster G — Supervised moment retrieval reference

### 19. QD-DETR (Moon et al., 2023, CVPR)
- **Axis:** moment retrieval comparison
- **Why:** Representative supervised DETR baseline on QVHighlights.
- **Omission risk:** Medium-high.

### 20. UniVTG (Lin et al., 2023, ICCV)
- **Axis:** moment retrieval comparison + reusable multi-task framing
- **Why:** Closest supervised multi-task grounding analog.
- **Omission risk:** Medium-high.

---

## What's NOT on the shortlist (and where to put them)

### `important_but_not_emphasized` (cite, but don't dwell)
Use in tables, brief related-work mentions, or as background context:
- **Other GEBD competitors:** Efficient GEBD (Zheng et al.), Online GEBD (Jung et al.), Boundary-aware SSL (Mun et al.)
- **Other supervised MR baselines:** SnAG (Mu et al., CVPR 2024), BAM-DETR (Lee & Byun), SimBase (Bao & Kot), 2D-TAN (Zhang et al.), TALL (Gao et al.), UnLoc (Yan et al.)
- **Long-video VLM compression (trained):** LongVU (Shen et al.), STORM (Jiang et al.), AdaReTaKe (Wang et al.), LVC (Wang et al.), Chapter-Llama (Ventura et al.), MovieChat (Song et al.), AdaVideoRAG, LVLR
- **Other zero-shot long-video reasoning:** VideoINSTA (Liao et al.), ChatVTG (Qu et al.), Prompts-to-Summaries (Barbara & Maalouf)
- **Datasets / benchmarks (context only):** EgoSchema (Mangalam et al.), MAD (Soldan et al.), MomentSeeker (Yuan et al.), LoVR (Cai et al.)
- **Method context:** Vid-Morp (Bao et al.), QV-M2/FlashMMR (Cao et al.), TEMPURA (Cheng et al.), Modality-Aware Shot Relating (Tan et al.), Self-Chained / SeViLA (Yu et al.), Measure Twice Cut Once (Pang et al.), TriDet (Shi et al.)
- **Foundations / analyses:** CLIP (Radford et al.), Uncovering Hidden Challenges (Otani et al.)
- **CPD reference (uncertain attribution):** Practical and Powerful Kernel-Based CPD (Song & Chen) — verify which CPD paper STITCH actually cites

### `background_or_peripheral` (probably not in related work)
- Towards Balanced Alignment, TEMPURA (re-evaluate), SCDM, LGI (LGVI), NExT-QA, Ego4D, VideoMAE V2, V-JEPA 2, Deep Video Discovery, DrVideo, Hierarchical Long Video, Video-RAG, Video-XL, Your Interest Your Summaries, MATR (Vid2VidMR — different task), Video Re-localization (already dropped)

### Already dropped (from v1)
CLIP4Clip, DreamerV3, Video Summarization Survey, Video-ColBERT, X-CLIP, Penguin-VL, Video Re-localization.
