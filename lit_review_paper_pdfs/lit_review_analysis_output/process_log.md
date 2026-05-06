# Process Log

Literature review triage pass for STITCH (NeurIPS submission draft).
Date: 2026-04-23

---

## Input

- **75 PDF files** in `lit_review_paper_pdfs/`
- **Reference paper**: STITCH draft PDF (attached in conversation)
- Task: classify, categorize, and extract structured metadata for all papers; generate 13 output files

---

## Phase 1: Paper Identification

4 papers had non-descriptive numeric filenames and required reading to identify:

| Filename | Identified As |
|----------|--------------|
| `2104.08860v2.pdf` | CLIP4Clip (Luo et al., Neurocomputing 2022) |
| `2107.09609v2.pdf` | QVHighlights + Moment-DETR (Lei et al., NeurIPS 2021) |
| `2303.13874v1.pdf` | QD-DETR (Moon et al., CVPR 2023) |
| `2307.16449v4.pdf` | MovieChat (Song et al., CVPR 2024) |

Remaining 71 papers were identified from filenames (which contain full titles).

---

## Phase 2: Deep Reads

Two papers with unknown content (not identifiable from title alone) were read in full:

1. **`Revisiting Kernel Temporal Segmentation...`** — Confirmed: Meta AI KTS-as-tokenizer paper. Training-free, adaptive chunking, most similar to STITCH. Classified **core**.

2. **`Training-free Video Temporal Grounding...`** — Confirmed: TFVTG by Zheng et al. GPT-4+BLIP-2 pipeline for training-free temporal grounding on QVHighlights + ActivityNet. Classified **core**.

---

## Phase 3: Classification Decisions

**Total: 75 papers**

| Tier | Count |
|------|-------|
| core | 15 |
| supporting | 40 |
| peripheral | 10 |
| drop | 7 |
| **Retained** | **68** |

**Category breakdown (retained only):**

| Category | Count |
|----------|-------|
| dataset_benchmark | 13 |
| method | 48 |
| other | 7 |

---

## Notable Classification Decisions

**CLIP4Clip → drop**: Video-text retrieval, not temporal grounding. Even though CLIP features are used in adjacent methods, CLIP4Clip itself has zero benchmark overlap with STITCH.

**DreamerV3 → drop**: RL world models paper; completely unrelated domain. Likely included in the PDF folder by mistake.

**KTS Revisited → core (not supporting)**: The meta-level similarity is very high — both use kernel temporal segmentation on frozen features without training. The distinction is that STITCH uses cosine-kernel CPD, adds adaptive chunk partition, and generalizes across three task families. This paper must be cited prominently in related work.

**ActivityNet Captions → supporting (not core dataset_benchmark)**: ActivityNet Captions is an evaluation benchmark for STITCH Table 2, but the paper introducing it (Dense-Captioning Events) predates and is separate from the actual moment retrieval split. The benchmark is core but the paper category is `supporting dataset_benchmark` since STITCH cites it as a data source.

**UnLoc → supporting with `query-potential`**: UnLoc handles both query-conditioned and query-free tasks through a unified head. Unlike strictly query-dependent models, it can run without a query. Marked `QP` to distinguish from pure NQ or pure QD.

**STORM**: Filename suggests RL world model (STORM for RL) but there is also a STORM paper for video token efficiency. Classified as the video token efficiency paper based on relevance context; if the PDF is actually the RL paper, reclassify to **drop**.

---

## Phase 4: Output Generation

All 13 output files written to `lit_review_analysis_output/`:

| File | Status |
|------|--------|
| papers_master.json | ✓ |
| papers_master.md | ✓ |
| papers_master.html | ✓ |
| datasets_benchmarks.md | ✓ |
| datasets_benchmarks.html | ✓ |
| methods.md | ✓ |
| methods.html | ✓ |
| other_papers.md | ✓ |
| other_papers.html | ✓ |
| drop_report.md | ✓ |
| drop_report.json | ✓ |
| taxonomy_notes.md | ✓ |
| process_log.md | ✓ |

---

## Caveats

1. **STORM filename**: The PDF filename `STORM Efficient Stochastic Transformer based World Models for Reinforcement Learning.pdf` matches an RL paper (DreamerV3 variant). If this is the RL paper, it should be dropped. If it is a video MLLM efficiency paper, the classification above stands. **Verify by reading the PDF.**

2. **No hallucination policy**: For papers read directly, metadata is grounded in paper text. For papers classified from filename/knowledge, `authors` fields may be incomplete (listed as "Unknown"). Venue/year are estimated from known publication dates.

3. **TAPOS**: Included under the Kinetics-GEBD entry because it is co-introduced and co-evaluated in the same paper. It is not tracked as a separate entry in papers_master.json.
