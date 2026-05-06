# Changes from v1 → v2

Summary of corrections and reclassifications introduced by the audit pass.

---

## 1. Headline Counts

| What | Count |
|------|------:|
| v1 retained entries | 68 |
| v2 retained entries | 62 |
| Entries with corrected metadata (title, authors, paper identification) | ~14 |
| Entries with changed `relevance` (tier moves) | 14 |
| Entries with changed `category` | 0 |
| Entries removed entirely (PDF not local) | 1 (LongVideoBench) |
| Entries with re-identified content (different paper than v1 thought) | 3 (TEMPURA, TAG, Modality-Aware Shot Relating) |
| Entries flagged with `evidence_basis` justification in v2 | 14 |
| Entries newly carrying explicit "year/venue unverified" marker | ~22 |

---

## 2. Relevance Tier Distribution

| Tier | v1 | v2 | Δ |
|------|----|----|---|
| core | 15 | 19 | +4 |
| supporting | 40 | 30 | −10 |
| peripheral | 10 | 13 | +3 |
| dropped | 7 | 7 | 0 |
| removed | 0 | 1 | +1 |

Net result: more papers concentrated in core (the closest-to-STITCH set sharpened) and peripheral (clearer cull of pre-DETR / off-task work).

---

## 3. Tier Moves

### Promoted → core (6)
- **TAG** (Lee et al.) — was supporting; v1 mis-described it as audio-visual. Actual paper does temporal coherence clustering, methodologically closest training-free MR competitor.
- **Wavelet Frame Selection / WFS-SB** (Chen et al.) — was supporting. Confirmed as training-free, query-agnostic, semantic-boundary-driven LVLM frame selection — closest method to App 3.
- **VideoTree** (Wang/Yu/Bansal et al.) — was supporting. Confirmed training-free hierarchical long-video LLM reasoning.
- **Moment Sampling in Video LLMs** (Chasmai et al.) — was supporting. Confirmed training-free, query-agnostic moment sampling for long-form Video LLM QA.
- **Point to Span** (Jeon et al.) — was supporting. Confirmed zero-shot long-video moment retrieval, sits at the same intersection STITCH targets.
- **TimeLoc** (Zhang et al.) — was supporting. Confirmed multi-task framework spanning TAL + temporal video grounding + moment retrieval + GEBD — closest published multi-task analog (supervised) to STITCH.

### Promoted → core (dataset)
- **ActivityNet Captions / Dense-Captioning Events** — was supporting; promoted to core because it is the source paper for one of the Table 2 benchmarks.

### Promoted → supporting (1)
- **ChatVTG** (Qu et al.) — was peripheral. Re-reading confirms it's a training-free zero-shot VTG method with multi-granularity LLM-generated segments, more relevant than peripheral.

### Demoted → supporting (1)
- **CPD reference paper** (was core) — v1 said this was Truong et al.'s ruptures library paper; the actual local PDF is Song & Chen's kerSeg paper, a different work. Demoted until STITCH's actual citation can be verified.

### Demoted → peripheral (4)
- **Aligning Moments in Time using Video Queries (MATR)** — was supporting. Actually does Vid2VidMR (video-to-video moment retrieval), not text-to-video — different task than STITCH Table 2.
- **SCDM** — was supporting. Pre-DETR-era TSG, superseded.
- **LGI / LGVI** — was supporting. Pre-DETR-era TSG, superseded.
- **NExT-QA** — was supporting. Not used in STITCH evaluation; less load-bearing than EgoSchema.

### Removed entirely
- **LongVideoBench** — PDF not present in `lit_review_paper_pdfs/`. Still a real STITCH benchmark; if added locally, re-include.

---

## 4. Most Substantially Corrected Entries

### Wrong paper identification (paper title/abstract differed from v1's description)
1. **TEMPURA** — v1 said "Object-level Semantic Context for TSG"; actual paper is "Temporal Event Masked Prediction and Understanding for Reasoning in Action" (Cheng et al., UW, arXiv 2505.01583, 2025).
2. **TAG** — v1 said "Boosting Text-Auditory-Visual Grounding with LLMs"; actual paper is "Simple Yet Effective Temporal-Aware Approach for Zero-Shot VTG" (Lee et al., Sungkyunkwan U).
3. **Modality-Aware Shot Relating (MASRC)** — v1 said "for Video Question Answering"; actual paper is "for Video Scene Detection" (Tan et al.) — the task is closer to GEBD than to VQA.

### Wrong attribution
4. **CPD reference** — v1 cited the local file as "Truong et al., Selective Review of Offline CPD Methods (ruptures)"; actual file is Song & Chen's "Practical and Powerful Kernel-Based CPD" (kerSeg), a different work.

### Wrong task
5. **Aligning Moments in Time using Video Queries (MATR)** — v1 listed as text-query moment retrieval baseline; actual task is Vid2VidMR (video-to-video).

### Title corrections (minor)
6. **MLVU** — v1 had "A Comprehensive Benchmark for Multi-Task Long Video Understanding"; actual title is "Benchmarking Multi-task Long Video Understanding".
7. **Kinetics-GEBD benchmark** — v1 had "A Benchmark Challenge for Event Segmentation"; actual is "A Benchmark for Event Segmentation".

### Confirmed (caveat resolved)
8. **STORM** — v1 flagged a possible RL-paper confusion (filename contained "Stochastic Transformer World Models for Reinforcement Learning"). Verified from page 1 that this is the video MLLM paper (Jiang et al., NVIDIA). v1 caveat now resolved.

### Author/venue verifications (~30 entries)
For ~30 entries previously listed as "Unknown" authors or guessed venues, full author lists and institutions are now confirmed against PDF first pages. See `metadata_audit.md` §2 for the table.

---

## 5. Evidence-Basis Annotations Added

These v2 entries now carry an explicit `evidence_basis` field justifying the most ambiguous classifications:

- `internvideo2`, `pelt` — clarify why `training_free`/`supervised` flags are set as they are.
- `kts_revisited`, `tfvtg`, `moment_gpt`, `zs_frozen_vlm`, `adaretake` — `training_free=true` justified from explicit abstract claims.
- `wfs_sb`, `tag`, `videotree`, `point_to_span`, `moment_sampling_vlm`, `timeloc` — promotion justifications.
- `boundary_ssl`, `vid_morp` — `unsupervised=true` justified.
- `snag` — `reusable_compute_style` choice justified (late fusion → reusable_repr_plus_scoring not rerun_query_specific).
- `on_gebd` — `long_video_framing=true` justified (streaming target).
- `cpd_song_chen` — demotion justified (attribution uncertain).
- `moment_alignment_matr` — demotion justified (Vid2VidMR ≠ STITCH task).
- `shot_relating_masrc`, `tempura`, `adavideorag`, `mlvu`, `storm` — task corrections noted.

---

## 6. Misleading Prior Entries (high priority to inspect)

Entries from v1 that were materially misleading and would have caused citation errors if used as-is:

| Entry | What was misleading | Risk |
|-------|---------------------|------|
| `cpd_song_chen` (was: ruptures CPD) | Wrong paper attributed; STITCH likely cites a different work | High — wrong methodological citation |
| `tempura` | Wrong paper described entirely | High — would have led to citing wrong work |
| `tag` | Method described as audio-visual when it's purely visual | Medium — would have framed competitor incorrectly |
| `shot_relating_masrc` | Task described as VQA when it's scene detection | Medium |
| `moment_alignment_matr` | Task described as text-query MR when it's video-query MR | Medium |
| `longvideobench` | Listed as core but PDF not local | Low (no PDF to cite anyway) |

---

## 7. What's Still Uncertain in v2

These items remain explicitly unverified and should be inspected before final paper writing:
- Year/venue for ~22 entries (mostly preprints) — flagged in metadata_audit.md §3.
- The CPD citation question — needs to be resolved against the STITCH draft.
- AdaVideoRAG, Deep Video Discovery, DrVideo, Hierarchical Long Video, Video-RAG, Video-XL, Your Interest Your Summaries — authors/venues not verified from PDFs in this audit pass; all currently peripheral so low priority.

---

## 8. New Files Added in This Pass

- `papers_master_v2.json` (corrected master)
- `papers_master_v2.md`
- `papers_master_v2.html` (with v1→v2 changes tab)
- `metadata_audit.md`
- `classification_audit.md`
- `core_related_work_shortlist.json`
- `core_related_work_shortlist.md`
- `core_related_work_shortlist.html`
- `changes_from_v1.md` (this file)

v1 outputs are retained alongside (untouched) for diff reference.
