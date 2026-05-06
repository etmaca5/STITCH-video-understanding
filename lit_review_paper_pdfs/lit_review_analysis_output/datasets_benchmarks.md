# Datasets & Benchmarks

**16 total** (6 core, 6 supporting, 1 peripheral)

---

## Core Benchmarks (directly evaluated in STITCH)

| Title | Venue | Task | Long Video | Notes |
|-------|-------|------|-----------|-------|
| **Kinetics-GEBD** (Shou et al.) | ICCV 2021 | Generic Event Boundary Detection | ✗ | Table 1 — Rel.Dis. and F1@0.05 |
| **QVHighlights** (Lei et al.) | NeurIPS 2021 | Moment retrieval + highlight detection | ✗ | Table 2 — R1@0.5, R1@0.7, mAP |
| **ActivityNet Captions** (Krishna et al.) | ICCV 2017 | Moment retrieval / dense captioning | ✗ | Table 2 — R1@0.5, R1@0.7 |
| **Video-MME** (Fu et al.) | CVPR 2024 | Multi-modal LLM QA (short/med/long) | ✓ | Table 3 — overall + long-video acc |
| **MLVU** (Zhou et al.) | NeurIPS 2024 | Multi-task long video understanding | ✓ | Table 3 — multiple subtask acc |
| **LongVideoBench** (Wu et al.) | NeurIPS 2024 | Long-context video-language QA | ✓ | Table 3 — validation split |

---

## Supporting Benchmarks (context / comparison)

| Title | Venue | Task | Long Video | Notes |
|-------|-------|------|-----------|-------|
| EgoSchema | NeurIPS 2023 | Egocentric long-form video QA | ✓ | 3-min egocentric clips, multi-choice QA |
| NExT-QA | CVPR 2021 | Causal/temporal video QA | ✗ | Temporal action reasoning |
| LoVR | 2024 | Long video text-to-video retrieval | ✓ | Adjacent to STITCH moment retrieval |
| MAD | CVPR 2022 | Movie-scale language grounding | ✓ | Feature-length movie temporal grounding |
| MomentSeeker | 2024 | Long video moment retrieval | ✓ | Comprehensive long-video moment dataset |
| TAPOS | ICCV 2021 | Action phase segmentation | ✗ | Part of the Kinetics-GEBD benchmark family |

---

## Peripheral

| Title | Venue | Task | Notes |
|-------|-------|------|-------|
| Ego4D | CVPR 2022 | Egocentric multi-task (NLQ, MQ, etc.) | Large-scale; STITCH not evaluated here |

---

## Notes

- **TAPOS** is referenced as part of the GEBD benchmark suite (Kinetics-GEBD paper).
- **ActivityNet Captions** provides the moment retrieval split; the original paper (dense captioning) is the correct citation.
- For Video-MME and MLVU, STITCH is applied as a frame selector for a downstream VLM (e.g., InternVL2); the benchmark itself is query-dependent but STITCH's frame selection is query-agnostic.
