# Classification Audit (v1 → v2)

Reclassification of relevance, category, and structured-flag fields after re-reading abstracts. Strictness deliberately increased from v1.

---

## 1. Relevance Changes

### Promotions (supporting → core)
| Entry | Reason |
|-------|--------|
| `tag` | After reading PDF: it's training-free zero-shot VTG with **temporal coherence clustering**, the methodologically closest training-free moment retrieval method to STITCH. Not the audio-visual paper v1 thought it was. |
| `wfs_sb` (Wavelet Frame Selection) | Training-free, query-agnostic, semantic-boundary-based long-video frame selection. The closest published method to STITCH App 3 along all axes. |
| `videotree` | Confirmed training-free, hierarchical, long-video VLM frame selection. Closest training-free competitor for the App 3 axis. |
| `moment_sampling_vlm` | Confirmed training-free, query-agnostic frame selection for long-video Video LLMs. App 3 direct competitor. |
| `point_to_span` | Confirmed zero-shot moment retrieval explicitly designed for hour-long videos — the union of training-free + long-video + moment retrieval. |
| `timeloc` | Confirmed: unifies temporal action localization, moment retrieval, temporal video grounding, and **GEBD** in a single framework — closest published multi-task analog (supervised) to STITCH's general framing. |

### Demotions (core → supporting)
| Entry | Reason |
|-------|--------|
| `cpd_song_chen` (was: ruptures CPD) | The local PDF is Song & Chen's kerSeg paper, not the Truong et al. ruptures paper that STITCH likely cites. Demoted until the actual STITCH citation is verified. |

### Demotions (supporting → peripheral)
| Entry | Reason |
|-------|--------|
| `scdm` | Pre-DETR-era TSG method; superseded by newer baselines (QD-DETR, BAM-DETR, UniVTG). |
| `lgvi` | Pre-DETR-era TSG; same logic as SCDM. |
| `nextqa` | Standard video QA benchmark not used in STITCH evaluation; less load-bearing than EgoSchema. |

### Demotions (supporting → peripheral)
| Entry | Reason |
|-------|--------|
| `moment_alignment_matr` | Task is video-to-video moment retrieval (Vid2VidMR), not text-to-video moment retrieval — outside STITCH's Table 2 scope. |

### Promotions (peripheral → supporting)
| Entry | Reason |
|-------|--------|
| `chatvtg` | Re-reading the abstract confirms this is a training-free zero-shot VTG pipeline with multi-granularity LLM-generated segments — relevant context, not peripheral. |

### Promotions (supporting → core, dataset)
| Entry | Reason |
|-------|--------|
| `activitynet_captions` | Source paper for ActivityNet Captions, which is a STITCH Table 2 evaluation dataset — should be core. |

### Removals
| Entry | Reason |
|-------|--------|
| `longvideobench` | PDF not present in `lit_review_paper_pdfs/`. Removed from master entirely; can be re-added if the PDF appears. |

---

## 2. Structured-Flag Audit

### 2.1 Dataset/benchmark papers — null-out method-style fields
For dataset papers, fields that don't meaningfully apply have been set to `null` in v2:
- `training_free`, `zero_shot`, `unsupervised`, `supervised`, `pretrained_backbone`, `general_multi_task_framing`, `reusable_compute_style` — all nulled for `qvhighlights`, `kinetics_gebd`, `mlvu`, `videomme`, `activitynet_captions`, `egoschema`, `nextqa`, `mad`, `momentseeker`, `lovr`, `ego4d`.
- The benchmarks themselves don't have a "training-free" property; only methods do.

For `uncovering_hidden` (analysis paper, category=other) — same null-out applied.

### 2.2 Long-video flag — tightened
v1 sometimes set `long_video_framing=true` when the paper merely mentioned long videos. v2 requires the paper to explicitly target hour-scale or otherwise emphasize long-video setting:
- `kts_revisited`: long_video_framing=**true** (explicit, "real-world videos are often several minutes long")
- `point_to_span`: long_video_framing=**true** (explicit "hour-long")
- `videotree`: long_video_framing=**true** (explicit "long videos")
- `moment_sampling_vlm`: **true** (explicit long-form video QA)
- `wfs_sb`: **true** (explicit "long video understanding")
- `chapter_llama`: **true** (explicit "hour-long videos")
- `longvu`, `storm`, `lvc`, `lvlr`, `videoinsta`, `adavideorag`, `video_rag`, `video_xl`, `adaretake`: all explicit long-video focus → **true**
- `mad`: **true** (movie-scale grounding, hour-long videos)
- `momentseeker`, `lovr`, `egoschema`, `ego4d`: **true** (long video benchmarks)
- `videomme`, `mlvu`: **true** (cover long video)
- Conversely, `nextqa`, `qd_detr`, `bam_detr`, `simbase`, `tag`, `tfvtg`, `qvhighlights`, etc. → **false** (operate on short clips by default)

### 2.3 General multi-task framing
Reserved for methods/benchmarks that actively span multiple temporal localization tasks:
- `kts_revisited`, `timeloc`, `unloc`, `univtg`, `internvideo2`, `clip`, `self_chained_sevila`, `mlvu`, `videomme` → **true**
- All others where it doesn't apply → **false** or null for dataset rows

### 2.4 query_awareness — tightened
- `query-dependent`: query is required at all stages (most VTG/VMR methods)
- `query-potential`: query optional (UnLoc, LongVU which can run query-free)
- `no-query`: produces output without any query (GEBD methods, KTS Revisited, WFS-SB, all backbones, scene segmentation methods)

Specific corrections: `wfs_sb` set to no-query (it's query-agnostic semantic boundary detection, not query-aware), `tag` set to query-dependent (it scores against the query for the final localization).

### 2.5 reusable_compute_style — disambiguated
Three buckets:
- `compute_once_reuse`: per-video computation can be reused for any task or query (KTS Revisited, InternVideo2 features, PELT, WFS-SB segmentation, AdaReTaKe, Moment Sampling VLM, LongVU, STORM, Chapter-Llama compute itself).
- `reusable_repr_plus_scoring`: per-video representation reusable, but per-query scoring still required (TFVTG, Moment-GPT, ZS-Frozen-VLM, SnAG late fusion, Vid-Morp, Point-to-Span, MovieChat memory, AdaVideoRAG, LVLR, Self-Chained, Boundary-aware SSL features, Prompts-to-Summaries).
- `rerun_query_specific`: every query forces full recomputation (QD-DETR, BAM-DETR, SimBase, UnLoc, UniVTG, 2D-TAN, TALL, SCDM, LGI, GEBD methods, TimeLoc, ChatVTG, Towards-Balanced, TEMPURA, MASRC, TriDet, Measure Twice, ChatVTG, Hierarchical Long, your_interest).

VideoTree was reclassified as `rerun_query_specific` because it builds a query-adaptive tree per query.

### 2.6 supervised vs unsupervised vs training_free
Cleaner separation in v2:
- `unsupervised=true` reserved for methods with no labeled supervision at any task-specific stage (KTS Revisited, WFS-SB, PELT, CPD, Boundary-aware SSL, VideoMAE V2, V-JEPA 2, Vid-Morp).
- `training_free=true` reserved for inference-only methods (TFVTG, KTS Revisited, WFS-SB, VideoTree, Moment Sampling VLM, AdaReTaKe, Prompts-to-Summaries, Moment-GPT, ZS-Frozen-VLM, Point-to-Span, VideoINSTA, ChatVTG, TAG (technically training-free since uses pretrained models without further training)).
- Most supervised methods retain `supervised=true` and the other two `false`.

---

## 3. Category Changes

| Entry | v1 category | v2 category | Reason |
|-------|-------------|-------------|--------|
| `tempura` | method (correctly) | method | But task changed substantially after re-reading. |
| `cpd_song_chen` | other (correctly) | other | Demoted relevance, kept category. |
| All other categories | unchanged | unchanged | |

No category changes were made — v1 categories were structurally correct even when relevance/metadata were wrong.

---

## 4. Net Effect on Relevance Distribution

| Tier | v1 count | v2 count |
|------|---------|---------|
| core | 15 | 19 |
| supporting | 40 | 30 |
| peripheral | 10 | 13 |
| dropped (kept dropped) | 7 | 7 |
| removed entirely | 0 | 1 (LongVideoBench) |
| **total** | 75 | 69 |

Net 14 papers moved between tiers; 6 dropped or removed. v2 core grew slightly because four highly-relevant training-free baselines (TAG, WFS-SB, VideoTree, Moment Sampling VLM) plus TimeLoc and Point-to-Span were promoted from supporting after re-reading their abstracts.
