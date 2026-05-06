# Taxonomy Notes

Notes on how the classification schema applies to this literature and the STITCH framing.

---

## Training-free vs. Zero-shot vs. Unsupervised

These three properties are distinct in this review:

- **training_free**: No gradient-based optimization at any stage for the video task. The model runs purely at inference using frozen pretrained weights. STITCH is training-free: it uses frozen InternVideo2 + a parameter-free CPD algorithm.
  - Edge case: TFVTG uses GPT-4 prompting (training-free) but GPT-4 itself was trained; we count the *method* as training-free.
  
- **zero_shot**: Evaluated on data distributions not seen during training, with no in-domain examples. STITCH is zero-shot in the sense that it has no exposure to Kinetics-GEBD, QVHighlights, etc. during any stage.
  - Distinction from training-free: A zero-shot model could still use a fine-tuned backbone (e.g., UniVTG is not zero-shot because it fine-tunes on grounding data).

- **unsupervised**: No labeled annotation used at any stage for the video task. STITCH is unsupervised: kernel CPD requires no annotated event boundaries.
  - InternVideo2 uses supervised pretraining, but the *application* in STITCH (CPD segmentation) is unsupervised.
  - Vid-Morp pretrains without labeled moments → marked unsupervised despite having a training stage.

---

## Query Awareness

- **query-dependent**: The method uses the natural language query at inference to produce temporally localized output. Examples: all moment retrieval methods, TFVTG, QD-DETR.
- **query-potential**: The method was designed for query-agnostic tasks but has a query-conditioning variant (e.g., UnLoc can handle both).
- **no-query**: The method produces temporal structure purely from visual content, without any natural language query. STITCH's segmentation step is no-query; it becomes query-potential when the output chunks are scored against a query for moment retrieval.

**Important STITCH distinction**: STITCH's CPD step is `no-query`, but when applied to moment retrieval (Table 2), the chunk scores use query matching. This is why we categorize STITCH as `query-potential` overall — compute-once, score-later.

---

## Temporal Unit

- **frame**: Method operates at the individual frame level (e.g., GEBD outputs a frame index).
- **window/clip**: Fixed-size sliding window or clip. Most LLM-based long video methods use this.
- **proposal/segment**: Variable-length segment proposals (e.g., DETR outputs span [start, end]).
- **adaptive chunk**: Data-driven variable-length temporal chunk (e.g., KTS, STITCH, AdaReTaKe).
- **hierarchical/multi-scale**: Multiple temporal granularities simultaneously (e.g., VideoTree, When One Moment Isn't Enough).

STITCH produces **adaptive chunks** as its primary temporal unit: variable-length, semantically coherent segments determined by kernel CPD.

---

## Reusable Compute Style

- **compute_once_reuse**: Video features / temporal structure computed once; reused for any downstream query or task without re-processing. STITCH, InternVideo2 embeddings, CLIP features.
- **reusable_repr_plus_scoring**: Representations computed once, but query-specific scoring happens per-query (e.g., retrieve chunk embeddings, then score against each query).
- **rerun_query_specific**: The entire forward pass must be rerun for each new query (e.g., QD-DETR, which conditions all encoder layers on the query).

STITCH's key efficiency claim is `compute_once_reuse` at the segmentation step. The moment retrieval application adds a `reusable_repr_plus_scoring` layer but the heavy compute (InternVideo2 + CPD) is reused.

---

## Long Video Framing

Papers are marked `long_video_framing = true` if they explicitly target videos longer than ~3 minutes or explicitly discuss scaling to hour-length videos. This includes:
- Video-MME (has long split), MLVU, LongVideoBench (core benchmarks)
- Chapter-Llama, MovieChat, VideoTree, LongVU (methods)
- MAD, MomentSeeker, LoVR (datasets)

STITCH's long video framing comes from its adaptive chunking enabling efficient processing of arbitrarily long videos, as demonstrated in the VLM reasoning application (Table 3).

---

## Category: method vs. dataset_benchmark vs. other

- **dataset_benchmark**: Primary contribution is annotated data + evaluation protocol. A paper like QVHighlights that also introduces Moment-DETR as a baseline is categorized `dataset_benchmark` because the dataset is the primary artifact we cite.
- **method**: Primary contribution is a new algorithm or model for temporal understanding.
- **other**: Foundation models, statistical algorithms, surveys, analysis papers.

---

## Relevance Tiers

- **core**: Paper is cited in STITCH body text (not just related work), contributes a benchmark/baseline/algorithm directly used or directly compared.
- **supporting**: Paper is in related work because it contextualizes STITCH's claims; a reviewer would expect it cited.
- **peripheral**: Paper is tangentially related; could be cited for completeness but absence wouldn't weaken the paper.
- **drop**: No connection to STITCH's tasks, benchmarks, or methods.

---

## Key Clusters for Related Work Sections

**Cluster 1: GEBD baselines** — GEBD Diffusion, Efficient GEBD, Online GEBD (all supervised; STITCH compares training-free)

**Cluster 2: Moment retrieval baselines** — Moment-DETR, QD-DETR, UniVTG, SnAG, BAM-DETR (supervised); TFVTG, ZS-Frozen-VLM, ZS-MLLM (training-free/zero-shot)

**Cluster 3: Training-free / adaptive temporal segmentation** — KTS Revisited, Wavelet Frame Selection, AdaReTaKe (all use boundary-based or adaptive temporal structure without labels)

**Cluster 4: VLM frame selection** — Moment Sampling VLM, VideoTree, LongVU, AdaReTaKe, STORM (all deal with selecting frames for downstream VLM; STITCH App 3)

**Cluster 5: Long video understanding** — MovieChat, LongVU, VideoINSTA, Chapter-Llama, AdaVideoRAG, LVC (long video context broadly)

**Cluster 6: Foundational algorithms** — PELT, CPD (ruptures) — the math behind STITCH's segmentation

**Cluster 7: Benchmarks** — QVHighlights, Kinetics-GEBD, ActivityNet Captions, Video-MME, MLVU, LongVideoBench — all directly evaluated in STITCH
