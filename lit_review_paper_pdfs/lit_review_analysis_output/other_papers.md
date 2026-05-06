# Other Papers (Non-Method, Non-Dataset)

**7 papers** in the "other" category: foundation models, algorithms, surveys, and analysis papers.

---

## Core

| Title | Authors | Venue | Relevance | Notes |
|-------|---------|-------|-----------|-------|
| **InternVideo2** | Wang et al. | ECCV 2024 | core | STITCH's frozen backbone. Multi-modal video foundation model. |
| **CPD (ruptures)** | Truong et al. | Signal Proc. 2020 | core | STITCH's kernel change-point detection algorithm library. |
| **PELT** | Killick et al. | JASA 2012 | core | O(n) efficient change-point search algorithm underpinning ruptures. |

---

## Supporting

| Title | Authors | Venue | Relevance | Notes |
|-------|---------|-------|-----------|-------|
| **CLIP** | Radford et al. | ICML 2021 | supporting | Foundation model underpinning many compared methods; InternVideo2 uses CLIP-style objectives. |
| **Uncovering Hidden Challenges** | Otani et al. | BMVC 2020 | supporting | Analysis of annotation biases in moment retrieval benchmarks. Relevant when interpreting Table 2 numbers. |

---

## Peripheral

| Title | Authors | Venue | Relevance | Notes |
|-------|---------|-------|-----------|-------|
| **VideoMAE V2** | Wang et al. | CVPR 2023 | peripheral | Masked autoencoder video backbone. Alternative to InternVideo2; not used in STITCH. |
| **V-JEPA 2** | Assran et al. | 2025 | peripheral | JEPA-based self-supervised video model. Alternative backbone; not used in STITCH. |

---

## Notes

- The "other" category covers papers that are neither a new method for temporal understanding/retrieval, nor a benchmark dataset. These include foundation models (InternVideo2, CLIP, VideoMAE V2, V-JEPA 2), core mathematical/statistical algorithms (CPD, PELT), and analysis papers.
- InternVideo2 and CLIP are also relevant as backbones for many compared methods, but their primary contribution is representation learning rather than temporal segmentation.
