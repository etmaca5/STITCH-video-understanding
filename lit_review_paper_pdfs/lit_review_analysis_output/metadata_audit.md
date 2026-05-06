# Metadata Audit (v1 → v2)

Verification pass over every previously core/supporting entry against the actual local PDF first page (authors, title, abstract). All corrections grounded in extracted PDF text.

---

## 1. Most Substantial Corrections

These v1 entries had wrong or hallucinated metadata that materially affected the classification.

### 1.1 CPD reference paper — wrong author attribution
- **Filename:** `Practical and powerful kernel-based change-point detection.pdf`
- **v1 said:** Truong et al., "Selective Review of Offline Change Point Detection Methods" — i.e., the ruptures library reference paper (Signal Processing 2020).
- **Actual PDF is:** Hoseung Song & Hao Chen, "Practical and Powerful Kernel-Based Change-Point Detection" (high-dim kernel CPD via permutation null; implemented in R package `kerSeg`). Different paper, different authors, different framework.
- **Implication:** STITCH likely cites the ruptures library (Truong et al., 2020), but only the Song-Chen paper is in the local folder. If STITCH actually cites ruptures, the right paper is missing locally; if STITCH actually cites Song-Chen / kerSeg, then v1's claim "we use the ruptures Python library" needs to be re-checked.
- **Action in v2:** Renamed to `cpd_song_chen`, demoted from core to supporting, attached `evidence_basis` flag noting the attribution issue.

### 1.2 TEMPURA — completely different paper than v1 described
- **Filename:** `TEMPURA_ Temporal Event Masked Prediction and Understanding for Reasoning in Action.pdf`
- **v1 said:** "Object-level Semantic Context for Temporal Sentence Grounding" — wrong paper entirely.
- **Actual PDF:** Cheng et al. (UW + CMU + NYCU + Microsoft), 2025, arXiv 2505.01583. Trains a model on the VER dataset (500K untrimmed videos, dense event descriptions) for masked event prediction and step-by-step action reasoning.
- **Action in v2:** Title, authors, year, venue, abstract-summary all rewritten. Kept as supporting (relevant as supervised event-segmentation reasoning model adjacent to GEBD framing).

### 1.3 TAG — wrong description; method is highly relevant
- **Filename:** `TAG_ A Simple Yet Effective Temporal-Aware Approach for Zero-Shot Video Temporal Grounding.pdf`
- **v1 said:** "Boosting Text-Auditory-Visual Grounding with LLMs" — multi-modal audio-visual grounding.
- **Actual PDF:** Lee et al. (Sungkyunkwan U). Visual zero-shot VTG using **temporal pooling + temporal coherence clustering + similarity adjustment**. No audio modality. Explicitly calls out "semantic fragmentation" — the same problem STITCH's chunk-based formulation addresses.
- **Action in v2:** Promoted from supporting to **core** (this is likely the methodologically closest training-free moment retrieval paper to STITCH).

### 1.4 Modality-Aware Shot Relating — wrong task
- **Filename:** `Modality-Aware Shot Relating and Comparing for Video Scene Detection.pdf`
- **v1 said:** "Shot Relating and Reasoning for Video QA"
- **Actual PDF:** Tan et al. (Chongqing U + XJTLU). Video Scene Detection (boundary-detection task adjacent to GEBD), not video QA.
- **Action in v2:** Title, authors, task, and evidence corrected. Kept as supporting.

### 1.5 Aligning Moments in Time using Video Queries — wrong task setup
- **Filename:** `Aligning Moments in Time using Video Queries.pdf`
- **v1 said:** "Moment retrieval baseline" with text-query retrieval like 2D-TAN.
- **Actual PDF:** Kumar et al. (IIT Jodhpur + Microsoft), MATR. The task is **video-to-video moment retrieval (Vid2VidMR)** — uses video queries to localize moments, NOT text queries. Different task setup from STITCH's Table 2.
- **Action in v2:** Demoted from supporting to **peripheral**. Renamed id to `moment_alignment_matr`.

### 1.6 LongVideoBench — included but PDF not present locally
- **v1 had:** LongVideoBench listed as core dataset.
- **Verification:** Not in `lit_review_paper_pdfs/`.
- **Action in v2:** Removed entirely. (Still a real STITCH benchmark, but cannot be classified from a local PDF that doesn't exist. If the PDF is added, re-add.)

---

## 2. Title / Year / Author Confirmations

For every previously-vague entry I now have verified author lists from the PDF. Updates worth noting:

| ID | v1 metadata | v2 metadata (verified) |
|----|-------------|------------------------|
| `kinetics_gebd` | "Shou et al., ICCV 2021" | Confirmed: Shou, Lei, Wang, Ghadiyaram, Feiszli (FAIR + NUS). Title "A Benchmark for Event Segmentation" (not "Benchmark Challenge"). |
| `mlvu` | "Zhou et al., NeurIPS 2024", "A Comprehensive Benchmark for Multi-Task Long Video Understanding" | Title is actually **"Benchmarking Multi-task Long Video Understanding"**. Authors verified. NeurIPS 2024 unconfirmed — venue field reduced to "arXiv 2024 (BAAI)". |
| `videomme` | "Fu et al., CVPR 2024" | arXiv 2405.21075. Likely **CVPR 2025**, not CVPR 2024 (preprint date 2024). 22+ authors, multi-institution. |
| `internvideo2` | "Wang et al., ECCV 2024" | First author Yi Wang, multiple ∗-equal contributors, OpenGVLab Shanghai AI Lab. Venue ECCV 2024 not confirmed from PDF first page; flagged. |
| `qvhighlights` | "Lei et al., NeurIPS 2021" | Confirmed Lei, Berg, Bansal — UNC. NeurIPS 2021. |
| `qd_detr` | "Moon et al., CVPR 2023" | Confirmed Moon, Hyun, Park, Park, Heo — Sungkyunkwan + Pyler. CVPR 2023. |
| `moviechat` | "Song et al., CVPR 2024" | Confirmed Song, Chai, Wang, et al. (Zhejiang U + UW + MSRA). |
| `kts_revisited` | "Afham et al. (Meta AI), arXiv 2023" | Confirmed full author list: Afham, Shukla, Poursaeed, Zhang, Shah, Lim. Meta AI. |
| `tfvtg` | "Zheng et al., arXiv 2024" | Confirmed Zheng, Cai, Chen, Peng, Liu — Peking U. ECCV 2024 likely. |
| `diffgebd` | "Unknown, 2023/24" | Confirmed Hwang, Gong, Kim, Cho — POSTECH + GenGenAI. |
| `on_gebd` | "Unknown, 2023" | Confirmed Jung, Kim, Lim, Son, Choi — GIST + SNU + POSTECH. |
| `gebd_efficient` | "Unknown, 2023" | Confirmed Zheng, Zhang, Wang, Song, Huang, Yang — Xi'an Jiaotong + Tsinghua. |
| `tag` | "Unknown, 2024" | Confirmed Lee et al., Sungkyunkwan U. Year ≥2025 likely. |
| `point_to_span` | "Unknown, 2024" | Confirmed Jeon et al., Chung-Ang U + KAIST + ETRI. |
| `wfs_sb` | "Unknown, 2024" | Confirmed Chen et al., Xiamen U. WFS-SB acronym. |
| `videotree` | "Wang et al., 2024" | Confirmed Wang, Yu, Stengel-Eskin, Yoon, Cheng, Bertasius, Bansal — UNC Chapel Hill. |
| `moment_sampling_vlm` | "Unknown, 2024" | Confirmed Chasmai, Jagatap, Gouthaman KV, Van Horn, Maji, Fanelli — UMass + Dolby. arXiv 2507.00033 (2025). |
| `adaretake` | "Unknown, 2024" | Confirmed Wang, Si, Wu, Zhu, Cao, Nie — HIT Shenzhen + Huawei. arXiv 2503.12559. |
| `longvu` | "Shen et al., NeurIPS 2024" | Confirmed first author Xiaoqian Shen, Meta AI + KAUST + Korea U. |
| `chapter_llama` | "Unknown, 2024" | Confirmed Ventura, Yang, Schmid, Varol — ENPC + Inria + Google DeepMind. |
| `videoinsta` | "Unknown, EMNLP 2024" | Confirmed Liao, Erler, Wang, Zhai, Zhang, Ma, Tresp — LMU Munich + MCML + TUM. EMNLP 2024 likely (not verified from page 1). |
| `tridet` | "Shi et al., CVPR 2023" | Confirmed Shi, Zhong, Cao, Ma, Li, Tao — Beihang + Meituan + JD. CVPR 2023. |
| `boundary_ssl` | "Unknown, 2023" | Confirmed Mun, Shin, Han, Lee, Ha, Lee, Kim — Kakao Brain + SNU + Hanyang. ICLR 2022 likely. |
| `lvc` | "Unknown, 2024" | Confirmed Wang, Wu, Rong, Jiang, et al. — CASIA + UCAS. |
| `lvlr` | "Unknown, 2024" | Confirmed Xu, Lan, Xie, Chen, Lu — likely Microsoft + USTC. |
| `simbase` | "Unknown, 2024" | Confirmed Bao, Kot — NTU. Same first author as Vid-Morp. Technical report. |
| `vid_morp` | "Unknown, 2024" | Confirmed Bao, Kong, Shao, Ng, Er, Kot — NTU. |
| `bam_detr` | "Unknown, 2024" | Confirmed Lee, Byun — Inha + Yonsei. ECCV 2024 likely. |
| `snag` | "Unknown, CVPR 2024" | Confirmed Mu, Mo, Li — UW-Madison. CVPR 2024. |
| `unloc` | "Yan et al., ICCV 2023" | Confirmed full author list: Yan, Xiong, Nagrani, Arnab, Wang, Ge, Ross, Schmid — Google Research. ICCV 2023. |
| `univtg` | "Lin et al., ICCV 2023" | Confirmed Lin, Zhang, Chen, Pramanick, Gao, Wang, Yan, Shou — NUS + Meta + JHU. ICCV 2023. |
| `self_chained_sevila` | "Unknown, NeurIPS 2023" | Confirmed Yu, Cho, Yadav, Bansal — UNC. NeurIPS 2023. Method = SeViLA. |
| `measure_twice` | "Unknown, 2024" | Confirmed Pang, Otani, Nakashima — U Osaka + CyberAgent. **ICLR 2026** explicitly stated. |
| `prompts_to_summaries` | "Unknown, 2024" | Confirmed Barbara, Maalouf — U Haifa. |
| `chatvtg` | "Unknown, 2024" | Confirmed Qu, Chen, Liu, Li, Zhao — BJTU + USTC + Horace Mann. |
| `mad` | "Soldan et al., CVPR 2022" | Confirmed: Soldan, Pardo, Alcázar, Caba Heilbron, Zhao, Giancola, Ghanem — KAUST + Adobe. |
| `momentseeker` | "Unknown, 2024" | Confirmed Yuan, Ni, Liu, Wang, Zhou, Liang, Zhao, Cao, Dou, Wen — Renmin U + BAAI. |
| `lovr` | "Unknown, 2024" | Confirmed Cai, Liang, Han, et al. — ECNU + PKU + HUST + Beihang + OceanBase. |
| `qvm2_flashmmr` | "Unknown, 2024" | Confirmed Cao, Du, Zhang, Yu, Li, Wang — U Queensland. Introduces QV-M2 dataset + FlashMMR method. |
| `towards_balanced` | "Unknown, 2024" | Confirmed Liu et al. — USTC + People's Daily. |
| `egoschema` | "Mangalam et al., NeurIPS 2023" | Confirmed Mangalam, Akshkulakov, Malik — UC Berkeley. NeurIPS 2023. |
| `nextqa` | "Xiao et al., CVPR 2021" | Confirmed Xiao, Shang, Yao, Chua — NUS. CVPR 2021. |
| `clip` | "Radford et al., ICML 2021" | Confirmed: full OpenAI author list. |
| `tall` | "Gao et al., ICCV 2017" | Confirmed Gao, Sun, Yang, Nevatia — USC + Google. |
| `two_d_tan` | "Zhang et al., AAAI 2020" | Confirmed Zhang, Peng, Fu, Luo — Rochester + MSR. |
| `lgvi` | "Unknown, CVPR 2020" | Confirmed Mun, Cho, Han — POSTECH + SNU. |
| `scdm` | "Yuan et al., NeurIPS 2019" | Confirmed Yuan, Ma, Wang, Liu, Zhu — Tsinghua + Tencent. |
| `activitynet_captions` | "Krishna et al., ICCV 2017" | Confirmed Krishna, Hata, Ren, Fei-Fei, Niebles — Stanford. |
| `uncovering_hidden` | "Otani et al., BMVC 2020" | Confirmed Otani, Nakashima, Rahtu, Heikkilä — CyberAgent + Osaka + Tampere + Oulu. |
| `tempura` | "Unknown" | Verified arXiv 2505.01583, Cheng et al. UW + CMU + NYCU + Microsoft. |

---

## 3. Suspicious / Uncertain Entries

These entries have remaining uncertainty after the audit pass and should be manually inspected before final citation.

| ID | Issue | What to verify |
|----|-------|----------------|
| `cpd_song_chen` | Likely wrong reference. STITCH probably cites Truong et al. (ruptures, 2020) or Harchaoui & Cappé (kernel CPD), not Song & Chen (kerSeg). | Open the STITCH draft and check the actual CPD citation. The local PDF may be the wrong file. |
| `pelt` | Verified, but PDF first page header says "November 26, 2024" — this is a re-typeset date for the 2012 JASA paper. Citation year should be **2012**. | Confirm — minor. |
| `internvideo2` | Venue "ECCV 2024" not confirmed from PDF first page. | Verify against the canonical citation. |
| `mlvu` | v1 said NeurIPS 2024 — not confirmed from PDF. | Verify venue. |
| `videomme` | arXiv preprint dated May 2024 → v3 May 2025. Likely CVPR 2025 not CVPR 2024. | Verify. |
| `tag`, `wfs_sb`, `point_to_span`, `lovr`, `momentseeker`, `lvc`, `lvlr`, `chatvtg`, `simbase`, `vid_morp`, `bam_detr`, `gebd_efficient`, `on_gebd`, `qvm2_flashmmr`, `prompts_to_summaries` | Year/venue not visible on PDF first page. | OK to cite as "arXiv 20XX" until canonical venue found. |
| `tempura` | v1 entry conflated with a different paper. v2 is the correct paper but adjacency to STITCH could be re-evaluated. | Inspect to decide whether to keep supporting or demote. |
| `moment_alignment_matr` | Entirely different task than v1 described. | Decide whether to drop entirely (since it's video-to-video, not text-to-video). |
| `adavideorag`, `deep_video_discovery`, `drvideo`, `hierarchical_long`, `video_rag`, `video_xl`, `your_interest` | Authors not verified from PDFs in this audit pass. All currently peripheral so low priority. | Defer. |
| Removed: `longvideobench` | PDF not present locally. | If you need to cite it, add the PDF and re-run. |

---

## 4. Filename / Title Mismatches

No major filename–title mismatches discovered: filenames in `lit_review_paper_pdfs/` correspond to the actual papers they contain. The only systematic issue is that v1 invented or guessed several full titles based on filename keywords.

---

## 5. Possible Duplicates / Same-Author Clusters

| Cluster | Entries | Notes |
|---------|---------|-------|
| Peijun Bao (NTU) | `simbase`, `vid_morp` | Same first author. Different methods, no duplication. |
| BAAI / Zheng Liu cluster | `mlvu`, `momentseeker` | Overlapping author groups across two benchmarks. No duplication. |
| Otani / Nakashima | `uncovering_hidden`, `measure_twice` | Same authors revisiting moment retrieval seven years later. No duplication. |
| Mohit Bansal lab (UNC) | `qvhighlights` (Lei), `videotree` (Wang+Yu+Bansal), `self_chained_sevila` (Yu+Cho+Yadav+Bansal) | Same lab, same Mohit Bansal. Three distinct papers. |
| Mun (POSTECH/SNU) | `lgvi`, `boundary_ssl` | Same first author Jonghwan Mun on both. No duplication. |

No actual duplicates found.

---

## 6. Verified vs. Inferred Distinction

In v2:
- `authors`, `title`, `arxiv_id` are taken from the PDF first page where possible.
- `year` and `venue` are sometimes left as `null` or "year/venue unverified" when not visible on page 1.
- All other fields (training_free, query_awareness, temporal_unit, etc.) are classification judgments grounded in the abstract.
