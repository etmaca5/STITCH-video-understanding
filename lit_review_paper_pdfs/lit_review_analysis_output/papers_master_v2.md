# STITCH Literature Review — Master Table v2

**v2 = corrected after metadata audit.** 69 retained entries (LongVideoBench removed since PDF not local; otherwise same 75 input files minus 7 dropped from v1).

| Tier | v1 | v2 |
|------|----|----|
| core | 15 | 19 |
| supporting | 40 | 30 |
| peripheral | 10 | 13 |
| dropped | 7 | 7 |
| removed (no local PDF) | 0 | 1 |

Legend — TF=training-free · ZS=zero-shot · UN=unsupervised · QA=query_awareness · TU=temporal_unit · LV=long_video · RC=reusable_compute_style.

---

## CORE (19)

| ID | Title | Authors | Year | Cat | TF | ZS | UN | QA | TU | LV | RC | v1→v2 |
|----|-------|---------|------|-----|----|----|----|----|-----|----|----|-------|
| qvhighlights | QVHighlights + Moment-DETR | Lei, Berg, Bansal | 2021 | dataset | – | – | – | QD | proposal | ✗ | – | unchanged |
| kinetics_gebd | Generic Event Boundary Detection (Kinetics-GEBD) | Shou, Lei, Wang, Ghadiyaram, Feiszli | 2021 | dataset | – | – | – | NQ | frame | ✗ | – | unchanged |
| activitynet_captions | Dense-Captioning Events in Videos (ActivityNet Captions) | Krishna, Hata, Ren, Fei-Fei, Niebles | 2017 | dataset | – | – | – | QD | proposal | ✗ | – | **promoted to core (was supporting)** |
| videomme | Video-MME | Fu et al. | 2024 | dataset | – | – | – | QD | frame | ✓ | – | unchanged |
| mlvu | MLVU | Zhou, Shu, Zhao et al. | 2024 | dataset | – | – | – | QD | frame | ✓ | – | unchanged (title corrected) |
| internvideo2 | InternVideo2 | Y. Wang et al. | 2024 | other | ✗ | ✗ | ✗ | NQ | window | ✓ | once | unchanged |
| pelt | PELT | Killick, Fearnhead, Eckley | 2012 | other | ✓ | – | ✓ | NQ | adaptive | ✗ | once | unchanged |
| kts_revisited | Revisiting KTS as Adaptive Tokenizer | Afham, Shukla, Poursaeed, Zhang, Shah, Lim | 2023 | method | ✓ | ✓ | ✓ | NQ | adaptive | ✓ | once | unchanged |
| diffgebd | GEBD via Denoising Diffusion (DiffGEBD) | Hwang, Gong, Kim, Cho | 2024 | method | ✗ | ✗ | ✗ | NQ | frame | ✗ | rerun | unchanged |
| gebd_efficient | Rethinking Architecture for Efficient GEBD | Zheng, Zhang, Wang, Song, Huang, Yang | – | method | ✗ | ✗ | ✗ | NQ | frame | ✗ | rerun | unchanged |
| on_gebd | Online GEBD (On-GEBD) | Jung, Kim, Lim, Son, Choi | – | method | ✗ | ✗ | ✗ | NQ | frame | ✓ | rerun | unchanged |
| tfvtg | Training-free Video Temporal Grounding | Zheng, Cai, Chen, Peng, Liu | 2024 | method | ✓ | ✓ | ✗ | QD | proposal | ✗ | repr+score | unchanged |
| moment_gpt | Zero-shot VMR via Off-the-shelf MLLMs (Moment-GPT) | Xu, Sun, Zhai, Li, Liang, Li, Du | 2024 | method | ✓ | ✓ | ✗ | QD | proposal | ✗ | rerun | unchanged |
| zs_frozen_vlm | Zero-Shot VMR from Frozen VLMs | Luo, Huang, Gong, Jin, Liu | – | method | ✓ | ✓ | ✗ | QD | proposal | ✗ | repr+score | unchanged |
| tag | TAG: Temporal-Aware Approach for ZS VTG | Lee et al. (Sungkyunkwan) | – | method | ✓ | ✓ | ✗ | QD | adaptive | ✗ | repr+score | **promoted (was supporting; v1 had wrong description)** |
| point_to_span | Point to Span: ZS MR for Hour-Long Videos | Jeon et al. | 2024 | method | ✓ | ✓ | ✗ | QD | proposal | ✓ | repr+score | **promoted (was supporting)** |
| wfs_sb | Wavelet-based Frame Selection (WFS-SB) | Chen et al. (Xiamen U) | – | method | ✓ | ✗ | ✓ | NQ | adaptive | ✓ | once | **promoted (was supporting)** |
| videotree | VideoTree: Adaptive Tree LLM Reasoning | Wang, Yu, Stengel-Eskin, Yoon, Cheng, Bertasius, Bansal | 2024 | method | ✓ | ✗ | ✗ | QD | hierarchical | ✓ | rerun | **promoted (was supporting)** |
| moment_sampling_vlm | Moment Sampling in Video LLMs | Chasmai, Jagatap, Gouthaman, Van Horn, Maji, Fanelli | 2025 | method | ✓ | ✗ | ✗ | NQ | adaptive | ✓ | once | **promoted (was supporting)** |
| timeloc | TimeLoc: Unified Timestamp Localization | Zhang, Sui, Liu, Mu, Wang, Ghanem | 2025 | method | ✗ | ✗ | ✗ | QP | proposal | ✓ | rerun | **promoted (was supporting)** |

---

## SUPPORTING (30)

| ID | Title | Authors | Year | Cat | TF | ZS | QA | TU | LV | v1→v2 |
|----|-------|---------|------|-----|----|----|----|----|-----|-------|
| qd_detr | QD-DETR | Moon, Hyun, Park, Park, Heo | 2023 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| univtg | UniVTG | Lin, Zhang, Chen, Pramanick, Gao, Wang, Yan, Shou | 2023 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| unloc | UnLoc | Yan, Xiong, Nagrani, Arnab, Wang, Ge, Ross, Schmid | 2023 | method | ✗ | ✗ | QP | proposal | ✗ | unchanged |
| snag | SnAG | Mu, Mo, Li | 2024 | method | ✗ | ✗ | QD | proposal | ✓ | unchanged (RC tightened) |
| bam_detr | BAM-DETR | Lee, Byun | 2024 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| simbase | SimBase | Bao, Kot | – | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| vid_morp | Vid-Morp | Bao, Kong, Shao, Ng, Er, Kot | – | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| qvm2_flashmmr | When One Moment Isn't Enough (QV-M2 + FlashMMR) | Cao, Du, Zhang, Yu, Li, Wang | 2024 | method | ✗ | ✗ | QD | hierarchical | ✗ | unchanged |
| towards_balanced | Towards Balanced Alignment | Liu et al. | – | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| chatvtg | ChatVTG | Qu, Chen, Liu, Li, Zhao | – | method | ✓ | ✓ | QD | proposal | ✗ | **promoted (was peripheral)** |
| measure_twice | Measure Twice, Cut Once | Pang, Otani, Nakashima | 2026 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| moviechat | MovieChat | Song, Chai, Wang et al. | 2024 | method | ✗ | ✗ | QD | window | ✓ | unchanged |
| longvu | LongVU | Shen et al. (Meta + KAUST) | 2024 | method | ✗ | ✗ | QP | adaptive | ✓ | unchanged |
| storm | STORM | Jiang et al. (NVIDIA) | 2025 | method | ✗ | ✗ | NQ | adaptive | ✓ | unchanged |
| adaretake | AdaReTaKe | Wang, Si, Wu, Zhu, Cao, Nie | 2025 | method | ✓ | ✗ | NQ | adaptive | ✓ | unchanged |
| chapter_llama | Chapter-Llama | Ventura, Yang, Schmid, Varol | 2024 | method | ✗ | ✗ | NQ | adaptive | ✓ | unchanged |
| videoinsta | VideoINSTA | Liao, Erler, Wang, Zhai, Zhang, Ma, Tresp | 2024 | method | ✓ | ✓ | QD | window | ✓ | unchanged |
| prompts_to_summaries | Prompts to Summaries | Barbara, Maalouf | – | method | ✓ | ✓ | QD | window | ✗ | unchanged |
| self_chained_sevila | Self-Chained ILM (SeViLA) | Yu, Cho, Yadav, Bansal | 2023 | method | ✗ | ✗ | QD | frame | ✗ | unchanged |
| boundary_ssl | Boundary-Aware SSL for Video Scene Segmentation | Mun, Shin, Han, Lee, Ha, Lee, Kim | 2022 | method | ✗ | ✗ | NQ | frame | ✗ | unchanged |
| tridet | TriDet | Shi, Zhong, Cao, Ma, Li, Tao | 2023 | method | ✗ | ✗ | NQ | proposal | ✗ | unchanged |
| shot_relating_masrc | Modality-Aware Shot Relating (MASRC) | Tan, Wang, Dang, Li, Ou | – | method | ✗ | ✗ | NQ | proposal | ✗ | unchanged (task corrected: Video Scene Detection, not VQA) |
| tempura | TEMPURA | Cheng et al. (UW + CMU + NYCU + Microsoft) | 2025 | method | ✗ | ✗ | NQ | proposal | ✗ | unchanged (paper completely re-identified) |
| two_d_tan | 2D-TAN | Zhang, Peng, Fu, Luo | 2020 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| tall | TALL | Gao, Sun, Yang, Nevatia | 2017 | method | ✗ | ✗ | QD | proposal | ✗ | unchanged |
| lvc | LVC | Wang, Wu, Rong, Jiang et al. (CASIA) | – | method | ✗ | ✗ | QD | window | ✓ | unchanged |
| lvlr | Long Video Understanding with Learnable Retrieval | Xu, Lan, Xie, Chen, Lu | – | method | ✗ | ✗ | QD | window | ✓ | unchanged |
| adavideorag | AdaVideoRAG | unverified | 2024 | method | ✗ | ✗ | QD | window | ✓ | unchanged |
| cpd_song_chen | Practical and Powerful Kernel-Based CPD | Song, Chen | – | other | ✓ | – | ✓ | NQ | adaptive | ✗ | **demoted (was core); attribution flagged** |
| clip | CLIP | Radford et al. | 2021 | other | ✗ | ✓ | QD | frame | ✗ | unchanged |
| uncovering_hidden | Uncovering Hidden Challenges in MR | Otani, Nakashima, Rahtu, Heikkilä | 2020 | other | – | – | QD | – | ✗ | unchanged |
| egoschema | EgoSchema | Mangalam, Akshkulakov, Malik | 2023 | dataset | – | – | QD | frame | ✓ | unchanged |
| mad | MAD | Soldan, Pardo, Alcázar, Caba Heilbron, Zhao, Giancola, Ghanem | 2022 | dataset | – | – | QD | proposal | ✓ | unchanged |
| momentseeker | MomentSeeker | Yuan, Ni, Liu, Wang, Zhou, Liang, Zhao, Cao, Dou, Wen | – | dataset | – | – | QD | proposal | ✓ | unchanged |
| lovr | LoVR | Cai, Liang, Han et al. | – | dataset | – | – | QD | proposal | ✓ | unchanged |

---

## PERIPHERAL (13)

| ID | Title | Authors | v1→v2 |
|----|-------|---------|-------|
| moment_alignment_matr | MATR (Aligning Moments via Video Queries) | Kumar, Agarwal, Gupta, Mishra | **demoted (Vid2VidMR ≠ STITCH's task)** |
| scdm | SCDM | Yuan, Ma, Wang, Liu, Zhu | **demoted (was supporting)** |
| lgvi | LGI / LGVI | Mun, Cho, Han | **demoted (was supporting)** |
| nextqa | NExT-QA | Xiao, Shang, Yao, Chua | **demoted (was supporting)** |
| ego4d | Ego4D | Grauman et al. | unchanged |
| videomae_v2 | VideoMAE V2 | L. Wang et al. | unchanged |
| vjepa2 | V-JEPA 2 | Assran et al. | unchanged |
| deep_video_discovery | Deep Video Discovery | unverified | unchanged |
| drvideo | DrVideo | unverified | unchanged |
| hierarchical_long | Hierarchical Long Video w/ Audiovisual Entity Cohesion | unverified | unchanged |
| video_rag | Video-RAG | unverified | unchanged |
| video_xl | Video-XL | unverified | unchanged |
| your_interest | Your Interest, Your Summaries | unverified | unchanged |

---

## REMOVED FROM MASTER

| Reason | Entry |
|--------|-------|
| PDF not present locally | LongVideoBench (still a real STITCH benchmark — re-add if PDF appears) |

## DROPPED (7, unchanged from v1)

CLIP4Clip · DreamerV3 · Video Summarization Survey · Video-ColBERT · X-CLIP · Penguin-VL · Video Re-localization
