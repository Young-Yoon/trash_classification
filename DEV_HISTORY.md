# Development history (v1–v16)

Sequential EgoWaste plastic-classification history: Colab exploration → packaged
scripts → segment/rerank versions → IEEE paper snapshots. Raw experiment dumps
are omitted; metrics live under `summaries/`.

## Problem

Material recovery facilities record first-person video of sorters holding
plastics while naming them aloud. Camera-1 plastics subset:

- **3,538** labeled frames, **463** clips (`timestamp_groups`)
- **15** taxonomy categories, **104** common phrases
- Vision branch: SAM3 masks → crop classifier (CLIP / SigLIP / EfficientNet-B0)
- Speech branch: Whisper large-v2 + LoRA → MiniLM phrase match + text rules
- Late fusion logistic regression joins image scores with speech class

## Notebook → package port

| Phase | Notebooks | Scripts |
|-------|-----------|---------|
| Data + early detect | `EgoWaste.ipynb` | `scripts/shared/`, `scripts/segment/` |
| SAM3 + IoU | `sam3.ipynb`, `sam3_260726.ipynb` | `scripts/segment/run_detect.py`, `evaluate_iou.py` |
| Crops / classify | `image.ipynb`, `imageClassify.ipynb` | `scripts/classify/` |
| ASR | `whisperV1.ipynb`, `audio.ipynb`, `lora_whisper_finetune.ipynb` | `scripts/whisper/` |
| Phrase semantics | `Semantic_retrieval.ipynb` | classify fusion / category map |

Full table: [docs/notebook_mapping.md](docs/notebook_mapping.md).

## Segment versions (canonical)

Defined in `configs/segment_versions.py` and `scripts/classify/segment_masks.py`
(`SEGMENT_VERSION_DESCRIPTIONS`, crop modes, rerank strategies).

### Phase 1 — Backbone masks (v1–v2)

| Ver | Idea | Fusion-EN frame Top-1 | Group any-frame final-prob |
|-----|------|----------------------:|---------------------------:|
| v1 | Hand ROI; in-hand max containment else largest bbox | 25.9% | 35.4% |
| v2 | Full-image plastic; SAM score Top-1 | 33.7% | 32.6% |

Paper-reported **mask IoU**: v1 **8.3%**, v2 **19.8%** (not the same column as
CSV `mean_max_iou_top1` in `summaries/segment_iou_summary.csv`).

### Phase 2 — Mask selection / crop (v3–v6, v8)

| Ver | Idea | Frame Top-1 | Group any-frame |
|-----|------|------------:|----------------:|
| v3 | Score Top-1 among all plastic candidates from v1 detections | 58.2% | 64.6% |
| v4 | Same masks as v3; **masked** crop | 53.2% | 61.8% |
| v5 | In-hand Top-1 else v2 fallback | 60.0% | 67.6% |
| v6 | Same masks as v5; **masked** crop | 54.4% | 64.4% |
| v8 | Same masks as v5; bbox + **15% padding** | 58.3% | 67.0% |

Bbox crops beat masked crops for this backbone; padding helps slightly vs v6.

### Phase 3 — Top-5 pools + confidence rerank (v7, v9, v10)

| Ver | Pool | Rerank | Frame Top-1 | Group any-frame |
|-----|------|--------|------------:|----------------:|
| v7 | Top-5 from v1 scores | confidence | 68.7% | 72.4% |
| v9 | v5-style pool (in-hand + v2 + ranked) | confidence | 64.8% | 73.7% |
| v10 | In-hand Top-5 | confidence | 66.1% | 75.2% |

Moving from a single mask to a shortlist is the largest classification jump.

### Phase 4 — Rerank strategies (v11–v16)

| Ver | Strategy | Frame Top-1 | Group any-frame |
|-----|----------|------------:|----------------:|
| v11 | margin (top1−top2), pool = v7 | 68.9% | 73.0% |
| v12 | margin, pool = v10 | 66.2% | 75.6% |
| v13 | composite 0.7 conf + 0.3 SAM | 66.9% | 71.1% |
| v14 | composite margin + SAM | 67.6% | 70.8% |
| **v15** | **margin + Whisper agreement bonus** | **70.0%** | **73.4%** |
| v16 | geo composite (margin + SAM + hand IoU) | 66.8% | 70.6% |

**Reported final setting: v15** (Fusion-EN group any-frame **73.4%** vs
EfficientNet vision-only **69.6%** in the paper table). Some intermediate group
cells (e.g. v10/v12) can look higher on one aggregation; the paper locks v15 as
the speech-aligned end-to-end configuration with stage-wise reporting.

## Class-balance ablation (Fusion-EN, final pipeline)

From `summaries/class_balance_experiments_summary.csv` (group Top-1 %):

| Variant | Group Top-1 | Macro-F1 Δ story |
|---------|------------:|------------------|
| A0 baseline | 73.4 | Even CE |
| A1 | 72.8 | Lower Other-#1 FP, higher Unnumbered recall |
| A2 / A3 | ~72.1–72.6 | Stronger reweight; rare-class tradeoffs |

## Paper snapshots

Workspace drafts (not vendored into this git tree by default):

1. `paper/main.tex` — title *Investigating Automated Plastic Waste Sorting Using Multimodal Classification*
2. `paper_v2_0919/diffs/` — incremental edits `00_fig1_ablation` → `01_affiliation` → `02_framing` → `03_related` → `04_method` → `05_exp_intro` → `08_discussion`
3. `paper_v1_0915/main.tex`, `paper_v2_0919/main.tex` — retitled *Mining Egocentric Video with Sorter Speech for Plastic Classification*; **v2 is the latest body**

## How to regenerate tables

```bash
./run_repro.sh summaries
# writes summaries/dataset_stats.csv, segment_version_catalog.csv,
# version_metrics_joined.csv, dataset_category_counts.csv, …
```

Joined metrics: `summaries/version_metrics_joined.csv`.
