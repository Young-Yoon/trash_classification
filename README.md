# EgoWaste trash classification (v1–v16)

Reproducible scripts, version registry, and summary metrics for multimodal
plastic sorting on the EgoWaste camera-1 plastics subset. Raw frames, SAM
pickles, candidate-crop PNGs, audio, and checkpoints are **not** in this
repository.

See [DEV_HISTORY.md](DEV_HISTORY.md) for the sequential v1–v16 story and
[docs/notebook_mapping.md](docs/notebook_mapping.md) for Colab → script ports.

## Layout

| Path | Purpose |
|------|---------|
| `scripts/segment/` | SAM3 detection + IoU |
| `scripts/classify/` | Crop classifiers, rerank v7–v16, late fusion |
| `scripts/whisper/` | Audio extract, denoise, LoRA Whisper, eval |
| `scripts/shared/` | Paths (`EGOWASTE_ROOT`, `SAM3_ROOT`, …) |
| `configs/segment_versions.py` | Version registry (descriptions, crop, rerank) |
| `analysis/build_summaries.py` | Rebuild compact CSV tables |
| `summaries/` | Shipped metrics only (no per-frame JSON) |

## Environment

```bash
export EGOWASTE_ROOT=/path/to/data_tree   # meta CSV, audio, segment/, checkpoints
export EGOWASTE_ASSET_ROOT=/path/to/assets  # optional frames/masks root
export SAM3_ROOT=/path/to/sam3            # detection only
# optional: CSIRE_CODE_ROOT if meta still lives under a csire_code checkout
```

Install Python deps:

```bash
pip install -r scripts/requirements.txt
pip install -r scripts/classify/requirements.txt
```

## Reproduce

**CPU / summaries only** (works with shipped `summaries/` plus local meta CSV):

```bash
chmod +x run_repro.sh
./run_repro.sh summaries
```

**GPU stages** (data + weights required; commands are commented inside the shell wrappers):

```bash
bash scripts/segment/run_pipeline.sh      # SAM3 detect + IoU
bash scripts/whisper/run_pipeline.sh      # metadata / LoRA / transcribe / eval
bash scripts/classify/run_pipeline.sh     # train or evaluate presets
# Segment-version sweep (v1–v16):
python scripts/classify/run_segment_classify_eval.py --help
```

Gemini classification remains an optional path in `scripts/classify/`; do not
commit API keys.

## Headline numbers (v15)

| Metric | Value |
|--------|------:|
| Frames / clips / classes / phrases | 3538 / 463 / 15 / 104 |
| Paper mask IoU v1 / v2 | 8.3% / 19.8% |
| Group any-frame final-prob Fusion-EN | **73.4%** |
| Group any-frame EfficientNet | 69.6% |
| Frame Top-1 Fusion-EN (segment summary) | 70.0% |

Paper mask IoU and CSV `mean_max_iou_top1` are different aggregations; see
`docs/notebook_mapping.md`.

## License / data

Dataset and model weights are external. Point `EGOWASTE_ROOT` at a licensed
local copy. This repo tracks code + summary tables only.
