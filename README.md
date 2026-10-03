# EgoWaste trash classification (v1–v16)

Reproducible scripts, version registry, summary metrics, paper snapshot, and
original Colab notebooks for multimodal plastic sorting on the EgoWaste
camera-1 plastics subset.

Raw frames, SAM pickles, candidate-crop PNGs, audio, and checkpoints are
**not** in this repository (~30 MB of meta + paper + notebooks are).

See [DEV_HISTORY.md](DEV_HISTORY.md) for the sequential v1–v16 story and
[docs/notebook_mapping.md](docs/notebook_mapping.md) for Colab → script ports.

## Layout

| Path | Purpose |
|------|---------|
| `data/` | Meta CSVs, taxonomy, training metadata (~10 MB) |
| `paper/` | Latest IEEE draft (`main.tex`, `refs.bib`, `main.pdf`, key figs) |
| `notebooks/` | Original Colab `.ipynb` (history / reference, ~12 MB) |
| `scripts/segment/` | SAM3 detection + IoU |
| `scripts/classify/` | Crop classifiers, rerank v7–v16, late fusion |
| `scripts/whisper/` | Audio extract, denoise, LoRA Whisper, eval |
| `scripts/shared/` | Paths (`EGOWASTE_ROOT`, `SAM3_ROOT`, …) |
| `configs/segment_versions.py` | Version registry (descriptions, crop, rerank) |
| `analysis/build_summaries.py` | Rebuild compact CSV tables |
| `summaries/` | Shipped metrics only (no per-frame JSON) |

## Environment

By default, meta CSVs resolve from `data/camera1_plastics_shareable_dataset/`.
Point asset/runtime trees elsewhere when needed:

```bash
export EGOWASTE_ASSET_ROOT=/path/to/assets  # frames/masks/clips
export EGOWASTE_ROOT=/path/to/runtime       # audio, segment/, checkpoints
export SAM3_ROOT=/path/to/sam3              # detection only
```

Install Python deps:

```bash
pip install -r requirements.txt
pip install -r scripts/requirements.txt
pip install -r scripts/classify/requirements.txt
```

## Reproduce

**CPU / summaries only** (shipped `data/` + `summaries/`):

```bash
chmod +x run_repro.sh
./run_repro.sh summaries
```

**GPU stages** (local weights + frames required):

```bash
bash scripts/segment/run_pipeline.sh
bash scripts/whisper/run_pipeline.sh
bash scripts/classify/run_pipeline.sh
python scripts/classify/run_segment_classify_eval.py --help
```

Gemini classification remains optional in `scripts/classify/`; do not commit API keys.
Notebooks under `notebooks/` are archival; prefer `scripts/` for repro.

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

Full video/frame assets and model weights stay external. This repo ships
metadata, paper figures/PDF, notebooks, and code only.
