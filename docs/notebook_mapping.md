# Notebook → script mapping for EgoWaste / trash_classification

Original Colab notebooks are checked in under [`notebooks/`](../notebooks/)
(~12 MB). They are exploratory history. Reproducible entrypoints are the
`scripts/` packages below — do not treat notebooks as the source of truth
for v1–v16.

| Notebook (`notebooks/`) | Role | Ported scripts | Notes |
|---|---|---|---|
| `EgoWaste.ipynb` | Dataset load, taxonomy, early detect_trash scaffolding | `scripts/shared/`, `scripts/segment/detect_trash.py`, `scripts/segment/run_detect.py` | Drive mounts → `EGOWASTE_ROOT` / shipped `data/` |
| `sam3.ipynb` | SAM3 hand/plastic detection, IoU | `scripts/segment/run_detect.py`, `evaluate_iou.py`, `run_pipeline.sh` | GPU + `SAM3_ROOT` + HF token |
| `sam3_260726.ipynb` | Earlier SAM3 IoU comparison | same as `sam3.ipynb` | Superseded by packaged segment pipeline |
| `image.ipynb` | Crop / image preprocess experiments | `scripts/classify/image_utils.py`, `segment_masks.py` | bbox vs masked crop modes |
| `imageClassify.ipynb` | CLIP/SigLIP/Gemini classification | `scripts/classify/classifiers.py`, `presets.py`, `run_classify.py`, `run_segment_classify_eval.py` | Gemini optional; no API keys in repo |
| `whisperV1.ipynb` | Audio extract + Whisper ASR | `scripts/whisper/extract_audio.py`, `transcribe.py`, `evaluate.py` | |
| `audio.ipynb` | Noise reduction | `scripts/whisper/denoise_audio.py` | |
| `lora_whisper_finetune.ipynb` | LoRA fine-tune sketch | `scripts/whisper/train_lora.py`, `build_training_metadata.py` | Checkpoints gitignored |
| `Semantic_retrieval.ipynb` | Phrase / semantic match | classify phrase matching + fusion (`fusion_model.py`, Whisper→category map) | MiniLM retrieval path |

## Analysis not covered by notebooks

| Need | Script |
|---|---|
| Dataset class / phrase counts | `analysis/build_summaries.py` |
| v1–v16 joined metrics table | `analysis/build_summaries.py` → `summaries/version_metrics_joined.csv` |
| Version registry | `configs/segment_versions.py` (mirrors `scripts/classify/segment_masks.py`) |

## Metric caveats

- **Paper mask IoU** (v1 8.3%, v2 19.8%) is the end-to-end / reported mask metric in the IEEE draft.
- **CSV `mean_max_iou_top1`** in `summaries/segment_iou_summary.csv` is a different aggregation (per-frame max IoU vs GT). Do not equate the two columns.
- **Frame Top-1** in `classification_segment_summary.csv` is not the same as **group any-frame final-prob** in the paper table (Fusion-EN 73.4%).
