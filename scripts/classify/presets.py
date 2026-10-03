"""Classification preset definitions."""

ALL_PRESET_NAMES: list[str] = []

PRESETS = {
    "clip_full_labels_top5": {
        "model": "clip",
        "label_set": "full",
        "use_mask": True,
        "output_dir": "clip_full_labels_top5",
    },
    "clip_concise_labels_top5": {
        "model": "clip",
        "label_set": "concise",
        "use_mask": True,
        "output_dir": "clip_concise_labels_top5",
    },
    "siglip_full_labels_top5": {
        "model": "siglip",
        "label_set": "full",
        "use_mask": True,
        "output_dir": "siglip_full_labels_top5",
    },
    "siglip_concise_labels_top5": {
        "model": "siglip",
        "label_set": "concise",
        "use_mask": True,
        "output_dir": "siglip_concise_labels_top5",
    },
    "clip_full_image_top5": {
        "model": "clip",
        "label_set": "full",
        "use_mask": False,
        "output_dir": "clip_full_labels",
    },
    "siglip_full_image_top5": {
        "model": "siglip",
        "label_set": "full",
        "use_mask": False,
        "output_dir": "siglip_full_labels",
    },
    "gemini_full_labels": {
        "model": "gemini",
        "label_set": "full",
        "use_mask": True,
        "output_dir": "gemini_full_labels",
        "top_k": 1,
    },
}

TRAINED_PRESETS = {
    "clip_linear_probe_crop": {
        "model_type": "vision",
        "method": "linear_probe",
        "backbone": "clip",
        "use_mask": True,
        "checkpoint_name": "clip_linear_probe_crop",
        "output_dir": "clip_linear_probe_crop",
    },
    "clip_lora_crop": {
        "model_type": "vision",
        "method": "lora",
        "backbone": "clip",
        "use_mask": True,
        "checkpoint_name": "clip_lora_crop",
        "output_dir": "clip_lora_crop",
    },
    "siglip_linear_probe_crop": {
        "model_type": "vision",
        "method": "linear_probe",
        "backbone": "siglip",
        "use_mask": True,
        "checkpoint_name": "siglip_linear_probe_crop",
        "output_dir": "siglip_linear_probe_crop",
    },
    "siglip_lora_crop": {
        "model_type": "vision",
        "method": "lora",
        "backbone": "siglip",
        "use_mask": True,
        "checkpoint_name": "siglip_lora_crop",
        "output_dir": "siglip_lora_crop",
    },
    "efficientnet_b0_crop": {
        "model_type": "efficientnet",
        "method": "efficientnet_b0",
        "backbone": "efficientnet_b0",
        "use_mask": True,
        "checkpoint_name": "efficientnet_b0_crop",
        "output_dir": "efficientnet_b0_crop",
    },
    "late_fusion_clip_whisper_crop": {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": "late_fusion_clip_whisper_crop",
        "output_dir": "late_fusion_clip_whisper_crop",
    },
    "late_fusion_efficientnet_whisper_crop_balance_a1": {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": "late_fusion_efficientnet_whisper_crop_balance_a1",
        "output_dir": "late_fusion_efficientnet_whisper_crop_balance_a1",
    },
    "late_fusion_efficientnet_whisper_crop_balance_a2": {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": "late_fusion_efficientnet_whisper_crop_balance_a2",
        "output_dir": "late_fusion_efficientnet_whisper_crop_balance_a2",
    },
    "late_fusion_efficientnet_whisper_crop_balance_a3": {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": "late_fusion_efficientnet_whisper_crop_balance_a3",
        "output_dir": "late_fusion_efficientnet_whisper_crop_balance_a3",
    },
    "late_fusion_efficientnet_whisper_crop": {
        "model_type": "fusion",
        "method": "late_fusion",
        "backbone": "fusion",
        "use_mask": True,
        "checkpoint_name": "late_fusion_efficientnet_whisper_crop",
        "output_dir": "late_fusion_efficientnet_whisper_crop",
    },
}

ALL_PRESETS = {**PRESETS, **TRAINED_PRESETS}
ALL_PRESET_NAMES = list(ALL_PRESETS.keys())
