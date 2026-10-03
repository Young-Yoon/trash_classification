EgoWaste Camera-1 Plastics Subset
===================================

Purpose
-------
This folder contains a Camera-1 subset of EgoWaste for reviewing:
1. Plastic-category image frames and object masks.
2. Short source-video clips with original audio.
3. Common descriptive audio phrases.
4. The Plastics label taxonomy and recycling-category metadata.
5. The corresponding plastic-only meta2 annotations.

Summary
-------
Plastic annotation rows: 3538
Unique plastic frames: 3002
Plastic categories: 15
Timestamp groups: 463
Generated audio-video clips: 463
Common phrase entries: 104
Taxonomy categories: 15

Main files
----------
camera1_plastics_meta2.csv
    All Camera-1 meta rows whose supercategory (material aware category) is Plastics.
    Additional columns include copied frame/mask paths and clip paths.

camera1_plastics_timestamp_groups.csv
    One row per GX/timestamp/phrase group.
    Includes frame ranges, current categories, transcript phrases, raw-video paths, and generated clip paths.

camera1_plastics_common_phrases.csv
    Frequency-ranked descriptive phrases associated with plastic timestamps.

camera1_plastics_taxonomy.json
    Plastic-only taxonomy with category names, abbreviations, recycling categories, descriptions, and dataset counts.

camera1_plastics_manifest.json
    Machine-readable summary of this exported subset.

Folders
-------
frames/
    Unique image frames associated with Plastics annotations.

masks/
    Object masks corresponding to the plastic meta2 rows.

audio_video_clips/
    Short MP4 clips containing the original source-video audio.

Important notes
---------------
- The original dataset is not modified.
- A clip is grouped by GX_id, begin_time, end_time, and phrase.
- Empty recycle-category or description fields mean that the supplied taxonomy JSON did not contain matching metadata for that category.
