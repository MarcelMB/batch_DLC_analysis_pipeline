# Batch DLC Analysis Pipeline

![Pipeline Flowchart](dlc_behavior_analysis_flowchart.png)

## Overview

Two-stage pipeline for batch processing DeepLabCut (DLC) tracked mouse behavior in an open-field arena across multiple days (D1-D5) and animals (M01-M09).

## Scripts

### 1. `DLC_batch_analysis.py` - Batch Wrapper

Handles video preparation and DLC inference in three phases:

- **Phase 1** - Find all `.mp4` videos per day tag, open a reference frame for interactive 4-corner arena selection (TL, TR, BR, BL), compute homography, and save corners as YAML
- **Phase 2** - Warp all videos using the homography (corrects 2D camera-angle distortion, straightening the arena from trapezoid to rectangle), capped at 15 min per video
- **Phase 3** - Run `deeplabcut.analyze_videos()` on all corrected videos

### 2. `dlc_behavior_analysis.py` - Post-Processing & Metrics

Takes the DLC CSV outputs and arena YAML from the batch wrapper and runs a full preprocessing + metrics pipeline:

**Preprocessing (pixel space):**
- Step 0: Likelihood fallback (tiered: midpoint of nose+tail > tail+offset > keep original)
- Step 1: Jump removal (threshold 100 mm, linear interpolation)
- Step 1b: Speed-based cleaning (flag > 50 cm/s, expand window, interpolate)
- Step 2: 3D parallax correction (corrects for mouse body height above arena floor - complements the 2D warp from the batch script)
- Step 3: Clip to arena bounds
- Step 3b: Gaussian smooth (sigma = 1 frame)

**Unit conversion:** pixel to cm (30 cm arena)

**Derived metrics:**
- Total distance (cm)
- Median speed (cm/s)
- Center zone distance (inner 25% of arena area)
- Arena coverage (% of 1 cm grid bins visited)
- Immobility time (seconds below 2 cm/s)

**Outputs:**
- 8-panel pipeline figure per animal (with quality warnings)
- `behavior_metrics.csv` (all animals x all days)
- Summary boxplots (5 metrics x D1-D5, color-coded per animal)

## DLC Model

`dlc_config.yaml` contains the DeepLabCut project configuration for the `topmouse_project` model (scorer: marcel). It tracks 26 body parts including nose, ears, eyes, neck, back, mouse_center, tail (6 segments), and bilateral shoulder/midside/hip points. The pipeline primarily uses `mouse_center`, `tail_base`, and `nose`.

## Requirements

- Python 3.x
- DeepLabCut
- OpenCV (`cv2`)
- NumPy, Pandas, Matplotlib, SciPy
- PyYAML
