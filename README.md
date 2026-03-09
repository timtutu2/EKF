# ECE 276A — Project 3: Visual-Inertial SLAM

## Overview

This project implements visual-inertial simultaneous localization and mapping (SLAM)
using an Extended Kalman Filter (EKF) with IMU and stereo-camera measurements.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main entry point — runs all parts on dataset00 |
| `pr3_utils.py` | Provided utility functions (SE(3) math, data loading, visualization) |
| `IMU_via_EKF/ekf_prediction.py` | **Part 1** — IMU localization via EKF prediction |

## How to Run

Make sure the `opencv-env` conda environment is active:

```bash
conda activate opencv-env
```

### Part 1 — IMU Localization via EKF Prediction

Run on both datasets and save trajectory plots:

```bash
python IMU_via_EKF/ekf_prediction.py
```

Or run via the main entry point (dataset00 only):

```bash
python main.py
```

**Expected outputs:**
- `IMU_via_EKF/dataset00_ekf_prediction.png` — 2D trajectory with orientation arrows
- `IMU_via_EKF/dataset01_ekf_prediction.png`

## Data

Place datasets at:
```
../dataset00/dataset00.npy
../dataset01/dataset01.npy
```

## Dependencies

```
numpy
matplotlib
transforms3d
```
