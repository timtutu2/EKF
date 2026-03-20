# ECE 276A Project 3: Visual-Inertial SLAM

## Overview

This `code/` directory contains the current implementation for ECE 276A Project 3:

1. IMU-only pose prediction with an EKF on `SE(3)`.
2. Stereo feature detection, matching, and temporal tracking.
3. Landmark mapping with an EKF update using a known IMU trajectory.
4. Full visual-inertial SLAM that jointly updates pose and landmarks.

The code is organized as standalone scripts for each part rather than a single unified pipeline.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Older starter script for Part 1 only; not the recommended entry point. |
| `pr3_utils.py` | Provided utilities for loading data, `SE(3)` math, and trajectory plotting. |
| `IMU_EKF/ekf_prediction.py` | Part 1: IMU localization via EKF prediction. |
| `Feature_detection_match/feature_detection.py` | Stereo matching plus temporal tracking to build feature tracks. |
| `Feature_detection_match/run_feature_detection_show_correspond.py` | Runs feature tracking on `dataset02` and saves stereo correspondence visualizations. |
| `Feature_detection_match/run_feature_detection_show_t+15.py` | Runs feature tracking on `dataset02` and saves temporal tracking visualizations. |
| `Landmark_mapping_EKF_update/landmark_mapping_ekf.py` | Part 3: landmark initialization and EKF landmark updates. |
| `Landmark_mapping_EKF_update/run_landmark_mapping.py` | Runner for landmark mapping on `dataset00`, `dataset01`, and `dataset02`. |
| `Visual-inertial SLAM/vi_slam.py` | Part 4: full VI-SLAM EKF. |
| `Visual-inertial SLAM/run_vi_slam.py` | Runner for full VI-SLAM on `dataset00`, `dataset01`, and `dataset02`. |
| `transforms3d/` | Local helper package used by the project. |

## Expected Data Layout

The runners expect the datasets to live at the repository root:

```text
PR3/
├── dataset00/dataset00.npy
├── dataset00/dataset00_imgs.npy
├── dataset01/dataset01.npy
├── dataset01/dataset01_imgs.npy
├── dataset02/dataset02.npy
└── dataset02/dataset02_imgs.npy
```

The `.npy` files loaded by `pr3_utils.load_data(...)` contain IMU, calibration, timestamps, and feature arrays. The feature-detection scripts additionally read `dataset02/dataset02_imgs.npy`.

## Dependencies

The current code imports:

- `numpy`
- `matplotlib`
- `opencv-python` (`cv2`)

A local copy of `transforms3d` is already included in this repository, so no external install is needed for that package.

## How To Run

Run commands from the `code/` directory unless noted otherwise.

### Part 1: IMU EKF Prediction

```bash
cd code
python IMU_EKF/ekf_prediction.py
```

What it does:
- Loads each of `dataset00`, `dataset01`, and `dataset02`.
- Propagates the IMU pose using linear and angular velocity measurements.
- Saves one trajectory plot per dataset next to the script.

Typical outputs:
- `IMU_EKF/dataset00_ekf_prediction.png`
- `IMU_EKF/dataset01_ekf_prediction.png`
- `IMU_EKF/dataset02_ekf_prediction.png`

### Feature Detection and Matching

Stereo matching and temporal tracking are implemented in `Feature_detection_match/feature_detection.py`.
Two runner scripts are provided for visualization on `dataset02`.

Stereo correspondence visualization:

```bash
cd code
python Feature_detection_match/run_feature_detection_show_correspond.py
```

Temporal tracking visualization:

```bash
cd code
python Feature_detection_match/run_feature_detection_show_t+15.py
```

Typical outputs:
- `Feature_detection_match/dataset02_features.npy`
- `Feature_detection_match/dataset02_features_correspond.png`
- `Feature_detection_match/dataset02_features_t+15.png`

### Part 3: Landmark Mapping EKF Update

```bash
cd code
python Landmark_mapping_EKF_update/run_landmark_mapping.py
```

What it does:
- Runs the Part 1 IMU EKF first.
- Uses the resulting trajectory as fixed input.
- Initializes landmarks by stereo triangulation.
- Refines landmarks with per-landmark EKF updates.

Typical outputs:
- `Landmark_mapping_EKF_update/dataset00_landmark_map_(20, 15).png`
- `Landmark_mapping_EKF_update/dataset01_landmark_map_(20, 15).png`
- `Landmark_mapping_EKF_update/dataset02_landmark_map_(20, 15).png`

The script prints a `.npy` result path as well, but saving the result dictionary is currently commented out in the runner.

### Part 4: Visual-Inertial SLAM

```bash
cd code
python "Visual-inertial SLAM/run_vi_slam.py"
```

What it does:
- Runs the IMU-only EKF as a baseline.
- Initializes landmarks from stereo observations.
- Performs a pose update using an information-form EKF step.
- Performs landmark EKF updates using the corrected pose.
- Saves trajectory comparison and landmark-map figures.

Typical outputs:
- `Visual-inertial SLAM/dataset00_vi_slam_(20, 15)_dt_0.png`
- `Visual-inertial SLAM/dataset00_trajectory_comparison_(20, 15)_dt_0.png`
- Same pattern for `dataset01` and `dataset02`

As with Part 3, the runner prints a `.npy` output path, but the actual `np.save(...)` block is commented out.

## Implementation Notes

- `IMU_EKF/ekf_prediction.py` uses right-perturbation EKF prediction on `SE(3)` and stores the full pose trajectory plus covariance history.
- `Landmark_mapping_EKF_update/landmark_mapping_ekf.py` filters tracks by minimum observation count, optionally applies grid subsampling, triangulates new landmarks, and updates landmarks with Joseph-form covariance updates.
- `Visual-inertial SLAM/vi_slam.py` adds pose correction using stereo measurements, marginalizes landmark uncertainty into the pose update, and then updates landmarks again with the corrected pose.
- The VI-SLAM code applies a ground-vehicle style correction mask that keeps only `x`, `y`, and yaw corrections during the pose update.

## Practical Notes

- `code/main.py` is not synchronized with the current folder names. It imports `IMU_via_EKF`, but the actual directory in this repo is `IMU_EKF`.
- Because the folder name `Visual-inertial SLAM` contains a space and a hyphen, the safest way to run Part 4 is by script path: `python "Visual-inertial SLAM/run_vi_slam.py"`.
- The feature-detection runner scripts currently hardcode the absolute path `/home/tim/Desktop/UCSD/ECE276A/PR3/dataset02`. If the repository is moved, update `DATASET_DIR` in those scripts.
- Several runners call `plt.show()`. If you are running on a headless machine, switch to a non-interactive backend or comment out the display call.
