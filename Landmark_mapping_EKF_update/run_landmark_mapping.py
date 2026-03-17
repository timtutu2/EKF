"""
Landmark Mapping via EKF Update — Runner Script
================================================
ECE 276A - Project 3, Part 3

Usage (from the Landmark_mapping_EKF_update/ directory):
    python run_landmark_mapping.py

Or from the code/ directory:
    python -m Landmark_mapping_EKF_update.run_landmark_mapping

Outputs (saved next to this script):
    datasetXX_landmark_map.png   — 2D top-view:  IMU trajectory + landmark map
    datasetXX_landmark_map.npy   — dict with keys:
                                      'landmarks'   (M, 3)   world positions
                                      'Sigma_lm'    (M, 3,3) covariances
                                      'initialized' (M,)     bool mask
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — allow imports from the parent code/ directory
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.join(BASE_DIR, '..')
sys.path.insert(0, CODE_DIR)

from pr3_utils import (
    load_data,
    visualize_trajectory_2d,
)
from IMU_EKF.ekf_prediction import ekf_imu_prediction
from landmark_mapping_ekf import ekf_landmark_mapping


# ===========================================================================
# Configuration
# ===========================================================================

DATASETS = {
    'dataset00': os.path.join(BASE_DIR, '../../dataset00/dataset00.npy'),
    'dataset01': os.path.join(BASE_DIR, '../../dataset01/dataset01.npy'),
}

# IMU process-noise covariance (tuned from Part 1)
W_IMU = np.diag([1e-4, 1e-4, 1e-4,   # translational
                 1e-4, 1e-4, 1e-4])   # rotational

# EKF landmark-mapping hyperparameters
EKF_PARAMS = dict(
    V_noise          = 4.0 * np.eye(4),   # observation noise: σ = 2 px per coord
    sigma_init       = 1.0,               # initial landmark position std dev [m]
    min_observations = 3,                 # min valid stereo obs to include a landmark
    max_depth        = 150.0,             # max triangulated depth at init [m]
    min_disparity    = 1.0,               # min stereo disparity at init [px]
    outlier_threshold= 20.0,              # chi-squared gate (4 DOF); None = disable
    verbose          = True,
)


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_landmark_map(world_T_imu, landmarks, initialized, dataset_name, save_path):
    """
    2D top-view plot of the IMU trajectory and the estimated landmark map.

    Landmark uncertainty ellipses (1-σ projected to the x-y plane) are drawn
    for every initialised landmark.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- IMU trajectory ---
    traj = world_T_imu[:, :2, 3]   # (N, 2) x-y positions
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, label='IMU trajectory', zorder=3)
    ax.scatter(traj[0, 0],  traj[0, 1],  c='g', marker='s', s=60, label='start', zorder=4)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='r', marker='o', s=60, label='end',   zorder=4)

    # --- Landmark positions ---
    lm_ok = landmarks[initialized]   # (K, 3)
    ax.scatter(lm_ok[:, 0], lm_ok[:, 1],
               c='orange', s=4, alpha=0.6, label=f'landmarks ({len(lm_ok)})', zorder=2)

    ax.set_xlabel('x [m]',  fontsize=12)
    ax.set_ylabel('y [m]',  fontsize=12)
    ax.set_title(
        f'Landmark Mapping via EKF Update — {dataset_name}\n'
        f'{initialized.sum()} / {len(initialized)} landmarks initialised',
        fontsize=12,
    )
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure : {save_path}")
    return fig, ax


def plot_landmark_uncertainty(landmarks, Sigma_lm, initialized, ax):
    """
    Overlay 1-σ uncertainty ellipses (x-y plane) for each initialised landmark.
    """
    from matplotlib.patches import Ellipse

    lm_ok  = landmarks[initialized]
    sig_ok = Sigma_lm[initialized]

    plotted = 0
    for j in range(len(lm_ok)):
        cov_xy = sig_ok[j, :2, :2]

        # Eigendecomposition for ellipse parameters
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        if np.any(eigvals <= 0):
            continue

        angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width   = 2.0 * np.sqrt(eigvals[0])
        height  = 2.0 * np.sqrt(eigvals[1])

        # Only draw reasonably sized ellipses to avoid cluttering the plot
        if max(width, height) > 20.0:
            continue

        ell = Ellipse(
            xy=(lm_ok[j, 0], lm_ok[j, 1]),
            width=width, height=height, angle=angle,
            edgecolor='darkorange', facecolor='none',
            linewidth=0.3, alpha=0.4, zorder=1,
        )
        ax.add_patch(ell)
        plotted += 1

    print(f"  Drew {plotted} uncertainty ellipses.")


# ===========================================================================
# Per-dataset processing
# ===========================================================================

def process_dataset(name, data_path):
    print(f"\n{'=' * 65}")
    print(f"  Dataset : {name}")
    print(f"{'=' * 65}")

    if not os.path.exists(data_path):
        print(f"  [SKIP] File not found: {data_path}")
        return

    # --- Load data ----------------------------------------------------------
    print("  Loading data …")
    (v_t, w_t, timestamps,
     features, K_l, K_r,
     extL_T_imu, extR_T_imu) = load_data(data_path)

    N, M_feat = timestamps.shape[0], features.shape[1]
    T_span    = timestamps[-1] - timestamps[0]
    print(f"  Timesteps        : {N}  ({T_span:.1f} s)")
    print(f"  Feature tracks   : {features.shape[1]}")

    # --- Part 1: IMU EKF prediction -----------------------------------------
    print("\n  [Part 1] Running IMU EKF prediction …")
    t0 = time.time()
    world_T_imu, _ = ekf_imu_prediction(v_t, w_t, timestamps, W_noise=W_IMU)
    print(f"  Done in {time.time()-t0:.1f} s")

    # --- Part 3: Landmark mapping via EKF update ----------------------------
    print("\n  [Part 3] Running EKF landmark mapping …")
    t0 = time.time()
    landmarks, Sigma_lm, initialized = ekf_landmark_mapping(
        world_T_imu, features, K_l, K_r, extL_T_imu, extR_T_imu,
        **EKF_PARAMS,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f} s")

    # --- Summary statistics -------------------------------------------------
    n_init = initialized.sum()
    if n_init > 0:
        lm_ok = landmarks[initialized]
        print(f"\n  Landmarks initialised   : {n_init} / {M_feat}")
        print(f"  x range  : [{lm_ok[:, 0].min():.1f}, {lm_ok[:, 0].max():.1f}] m")
        print(f"  y range  : [{lm_ok[:, 1].min():.1f}, {lm_ok[:, 1].max():.1f}] m")
        print(f"  z range  : [{lm_ok[:, 2].min():.1f}, {lm_ok[:, 2].max():.1f}] m")
        median_std = np.sqrt(np.median(Sigma_lm[initialized, 0, 0]))
        print(f"  Median σ_x : {median_std:.3f} m")
    else:
        print("  WARNING: No landmarks were initialised!")

    # --- Save results -------------------------------------------------------
    result_path = os.path.join(BASE_DIR, f'{name}_landmark_map.npy')
    np.save(result_path, {
        'landmarks':   landmarks,
        'Sigma_lm':    Sigma_lm,
        'initialized': initialized,
    })
    print(f"\n  Saved results : {result_path}")

    # --- Visualisation ------------------------------------------------------
    fig_path = os.path.join(BASE_DIR, f'{name}_landmark_map.png')
    fig, ax  = plot_landmark_map(
        world_T_imu, landmarks, initialized, name, fig_path
    )
    if n_init > 0:
        plot_landmark_uncertainty(landmarks, Sigma_lm, initialized, ax)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    for ds_name, ds_path in DATASETS.items():
        process_dataset(ds_name, ds_path)

    print("\nAll datasets processed.")
