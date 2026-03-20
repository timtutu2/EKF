"""
Visual-Inertial SLAM — Runner Script
=====================================
ECE 276A - Project 3, Part 4

Usage (from the code/ directory):
    python -m "Visual-inertial SLAM.run_vi_slam"

Or from the Visual-inertial SLAM/ directory:
    python run_vi_slam.py

Outputs (saved next to this script):
    datasetXX_vi_slam.png    — 2D top-view: VI-SLAM trajectory + landmark map
                               (also overlays the pure-IMU trajectory for comparison)
    datasetXX_vi_slam.npy    — dict with keys:
                                   'world_T_imu'  (N, 4, 4)  corrected trajectory
                                   'landmarks'    (M, 3)     landmark positions
                                   'Sigma_lm'     (M, 3, 3)  landmark covariances
                                   'initialized'  (M,)       bool mask
"""

from configparser import MAX_INTERPOLATION_DEPTH
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, CODE_DIR)

from pr3_utils import load_data, visualize_trajectory_2d
from IMU_EKF.ekf_prediction import ekf_imu_prediction
from vi_slam import vi_slam_ekf


# ===========================================================================
# Configuration
# ===========================================================================

DATASETS = {
    'dataset00': os.path.join(BASE_DIR, '../../dataset00/dataset00.npy'),
    'dataset01': os.path.join(BASE_DIR, '../../dataset01/dataset01.npy'),
    'dataset02': os.path.join(BASE_DIR, '../../dataset02/dataset02.npy'),
}

# IMU process-noise covariance  [translational, rotational]
W_IMU = np.diag([1e-4, 1e-4, 1e-4,   # translational noise variance [m²/s²·s²]
                 1e-4, 1e-4, 1e-4])   # rotational noise variance    [rad²/s²·s²]

# VI-SLAM hyper-parameters — tune these for best results
VI_SLAM_PARAMS = dict(
    W_noise          = W_IMU,
    V_noise          = 4.0 * np.eye(4),  # stereo obs. noise: σ = 2 px per coord
    sigma_init       = 1.0,              # initial landmark position std dev [m]
    min_observations = 3,                # min stereo obs. to include a landmark
    lm_grid          = (20, 15),         # spatial grid (rows×cols): 1 best track/cell
    max_depth        = 150.0,            # max DLT triangulation depth [m]
    min_disparity    = 1.0,              # min stereo disparity at init [px]
    outlier_threshold= 20.0,             # chi-squared gate (4 DOF); None = off
    max_lm_per_step  = 200,              # max landmarks used per pose-update step
    verbose          = True,
)


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_vi_slam_results(
    world_T_imu_slam,
    world_T_imu_imu,
    landmarks,
    initialized,
    dataset_name,
    save_path,
):
    """
    2D top-view of:
      - Pure-IMU trajectory (grey dashed)
      - VI-SLAM corrected trajectory (blue)
      - Estimated landmark map (orange dots)
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Pure-IMU trajectory (reference)
    traj_imu  = world_T_imu_imu[:, :2, 3]
    ax.plot(traj_imu[:, 0], traj_imu[:, 1],
            color='grey', linewidth=1.0, linestyle='--',
            alpha=0.6, label='IMU-only trajectory', zorder=2)

    # VI-SLAM trajectory
    traj_slam = world_T_imu_slam[:, :2, 3]
    ax.plot(traj_slam[:, 0], traj_slam[:, 1],
            color='steelblue', linewidth=1.8, label='VI-SLAM trajectory', zorder=3)
    ax.scatter(traj_slam[0, 0],  traj_slam[0, 1],
               c='green', marker='s', s=80, zorder=5, label='start')
    ax.scatter(traj_slam[-1, 0], traj_slam[-1, 1],
               c='red',   marker='o', s=80, zorder=5, label='end')

    # Landmark map
    if initialized.any():
        lm_ok = landmarks[initialized]
        ax.scatter(lm_ok[:, 0], lm_ok[:, 1],
                   c='orange', s=3, alpha=0.5,
                   label=f'landmarks ({initialized.sum()})', zorder=1)

    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title(
        f'Visual-Inertial SLAM — {dataset_name}\n'
        f'{initialized.sum()} landmarks initialised',
        fontsize=13,
    )
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure : {save_path}")
    return fig, ax


def plot_trajectory_comparison(
    world_T_imu_slam,
    world_T_imu_imu,
    dataset_name,
    save_path,
):
    """Side-by-side comparison of IMU-only vs VI-SLAM trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, traj, title, colour in [
        (axes[0], world_T_imu_imu,  'IMU-only EKF prediction', 'grey'),
        (axes[1], world_T_imu_slam, 'VI-SLAM (corrected)',      'steelblue'),
    ]:
        xy = traj[:, :2, 3]
        ax.plot(xy[:, 0], xy[:, 1], color=colour, linewidth=1.5)
        ax.scatter(xy[0, 0],  xy[0, 1],  c='green', marker='s', s=60, zorder=5)
        ax.scatter(xy[-1, 0], xy[-1, 1], c='red',   marker='o', s=60, zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Trajectory Comparison — {dataset_name}', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved comparison : {save_path}")
    return fig


# ===========================================================================
# Per-dataset processing
# ===========================================================================

def process_dataset(name: str, data_path: str):
    print(f"\n{'=' * 70}")
    print(f"  Dataset : {name}")
    print(f"{'=' * 70}")

    if not os.path.exists(data_path):
        print(f"  [SKIP] Data file not found: {data_path}")
        return

    # --- Load data ----------------------------------------------------------
    print("  Loading data …")
    (v_t, w_t, timestamps,
     features, K_l, K_r,
     extL_T_imu, extR_T_imu) = load_data(data_path)

    N      = timestamps.shape[0]
    M_feat = features.shape[1]
    T_span = timestamps[-1] - timestamps[0]
    print(f"  Timesteps      : {N}  ({T_span:.1f} s)")
    print(f"  Feature tracks : {M_feat}")

    # --- Pure IMU EKF (for comparison) ------------------------------------
    print("\n  [Reference] Running pure-IMU EKF prediction …")
    t0 = time.time()
    world_T_imu_imu, _ = ekf_imu_prediction(
        v_t, w_t, timestamps, W_noise=W_IMU
    )
    print(f"  Done in {time.time() - t0:.1f} s")

    # --- VI-SLAM -----------------------------------------------------------
    print("\n  [Part 4] Running VI-SLAM EKF …")
    t0 = time.time()
    world_T_imu_slam, Sigma_T, landmarks, Sigma_lm, initialized = vi_slam_ekf(
        v_t, w_t, timestamps, features, K_l, K_r,
        extL_T_imu, extR_T_imu,
        **VI_SLAM_PARAMS,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f} s")

    # --- Summary -----------------------------------------------------------
    n_init = initialized.sum()
    print(f"\n  Landmarks initialised  : {n_init} / {M_feat}")
    if n_init > 0:
        lm_ok = landmarks[initialized]
        print(f"  x range : [{lm_ok[:,0].min():.1f}, {lm_ok[:,0].max():.1f}] m")
        print(f"  y range : [{lm_ok[:,1].min():.1f}, {lm_ok[:,1].max():.1f}] m")
        print(f"  z range : [{lm_ok[:,2].min():.1f}, {lm_ok[:,2].max():.1f}] m")

    imu_final  = world_T_imu_imu[-1, :3, 3]
    slam_final = world_T_imu_slam[-1, :3, 3]
    print(f"\n  Final IMU-only pos  : x={imu_final[0]:.2f}  y={imu_final[1]:.2f}  z={imu_final[2]:.2f} m")
    print(f"  Final VI-SLAM  pos  : x={slam_final[0]:.2f}  y={slam_final[1]:.2f}  z={slam_final[2]:.2f} m")

    # --- Save results ------------------------------------------------------
    result_path = os.path.join(BASE_DIR, f'{name}_vi_slam_{VI_SLAM_PARAMS["lm_grid"]}.npy')
    # np.save(result_path, {
    #     'world_T_imu':  world_T_imu_slam,
    #     'landmarks':    landmarks,
    #     'Sigma_lm':     Sigma_lm,
    #     'initialized':  initialized,
    # })
    print(f"\n  Saved results : {result_path}")

    # --- Plots -------------------------------------------------------------
    # Main plot: VI-SLAM trajectory + landmark map
    fig1_path = os.path.join(BASE_DIR, f'{name}_vi_slam_{VI_SLAM_PARAMS["lm_grid"]}_dt_0.png')
    fig1, _ = plot_vi_slam_results(
        world_T_imu_slam, world_T_imu_imu,
        landmarks, initialized, name, fig1_path,
    )

    # Comparison plot: IMU-only vs VI-SLAM side by side
    fig2_path = os.path.join(BASE_DIR, f'{name}_trajectory_comparison_{VI_SLAM_PARAMS["lm_grid"]}_dt_0.png')
    plot_trajectory_comparison(
        world_T_imu_slam, world_T_imu_imu, name, fig2_path,
    )

    plt.show()
    plt.close('all')


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == '__main__':
    for ds_name, ds_path in DATASETS.items():
        process_dataset(ds_name, ds_path)

    print("\nAll datasets processed.")
