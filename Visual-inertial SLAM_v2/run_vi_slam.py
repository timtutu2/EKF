"""
Runner for the Visual-inertial SLAM EKF in Visual-inertial SLAM_v2.

Run from the code directory with:
    python "Visual-inertial SLAM_v2/run_vi_slam.py"
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, CODE_DIR)

from IMU_EKF.ekf_prediction import ekf_imu_prediction
from pr3_utils import load_data
from vi_slam import vi_slam_ekf


DATASETS = {
    "dataset00": os.path.join(BASE_DIR, "../../dataset00/dataset00.npy"),
    "dataset01": os.path.join(BASE_DIR, "../../dataset01/dataset01.npy"),
}

W_IMU = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])

VI_SLAM_PARAMS = dict(
    W_noise=W_IMU,
    V_noise=4.0 * np.eye(4),
    sigma_init=1.0,
    min_observations=3,
    max_depth=150.0,
    min_disparity=1.0,
    outlier_threshold=20.0,
    max_lm_per_step=200,
    verbose=True,
)


def plot_results(
    world_T_imu_slam: np.ndarray,
    world_T_imu_imu: np.ndarray,
    landmarks: np.ndarray,
    initialized: np.ndarray,
    dataset_name: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 9))

    traj_imu = world_T_imu_imu[:, :2, 3]
    traj_slam = world_T_imu_slam[:, :2, 3]

    ax.plot(
        traj_imu[:, 0],
        traj_imu[:, 1],
        color="0.6",
        linestyle="--",
        linewidth=1.0,
        label="IMU-only",
    )
    ax.plot(
        traj_slam[:, 0],
        traj_slam[:, 1],
        color="steelblue",
        linewidth=1.8,
        label="VI-SLAM",
    )
    ax.scatter(traj_slam[0, 0], traj_slam[0, 1], c="green", marker="s", s=70, label="start")
    ax.scatter(traj_slam[-1, 0], traj_slam[-1, 1], c="red", marker="o", s=70, label="end")

    if np.any(initialized):
        lm = landmarks[initialized]
        ax.scatter(lm[:, 0], lm[:, 1], c="orange", s=3, alpha=0.5, label=f"landmarks ({lm.shape[0]})")

    ax.set_title(f"Visual-inertial SLAM - {dataset_name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_dataset(name: str, data_path: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"Dataset: {name}")
    print(f"{'=' * 70}")

    if not os.path.exists(data_path):
        print(f"Missing dataset file: {data_path}")
        return

    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(data_path)

    print("Running IMU-only prediction...")
    t0 = time.time()
    world_T_imu_imu, _ = ekf_imu_prediction(v_t, w_t, timestamps, W_noise=W_IMU)
    print(f"  IMU-only done in {time.time() - t0:.1f} s")

    print("Running visual-inertial SLAM...")
    t0 = time.time()
    world_T_imu_slam, Sigma_T, landmarks, Sigma_lm, initialized = vi_slam_ekf(
        v_t,
        w_t,
        timestamps,
        features,
        K_l,
        K_r,
        extL_T_imu,
        extR_T_imu,
        **VI_SLAM_PARAMS,
    )
    print(f"  VI-SLAM done in {time.time() - t0:.1f} s")

    imu_final = world_T_imu_imu[-1, :3, 3]
    slam_final = world_T_imu_slam[-1, :3, 3]
    print(f"  Final IMU-only position: {imu_final}")
    print(f"  Final VI-SLAM position:  {slam_final}")
    print(f"  Final initialized landmarks: {initialized.sum()} / {features.shape[1]}")
    print(f"  Final pose covariance diagonal: {np.diag(Sigma_T)}")

    result_path = os.path.join(BASE_DIR, f"{name}_vi_slam_v2.npy")
    np.save(
        result_path,
        {
            "world_T_imu": world_T_imu_slam,
            "Sigma_T": Sigma_T,
            "landmarks": landmarks,
            "Sigma_lm": Sigma_lm,
            "initialized": initialized,
        },
    )
    print(f"  Saved results to {result_path}")

    plot_path = os.path.join(BASE_DIR, f"{name}_vi_slam_v2.png")
    plot_results(
        world_T_imu_slam,
        world_T_imu_imu,
        landmarks,
        initialized,
        name,
        plot_path,
    )
    print(f"  Saved plot to {plot_path}")


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASETS.items():
        process_dataset(dataset_name, dataset_path)
