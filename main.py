import numpy as np
import matplotlib.pyplot as plt
from pr3_utils import *
from IMU_via_EKF.ekf_prediction import ekf_imu_prediction


if __name__ == '__main__':

    # Load the measurements
    filename = "../dataset00/dataset00.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filename)

    # (a) IMU Localization via EKF Prediction
    # Process noise covariance W (6×6): [v_x, v_y, v_z, ω_x, ω_y, ω_z] variances
    W_noise = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    world_T_imu, Sigma = ekf_imu_prediction(v_t, w_t, timestamps, W_noise=W_noise)

    # Visualize the IMU trajectory
    fig, ax = visualize_trajectory_2d(world_T_imu,
                                      path_name="dataset00 EKF IMU prediction",
                                      show_ori=True)
    ax.set_title("IMU Localization via EKF Prediction — dataset00")
    plt.tight_layout()
    plt.savefig("dataset00_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()

    # (b) Landmark Mapping via EKF Update
    # TODO: implement in Part 3

    # (c) Visual-Inertial SLAM
    # TODO: implement in Part 4

