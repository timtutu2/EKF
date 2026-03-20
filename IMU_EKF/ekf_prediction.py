import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow imports from the parent code directory (pr3_utils.py lives there)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pr3_utils import (
    load_data,
    visualize_trajectory_2d,
    axangle2twist,   # [v; ω]  →  4×4 se(3) twist matrix
    twist2pose,      # 4×4 se(3) twist matrix  →  4×4 SE(3) matrix (matrix exp)
    pose2adpose,     # 4×4 SE(3)  →  6×6 adjoint Ad(T)
    inversePose,     # 4×4 SE(3)  →  its inverse in SE(3)
)


# ---------------------------------------------------------------------------
# Core EKF prediction function
# ---------------------------------------------------------------------------

def ekf_imu_prediction(v_t: np.ndarray,
                       w_t: np.ndarray,
                       timestamps: np.ndarray,
                       W_noise: np.ndarray = None) -> tuple:
    N = v_t.shape[0]

    # --- Default process noise covariance -----------------------------------
    # Tuned values: small but non-zero to model IMU noise.
    # Translational noise σ_v ≈ 0.01 m/s  →  σ_v^2 ≈ 1e-4
    # Rotational noise    σ_ω ≈ 0.01 rad/s →  σ_ω^2 ≈ 1e-4
    a=1e-4
    if W_noise is None:
        W_noise = np.diag([a, a, a,   # translational
                            a, a, a])   # rotational

    world_T_imu = np.zeros((N, 4, 4))
    Sigma       = np.zeros((N, 6, 6))

    #Initial conditions
    world_T_imu[0] = np.eye(4)     # World frame coincides with IMU at t=0
    Sigma[0]       = np.zeros((6, 6))  # Initial pose perfectly known

    #Prediction loop
    for t in range(N - 1):
        # Time step duration
        dt = timestamps[t + 1] - timestamps[t]

        # Body-frame twist:  u_t = [v_t; ω_t]
        u_t = np.concatenate([v_t[t], w_t[t]])   # shape (6,)

        # Scale twist by Δt  →  u_t · Δt  ∈  se(3)
        u_dt = u_t * dt                            # shape (6,)

        # Matrix form of the twist:  û_{Δt} ∈ se(3)  (4×4 skew-symmetric block)
        twist_mat = axangle2twist(u_dt)            # shape (4, 4)

        # Motion increment:  Φ_t = exp(û_t · Δt)  ∈  SE(3)
        Phi_t = twist2pose(twist_mat)              # shape (4, 4)

        # ---- Mean update -----------------------------------------------
        # T_{t+1} = T_t · Φ_t
        world_T_imu[t + 1] = world_T_imu[t] @ Phi_t

        # ---- Covariance update -----------------------------------------
        # F_t = Ad(Φ_t^{-1}) = Ad(exp(-û_t · Δt))
        Phi_t_inv = inversePose(Phi_t)             # shape (4, 4)  [SE(3) inverse]
        F_t       = pose2adpose(Phi_t_inv)         # shape (6, 6)  [adjoint]

        # Σ_{t+1} = F_t · Σ_t · F_t^T + W
        Sigma[t + 1] = F_t @ Sigma[t] @ F_t.T + W_noise

    return world_T_imu, Sigma


# ---------------------------------------------------------------------------
# Main script: run on both datasets and save trajectory plots
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Paths are relative to this script's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    datasets = {
        'dataset00': os.path.join(BASE_DIR, '../../dataset00/dataset00.npy'),
        'dataset01': os.path.join(BASE_DIR, '../../dataset01/dataset01.npy'),
        'dataset02': os.path.join(BASE_DIR, '../../dataset02/dataset02.npy'),
    }

    # Process noise covariance (tuned for good trajectory estimation)
    # Each diagonal entry is the variance of the corresponding noise component.
    W_noise = np.diag([
        1e-4, 1e-4, 1e-4,   # translational noise variance [m²/s²·s²]
        1e-4, 1e-4, 1e-4    # rotational noise variance    [rad²/s²·s²]
    ])

    for name, data_path in datasets.items():
        print(f"\n{'='*60}")
        print(f"  Dataset : {name}")
        print(f"{'='*60}")

        if not os.path.exists(data_path):
            print(f"  [SKIP] Data file not found: {data_path}")
            continue

        # Load data
        v_t, w_t, timestamps, _, K_l, K_r, extL_T_imu, extR_T_imu = \
            load_data(data_path)
        
        N = timestamps.shape[0]
        duration = timestamps[-1] - timestamps[0] #last -first  
        avg_dt = duration / (N - 1)

        print(f"  Timesteps : {N}")
        print(f"  Duration  : {duration:.2f} s  (avg Δt = {avg_dt*1000:.2f} ms)")
        print(f"  v_t shape : {v_t.shape}")
        print(f"  w_t shape : {w_t.shape}")

        # Run EKF prediction
        print(f"\n  Running EKF prediction ...")
        world_T_imu, Sigma = ekf_imu_prediction(v_t, w_t, timestamps, W_noise=W_noise)

        # Summary statistics
        final_pos = world_T_imu[-1, :3, 3]
        final_std = np.sqrt(np.diag(Sigma[-1]))
        print(f"  Final position  : x={final_pos[0]:.3f}  y={final_pos[1]:.3f}  z={final_pos[2]:.3f} [m]")
        print(f"  Final std (pos) : {final_std[:3]}")
        print(f"  Final std (rot) : {final_std[3:]}")

        # ---- Visualization ------------------------------------------------
        fig, ax = visualize_trajectory_2d(
            world_T_imu,
            path_name=f"{name} — EKF IMU prediction",
            show_ori=True
        )
        ax.set_title(
            f'IMU Localization via EKF Prediction\n'
            f'{name}  |  duration={duration:.1f}s  |  N={N} steps',
            fontsize=11
        )
        plt.tight_layout()

        # Save figure next to this script
        fig_path = os.path.join(BASE_DIR, f'{name}_ekf_prediction.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  Trajectory plot : {fig_path}")

        plt.show()

    print("\nAll datasets processed.")
