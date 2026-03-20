"""
Visual-Inertial SLAM via EKF
============================
ECE 276A - Project 3, Part 4

Combines the IMU EKF prediction (Part 1) with a joint EKF update step that
simultaneously corrects both the IMU pose T_t ∈ SE(3) and the 3-D landmark
positions m_j ∈ R^3 using stereo-camera observations.

Mathematical Background
-----------------------
State:
    T_t  ∈ SE(3)    — IMU pose (mean),  Σ_T  ∈ R^{6×6}  covariance
    m_j  ∈ R^3      — landmark positions, Σ_j ∈ R^{3×3}

Prediction (same as Part 1):
    T̄_{t+1}  = T̄_t · exp(û_t · Δt)
    Σ_T{t+1} = F_t · Σ_T_t · F_t^T + W,   F_t = Ad(exp(-û_t·Δt))

Pose Update (information-filter form, marginalising landmark uncertainty):

    For each observed landmark j, the effective observation noise is:
        V_eff_j = V + H_m_j · Σ_j · H_m_j^T          (4×4)

    Information accumulation (summed over all K_v valid observations):
        Ω_T ← Σ_T^{-1} + Σ_j  H_T_j^T · V_eff_j^{-1} · H_T_j
        b   ← Σ_j  H_T_j^T · V_eff_j^{-1} · innov_j

    Pose correction:
        Σ_T  ← Ω_T^{-1}
        δξ   ← Σ_T · b
        T̄_t  ← T̄_t · exp(δξ̂)

Landmark Update (standard per-landmark EKF, Joseph form; with corrected pose):
    Same as Part 3 — see landmark_mapping_ekf.py for details.

Pose Jacobian  H_T_j  (4×6):
    Under right-perturbation  T_t = T̄_t · exp(ξ̂):
        T_imu_world(ξ) = exp(-ξ̂) · T̄_imu_world

    extL_T_imu (= _IT_L) maps FROM the left regular-camera frame TO the IMU frame.
    Let oTr be the regular→optical rotation, and let:
        oT_L = oTr @ inv(extL_T_imu)   (IMU → optical left camera)

    Let q_imu = T̄_imu_world · m_h   (landmark in IMU frame).
    Then ∂p_L/∂ξ = -oT_L · q⊙   where q⊙ is the 4×6 "odot" operator:

        q⊙ = [[q4,  0,  0,   0,  q3, -q2],
               [ 0, q4,  0, -q3,   0,  q1],
               [ 0,  0, q4,  q2, -q1,   0],
               [ 0,  0,  0,   0,   0,   0]]

    H_T[:2, :] = (∂z_L/∂p_L) · (-oT_L · q⊙)
    H_T[2:, :] = (∂z_R/∂p_R) · (-oT_R · q⊙)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pr3_utils import (
    axangle2twist,
    twist2pose,
    pose2adpose,
    inversePose,
)

# Rotation from the "regular" camera frame (same axes as IMU: x=fwd, y=left, z=up)
# to the optical camera frame (x=right, y=down, z=fwd) used by the K matrix.
_oTr = np.array([[0., -1.,  0., 0.],
                 [0.,  0., -1., 0.],
                 [1.,  0.,  0., 0.],
                 [0.,  0.,  0., 1.]])
from Landmark_mapping_EKF_update.landmark_mapping_ekf import (
    triangulate_batch,
    observations_batch,
    jacobians_batch,
    _grid_subsample,
)


# ---------------------------------------------------------------------------
# Odot operator  q⊙  (4×6):  ξ̂ · q  =  q⊙ · ξ
# ---------------------------------------------------------------------------

def odot_operator_batch(q_batch: np.ndarray) -> np.ndarray:
    """
    Compute the 4×6 "odot" matrix Q such that  ξ̂ · q  =  Q · ξ.

    For a homogeneous point q = [q1, q2, q3, q4]:
        Q = [[q4,  0,  0,   0,  q3, -q2],
             [ 0, q4,  0, -q3,   0,  q1],
             [ 0,  0, q4,  q2, -q1,   0],
             [ 0,  0,  0,   0,   0,   0]]

    Parameters
    ----------
    q_batch : (K, 4)  batch of homogeneous points

    Returns
    -------
    Q : (K, 4, 6)
    """
    K = q_batch.shape[0]
    q1 = q_batch[:, 0]
    q2 = q_batch[:, 1]
    q3 = q_batch[:, 2]
    q4 = q_batch[:, 3]

    Q = np.zeros((K, 4, 6))
    # Row 0: [q4, 0, 0, 0, q3, -q2]
    Q[:, 0, 0] = q4;  Q[:, 0, 4] = q3;  Q[:, 0, 5] = -q2
    # Row 1: [0, q4, 0, -q3, 0, q1]
    Q[:, 1, 1] = q4;  Q[:, 1, 3] = -q3; Q[:, 1, 5] = q1
    # Row 2: [0, 0, q4, q2, -q1, 0]
    Q[:, 2, 2] = q4;  Q[:, 2, 3] = q2;  Q[:, 2, 4] = -q1
    # Row 3: all zeros

    return Q


# ---------------------------------------------------------------------------
# Pose Jacobian  H_T  (K, 4, 6):  ∂z / ∂ξ
# ---------------------------------------------------------------------------

def pose_jacobians_batch(
    m_batch: np.ndarray,
    T_imu_world: np.ndarray,
    extL_T_imu: np.ndarray,
    extR_T_imu: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
) -> np.ndarray:
    """
    4×6 Jacobian of the stereo observation w.r.t. the pose perturbation ξ
    (right-perturbation: T = T̄ · exp(ξ̂)).

    Parameters
    ----------
    m_batch     : (K, 3)  landmark world-frame positions
    T_imu_world : (4, 4)  current estimate of world → IMU transform
    extL_T_imu  : (4, 4)  left camera regular frame → IMU  (_IT_L)
    extR_T_imu  : (4, 4)  right camera regular frame → IMU (_IT_R)
    K_l, K_r    : (3, 3)  camera intrinsics

    Returns
    -------
    H_T : (K, 4, 6)
    """
    K = m_batch.shape[0]
    m_h = np.hstack([m_batch, np.ones((K, 1))])   # (K, 4)

    # Landmark in IMU frame:  q_imu = T̄_imu_world · m_h
    q_imu = (T_imu_world @ m_h.T).T               # (K, 4)

    # IMU → optical camera transforms  (using _oTr and inverse of the stored extrinsics)
    oT_L = _oTr @ inversePose(extL_T_imu)   # (4, 4)
    oT_R = _oTr @ inversePose(extR_T_imu)   # (4, 4)

    # Landmark in optical camera frames (for projection Jacobians)
    p_L = (oT_L @ q_imu.T).T                      # (K, 4)
    p_R = (oT_R @ q_imu.T).T                      # (K, 4)

    # Odot operator  Q = q_imu⊙  (K, 4, 6)
    Q = odot_operator_batch(q_imu)

    # ∂p_L/∂ξ = -oT_L · Q
    dp_L = -np.einsum('ij,kjl->kil', oT_L, Q)     # (K, 4, 6)
    dp_R = -np.einsum('ij,kjl->kil', oT_R, Q)     # (K, 4, 6)

    # ∂z_L/∂p_L  (K, 2, 4)
    dz_L = np.zeros((K, 2, 4))
    dz_L[:, 0, 0] =  K_l[0, 0] / p_L[:, 2]
    dz_L[:, 0, 2] = -K_l[0, 0] * p_L[:, 0] / p_L[:, 2] ** 2
    dz_L[:, 1, 1] =  K_l[1, 1] / p_L[:, 2]
    dz_L[:, 1, 2] = -K_l[1, 1] * p_L[:, 1] / p_L[:, 2] ** 2

    # ∂z_R/∂p_R  (K, 2, 4)
    dz_R = np.zeros((K, 2, 4))
    dz_R[:, 0, 0] =  K_r[0, 0] / p_R[:, 2]
    dz_R[:, 0, 2] = -K_r[0, 0] * p_R[:, 0] / p_R[:, 2] ** 2
    dz_R[:, 1, 1] =  K_r[1, 1] / p_R[:, 2]
    dz_R[:, 1, 2] = -K_r[1, 1] * p_R[:, 1] / p_R[:, 2] ** 2

    # Chain rule:  H_T = [dz_L · dp_L ; dz_R · dp_R]
    H_T = np.zeros((K, 4, 6))
    H_T[:, :2, :] = dz_L @ dp_L   # (K, 2, 4) @ (K, 4, 6) → (K, 2, 6)
    H_T[:, 2:, :] = dz_R @ dp_R   # (K, 2, 4) @ (K, 4, 6) → (K, 2, 6)

    return H_T


# ---------------------------------------------------------------------------
# Main VI-SLAM EKF
# ---------------------------------------------------------------------------

def vi_slam_ekf(
    v_t: np.ndarray,
    w_t: np.ndarray,
    timestamps: np.ndarray,
    features: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
    extL_T_imu: np.ndarray,
    extR_T_imu: np.ndarray,
    W_noise: np.ndarray = None,
    V_noise: np.ndarray = None,
    sigma_init: float = 1.0,
    min_observations: int = 3,
    lm_grid: tuple = (20, 15),
    max_depth: float = 150.0,
    min_disparity: float = 1.0,
    outlier_threshold: float = 20.0,
    max_lm_per_step: int = 200,
    verbose: bool = True,
) -> tuple:
    """
    Full EKF Visual-Inertial SLAM.

    The algorithm performs at every timestep t:
    1. **Predict** — propagate IMU pose and covariance using SE(3) kinematics.
    2. **Initialise** — triangulate newly seen landmarks via DLT stereo.
    3. **Pose update** — correct pose via information-filter accumulation over
       all valid observations (landmark uncertainty marginalised into V_eff).
    4. **Landmark update** — standard per-landmark EKF using the corrected pose
       (same as Part 3, but using the updated T_cam).

    Parameters
    ----------
    v_t, w_t     : (N, 3)  IMU linear / angular velocity [body frame]
    timestamps   : (N,)    UNIX timestamps [s]
    features     : (4, M, N)  stereo pixel observations; -1 for missing
    K_l, K_r     : (3, 3)  camera intrinsics
    extL_T_imu   : (4, 4)  left camera regular frame → IMU  (_IT_L)
    extR_T_imu   : (4, 4)  right camera regular frame → IMU (_IT_R)
    W_noise      : (6, 6)  IMU process noise covariance
    V_noise      : (4, 4)  stereo observation noise covariance
    sigma_init   : float   initial landmark position std dev [m]
    min_observations : int min valid stereo obs to include a landmark
    lm_grid      : (rows, cols) | None  spatial grid for landmark subsampling;
                                   one best-observed track kept per image cell
    max_depth    : float   max triangulation depth at init [m]
    min_disparity: float   min stereo disparity at init [px]
    outlier_threshold : float  chi-squared gate (4 DOF); None disables
    max_lm_per_step   : int    cap on landmarks used per pose-update step
    verbose      : bool    print progress

    Returns
    -------
    world_T_imu : (N, 4, 4)  corrected IMU trajectory
    Sigma_T     : (6, 6)     final pose covariance
    landmarks   : (M, 3)     estimated landmark world positions (NaN if uninit)
    Sigma_lm    : (M, 3, 3)  per-landmark covariances
    initialized : (M,)       bool mask of successfully initialised landmarks
    """
    N = timestamps.shape[0]
    M = features.shape[1]

    # ---- Default noise parameters ----------------------------------------
    if W_noise is None:
        W_noise = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    if V_noise is None:
        V_noise = 4.0 * np.eye(4)

    Sigma0 = sigma_init ** 2 * np.eye(3)

    # ---- Pre-filter landmarks: min-observation count + spatial grid ----------
    valid_obs = np.all(features >= 0, axis=0)   # (M, N) — True when all 4 coords ≥ 0
    obs_count = valid_obs.sum(axis=1)            # (M,)
    keep_lm   = obs_count >= min_observations    # (M,)

    # Spatial grid subsampling: one best-observed track per image-space cell.
    if lm_grid is not None:
        keep_lm = _grid_subsample(features, valid_obs, obs_count, keep_lm, grid=lm_grid)

    if verbose:
        print(f"  Total landmarks             : {M}")
        print(f"  After min-obs filter (≥{min_observations}) : {(obs_count >= min_observations).sum()}")
        if lm_grid is not None:
            print(f"  After grid subsampling {lm_grid}  : {keep_lm.sum()}")

    # ---- Allocate storage ------------------------------------------------
    world_T_imu = np.zeros((N, 4, 4))
    world_T_imu[0] = np.eye(4)
    Sigma_T = np.zeros((6, 6))               # running pose covariance

    landmarks   = np.full((M, 3), np.nan)
    Sigma_lm    = np.tile(Sigma0, (M, 1, 1))
    initialized = np.zeros(M, dtype=bool)

    # Pre-compute IMU→optical-camera transforms (fixed calibration, not time-varying)
    oT_L = _oTr @ inversePose(extL_T_imu)   # IMU → optical left-cam  (4, 4)
    oT_R = _oTr @ inversePose(extR_T_imu)   # IMU → optical right-cam (4, 4)

    # ---- Main loop -------------------------------------------------------
    for t in range(N):

        # ------------------------------------------------------------------
        # Step 1: EKF Prediction
        # ------------------------------------------------------------------
        if t > 0:
            dt    = timestamps[t] - timestamps[t - 1]
            u_dt  = np.concatenate([v_t[t - 1], w_t[t - 1]]) * dt
            Phi_t = twist2pose(axangle2twist(u_dt))          # SE(3)

            world_T_imu[t] = world_T_imu[t - 1] @ Phi_t

            F_t    = pose2adpose(inversePose(Phi_t))         # Ad(Φ^{-1})
            Sigma_T = F_t @ Sigma_T @ F_t.T + W_noise

        # Camera transforms for this timestep
        T_imu_world = inversePose(world_T_imu[t])
        T_cam_L     = oT_L @ T_imu_world                    # world → optical left cam
        T_cam_R     = oT_R @ T_imu_world                    # world → optical right cam
        P_L         = K_l @ T_cam_L[:3, :]
        P_R         = K_r @ T_cam_R[:3, :]

        obs_at_t = valid_obs[:, t] & keep_lm

        # ------------------------------------------------------------------
        # Step 2: Initialise new landmarks via DLT triangulation
        # ------------------------------------------------------------------
        new_idx = np.where(obs_at_t & ~initialized)[0]
        if len(new_idx) > 0:
            lx_n = features[0, new_idx, t]
            ly_n = features[1, new_idx, t]
            rx_n = features[2, new_idx, t]
            ry_n = features[3, new_idx, t]

            disp = lx_n - rx_n
            good = disp >= min_disparity

            if good.any():
                ni    = new_idx[good]
                m_new = triangulate_batch(
                    lx_n[good], ly_n[good], rx_n[good], ry_n[good], P_L, P_R
                )

                m_h      = np.hstack([m_new, np.ones((len(ni), 1))])
                p_check  = (T_cam_L @ m_h.T).T
                depth_ok = (p_check[:, 2] > 0) & (p_check[:, 2] < max_depth)

                if depth_ok.any():
                    ni_ok              = ni[depth_ok]
                    landmarks[ni_ok]   = m_new[depth_ok]
                    Sigma_lm[ni_ok]    = Sigma0
                    initialized[ni_ok] = True

        # ------------------------------------------------------------------
        # Step 3: Gather observations of initialised landmarks
        # ------------------------------------------------------------------
        upd_idx = np.where(obs_at_t & initialized)[0]
        if len(upd_idx) == 0:
            if verbose and (t % 500 == 0 or t == N - 1):
                print(f"  t={t:5d}/{N}  init={initialized.sum():5d}  no updates")
            continue

        # Prioritise landmarks with more observations (better estimates)
        if len(upd_idx) > max_lm_per_step:
            counts = obs_count[upd_idx]
            sel    = np.argsort(-counts)[:max_lm_per_step]
            upd_idx = upd_idx[sel]

        m_b     = landmarks[upd_idx]              # (K, 3)
        z_obs_b = features[:, upd_idx, t].T       # (K, 4)

        # Predicted observations and valid-projection mask
        z_hat, valid_proj = observations_batch(m_b, T_cam_L, T_cam_R, K_l, K_r)
        if not valid_proj.any():
            continue

        vi    = np.where(valid_proj)[0]
        idx_v = upd_idx[vi]
        m_v   = m_b[vi]
        z_h_v = z_hat[vi]
        z_o_v = z_obs_b[vi]
        innov = z_o_v - z_h_v                      # (Kv, 4)

        # ------------------------------------------------------------------
        # Drop landmarks whose state is numerically corrupted (NaN / Inf)
        # ------------------------------------------------------------------
        Sig_m_v_raw = Sigma_lm[idx_v]                             # (Kv, 3, 3)
        finite_mask = (
            np.all(np.isfinite(m_v), axis=1)
            & np.all(np.isfinite(Sig_m_v_raw.reshape(len(idx_v), -1)), axis=1)
        )
        if not finite_mask.all():
            # Mark corrupted landmarks for reinitialisation
            bad_idx = idx_v[~finite_mask]
            landmarks[bad_idx]   = np.nan
            initialized[bad_idx] = False
            # Keep only the good subset
            vi    = vi[finite_mask];    idx_v = idx_v[finite_mask]
            m_v   = m_v[finite_mask];   innov = innov[finite_mask]
            z_o_v = z_o_v[finite_mask]
        if len(idx_v) == 0:
            continue

        # Jacobians at predicted pose
        H_m   = jacobians_batch(m_v, T_cam_L, T_cam_R, K_l, K_r)          # (Kv, 4, 3)
        H_T_m = pose_jacobians_batch(m_v, T_imu_world, extL_T_imu,
                                     extR_T_imu, K_l, K_r)                 # (Kv, 4, 6)

        # Drop any rows where the Jacobians themselves are non-finite
        finite_J = (
            np.all(np.isfinite(H_m.reshape(len(idx_v), -1)), axis=1)
            & np.all(np.isfinite(H_T_m.reshape(len(idx_v), -1)), axis=1)
        )
        if not finite_J.all():
            vi    = vi[finite_J];    idx_v = idx_v[finite_J]
            m_v   = m_v[finite_J];   innov = innov[finite_J]
            z_o_v = z_o_v[finite_J]
            H_m   = H_m[finite_J];   H_T_m = H_T_m[finite_J]
        if len(idx_v) == 0:
            continue

        # ------------------------------------------------------------------
        # Outlier rejection (Mahalanobis gating, marginal per landmark)
        # ------------------------------------------------------------------
        if outlier_threshold is not None:
            Sig_v = Sigma_lm[idx_v]                                            # (Kv, 3, 3)
            S_marg = (
                H_T_m @ Sigma_T[None] @ H_T_m.transpose(0, 2, 1)
                + H_m   @ Sig_v        @ H_m.transpose(0, 2, 1)
                + V_noise[None]
                + 1e-6 * np.eye(4)[None]                                       # regularisation
            )                                                                  # (Kv, 4, 4)
            try:
                S_inv = np.linalg.inv(S_marg)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S_marg)
            mahal  = np.einsum('ki,kij,kj->k', innov, S_inv, innov)
            inlier = np.isfinite(mahal) & (mahal < outlier_threshold)

            if not inlier.any():
                if verbose and (t % 500 == 0 or t == N - 1):
                    print(f"  t={t:5d}/{N}  all outliers")
                continue

            vi    = vi[inlier];    idx_v  = idx_v[inlier]
            m_v   = m_v[inlier];   innov  = innov[inlier]
            z_o_v = z_o_v[inlier]
            H_m   = H_m[inlier];   H_T_m  = H_T_m[inlier]

        Kv = len(idx_v)

        # ------------------------------------------------------------------
        # Step 3a: Pose update (information-filter, marginalised landmark σ)
        # ------------------------------------------------------------------
        Sig_m_v = Sigma_lm[idx_v]                                  # (Kv, 3, 3)

        # Effective noise:  V_eff_j = V + H_m_j · Σ_j · H_m_j^T   (Kv, 4, 4)
        # Small regularisation keeps the matrix well-conditioned even when
        # Σ_m is near-zero (fully converged landmark).
        V_eff = (
            V_noise[None]
            + H_m @ Sig_m_v @ H_m.transpose(0, 2, 1)
            + 1e-6 * np.eye(4)[None]
        )
        try:
            V_eff_inv = np.linalg.inv(V_eff)                       # (Kv, 4, 4)
        except np.linalg.LinAlgError:
            V_eff_inv = np.linalg.pinv(V_eff)

        # H_T^T · V_eff^{-1}   (Kv, 6, 4)
        HT_Vinv = H_T_m.transpose(0, 2, 1) @ V_eff_inv

        # Accumulate information (vectorised sum over Kv):
        # Ω_T ← Σ_T^{-1} + Σ_j H_T_j^T V_eff_j^{-1} H_T_j
        reg     = max(1e-9, 1e-10 * np.linalg.norm(Sigma_T))
        Omega_T = np.linalg.inv(Sigma_T + reg * np.eye(6))
        Omega_T += np.einsum('kij,kjl->il', HT_Vinv, H_T_m)       # (6, 6)

        # Information vector accumulation
        xi_info = np.einsum('kij,kj->i', HT_Vinv, innov)           # (6,)

        # Update pose covariance and compute correction
        Sigma_T = np.linalg.inv(Omega_T)
        dxi     = Sigma_T @ xi_info                                 # (6,)

        # Apply correction:  T̄_t ← T̄_t · exp(δξ̂)
        world_T_imu[t] = world_T_imu[t] @ twist2pose(axangle2twist(dxi))

        # ------------------------------------------------------------------
        # Step 3b: Landmark update (standard EKF, Joseph form, updated pose)
        # ------------------------------------------------------------------
        T_imu_world = inversePose(world_T_imu[t])
        T_cam_L     = oT_L @ T_imu_world                    # world → optical left cam
        T_cam_R     = oT_R @ T_imu_world                    # world → optical right cam

        # Re-project with corrected pose
        z_hat_new, valid_new = observations_batch(m_v, T_cam_L, T_cam_R, K_l, K_r)
        vi_new = np.where(valid_new)[0]
        if len(vi_new) == 0:
            if verbose and (t % 500 == 0 or t == N - 1):
                print(f"  t={t:5d}/{N}  no valid projections after pose update")
            continue

        idx_v2  = idx_v[vi_new]
        m_v2    = m_v[vi_new]
        z_h_v2  = z_hat_new[vi_new]
        z_o_v2  = z_o_v[vi_new]
        innov2  = z_o_v2 - z_h_v2
        Sig_v2  = Sigma_lm[idx_v2]

        H_m2 = jacobians_batch(m_v2, T_cam_L, T_cam_R, K_l, K_r)  # (Kv2, 4, 3)

        # Innovation covariance  S = H Σ H^T + V
        HS  = H_m2 @ Sig_v2                                        # (Kv2, 4, 3)
        S2  = HS @ H_m2.transpose(0, 2, 1) + V_noise[None]        # (Kv2, 4, 4)

        # Kalman gain  K = Σ H^T S^{-1}
        SigHt  = Sig_v2 @ H_m2.transpose(0, 2, 1)                 # (Kv2, 3, 4)
        Kt     = np.linalg.solve(S2, SigHt.transpose(0, 2, 1))    # (Kv2, 4, 3)
        K_gain = Kt.transpose(0, 2, 1)                             # (Kv2, 3, 4)

        # Mean update
        landmarks[idx_v2] = m_v2 + (K_gain @ innov2[:, :, None]).squeeze(-1)

        # Covariance update — Joseph form
        I3    = np.eye(3)[None]
        KH    = K_gain @ H_m2
        I_KH  = I3 - KH
        Sigma_lm[idx_v2] = (
            I_KH @ Sig_v2 @ I_KH.transpose(0, 2, 1)
            + K_gain @ V_noise[None] @ K_gain.transpose(0, 2, 1)
        )

        # Sanitise: reset any landmarks whose covariance or position became
        # non-finite (can happen in extreme numerical cases).
        bad_after = ~np.all(
            np.isfinite(Sigma_lm[idx_v2].reshape(len(idx_v2), -1)), axis=1
        ) | ~np.all(np.isfinite(landmarks[idx_v2]), axis=1)
        if bad_after.any():
            bad_lm_idx = idx_v2[bad_after]
            Sigma_lm[bad_lm_idx]    = Sigma0
            landmarks[bad_lm_idx]   = np.nan
            initialized[bad_lm_idx] = False

        if verbose and (t % 500 == 0 or t == N - 1):
            print(
                f"  t={t:5d}/{N}  init={initialized.sum():5d}  "
                f"pose_upd={Kv:4d}  lm_upd={len(vi_new):4d}"
            )

    return world_T_imu, Sigma_T, landmarks, Sigma_lm, initialized
