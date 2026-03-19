"""
Visual-inertial SLAM via EKF.

This implementation follows the PR3 specification:
1. Predict the IMU pose with SE(3) kinematics from the measured body twist.
2. Triangulate newly observed stereo features to initialise landmarks.
3. Update the pose using the stereo observation model while marginalising the
   current landmark uncertainty into the observation covariance.
4. Update the observed landmarks with a standard EKF correction.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pr3_utils import axangle2twist, inversePose, pose2adpose, twist2pose

# Rotation from the "regular" camera frame (same axes as IMU: x=fwd, y=left, z=up)
# to the optical camera frame (x=right, y=down, z=fwd) used by the K matrix.
_oTr = np.array([[0., -1.,  0., 0.],
                 [0.,  0., -1., 0.],
                 [1.,  0.,  0., 0.],
                 [0.,  0.,  0., 1.]])
from Landmark_mapping_EKF_update.landmark_mapping_ekf import (
    jacobians_batch,
    observations_batch,
    triangulate_batch,
)


def odot_operator_batch(q_batch: np.ndarray) -> np.ndarray:
    """
    Build the homogeneous-point odot operator q^⊙ for a batch of points.

    Each returned matrix Q satisfies:
        xi_hat @ q = Q @ xi
    for xi = [rho, phi] in R^6.
    """
    q_batch = np.asarray(q_batch)
    q1 = q_batch[:, 0]
    q2 = q_batch[:, 1]
    q3 = q_batch[:, 2]
    q4 = q_batch[:, 3]

    Q = np.zeros((q_batch.shape[0], 4, 6), dtype=q_batch.dtype)
    Q[:, 0, 0] = q4
    Q[:, 0, 4] = q3
    Q[:, 0, 5] = -q2
    Q[:, 1, 1] = q4
    Q[:, 1, 3] = -q3
    Q[:, 1, 5] = q1
    Q[:, 2, 2] = q4
    Q[:, 2, 3] = q2
    Q[:, 2, 4] = -q1
    return Q


def pose_jacobians_batch(
    m_batch: np.ndarray,
    T_imu_world: np.ndarray,
    extL_T_imu: np.ndarray,
    extR_T_imu: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
) -> np.ndarray:
    """
    Stereo observation Jacobian with respect to the right-invariant pose error.

    The linearisation uses:
        world_T_imu = world_T_imu_bar @ exp(xi_hat)
        imu_T_world = exp(-xi_hat) @ imu_T_world_bar
    """
    m_h = np.hstack([m_batch, np.ones((m_batch.shape[0], 1))])
    q_imu = (T_imu_world @ m_h.T).T

    # IMU → optical camera transforms
    oT_L = _oTr @ inversePose(extL_T_imu)
    oT_R = _oTr @ inversePose(extR_T_imu)

    p_L = (oT_L @ q_imu.T).T
    p_R = (oT_R @ q_imu.T).T

    Q = odot_operator_batch(q_imu)
    dp_L = -np.einsum("ij,kjl->kil", oT_L, Q)
    dp_R = -np.einsum("ij,kjl->kil", oT_R, Q)

    H = np.zeros((m_batch.shape[0], 4, 6), dtype=m_batch.dtype)

    dz_L = np.zeros((m_batch.shape[0], 2, 4), dtype=m_batch.dtype)
    dz_L[:, 0, 0] = K_l[0, 0] / p_L[:, 2]
    dz_L[:, 0, 2] = -K_l[0, 0] * p_L[:, 0] / (p_L[:, 2] ** 2)
    dz_L[:, 1, 1] = K_l[1, 1] / p_L[:, 2]
    dz_L[:, 1, 2] = -K_l[1, 1] * p_L[:, 1] / (p_L[:, 2] ** 2)

    dz_R = np.zeros((m_batch.shape[0], 2, 4), dtype=m_batch.dtype)
    dz_R[:, 0, 0] = K_r[0, 0] / p_R[:, 2]
    dz_R[:, 0, 2] = -K_r[0, 0] * p_R[:, 0] / (p_R[:, 2] ** 2)
    dz_R[:, 1, 1] = K_r[1, 1] / p_R[:, 2]
    dz_R[:, 1, 2] = -K_r[1, 1] * p_R[:, 1] / (p_R[:, 2] ** 2)

    H[:, :2, :] = dz_L @ dp_L
    H[:, 2:, :] = dz_R @ dp_R
    return H


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _safe_inverse(M: np.ndarray, reg: float = 1e-9) -> np.ndarray:
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(M + reg * np.eye(M.shape[0]))


def _safe_batch_inverse(M: np.ndarray, reg: float = 1e-9) -> np.ndarray:
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        eye = np.eye(M.shape[-1])[None]
        return np.linalg.inv(M + reg * eye)


def vi_slam_ekf(
    v_t: np.ndarray,
    w_t: np.ndarray,
    timestamps: np.ndarray,
    features: np.ndarray,
    K_l: np.ndarray,
    K_r: np.ndarray,
    extL_T_imu: np.ndarray,
    extR_T_imu: np.ndarray,
    W_noise: np.ndarray | None = None,
    V_noise: np.ndarray | None = None,
    sigma_init: float = 1.0,
    min_observations: int = 3,
    max_depth: float = 150.0,
    min_disparity: float = 1.0,
    outlier_threshold: float | None = 20.0,
    max_lm_per_step: int = 200,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run joint visual-inertial EKF SLAM over the full trajectory.

    Returns:
        world_T_imu: (N, 4, 4) corrected IMU trajectory
        Sigma_T:     (6, 6) final pose covariance
        landmarks:   (M, 3) landmark means
        Sigma_lm:    (M, 3, 3) landmark covariances
        initialized: (M,) initialisation mask
    """
    if W_noise is None:
        W_noise = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    if V_noise is None:
        V_noise = 4.0 * np.eye(4)

    N = timestamps.shape[0]
    M = features.shape[1]
    Sigma0 = (sigma_init ** 2) * np.eye(3)

    valid_obs = np.all(features >= 0, axis=0)
    obs_count = valid_obs.sum(axis=1)
    keep_lm = obs_count >= min_observations

    world_T_imu = np.zeros((N, 4, 4))
    world_T_imu[0] = np.eye(4)
    Sigma_T = np.zeros((6, 6))

    landmarks = np.full((M, 3), np.nan)
    Sigma_lm = np.tile(Sigma0, (M, 1, 1))
    initialized = np.zeros(M, dtype=bool)

    if verbose:
        print(f"  Total landmarks: {M}")
        print(f"  Landmarks kept after min-observation filter: {keep_lm.sum()}")

    # Pre-compute IMU→optical-camera transforms (fixed calibration, not time-varying)
    oT_L = _oTr @ inversePose(extL_T_imu)   # IMU → optical left-cam  (4, 4)
    oT_R = _oTr @ inversePose(extR_T_imu)   # IMU → optical right-cam (4, 4)

    for t in range(N):
        if t > 0:
            dt = timestamps[t] - timestamps[t - 1]
            twist_dt = np.concatenate([v_t[t - 1], w_t[t - 1]]) * dt
            Phi_t = twist2pose(axangle2twist(twist_dt))

            world_T_imu[t] = world_T_imu[t - 1] @ Phi_t
            F_t = pose2adpose(inversePose(Phi_t))
            Sigma_T = _symmetrize(F_t @ Sigma_T @ F_t.T + W_noise)

        T_imu_world = inversePose(world_T_imu[t])
        T_cam_L = oT_L @ T_imu_world                        # world → optical left cam
        T_cam_R = oT_R @ T_imu_world                        # world → optical right cam
        P_L = K_l @ T_cam_L[:3, :]
        P_R = K_r @ T_cam_R[:3, :]

        obs_at_t = valid_obs[:, t] & keep_lm

        new_idx = np.where(obs_at_t & ~initialized)[0]
        if new_idx.size:
            z_new = features[:, new_idx, t]
            disparity = z_new[0] - z_new[2]
            good_init = disparity >= min_disparity
            if np.any(good_init):
                init_idx = new_idx[good_init]
                m_new = triangulate_batch(
                    z_new[0, good_init],
                    z_new[1, good_init],
                    z_new[2, good_init],
                    z_new[3, good_init],
                    P_L,
                    P_R,
                )
                m_new_h = np.hstack([m_new, np.ones((m_new.shape[0], 1))])
                p_L_new = (T_cam_L @ m_new_h.T).T
                p_R_new = (T_cam_R @ m_new_h.T).T
                depth_ok = (
                    (p_L_new[:, 2] > 0)
                    & (p_R_new[:, 2] > 0)
                    & (p_L_new[:, 2] < max_depth)
                    & (p_R_new[:, 2] < max_depth)
                )
                if np.any(depth_ok):
                    good_idx = init_idx[depth_ok]
                    landmarks[good_idx] = m_new[depth_ok]
                    Sigma_lm[good_idx] = Sigma0
                    initialized[good_idx] = True

        upd_idx = np.where(obs_at_t & initialized)[0]
        if not upd_idx.size:
            if verbose and (t % 500 == 0 or t == N - 1):
                print(f"  t={t:5d}/{N} init={initialized.sum():5d} no updates")
            continue

        if upd_idx.size > max_lm_per_step:
            priority = np.argsort(-obs_count[upd_idx])[:max_lm_per_step]
            upd_idx = upd_idx[priority]

        m_batch = landmarks[upd_idx]
        z_obs = features[:, upd_idx, t].T

        finite_state = (
            np.all(np.isfinite(m_batch), axis=1)
            & np.all(np.isfinite(Sigma_lm[upd_idx].reshape(upd_idx.size, -1)), axis=1)
        )
        if not np.all(finite_state):
            bad_idx = upd_idx[~finite_state]
            landmarks[bad_idx] = np.nan
            Sigma_lm[bad_idx] = Sigma0
            initialized[bad_idx] = False
            upd_idx = upd_idx[finite_state]
            m_batch = m_batch[finite_state]
            z_obs = z_obs[finite_state]
            if not upd_idx.size:
                continue

        z_hat, valid_proj = observations_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r)
        if not np.any(valid_proj):
            continue

        upd_idx = upd_idx[valid_proj]
        m_batch = m_batch[valid_proj]
        z_obs = z_obs[valid_proj]
        z_hat = z_hat[valid_proj]
        innov = z_obs - z_hat

        H_m = jacobians_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r)
        H_T = pose_jacobians_batch(m_batch, T_imu_world, extL_T_imu, extR_T_imu, K_l, K_r)

        finite_jac = (
            np.all(np.isfinite(H_m.reshape(upd_idx.size, -1)), axis=1)
            & np.all(np.isfinite(H_T.reshape(upd_idx.size, -1)), axis=1)
            & np.all(np.isfinite(innov), axis=1)
        )
        if not np.all(finite_jac):
            upd_idx = upd_idx[finite_jac]
            m_batch = m_batch[finite_jac]
            z_obs = z_obs[finite_jac]
            z_hat = z_hat[finite_jac]
            innov = innov[finite_jac]
            H_m = H_m[finite_jac]
            H_T = H_T[finite_jac]
            if not upd_idx.size:
                continue

        Sigma_batch = Sigma_lm[upd_idx]

        if outlier_threshold is not None:
            S_marg = (
                H_T @ Sigma_T[None] @ H_T.transpose(0, 2, 1)
                + H_m @ Sigma_batch @ H_m.transpose(0, 2, 1)
                + V_noise[None]
                + 1e-6 * np.eye(4)[None]
            )
            S_marg_inv = _safe_batch_inverse(S_marg, reg=1e-6)
            mahal = np.einsum("ki,kij,kj->k", innov, S_marg_inv, innov)
            inlier = np.isfinite(mahal) & (mahal < outlier_threshold)
            if not np.any(inlier):
                if verbose and (t % 500 == 0 or t == N - 1):
                    print(f"  t={t:5d}/{N} init={initialized.sum():5d} all outliers")
                continue
            upd_idx = upd_idx[inlier]
            m_batch = m_batch[inlier]
            z_obs = z_obs[inlier]
            z_hat = z_hat[inlier]
            innov = innov[inlier]
            H_m = H_m[inlier]
            H_T = H_T[inlier]
            Sigma_batch = Sigma_batch[inlier]

        V_eff = (
            V_noise[None]
            + H_m @ Sigma_batch @ H_m.transpose(0, 2, 1)
            + 1e-6 * np.eye(4)[None]
        )
        V_eff_inv = _safe_batch_inverse(V_eff, reg=1e-6)
        Ht_Vinv = H_T.transpose(0, 2, 1) @ V_eff_inv

        prior_reg = 1e-9 * np.eye(6)
        Omega_T = _safe_inverse(Sigma_T + prior_reg, reg=1e-9)
        Omega_T += np.einsum("kij,kjl->il", Ht_Vinv, H_T)
        xi_info = np.einsum("kij,kj->i", Ht_Vinv, innov)

        Sigma_T = _symmetrize(_safe_inverse(Omega_T, reg=1e-9))
        dxi = Sigma_T @ xi_info
        world_T_imu[t] = world_T_imu[t] @ twist2pose(axangle2twist(dxi))

        T_imu_world = inversePose(world_T_imu[t])
        T_cam_L = oT_L @ T_imu_world                        # world → optical left cam
        T_cam_R = oT_R @ T_imu_world                        # world → optical right cam

        z_hat_corr, valid_corr = observations_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r)
        if not np.any(valid_corr):
            if verbose and (t % 500 == 0 or t == N - 1):
                print(f"  t={t:5d}/{N} init={initialized.sum():5d} no valid projections")
            continue

        upd_idx = upd_idx[valid_corr]
        m_batch = m_batch[valid_corr]
        z_obs = z_obs[valid_corr]
        z_hat_corr = z_hat_corr[valid_corr]
        innov_corr = z_obs - z_hat_corr
        Sigma_batch = Sigma_lm[upd_idx]

        H_m_corr = jacobians_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r)
        S = H_m_corr @ Sigma_batch @ H_m_corr.transpose(0, 2, 1) + V_noise[None]
        SigHt = Sigma_batch @ H_m_corr.transpose(0, 2, 1)
        K_gain = np.linalg.solve(S, SigHt.transpose(0, 2, 1)).transpose(0, 2, 1)

        landmarks[upd_idx] = m_batch + (K_gain @ innov_corr[:, :, None]).squeeze(-1)

        I3 = np.eye(3)[None]
        KH = K_gain @ H_m_corr
        I_KH = I3 - KH
        Sigma_lm[upd_idx] = (
            I_KH @ Sigma_batch @ I_KH.transpose(0, 2, 1)
            + K_gain @ V_noise[None] @ K_gain.transpose(0, 2, 1)
        )
        Sigma_lm[upd_idx] = 0.5 * (
            Sigma_lm[upd_idx] + Sigma_lm[upd_idx].transpose(0, 2, 1)
        )

        finite_after = (
            np.all(np.isfinite(landmarks[upd_idx]), axis=1)
            & np.all(np.isfinite(Sigma_lm[upd_idx].reshape(upd_idx.size, -1)), axis=1)
        )
        if not np.all(finite_after):
            bad_idx = upd_idx[~finite_after]
            landmarks[bad_idx] = np.nan
            Sigma_lm[bad_idx] = Sigma0
            initialized[bad_idx] = False

        if verbose and (t % 500 == 0 or t == N - 1):
            print(
                f"  t={t:5d}/{N} init={initialized.sum():5d} "
                f"pose_upd={H_T.shape[0]:4d} lm_upd={upd_idx.size:4d}"
            )

    return world_T_imu, Sigma_T, landmarks, Sigma_lm, initialized
