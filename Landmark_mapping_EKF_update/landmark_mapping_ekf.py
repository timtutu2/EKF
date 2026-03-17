"""
Landmark Mapping via EKF Update
================================
ECE 276A - Project 3, Part 3

Assumes the IMU trajectory from Part 1 (ekf_imu_prediction) is correct (given).
Estimates the 3D world-frame positions  m_j ∈ R^3  of M visual landmarks observed
by the stereo camera, using EKF update steps based on the stereo observation model.

Mathematical Background
-----------------------
Landmark State (independent per landmark j):
    m_j  ∈ R^3      — 3D world-frame position
    Σ_j  ∈ R^{3×3}  — position covariance

Stereo Observation Model for landmark j at time t:
    z_{t,j} = h(m_j; T_t) + n_t,    n_t ~ N(0, V)

    where T_t = world_T_imu[t] is the known IMU pose and:

        p_L = extL_T_imu @ inv(T_t) @ [m_j; 1]   (landmark in left-cam frame)
        p_R = extR_T_imu @ inv(T_t) @ [m_j; 1]   (landmark in right-cam frame)

        h(m_j; T_t) = [ fu_l · p_L[0]/p_L[2] + cu_l,
                         fv_l · p_L[1]/p_L[2] + cv_l,
                         fu_r · p_R[0]/p_R[2] + cu_r,
                         fv_r · p_R[1]/p_R[2] + cv_r ]

Observation Jacobian  H_j = ∂h/∂m_j  (4×3):
    Let A_L = extL_T_imu @ T_imu_world,  A_R = extR_T_imu @ T_imu_world
        p_L = A_L @ [m_j; 1],  p_R = A_R @ [m_j; 1]

    ∂h_L / ∂p_L  (2×4):
        row 0: [ fu_l/p_L[2],  0,  -fu_l·p_L[0]/p_L[2]²,  0 ]
        row 1: [ 0,  fv_l/p_L[2],  -fv_l·p_L[1]/p_L[2]²,  0 ]

    ∂p_L / ∂m_j = A_L[:, :3]  (4×3)

    H_j[:2] = (∂h_L/∂p_L) @ A_L[:, :3]   (2×3)
    H_j[2:] = (∂h_R/∂p_R) @ A_R[:, :3]   (2×3)

EKF Update (vectorized over all observed landmarks K at time t):
    S     = H Σ H^T + V                   (K×4×4 innovation covariance)
    K     = Σ H^T S^{-1}                  (K×3×4 Kalman gain)
    m     ← m + K (z − ĥ)                (K×3)
    Σ     ← (I−KH) Σ (I−KH)^T + KVK^T   (K×3×3, Joseph form)

Landmark Initialization:
    On first valid observation at time t, the 3D position is estimated by
    DLT triangulation (normal-equations form) using the stereo pair.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pr3_utils import inversePose


# ---------------------------------------------------------------------------
# DLT Stereo Triangulation (batch)
# ---------------------------------------------------------------------------

def triangulate_batch(lx, ly, rx, ry, P_L, P_R):
    """
    Batch DLT triangulation: find 3D world points from stereo observations.

    The projection equation  λ [u; v; 1] = P @ [m; 1]  yields two linear
    constraints per camera view.  For left and right cameras this produces
    a 4×3 linear system  A m = b  that is solved via normal equations A^T A m = A^T b.

    Parameters
    ----------
    lx, ly  : (K,) float  left-image pixel coordinates
    rx, ry  : (K,) float  right-image pixel coordinates
    P_L     : (3, 4)      left  projection matrix  = K_l @ extL_T_imu @ T_imu_world
    P_R     : (3, 4)      right projection matrix  = K_r @ extR_T_imu @ T_imu_world

    Returns
    -------
    m : (K, 3) world-frame 3D positions
    """
    # Build the (K, 4, 3) coefficient matrix A (homogeneous 4th col moved to rhs)
    A = np.stack([
        lx[:, None] * P_L[2:3, :3] - P_L[0:1, :3],   # (K, 3)
        ly[:, None] * P_L[2:3, :3] - P_L[1:2, :3],   # (K, 3)
        rx[:, None] * P_R[2:3, :3] - P_R[0:1, :3],   # (K, 3)
        ry[:, None] * P_R[2:3, :3] - P_R[1:2, :3],   # (K, 3)
    ], axis=1)  # (K, 4, 3)

    b = np.stack([
        -(lx * P_L[2, 3] - P_L[0, 3]),   # (K,)
        -(ly * P_L[2, 3] - P_L[1, 3]),
        -(rx * P_R[2, 3] - P_R[0, 3]),
        -(ry * P_R[2, 3] - P_R[1, 3]),
    ], axis=1)  # (K, 4)

    # Normal equations:  (A^T A) m = A^T b
    At   = A.transpose(0, 2, 1)                       # (K, 3, 4)
    AtA  = At @ A                                      # (K, 3, 3)
    Atb  = At @ b[:, :, None]                          # (K, 3, 1)  keep as column vector

    # np.linalg.solve gufunc signature is (m,m),(m,n)->(m,n);
    # Atb must be 3-D so the batch dim K is not mistaken for a core dim.
    m = np.linalg.solve(AtA, Atb).squeeze(-1)         # (K, 3)
    return m


# ---------------------------------------------------------------------------
# Stereo Observation Model (batch)
# ---------------------------------------------------------------------------

def observations_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r):
    """
    Predicted stereo pixel observations for a batch of landmarks.

    Parameters
    ----------
    m_batch  : (K, 3)  landmark world-frame positions
    T_cam_L  : (4, 4)  world → left-camera  transform = extL_T_imu @ T_imu_world
    T_cam_R  : (4, 4)  world → right-camera transform = extR_T_imu @ T_imu_world
    K_l, K_r : (3, 3)  camera intrinsic matrices

    Returns
    -------
    z_hat : (K, 4)  predicted [lx, ly, rx, ry]
    valid : (K,)    True when landmark is in front of both cameras
    """
    K   = m_batch.shape[0]
    m_h = np.hstack([m_batch, np.ones((K, 1))])   # (K, 4) homogeneous

    p_L = (T_cam_L @ m_h.T).T   # (K, 4) in left-cam frame
    p_R = (T_cam_R @ m_h.T).T   # (K, 4) in right-cam frame

    valid = (p_L[:, 2] > 0) & (p_R[:, 2] > 0)

    z_hat     = np.zeros((K, 4))
    v         = valid
    z_hat[v, 0] = K_l[0, 0] * p_L[v, 0] / p_L[v, 2] + K_l[0, 2]
    z_hat[v, 1] = K_l[1, 1] * p_L[v, 1] / p_L[v, 2] + K_l[1, 2]
    z_hat[v, 2] = K_r[0, 0] * p_R[v, 0] / p_R[v, 2] + K_r[0, 2]
    z_hat[v, 3] = K_r[1, 1] * p_R[v, 1] / p_R[v, 2] + K_r[1, 2]

    return z_hat, valid


# ---------------------------------------------------------------------------
# Observation Jacobian  ∂h/∂m  (batch)
# ---------------------------------------------------------------------------

def jacobians_batch(m_batch, T_cam_L, T_cam_R, K_l, K_r):
    """
    4×3 Jacobians of the stereo observation w.r.t. landmark world-frame positions.

    Parameters
    ----------
    m_batch  : (K, 3)  landmark world-frame positions
    T_cam_L  : (4, 4)  world → left-camera  transform
    T_cam_R  : (4, 4)  world → right-camera transform
    K_l, K_r : (3, 3)  camera intrinsics

    Returns
    -------
    H : (K, 4, 3)  Jacobians  ∂[lx, ly, rx, ry] / ∂m_j
    """
    K   = m_batch.shape[0]
    m_h = np.hstack([m_batch, np.ones((K, 1))])   # (K, 4)

    p_L = (T_cam_L @ m_h.T).T   # (K, 4)
    p_R = (T_cam_R @ m_h.T).T   # (K, 4)

    # ∂z_L / ∂p_L  (K, 2, 4)
    dz_L = np.zeros((K, 2, 4))
    dz_L[:, 0, 0] =  K_l[0, 0] / p_L[:, 2]
    dz_L[:, 0, 2] = -K_l[0, 0] * p_L[:, 0] / p_L[:, 2] ** 2
    dz_L[:, 1, 1] =  K_l[1, 1] / p_L[:, 2]
    dz_L[:, 1, 2] = -K_l[1, 1] * p_L[:, 1] / p_L[:, 2] ** 2

    # ∂z_R / ∂p_R  (K, 2, 4)
    dz_R = np.zeros((K, 2, 4))
    dz_R[:, 0, 0] =  K_r[0, 0] / p_R[:, 2]
    dz_R[:, 0, 2] = -K_r[0, 0] * p_R[:, 0] / p_R[:, 2] ** 2
    dz_R[:, 1, 1] =  K_r[1, 1] / p_R[:, 2]
    dz_R[:, 1, 2] = -K_r[1, 1] * p_R[:, 1] / p_R[:, 2] ** 2

    # ∂p_L / ∂m_j = T_cam_L[:, :3]   (4×3) — identical for all K landmarks
    dA_L = T_cam_L[:, :3]   # (4, 3)
    dA_R = T_cam_R[:, :3]   # (4, 3)

    # Chain rule: H[:, :2, :] = dz_L @ dA_L,   H[:, 2:, :] = dz_R @ dA_R
    H          = np.zeros((K, 4, 3))
    H[:, :2, :] = dz_L @ dA_L   # (K, 2, 4) @ (4, 3) → (K, 2, 3)
    H[:, 2:, :] = dz_R @ dA_R   # (K, 2, 4) @ (4, 3) → (K, 2, 3)

    return H


# ---------------------------------------------------------------------------
# Main EKF Landmark Mapping
# ---------------------------------------------------------------------------

def ekf_landmark_mapping(
    world_T_imu,
    features,
    K_l, K_r,
    extL_T_imu, extR_T_imu,
    V_noise=None,
    sigma_init=1.0,
    min_observations=3,
    max_depth=150.0,
    min_disparity=1.0,
    outlier_threshold=20.0,
    verbose=True,
):
    """
    EKF landmark mapping with known IMU trajectory.

    No EKF prediction step is needed since landmarks are assumed static and the
    IMU poses world_T_imu are treated as ground truth.

    Since each landmark is independent (no cross-covariance through the robot pose
    in pure landmark mapping), the joint EKF update decouples into per-landmark
    updates — equivalent to maintaining a block-diagonal covariance.

    Parameters
    ----------
    world_T_imu      : (N, 4, 4)   SE(3) IMU poses from EKF prediction
    features         : (4, M, N)   Stereo observations [lx,ly,rx,ry] × M × N
                                   Invalid observations are set to -1.
    K_l, K_r         : (3, 3)      Left / right camera intrinsic matrices
    extL_T_imu       : (4, 4)      SE(3) transform: IMU → left camera frame
    extR_T_imu       : (4, 4)      SE(3) transform: IMU → right camera frame
    V_noise          : (4, 4)      Pixel observation noise covariance.
                                   Default: 4·I₄  (σ = 2 px per coordinate)
    sigma_init       : float        Initial landmark position std dev [m].
                                   Σ₀ = sigma_init² · I₃
    min_observations : int          Minimum valid stereo observations required to
                                   include a landmark (coarse quality filter).
    max_depth        : float        Max allowed initial depth at triangulation [m].
    min_disparity    : float        Min required stereo disparity at init [px].
                                   Filters out very distant (unreliable) landmarks.
    outlier_threshold: float | None Chi-squared threshold (4 DOF) for innovation
                                   gating.  None disables outlier rejection.
                                   χ²(4, 0.9999) ≈ 18.5; using 20.0 as default.
    verbose          : bool         Print per-step progress to stdout.

    Returns
    -------
    landmarks   : (M, 3)      Estimated 3D world positions; NaN for uninitialised.
    Sigma_lm    : (M, 3, 3)   Per-landmark position covariances.
    initialized : (M,)        Boolean mask: True for successfully initialised lm.
    """
    N = world_T_imu.shape[0]
    M = features.shape[1]

    if V_noise is None:
        V_noise = 4.0 * np.eye(4)   # σ = 2 px per coordinate

    Sigma0 = sigma_init ** 2 * np.eye(3)

    # -----------------------------------------------------------------------
    # Pre-filter landmarks by minimum observation count
    # -----------------------------------------------------------------------
    valid_obs  = np.all(features >= 0, axis=0)   # (M, N) True when all 4 coords valid
    obs_count  = valid_obs.sum(axis=1)            # (M,)
    keep_lm    = obs_count >= min_observations    # (M,)

    if verbose:
        print(f"  Total landmarks             : {M}")
        print(f"  After min-obs filter (≥{min_observations}) : {keep_lm.sum()}")

    # -----------------------------------------------------------------------
    # Allocate per-landmark state
    # -----------------------------------------------------------------------
    landmarks   = np.full((M, 3), np.nan)
    Sigma_lm    = np.tile(Sigma0, (M, 1, 1))      # (M, 3, 3)
    initialized = np.zeros(M, dtype=bool)

    # Pre-compute all inverse IMU poses once  (N, 4, 4)
    T_imu_world_all = inversePose(world_T_imu)

    # -----------------------------------------------------------------------
    # Main EKF loop over timesteps
    # -----------------------------------------------------------------------
    for t in range(N):
        T_imu_world = T_imu_world_all[t]                    # (4, 4)
        T_cam_L     = extL_T_imu @ T_imu_world              # (4, 4)  world→left cam
        T_cam_R     = extR_T_imu @ T_imu_world              # (4, 4)  world→right cam

        # Projection matrices for DLT triangulation  (3, 4)
        P_L = K_l @ T_cam_L[:3, :]
        P_R = K_r @ T_cam_R[:3, :]

        # Landmarks with valid observations at this timestep (that pass the filter)
        obs_at_t = valid_obs[:, t] & keep_lm                # (M,)

        # ------------------------------------------------------------------
        # Initialization pass: triangulate landmarks seen for the first time
        # ------------------------------------------------------------------
        new_idx = np.where(obs_at_t & ~initialized)[0]

        if len(new_idx) > 0:
            lx_n = features[0, new_idx, t]
            ly_n = features[1, new_idx, t]
            rx_n = features[2, new_idx, t]
            ry_n = features[3, new_idx, t]

            disp = lx_n - rx_n                              # stereo disparity
            good = disp >= min_disparity

            if good.any():
                ni    = new_idx[good]
                m_new = triangulate_batch(
                    lx_n[good], ly_n[good], rx_n[good], ry_n[good], P_L, P_R
                )  # (K', 3)

                # Check: landmark must be in front of left camera and within max_depth
                m_h      = np.hstack([m_new, np.ones((len(ni), 1))])   # (K', 4)
                p_check  = (T_cam_L @ m_h.T).T                         # (K', 4)
                depth_ok = (p_check[:, 2] > 0) & (p_check[:, 2] < max_depth)

                if depth_ok.any():
                    ni_ok                    = ni[depth_ok]
                    landmarks[ni_ok]         = m_new[depth_ok]
                    Sigma_lm[ni_ok]          = Sigma0
                    initialized[ni_ok]       = True

        # ------------------------------------------------------------------
        # EKF update pass: update landmarks already initialised
        # ------------------------------------------------------------------
        upd_idx = np.where(obs_at_t & initialized)[0]

        if len(upd_idx) == 0:
            if verbose and (t % 500 == 0 or t == N - 1):
                print(f"  t={t:5d}/{N}  init={initialized.sum():5d}  obs=0")
            continue

        m_b   = landmarks[upd_idx]                    # (K, 3)
        Sig_b = Sigma_lm[upd_idx]                     # (K, 3, 3)
        z_obs = features[:, upd_idx, t].T             # (K, 4)

        # Predicted observations and in-front-of-camera validity
        z_hat, valid_proj = observations_batch(m_b, T_cam_L, T_cam_R, K_l, K_r)

        if not valid_proj.any():
            continue

        vi     = np.where(valid_proj)[0]
        idx_v  = upd_idx[vi]
        m_v    = m_b[vi]          # (Kv, 3)
        Sig_v  = Sig_b[vi]        # (Kv, 3, 3)
        z_h_v  = z_hat[vi]        # (Kv, 4)
        z_o_v  = z_obs[vi]        # (Kv, 4)
        innov  = z_o_v - z_h_v    # (Kv, 4)

        # Jacobians  H (Kv, 4, 3)
        H_v = jacobians_batch(m_v, T_cam_L, T_cam_R, K_l, K_r)

        # Innovation covariance  S = H Σ H^T + V  (Kv, 4, 4)
        HS  = H_v @ Sig_v                                        # (Kv, 4, 3)
        S   = HS @ H_v.transpose(0, 2, 1) + V_noise[None]       # (Kv, 4, 4)

        # ------------------------------------------------------------------
        # Optional chi-squared gating (Mahalanobis outlier rejection)
        # ------------------------------------------------------------------
        if outlier_threshold is not None:
            S_inv  = np.linalg.inv(S)                                          # (Kv, 4, 4)
            mahal  = np.einsum('ki,kij,kj->k', innov, S_inv, innov)            # (Kv,)
            inlier = mahal < outlier_threshold

            if not inlier.any():
                if verbose and (t % 500 == 0 or t == N - 1):
                    print(f"  t={t:5d}/{N}  init={initialized.sum():5d}  all outliers")
                continue

            vi    = vi[inlier];    idx_v = idx_v[inlier]
            m_v   = m_v[inlier];   Sig_v = Sig_v[inlier]
            innov = innov[inlier]; H_v   = H_v[inlier]
            HS    = HS[inlier];    S     = S[inlier]

        # ------------------------------------------------------------------
        # Kalman gain  K = Σ H^T S^{-1}
        #   Solved as: K^T = S^{-1} (H Σ^T) = S^{-1} (H Σ)  [Σ symmetric]
        # ------------------------------------------------------------------
        SigHt  = Sig_v @ H_v.transpose(0, 2, 1)                  # (Kv, 3, 4)
        Kt     = np.linalg.solve(S, SigHt.transpose(0, 2, 1))    # (Kv, 4, 3)
        K_gain = Kt.transpose(0, 2, 1)                            # (Kv, 3, 4)

        # Mean update
        landmarks[idx_v] = m_v + (K_gain @ innov[:, :, None]).squeeze(-1)  # (Kv, 3)

        # Covariance update — Joseph form for numerical stability:
        #   Σ ← (I−KH) Σ (I−KH)^T + K V K^T
        I3      = np.eye(3)[None]                                 # (1, 3, 3)
        KH      = K_gain @ H_v                                    # (Kv, 3, 3)
        I_KH    = I3 - KH                                         # (Kv, 3, 3)
        Sigma_lm[idx_v] = (
            I_KH @ Sig_v @ I_KH.transpose(0, 2, 1)
            + K_gain @ V_noise[None] @ K_gain.transpose(0, 2, 1)
        )  # (Kv, 3, 3)

        if verbose and (t % 500 == 0 or t == N - 1):
            n_upd = len(idx_v)
            print(
                f"  t={t:5d}/{N}  init={initialized.sum():5d}  "
                f"updated={n_upd:4d}/{len(upd_idx):4d}"
            )

    return landmarks, Sigma_lm, initialized
