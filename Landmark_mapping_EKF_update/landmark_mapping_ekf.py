import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pr3_utils import inversePose

# Rotation from the "regular" camera frame (same axes as IMU: x=fwd, y=left, z=up)
# to the optical camera frame (x=right, y=down, z=fwd) used by the K matrix.
_oTr = np.array([[0., -1.,  0., 0.],
                 [0.,  0., -1., 0.],
                 [1.,  0.,  0., 0.],
                 [0.,  0.,  0., 1.]])


# ---------------------------------------------------------------------------
# Spatial grid subsampling of landmark tracks
# ---------------------------------------------------------------------------

def _grid_subsample(features, valid_obs, obs_count, keep_lm, grid=(20, 15)):
    M    = features.shape[1]
    rows, cols = grid

    # Mean left-image position per track across all valid frames
    lx_vals = np.where(valid_obs, features[0], np.nan)   # (M, N)
    ly_vals = np.where(valid_obs, features[1], np.nan)   # (M, N)
    mean_lx = np.nanmean(lx_vals, axis=1)                # (M,)
    mean_ly = np.nanmean(ly_vals, axis=1)                # (M,)

    eligible = keep_lm & np.isfinite(mean_lx) & np.isfinite(mean_ly)
    if not eligible.any():
        return keep_lm.copy()

    x_min = mean_lx[eligible].min();  x_max = mean_lx[eligible].max()
    y_min = mean_ly[eligible].min();  y_max = mean_ly[eligible].max()

    col_idx = np.clip(
        ((mean_lx - x_min) / (x_max - x_min + 1e-9) * cols).astype(int),
        0, cols - 1,
    )
    row_idx = np.clip(
        ((mean_ly - y_min) / (y_max - y_min + 1e-9) * rows).astype(int),
        0, rows - 1,
    )
    cell_id = row_idx * cols + col_idx   # (M,)

    new_keep = np.zeros(M, dtype=bool)
    for cell in range(rows * cols):
        candidates = np.where(eligible & (cell_id == cell))[0]
        if len(candidates) == 0:
            continue
        best = candidates[np.argmax(obs_count[candidates])]
        new_keep[best] = True

    return new_keep


# ---------------------------------------------------------------------------
# DLT Stereo Triangulation (batch)
# ---------------------------------------------------------------------------

def triangulate_batch(lx, ly, rx, ry, P_L, P_R):
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
    lm_grid=(20, 15),
    max_depth=150.0,
    min_disparity=1.0,
    outlier_threshold=20.0,
    verbose=True,
):
    N = world_T_imu.shape[0]
    M = features.shape[1]

    if V_noise is None:
        V_noise = 4.0 * np.eye(4)   # σ = 2 px per coordinate

    Sigma0 = sigma_init ** 2 * np.eye(3)

    # -----------------------------------------------------------------------
    # Pre-filter landmarks: min-observation count + spatial grid subsampling
    # -----------------------------------------------------------------------
    valid_obs = np.all(features >= 0, axis=0)   # (M, N) True when all 4 coords valid
    obs_count = valid_obs.sum(axis=1)            # (M,)
    keep_lm   = obs_count >= min_observations    # (M,)

    # Spatial grid subsampling: divide the left image into a (rows × cols) grid
    # and keep only the best-observed track per cell.  This distributes landmarks
    # evenly across the field of view — much better geometric coverage than
    # retaining an arbitrary top-N count.
    if lm_grid is not None:
        keep_lm = _grid_subsample(features, valid_obs, obs_count, keep_lm, grid=lm_grid)

    if verbose:
        print(f"  Total landmarks             : {M}")
        print(f"  After min-obs filter (≥{min_observations}) : {(obs_count >= min_observations).sum()}")
        if lm_grid is not None:
            print(f"  After grid subsampling {lm_grid}  : {keep_lm.sum()}")

    # -----------------------------------------------------------------------
    # Allocate per-landmark state
    # -----------------------------------------------------------------------
    landmarks   = np.full((M, 3), np.nan)
    Sigma_lm    = np.tile(Sigma0, (M, 1, 1))      # (M, 3, 3)
    initialized = np.zeros(M, dtype=bool)

    # Pre-compute all inverse IMU poses once  (N, 4, 4)
    T_imu_world_all = inversePose(world_T_imu)

    # oTr @ inv(extL/R_T_imu): maps from IMU frame to optical camera frame.
    # extL_T_imu = _IT_L maps from left-cam regular frame to IMU, so its
    # inverse maps from IMU to left-cam regular, and oTr converts regular→optical.
    oT_L = _oTr @ inversePose(extL_T_imu)   # IMU → optical left-cam  (4, 4)
    oT_R = _oTr @ inversePose(extR_T_imu)   # IMU → optical right-cam (4, 4)

    # -----------------------------------------------------------------------
    # Main EKF loop over timesteps
    # -----------------------------------------------------------------------
    for t in range(N):
        T_imu_world = T_imu_world_all[t]                    # (4, 4)
        T_cam_L     = oT_L @ T_imu_world                    # (4, 4)  world→optical left cam
        T_cam_R     = oT_R @ T_imu_world                    # (4, 4)  world→optical right cam

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
