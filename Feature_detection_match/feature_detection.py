"""
(a) Stereo matching: detect Shi-Tomasi corners in left image, find
    correspondences in right image by searching along the epipolar line
    using normalized cross-correlation (NCC / matchTemplate).  Pure LK is
    unsuitable for stereo here because it collapses to zero-disparity for
    the many far-away features in this dataset; exhaustive epipolar-line
    search is the standard fix and is explicitly allowed by the spec
    ("Feel free to use any function from OpenCV").

(b) Temporal tracking: track features in left image across time using
    calcOpticalFlowPyrLK with forward-backward error checking.

Output: features array of shape (4, M, T)
    Row 0: left  x-pixel (l_x)
    Row 1: left  y-pixel (l_y)
    Row 2: right x-pixel (r_x)
    Row 3: right y-pixel (r_y)
Missing observations are set to -1 (all four rows).
"""

import numpy as np
import cv2

# ── Feature detection ────────────────────────────────────────────────────────
MAX_FEATURES    = 200    # maximum tracked features at any one time
MIN_FEATURES    = 80     # re-detect new corners below this count
QUALITY_LEVEL   = 0.01
MIN_DISTANCE    = 10     # min pixel distance between detected corners
BLOCK_SIZE      = 7      # Shi-Tomasi block size

# ── Temporal optical flow (LK) ────────────────────────────────────────────────
LK_PARAMS = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FB_ERROR_THRESH = 1.0    # max forward-backward pixel error

# ── Stereo NCC matching ───────────────────────────────────────────────────────
STEREO_PATCH    = 11     # half-size of NCC patch (full patch = 2*STEREO_PATCH+1)
NCC_THRESH      = 0.75   # minimum NCC score to accept a stereo match
MAX_HORIZ_DISP  = 128    # max l_x - r_x search range (pixels)
MAX_VERT_DISP   = 5      # |l_y - r_y| search strip half-height (pixels)

# ── EKF-SLAM filtering (optional, applied when filter_for_slam=True) ──────────
MIN_DISP_SLAM   = 1.0    # discard features with l_x - r_x < this (near infinity)
MAX_DISP_SLAM   = 100.0  # discard features with l_x - r_x > this (too close)

# ── Image-boundary margin ─────────────────────────────────────────────────────
MARGIN = STEREO_PATCH + 2   # keep features away from the image border


def _in_bounds(pts: np.ndarray, h: int, w: int) -> np.ndarray:
    """Boolean mask: True where (x, y) is strictly inside the image."""
    x, y = pts[:, 0], pts[:, 1]
    return (x >= MARGIN) & (x < w - MARGIN) & (y >= MARGIN) & (y < h - MARGIN)


def detect_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Shi-Tomasi corner detection.
    Returns (N, 2) float32 array of (x, y) pixel coordinates.
    """
    pts = cv2.goodFeaturesToTrack(
        img,
        maxCorners  = MAX_FEATURES,
        qualityLevel= QUALITY_LEVEL,
        minDistance = MIN_DISTANCE,
        blockSize   = BLOCK_SIZE,
        mask        = mask,
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.float32)
    return pts.reshape(-1, 2)


def track_temporal(
    img_prev: np.ndarray,
    img_curr: np.ndarray,
    pts_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track features from img_prev to img_curr using LK optical flow (part b)
    with forward-backward error checking for robustness.

    Returns
    -------
    pts_curr : (N, 2) float32 — tracked positions in current frame
    valid    : (N,)   bool    — True where tracking succeeded
    """
    if len(pts_prev) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    p0 = pts_prev.reshape(-1, 1, 2).astype(np.float32)

    p1,  st_fwd, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, p0, None, **LK_PARAMS)
    p0b, st_bwd, _ = cv2.calcOpticalFlowPyrLK(img_curr, img_prev, p1, None, **LK_PARAMS)

    fb_err = np.abs(p0 - p0b).reshape(-1, 2).max(axis=1)
    valid  = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_err < FB_ERROR_THRESH)

    return p1.reshape(-1, 2), valid


def stereo_match(
    img_left:  np.ndarray,
    img_right: np.ndarray,
    pts_left:  np.ndarray,
    filter_for_slam: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find right-image correspondences for left-image feature points (part a).

    Strategy: for each feature at (lx, ly) extract a patch from the left
    image and slide it along the epipolar strip (same row ± MAX_VERT_DISP)
    in the right image using normalized cross-correlation (matchTemplate).
    The best-scoring position is accepted if NCC ≥ NCC_THRESH and the
    stereo geometry constraints are satisfied.

    Parameters
    ----------
    filter_for_slam : if True apply MIN/MAX_DISP_SLAM to keep only features
                      with reliable depth estimates.

    Returns
    -------
    pts_right : (N, 2) float32 — matched positions in right frame
    valid     : (N,)   bool    — True where stereo match is reliable
    """
    h, w = img_left.shape[:2]
    N = len(pts_left)
    if N == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)

    pts_right = pts_left.copy().astype(np.float32)
    valid     = np.zeros(N, dtype=bool)

    for i, (lx, ly) in enumerate(pts_left):
        xi = int(round(float(lx)))
        yi = int(round(float(ly)))

        # Extract NCC patch from left image
        y0_p = yi - STEREO_PATCH
        y1_p = yi + STEREO_PATCH + 1
        x0_p = xi - STEREO_PATCH
        x1_p = xi + STEREO_PATCH + 1
        if y0_p < 0 or y1_p > h or x0_p < 0 or x1_p > w:
            continue
        patch = img_left[y0_p:y1_p, x0_p:x1_p]

        # Define the search strip in the right image along the epipolar line
        # For standard horizontal stereo: r_x ∈ [lx - MAX_HORIZ_DISP, lx]
        # Allow a small +5 px slack in case of slight non-rectification
        sx0 = max(STEREO_PATCH, xi - MAX_HORIZ_DISP)
        sx1 = min(w - STEREO_PATCH, xi + 5)
        sy0 = max(STEREO_PATCH, yi - MAX_VERT_DISP)
        sy1 = min(h - STEREO_PATCH, yi + MAX_VERT_DISP + 1)
        if sx1 - sx0 < 1 or sy1 - sy0 < 1:
            continue

        # Extract strip from right image (pad by patch size for matchTemplate)
        strip = img_right[sy0 - STEREO_PATCH : sy1 + STEREO_PATCH,
                          sx0 - STEREO_PATCH : sx1 + STEREO_PATCH]
        if strip.shape[0] < patch.shape[0] or strip.shape[1] < patch.shape[1]:
            continue

        res = cv2.matchTemplate(strip, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < NCC_THRESH:
            continue

        rx = float(sx0 + max_loc[0])
        ry = float(sy0 + max_loc[1])

        # Geometric constraints
        horiz_disp = lx - rx     # l_x - r_x, positive for valid stereo
        vert_disp  = abs(ly - ry)

        if vert_disp > MAX_VERT_DISP:
            continue
        if filter_for_slam and (horiz_disp < MIN_DISP_SLAM or horiz_disp > MAX_DISP_SLAM):
            continue
        if rx < MARGIN or rx >= w - MARGIN or ry < MARGIN or ry >= h - MARGIN:
            continue

        pts_right[i] = [rx, ry]
        valid[i]     = True

    return pts_right, valid


def build_feature_tracks(
    cam_imgs_L: list,
    cam_imgs_R: list,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build feature tracks for all T frames.

    Steps per frame
    ---------------
    1. If t == 0: detect initial Shi-Tomasi corners in left image.
    2. Else: track active features from previous left frame (LK + FB check).
             Discard lost tracks. If count < MIN_FEATURES, detect fresh corners
             in regions not already covered.
    3. Stereo-match surviving left features → right image.
    4. Record (l_x, l_y, r_x, r_y) for each valid stereo pair.

    Returns
    -------
    features : ndarray, shape (4, M, T), dtype float64
        -1 wherever the j-th feature is absent in frame t.
    """
    T    = len(cam_imgs_L)
    h, w = np.array(cam_imgs_L[0]).shape[:2]

    # Per-frame dict: feature_id → (lx, ly, rx, ry)
    observations: list[dict[int, tuple]] = [{} for _ in range(T)]

    active_ids: list[int]     = []
    active_pts: np.ndarray    = np.empty((0, 2), dtype=np.float32)
    next_id:    int           = 0
    prev_left:  np.ndarray | None = None

    for t in range(T):
        if verbose and t % 200 == 0:
            print(f"  frame {t:5d}/{T}  active={len(active_ids)}")

        img_left  = np.array(cam_imgs_L[t])
        img_right = np.array(cam_imgs_R[t])

        # ── Step 1 / 2: initialise or track ──────────────────────────────────
        if t == 0:
            pts = detect_features(img_left)
            pts = pts[_in_bounds(pts, h, w)]
            active_ids = list(range(len(pts)))
            active_pts = pts
            next_id    = len(pts)
        else:
            # Temporal tracking
            if len(active_pts) > 0:
                curr_pts, valid = track_temporal(prev_left, img_left, active_pts)
                # Also require tracked points to be inside image
                valid &= _in_bounds(curr_pts, h, w)
                active_ids = [active_ids[i] for i in range(len(active_ids)) if valid[i]]
                active_pts = curr_pts[valid]

            # Augment with fresh detections if needed
            if len(active_pts) < MIN_FEATURES:
                mask = np.full((h, w), 255, dtype=np.uint8)
                for pt in active_pts:
                    cv2.circle(mask, (int(round(pt[0])), int(round(pt[1]))),
                               MIN_DISTANCE, 0, -1)
                new_pts = detect_features(img_left, mask=mask)
                new_pts = new_pts[_in_bounds(new_pts, h, w)]
                if len(new_pts) > 0:
                    new_ids    = list(range(next_id, next_id + len(new_pts)))
                    next_id   += len(new_pts)
                    active_ids = active_ids + new_ids
                    active_pts = (np.vstack([active_pts, new_pts])
                                  if len(active_pts) > 0 else new_pts)

        # ── Step 3: stereo matching ───────────────────────────────────────────
        if len(active_pts) > 0:
            right_pts, stereo_ok = stereo_match(img_left, img_right, active_pts)

            # ── Step 4: record observations ───────────────────────────────────
            for fid, lpt, rpt, ok in zip(active_ids, active_pts, right_pts, stereo_ok):
                if ok:
                    observations[t][fid] = (float(lpt[0]), float(lpt[1]),
                                            float(rpt[0]), float(rpt[1]))

        prev_left = img_left

    # ── Assemble output (4, M, T) ─────────────────────────────────────────────
    M        = next_id
    features = np.full((4, M, T), -1.0, dtype=np.float64)

    for t, obs in enumerate(observations):
        for fid, (lx, ly, rx, ry) in obs.items():
            features[0, fid, t] = lx
            features[1, fid, t] = ly
            features[2, fid, t] = rx
            features[3, fid, t] = ry

    if verbose:
        n_obs   = np.sum(features[0] != -1)
        avg_per = n_obs / T
        print(f"\n  Done. Total unique features: {M}")
        print(f"  Total observations: {n_obs}  ({avg_per:.1f} per frame)")

    return features
