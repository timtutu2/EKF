"""
Run feature detection and matching on dataset02.

Outputs
-------
dataset02_features.npy  — dict with key 'features', shape (4, M, T)
dataset02_features.png  — visualization of tracks at a few selected frames
"""

import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from feature_detection import build_feature_tracks

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = "/home/tim/Desktop/UCSD/ECE276A/PR3/dataset02"
IMGS_FILE    = os.path.join(DATASET_DIR, "dataset02_imgs.npy")
DATA_FILE    = os.path.join(DATASET_DIR, "dataset02.npy")
OUT_NPY      = os.path.join(os.path.dirname(__file__), "dataset02_features.npy")
OUT_PNG      = os.path.join(os.path.dirname(__file__), "dataset02_features_t+15.png")
MAX_FRAMES = None   # 500 for a quick test


def draw_stereo_on_single(bg_img, lpts, rpts, max_draw=60, seed=0):
    """
    Overlay left (blue) and right (red) stereo features on a single background
    image, connected by green lines showing the stereo disparity.
    """
    canvas_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lpts), size=min(max_draw, len(lpts)), replace=False)
    for i in idx:
        lp = (int(round(lpts[i, 0])), int(round(lpts[i, 1])))
        rp = (int(round(rpts[i, 0])), int(round(rpts[i, 1])))
        cv2.line(canvas_bgr, lp, rp, (0, 220, 0), 1)
        cv2.circle(canvas_bgr, lp, 4, (255, 50, 0), -1)   # blue  = left feature
        cv2.circle(canvas_bgr, rp, 4, (0, 50, 255), -1)   # red   = right feature
    return canvas_bgr


def draw_temporal_on_single(bg_img, lpts_t, lpts_tp, max_draw=60, seed=1):
    """
    Overlay features at time t (blue) and time t+dt (red) on a single left
    camera image, connected by green lines showing the temporal displacement.
    """
    canvas_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lpts_t), size=min(max_draw, len(lpts_t)), replace=False)
    for i in idx:
        p0 = (int(round(lpts_t[i,  0])), int(round(lpts_t[i,  1])))
        p1 = (int(round(lpts_tp[i, 0])), int(round(lpts_tp[i, 1])))
        cv2.line(canvas_bgr, p0, p1, (0, 220, 0), 1)
        cv2.circle(canvas_bgr, p0, 4, (255, 50, 0), -1)   # blue = features at t
        cv2.circle(canvas_bgr, p1, 4, (0, 50, 255), -1)   # red  = features at t+dt
    return canvas_bgr


def main():
    print("Loading images …")
    imgs_data   = np.load(IMGS_FILE, allow_pickle=True).item()
    cam_imgs_L  = imgs_data["cam_imgs_L"]
    cam_imgs_R  = imgs_data["cam_imgs_R"]

    if MAX_FRAMES is not None:
        cam_imgs_L = cam_imgs_L[:MAX_FRAMES]
        cam_imgs_R = cam_imgs_R[:MAX_FRAMES]
        print(f"  (using first {MAX_FRAMES} frames)")

    T = len(cam_imgs_L)
    print(f"  {T} frame pairs  ({np.array(cam_imgs_L[0]).shape})")

    # ── Run feature detection & tracking ─────────────────────────────────────
    print("\nRunning feature detection and tracking …")
    features = build_feature_tracks(cam_imgs_L, cam_imgs_R, verbose=True)
    print(f"features shape: {features.shape}")   # (4, M, T)

    # ── Save result ───────────────────────────────────────────────────────────
    np.save(OUT_NPY, {"features": features})
    print(f"\nSaved features → {OUT_NPY}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    print("Creating visualization …")

    dt_vis = 15   # temporal gap for the temporal-tracking panels

    # Pick one stereo frame and two temporal frame pairs
    stereo_t   = 1808
    temporal_t = [T // 4, 3 * T // 4]
    print(f"stereo_t: {stereo_t}")
    print(f"temporal_t: {temporal_t[0]}")
    print(f"temporal_t: {temporal_t[1]}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Feature Detection & Matching — dataset02", fontsize=13)

    # ── Top row: stereo matching (left features blue, right features red) ──
    for col, (ax, use_right_bg) in enumerate(zip(axes[0], [False, True])):
        img_l = np.array(cam_imgs_L[stereo_t])
        img_r = np.array(cam_imgs_R[stereo_t])
        bg    = img_r if use_right_bg else img_l

        valid_mask = features[0, :, stereo_t] != -1
        valid_ids  = np.where(valid_mask)[0]

        if len(valid_ids) == 0:
            ax.imshow(bg, cmap="gray")
            ax.set_title(f"t={stereo_t}  (no features)")
            ax.axis("off")
            continue

        lpts = features[:2, valid_ids, stereo_t].T   # (N, 2)
        rpts = features[2:, valid_ids, stereo_t].T   # (N, 2)

        panel = draw_stereo_on_single(bg, lpts, rpts, seed=col)
        bg_label = "right cam" if use_right_bg else "left cam"
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Stereo — {bg_label} bg  t={stereo_t}  pairs={len(valid_ids)}\n"
            "● blue=left feat  ● red=right feat  — green=disparity",
            fontsize=9,
        )
        ax.axis("off")

    # ── Bottom row: temporal tracking across two time pairs ────────────────
    for col, (ax, t0) in enumerate(zip(axes[1], temporal_t)):
        t1 = min(t0 + dt_vis, T - 1)
        img_l0 = np.array(cam_imgs_L[t0])

        # Features visible at both t0 and t1
        mask_t0 = features[0, :, t0] != -1
        mask_t1 = features[0, :, t1] != -1
        common  = np.where(mask_t0 & mask_t1)[0]

        if len(common) == 0:
            ax.imshow(img_l0, cmap="gray")
            ax.set_title(f"t={t0}→{t1}  (no common features)")
            ax.axis("off")
            continue

        lpts_t0 = features[:2, common, t0].T   # (N, 2)
        lpts_t1 = features[:2, common, t1].T   # (N, 2)

        panel = draw_temporal_on_single(img_l0, lpts_t0, lpts_t1, seed=col + 10)
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"Temporal — left cam  t={t0} → t={t1}  tracks={len(common)}\n"
            f"● blue=t={t0}  ● red=t={t1}  — green=motion",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved visualization → {OUT_PNG}")
    plt.close()

    # ── Summary statistics ────────────────────────────────────────────────────
    obs_per_frame = np.sum(features[0] != -1, axis=0)
    print(f"\nObservations per frame  min={obs_per_frame.min()}  "
          f"max={obs_per_frame.max()}  mean={obs_per_frame.mean():.1f}")

    # Track lengths (number of frames each feature is visible)
    track_lengths = np.sum(features[0] != -1, axis=1)
    print(f"Track lengths           min={track_lengths.min()}  "
          f"max={track_lengths.max()}  mean={track_lengths.mean():.1f}")


if __name__ == "__main__":
    main()
