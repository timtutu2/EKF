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
OUT_PNG      = os.path.join(os.path.dirname(__file__), "dataset02_features.png")

# ── Optional: process only a prefix of frames for a quick test ────────────────
#   Set to None to process all 6393 frames.
MAX_FRAMES = None   # e.g. 500 for a quick test


def draw_stereo_matches(img_l, img_r, lpts, rpts, max_draw=60):
    """Side-by-side left+right image with matched feature pairs."""
    h, w  = img_l.shape
    canvas = np.zeros((h, 2 * w), dtype=np.uint8)
    canvas[:, :w]  = img_l
    canvas[:, w:]  = img_r
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(lpts), size=min(max_draw, len(lpts)), replace=False)
    for i in idx:
        lp = (int(round(lpts[i, 0])), int(round(lpts[i, 1])))
        rp = (int(round(rpts[i, 0])) + w, int(round(rpts[i, 1])))
        color = tuple(int(c) for c in rng.integers(80, 256, 3))
        cv2.circle(canvas_bgr, lp, 3, (255, 100, 0), -1)
        cv2.circle(canvas_bgr, rp, 3, (0, 100, 255), -1)
        cv2.line(canvas_bgr, lp, rp, color, 1)
    return canvas_bgr


def draw_temporal_tracks(img_t, img_tp, lpts_t, lpts_tp, max_draw=60):
    """Side-by-side frames t and t+dt with feature displacement arrows."""
    h, w  = img_t.shape
    canvas = np.zeros((h, 2 * w), dtype=np.uint8)
    canvas[:, :w]  = img_t
    canvas[:, w:]  = img_tp
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    rng = np.random.default_rng(1)
    idx = rng.choice(len(lpts_t), size=min(max_draw, len(lpts_t)), replace=False)
    for i in idx:
        p0 = (int(round(lpts_t[i,  0])),      int(round(lpts_t[i,  1])))
        p1 = (int(round(lpts_tp[i, 0])) + w,  int(round(lpts_tp[i, 1])))
        cv2.circle(canvas_bgr, p0, 3, (255, 100, 0), -1)
        cv2.circle(canvas_bgr, p1, 3, (0, 100, 255), -1)
        cv2.arrowedLine(canvas_bgr, p0, p1, (0, 220, 0), 1, tipLength=0.02)
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

    # Pick 4 display frames spread across the sequence
    vis_frames = [0, T // 4, T // 2, 3 * T // 4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Feature Detection & Matching — dataset02", fontsize=13)

    dt_vis = 15   # temporal gap for the temporal-tracking panel

    for ax, t in zip(axes.ravel(), vis_frames):
        img_l = np.array(cam_imgs_L[t])
        img_r = np.array(cam_imgs_R[t])

        # Gather valid stereo pairs at this timestep
        valid_mask  = features[0, :, t] != -1
        valid_ids   = np.where(valid_mask)[0]

        if len(valid_ids) == 0:
            ax.imshow(img_l, cmap="gray")
            ax.set_title(f"t={t}  (no features)")
            ax.axis("off")
            continue

        lpts = features[:2, valid_ids, t].T      # (N, 2)
        rpts = features[2:, valid_ids, t].T      # (N, 2)

        panel = draw_stereo_matches(img_l, img_r, lpts, rpts)
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(f"t={t}  stereo pairs={len(valid_ids)}", fontsize=10)
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
