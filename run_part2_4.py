import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize as imresize

# ---- helpers from src/filters.py ----
from src.filters import (
    gaussian_blur_cv_2d,   # size-preserving Gaussian blur (normalized kernel, reflect padding)
    gaussian_stack,        # no downsampling
    laplacian_stack,       # returns [L0..L{L-1}] with last = residual
)

# ================== USER SETTINGS ==================
IMG_A = "data/orange.jpeg"     # base image
IMG_B = "data/apple.jpeg"    # insert image

# Mask mode: "step_vertical" | "step_horizontal" | "ellipse"
MASK_MODE = "step_vertical"

# Ellipse params (used only when MASK_MODE == "ellipse")
SCALE_B      = 0.38          # B width as fraction of A width (0.25–0.40 typical)
POS_B        = (0.53, 0.60)  # (cx, cy) in [0,1] on A
ELLIPSE_PAD  = (0.06, 0.12)  # inner padding inside ellipse (frac of tw, th)
FEATHER_FRAC = 0.05          # mask feather sigma as frac of min(H,W)

# Step-mask controls (used only for step modes)
STEP_DIRECTION = "horizontal"  # not strictly needed; derived from MASK_MODE
SEAM_FRAC      = 0.50          # step location as fraction of width/height
PREBLUR_FRAC   = 0.08          # pre-blur the binary step (0.06–0.12 good)

# Stacks
LEVELS = 10
SIGMA  = 8.0

# Optional local tone match (helps remove residual color jumps)
COLOR_MATCH = False

# Outputs
RUN_NAME   = "oraple"   # e.g. "eye_hand", "orapple"
OUT_DIR    = f"results/part2_4/{RUN_NAME}"
BLEND_PATH = f"{OUT_DIR}/blend.png"
SAVE_PANEL = True
PANEL_PATH = f"{OUT_DIR}/panel.png"
# ====================================================


# ----------------– utilities ----------------–
def rgb01(im):
    im = img_as_float(im)
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)
    elif im.shape[2] > 3:
        im = im[..., :3]
    return np.clip(im.astype(np.float32), 0.0, 1.0)

def build_color_lap_stack(img, levels, sigma):
    Lr = laplacian_stack(img[..., 0], levels=levels, sigma=sigma)
    Lg = laplacian_stack(img[..., 1], levels=levels, sigma=sigma)
    Lb = laplacian_stack(img[..., 2], levels=levels, sigma=sigma)
    return [np.dstack([Lr[i], Lg[i], Lb[i]]).astype(np.float32) for i in range(levels)]

def vis01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# ---- ellipse helpers ----
def make_fullsize_ellipse_mask(A, tw, th, cx01, cy01, pad_x_frac, pad_y_frac, feather_frac):
    """Return (M_soft, (x0,y0)) full-size mask with a feathered ellipse; white selects B."""
    H, W = A.shape[:2]
    cx, cy = int(cx01 * W), int(cy01 * H)
    x0, y0 = cx - tw // 2, cy - th // 2

    # ellipse radii after inner padding
    rx = max(1, int(0.5 * tw * (1 - 2 * pad_x_frac)))
    ry = max(1, int(0.5 * th * (1 - 2 * pad_y_frac)))

    M_hard = np.zeros((H, W), dtype=np.float32)
    x_min, x_max = max(0, x0), min(W, x0 + tw)
    y_min, y_max = max(0, y0), min(H, y0 + th)
    if x_max > x_min and y_max > y_min:
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        ex = (xx - cx) / float(rx)
        ey = (yy - cy) / float(ry)
        M_hard[y_min:y_max, x_min:x_max][(ex * ex + ey * ey) <= 1.0] = 1.0

    sigma = max(1.0, float(feather_frac) * min(H, W))
    M_soft = gaussian_blur_cv_2d(M_hard, sigma=sigma, ksize=None)
    return np.clip(M_soft, 0.0, 1.0).astype(np.float32), (x0, y0)

def place_B_on_fullsize_canvas_masked(A, Bsrc, tw, th, x0, y0, M_soft):
    """Pre-compose B over A with the SAME soft mask, so B==A outside ellipse (no square)."""
    H, W = A.shape[:2]
    B_resized = imresize(Bsrc, (th, tw), anti_aliasing=True, preserve_range=True).astype(np.float32)

    # Put resized B on a zero canvas at (x0,y0)
    B_patch = np.zeros_like(A, dtype=np.float32)
    ys0, ys1 = max(0, y0), min(H, y0 + th)
    xs0, xs1 = max(0, x0), min(W, x0 + tw)
    by0, by1 = max(0, -y0), th - max(0, (y0 + th) - H)
    bx0, bx1 = max(0, -x0), tw - max(0, (x0 + tw) - W)
    if ys1 > ys0 and xs1 > xs0 and by1 > by0 and bx1 > bx0:
        B_patch[ys0:ys1, xs0:xs1, :] = B_resized[by0:by1, bx0:bx1, :]

    M3 = np.dstack([M_soft] * 3).astype(np.float32)
    return (1.0 - M3) * A + M3 * B_patch

def local_color_match(B, A, M_soft):
    """Simple mean/std match inside mask so B doesn't look pale against A."""
    eps = 1e-6
    M = M_soft[..., None]
    w = M.mean() + eps
    for c in range(3):
        a_mean = (A[..., c] * M[..., 0]).sum() / w
        b_mean = (B[..., c] * M[..., 0]).sum() / w
        a_std  = np.sqrt(((A[..., c] - a_mean) ** 2 * M[..., 0]).sum() / w + eps)
        b_std  = np.sqrt(((B[..., c] - b_mean) ** 2 * M[..., 0]).sum() / w + eps)
        B[..., c] = a_mean + (B[..., c] - b_mean) * (a_std / max(b_std, 1e-3))
    return np.clip(B, 0.0, 1.0)

def make_step_mask(h, w, orientation="vertical", frac=0.5, preblur_frac=0.0):
    m = np.zeros((h, w), np.float32)
    if orientation == "vertical":
        m[:, : int(w * frac)] = 1.0
    else:  # horizontal
        m[: int(h * frac), :] = 1.0

    if preblur_frac and preblur_frac > 0:
        sigma = float(preblur_frac) * min(h, w)
        m = gaussian_blur_cv_2d(m, sigma=sigma, ksize=None)
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
    return m

# ---- panel (optional) ----
def save_panel(A, Bsrc, GM, LA, LB, blended, path):
    idxs   = [0, len(GM)//2, len(GM)-1]
    labels = ["High band", "Mid band", "Low band (residual)"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.04, hspace=0.10)
    titles = ["A × M", "B × (1−M)", "Sum"]

    for r, (idx, label) in enumerate(zip(idxs, labels)):
        axes[r, 0].imshow(GM[idx], cmap="gray")
        axes[r, 0].set_title("Mask band", fontsize=11)
        axes[r, 0].set_ylabel(label, fontsize=11)
        axes[r, 0].axis("off")

        M3 = np.dstack([GM[idx]] * 3)
        left  = LA[idx] * (1.0 - M3)      # A’s contribution at this band
        right = LB[idx] * M3              # B’s contribution at this band
        for c, img in enumerate([vis01(left), vis01(right), vis01(left + right)]):
            ax = axes[r, c+1]
            ax.imshow(np.clip(img, 0, 1))
            if r == 0: ax.set_title(titles[c], fontsize=11)
            ax.axis("off")

    axes[3, 0].imshow(A);    axes[3, 0].set_title("A");        axes[3, 0].axis("off")
    axes[3, 1].imshow(Bsrc); axes[3, 1].set_title("B (src)");  axes[3, 1].axis("off")
    axes[3, 2].imshow(np.clip(np.dstack([GM[-1]]*3), 0, 1)); axes[3, 2].set_title("Mask M0"); axes[3, 2].axis("off")
    axes[3, 3].imshow(blended); axes[3, 3].set_title("Blended"); axes[3, 3].axis("off")

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

# ----------------– main ----------------–
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load inputs
    A = rgb01(io.imread(IMG_A))
    Bsrc = rgb01(io.imread(IMG_B))
    H, W = A.shape[:2]

    # Build M0 (HxW, white selects B) and full-size B
    if MASK_MODE == "ellipse":
        tw = int(SCALE_B * W)
        th = max(1, int(tw * (Bsrc.shape[0] / Bsrc.shape[1])))
        M0, (x0, y0) = make_fullsize_ellipse_mask(
            A, tw, th, POS_B[0], POS_B[1], ELLIPSE_PAD[0], ELLIPSE_PAD[1], FEATHER_FRAC
        )
        # light tail-trim, if you like (0.08–0.14 reasonable)
        TAIL = 0.10
        M0 = np.clip((M0 - TAIL) / (1.0 - TAIL), 0.0, 1.0).astype(np.float32)
        B = place_B_on_fullsize_canvas_masked(A, Bsrc, tw, th, x0, y0, M0)

    elif MASK_MODE in ("step_vertical", "step_horizontal"):
        orientation = "vertical" if MASK_MODE == "step_vertical" else "horizontal"
        M0 = make_step_mask(H, W, orientation=orientation,
                            frac=SEAM_FRAC, preblur_frac=PREBLUR_FRAC)
        B  = imresize(Bsrc, (H, W), anti_aliasing=True, preserve_range=True).astype(np.float32)

    else:
        raise ValueError(f"Unknown MASK_MODE: {MASK_MODE}")

    # Optional local color match (works for both modes)
    if COLOR_MATCH:
        B = local_color_match(B, A, M0)

    # Build stacks (no downsampling)
    LA = build_color_lap_stack(A, LEVELS, SIGMA)
    LB = build_color_lap_stack(B, LEVELS, SIGMA)
    GM = gaussian_stack(M0.astype(np.float32), levels=LEVELS, sigma=SIGMA)

    # Optional tiny bias toward B (uncomment if useful, e.g., sky dominance)
    # GM = [np.clip(g, 0, 1)**0.95 for g in GM]

    # Blend: white selects B, black selects A
    L_blend = []
    for i in range(LEVELS):
        M3 = np.dstack([GM[i]] * 3).astype(np.float32)
        L_blend.append((1.0 - M3) * LA[i] + M3 * LB[i])

    blended = np.clip(sum(L_blend), 0.0, 1.0).astype(np.float32)

    # Save
    plt.imsave(BLEND_PATH, np.ascontiguousarray(blended))
    print(f"Saved blend: {BLEND_PATH}")

    if SAVE_PANEL:
        save_panel(A, Bsrc, GM, LA, LB, blended, PANEL_PATH)
        print(f"Saved panel: {PANEL_PATH}")


if __name__ == "__main__":
    main()