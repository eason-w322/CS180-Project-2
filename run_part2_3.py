# run_part2_3.py — Gaussian & Laplacian STACKS with *soft* mask (color)
# Produces a Fig-3.42-like panel + the final blend. No windows pop up.

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float

from src.filters import (
    gaussian_blur_cv_2d,     # your 2D Gaussian (no downsample)
    laplacian_stack,         # works per channel; returns [L0..L_{L-1}]
)

# ---------------- settings you can tweak ----------------
IMG_A    = "data/orange.jpeg"
IMG_B    = "data/apple.jpeg"
OUT_DIR  = "results/part2_3"
PANEL    = f"{OUT_DIR}/panel_soft_seam.png"
BLEND    = f"{OUT_DIR}/blend_soft.png"

LEVELS   = 8         # more levels -> smoother low-band mask
SIGMA    = 6       
MASK_DIR = "vertical"  # "vertical" or "horizontal"
RAMP_FRAC = 0.35     # cosine feather width as fraction of image size
# --------------------------------------------------------


# ------------ small utilities ------------
def rgb01(im):
    im = img_as_float(im)
    if im.ndim == 2:  # gray -> 3ch
        im = np.stack([im, im, im], axis=2)
    elif im.shape[2] > 3:
        im = im[..., :3]
    return np.clip(im.astype(np.float32), 0.0, 1.0)

def center_crop_to_match(a, b):
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    def crop(x):
        h, w = x.shape[:2]
        r0 = (h - H)//2; c0 = (w - W)//2
        return x[r0:r0+H, c0:c0+W, :]
    return crop(a), crop(b)

def vis01(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# cosine-ramped step (pre-feather) so even high band isn’t razor sharp
def feathered_step(h, w, direction="vertical", ramp_frac=0.06):
    m = np.zeros((h, w), np.float32)
    if direction == "vertical":
        mid = w // 2
        k   = max(1, int(round(ramp_frac * w)))
        left  = max(0, mid - k//2)
        right = min(w, mid + k//2)
        m[:, :left]  = 1.0
        m[:, right:] = 0.0
        if right > left:
            t = np.linspace(0, np.pi, right - left, dtype=np.float32)
            ramp = 0.5 * (1 + np.cos(t))  # 1 -> 0
            m[:, left:right] = ramp
    else:
        mid = h // 2
        k   = max(1, int(round(ramp_frac * h)))
        top = max(0, mid - k//2)
        bot = min(h, mid + k//2)
        m[:top, :] = 1.0
        m[bot:, :] = 0.0
        if bot > top:
            t = np.linspace(0, np.pi, bot - top, dtype=np.float32)
            ramp = 0.5 * (1 + np.cos(t))  # 1 -> 0
            m[top:bot, :] = ramp[:, None]
    return m

def gaussian_stack_cumulative(x, levels, sigma):
    out = [x.astype(np.float32)]
    cur = out[0]
    for _ in range(1, levels):
        cur = gaussian_blur_cv_2d(cur, sigma=sigma, ksize=None)
        out.append(cur.astype(np.float32))
    return out

def build_color_lap_stack(img, levels, sigma):
    Ls = []
    for c in range(3):
        Ls.append(laplacian_stack(img[..., c], levels=levels, sigma=sigma))
    return Ls

def color_band(Ls, idx):
    return np.dstack([Ls[0][idx], Ls[1][idx], Ls[2][idx]])
# -----------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    A = rgb01(io.imread(IMG_A))
    B = rgb01(io.imread(IMG_B))
    A, B = center_crop_to_match(A, B)
    H, W = A.shape[:2]

    # Color Laplacian stacks (no downsampling)
    LA = build_color_lap_stack(A, LEVELS, SIGMA)
    LB = build_color_lap_stack(B, LEVELS, SIGMA)

    # Soft mask stack: feathered step -> cumulative Gaussian stack
    M0  = feathered_step(H, W, MASK_DIR, ramp_frac=RAMP_FRAC)
    Ms  = gaussian_stack_cumulative(M0, LEVELS, SIGMA)

    # pick three bands to show
    idx_high, idx_mid, idx_low = 0, LEVELS//2, LEVELS-1
    idxs   = [idx_high, idx_mid, idx_low]
    labels = ["High band", "Mid band", "Low band (residual)"]

    # rows: (A*Mi, B*(1-Mi), sum)
    rows = []
    for idx in idxs:
        M  = Ms[idx]; M3 = np.dstack([M, M, M])
        Ab = color_band(LA, idx)
        Bb = color_band(LB, idx)
        left  = Ab * M3
        mid   = Bb * (1.0 - M3)
        right = left + mid
        rows.append((vis01(left), vis01(mid), vis01(right)))

    # reconstruct full blend using *raw* bands (no vis scaling)
    blended = np.zeros_like(A, dtype=np.float32)
    for i in range(LEVELS):
        M3 = np.dstack([Ms[i], Ms[i], Ms[i]])
        blended += color_band(LA, i) * M3 + color_band(LB, i) * (1.0 - M3)
    blended = np.clip(blended, 0.0, 1.0)

    # panel
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    for r, (L, R, S) in enumerate(rows):
        for c, img in enumerate((L, R, S)):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            if r == 0:
                ax.set_title(["Apple × M", "Orange × (1−M)", "Sum"][c], fontsize=11)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=11)
            ax.axis("off")

    axes[3, 0].imshow(A);        axes[3, 0].set_title("Apple");   axes[3, 0].axis("off")
    axes[3, 1].imshow(B);        axes[3, 1].set_title("Orange");  axes[3, 1].axis("off")
    axes[3, 2].imshow(blended);  axes[3, 2].set_title("Blended"); axes[3, 2].axis("off")
    axes[3, 0].set_ylabel("Full image", fontsize=11)

    fig.savefig(PANEL, dpi=220, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(BLEND, blended)
    print(f"Saved panel to {PANEL}")
    print(f"Saved blended image to {BLEND}")
    print(f"(levels={LEVELS}, sigma={SIGMA}, ramp_frac={RAMP_FRAC})")

if __name__ == "__main__":
    main()