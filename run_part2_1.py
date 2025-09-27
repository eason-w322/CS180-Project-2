import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float

from src.filters import (
    gaussian_blur,
    unsharp_mask,
    unsharp_kernel,
    apply_kernel,
)

# ---- settings ----
IN_PATH       = "data/pyramid.jpg"   # change this for each run
SAVE_ROOT     = "results/part2_1"
NAME          = "pyramid"       # <-- give each run a unique name (e.g., "taj", "dog", "custom1")
SIGMA         = 1.5              # blur scale
AMOUNT        = 1.0              # detail boost (unsharp strength)
TEST_BLUR_SIG = 2.0              # for the blur→resharpen demo
# -------------------

def to_float01(img):
    img = img_as_float(img)
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[..., :3]  # drop alpha
    return np.clip(img.astype(np.float32), 0.0, 1.0)

def save_img(path, arr, cmap=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.ndim == 2:
        plt.imsave(path, arr, cmap=cmap or "gray")
    else:
        plt.imsave(path, np.clip(arr, 0.0, 1.0))

def main():
    SAVE_DIR = os.path.join(SAVE_ROOT, NAME)  # make unique subfolder
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load; keep color if present
    img = to_float01(io.imread(IN_PATH))

    # A) Standard unsharp (two-step)
    low, high, sharp = unsharp_mask(img, sigma=SIGMA, amount=AMOUNT, ksize=None)

    # B) Single-convolution unsharp (equivalent)
    K = unsharp_kernel(sigma=SIGMA, amount=AMOUNT, ksize=None)
    sharp_one = apply_kernel(img, K)

    # Save main results
    save_img(f"{SAVE_DIR}/0_original.png", img)
    save_img(f"{SAVE_DIR}/1_blurred.png",  low)

    # High-frequency visualization
    hf = high
    hf_vis = (hf - hf.min()) / (hf.max() - hf.min() + 1e-8)
    save_img(f"{SAVE_DIR}/2_highfreq_vis.png", hf_vis)
    save_img(f"{SAVE_DIR}/3_sharp_two_step.png",   sharp)
    save_img(f"{SAVE_DIR}/4_sharp_single_conv.png", sharp_one)

    # C) Demo: blur a sharp image, then try to recover sharpness
    img_blurred   = gaussian_blur(img, sigma=TEST_BLUR_SIG)
    _, _, img_res = unsharp_mask(img_blurred, sigma=TEST_BLUR_SIG, amount=AMOUNT)
    save_img(f"{SAVE_DIR}/5_demo_blurred.png",     img_blurred)
    save_img(f"{SAVE_DIR}/6_demo_resharpened.png", img_res)

    print(f"Saved results to {SAVE_DIR}/")
    print(f"Params: sigma={SIGMA}, amount={AMOUNT}")

if __name__ == "__main__":
    main()
