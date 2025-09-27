# run_part1_3.py — CS180 Project 2, Part 1.3 (edge comparison, minimal)
# Saves: gradient_blur.png, edges_blur.png, gradient_DoG.png, edges_DoG.png
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_float

from src.filters import (
    conv2d_two_forloops,
    finite_differences,
    gradients,
    binarize_edges,       # edges = (G >= thresh).astype(float32)
    gaussian_blur_cv_2d,  # single 2-D Gaussian blur (OpenCV-based kernel)
    dog_kernels_cv,       # DoG-x / DoG-y kernels from cv2 Gaussian
)

# ---- settings ----
CAM_PATH   = "data/cameraman.png"
SAVE_DIR   = "results/part1_3"
SIGMA      = 1.5          # blur strength
KSIZE      = None         # None => auto ~ 6*sigma + 1 (odd)
THRESHOLD  = 0.20         # ABSOLUTE threshold applied to gradient magnitudes
# -------------------

def to_gray01(im):
    im = img_as_float(im)
    im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
    im = np.clip(im, 0.0, 1.0)
    if im.ndim == 3:
        if im.shape[2] > 3:
            im = im[..., :3]  # drop alpha
        im = color.rgb2gray(im)
    return im.astype(np.float32)

def save_gray(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr, cmap="gray")

def main():
    img = to_gray01(io.imread(CAM_PATH))

    # ----- Pipeline A: Blur (2-D) -> finite differences -> magnitude -> threshold
    img_blur = gaussian_blur_cv_2d(img, sigma=SIGMA, ksize=KSIZE, conv2d=conv2d_two_forloops)
    Dx, Dy   = finite_differences()
    Ix_b, Iy_b, G_b = gradients(img_blur, Dx, Dy, conv2d=conv2d_two_forloops)
    edges_blur = binarize_edges(G_b, THRESHOLD)

    # ----- Pipeline B: DoG (single-step derivatives) -> magnitude -> threshold
    kx, ky = dog_kernels_cv(SIGMA, KSIZE)     # DoG-x / DoG-y kernels
    Ix_d   = conv2d_two_forloops(img, kx)
    Iy_d   = conv2d_two_forloops(img, ky)
    G_d    = np.sqrt(Ix_d * Ix_d + Iy_d * Iy_d).astype(np.float32)
    edges_dog = binarize_edges(G_d, THRESHOLD)

    # Save outputs
    save_gray(f"{SAVE_DIR}/gradient_blur.png",   G_b)
    save_gray(f"{SAVE_DIR}/edges_blur.png",      edges_blur)
    save_gray(f"{SAVE_DIR}/gradient_DoG.png",    G_d)
    save_gray(f"{SAVE_DIR}/edges_DoG.png",       edges_dog)

    print(f"Saved to {SAVE_DIR}/")
    print(f"Absolute threshold used: {THRESHOLD:.4f}  |  sigma={SIGMA}")

if __name__ == "__main__":
    main()