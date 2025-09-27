# run2.py — CS180 Project 2, Part 1.2 (minimal)
# Quiet (no popups), saves only edges + a visual of G.

import warnings; warnings.filterwarnings("ignore")
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import img_as_float

from src.filters import (
    conv2d_two_forloops,
    finite_differences,
    gradients,          # uses your chosen conv2d
    binarize_edges,     # explicit threshold version
    rescale01,
)

# ---- settings ----
CAM_PATH  = "data/cameraman.png"   # your file
SAVE_DIR  = "results/part1_2"
THRESH    = 0.34                  # tweak by eye

def to_gray01(im):
    im = img_as_float(im)
    im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
    im = np.clip(im, 0.0, 1.0)
    if im.ndim == 3:
        if im.shape[2] > 3:
            im = im[..., :3]      # drop alpha if present
        im = color.rgb2gray(im)
    return im.astype(np.float32)

def save_gray(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr, cmap="gray")

def main():
    img = to_gray01(io.imread(CAM_PATH))
    Dx, Dy = finite_differences()

    # derivatives + grad magnitude (two-for-loops conv)
    Ix, Iy, G = gradients(img, Dx, Dy, conv2d=conv2d_two_forloops)

    # binary edge map with explicit threshold
    edges = binarize_edges(G, thresh=THRESH)

    # save only what's essential
    save_gray(f"{SAVE_DIR}/binarize_edges.png", edges)      
    save_gray(f"{SAVE_DIR}/gradient.png", G)  
    save_gray(f"{SAVE_DIR}/Ix.png", Ix)
    save_gray(f"{SAVE_DIR}/Iy.png", Iy)

    print(f"Saved edges and G_vis to {SAVE_DIR}/ (THRESH={THRESH})")

if __name__ == "__main__":
    main()