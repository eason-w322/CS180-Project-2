import os
import numpy as np
import warnings
warnings.filterwarnings("ignore") # ignores the runtime warnings
# save-only (no pop-up windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.util import img_as_float

from src.filters import (
    conv2d_four_forloops,   
    conv2d_two_forloops,     
    conv2d_scipy_same,       
    box_kernel,
    finite_differences,
)

# -------- Config --------
IMAGE_PATH = "data/selfie.jpg"
BOX_SIZE   = 9
SAVE_DIR   = "results/part1_1"
# ------------------------

def to_gray01(im):
    """Return grayscale float32 in [0,1], robust to dtype/alpha/NaNs."""
    im = img_as_float(im)
    im = np.nan_to_num(im, nan=0.0, posinf=1.0, neginf=0.0)
    im = np.clip(im, 0.0, 1.0)
    if im.ndim == 3:
        if im.shape[2] > 3:
            im = im[..., :3]    # drop alpha
        im = color.rgb2gray(im)
    return im.astype(np.float32)

def save_gray(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, arr, cmap="gray")

def main():
    img = to_gray01(io.imread(IMAGE_PATH))
    k   = box_kernel(BOX_SIZE)

    # Convolutions
    y_four = conv2d_four_forloops(img, k) 
    y_two  = conv2d_two_forloops(img, k)
    y_sp   = conv2d_scipy_same(img, k)

    print("Box conv diffs (max abs):")
    if y_four is not None:
        print("  |four-forloops - scipy|:", float(np.max(np.abs(y_four - y_sp))))
    print("  |two-forloops  - scipy|:", float(np.max(np.abs(y_two  - y_sp))))

    # Finite differences (for preview into 1.2)
    Dx, Dy = finite_differences()
    Ix = conv2d_two_forloops(img, Dx)
    Iy = conv2d_two_forloops(img, Dy)

    # Save results
    save_gray(f"{SAVE_DIR}/selfie_gray.png",           img)
    save_gray(f"{SAVE_DIR}/selfie_box_four.png",       y_four)
    save_gray(f"{SAVE_DIR}/selfie_box_two.png",        y_two)
    save_gray(f"{SAVE_DIR}/selfie_box_scipy.png",      y_sp)
    save_gray(f"{SAVE_DIR}/selfie_Ix.png",             Ix)
    save_gray(f"{SAVE_DIR}/selfie_Iy.png",             Iy)

    print(f"Saved images to {SAVE_DIR}/")

if __name__ == "__main__":
    main()