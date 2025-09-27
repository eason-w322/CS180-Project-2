import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from pathlib import Path

from align_image_code import align_images

# ------------------------ Config ------------------------
IMG1_PATH = "data/joker.jpg"   # high-frequency source
IMG2_PATH = "data/batman.jpg"    # low-frequency source

# Name each run so outputs don’t overwrite
RUN_NAME  = "batman_joker"      
OUT_ROOT  = "results/part2_2"   # final OUTDIR will be results/part2_2/<RUN_NAME>

# Parameters
SIGMA_LOW  = 1    # Gaussian sigma for low-pass (im2)
SIGMA_HIGH = 2   # Gaussian sigma inside high-pass (im1’s blur)
ALPHA_HIGH = 1.0    # scale for high-pass before combining
BETA_LOW   = 1.0    # scale for low-pass before combining
# --------------------------------------------------------


# ---------------------- Small utils ---------------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_gray01(im):
    """Return grayscale float32 in [0,1]."""
    im = im.astype(np.float32, copy=False)
    if im.ndim == 3 and im.shape[2] >= 3:
        im = im[..., :3]
        w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        im = np.tensordot(im, w, axes=([-1], [0]))
    if im.size == 0:
        raise ValueError("to_gray01 received an empty image (size 0).")
    if im.max() > 1.5:
        im = im / 255.0
    return np.clip(im, 0.0, 1.0).astype(np.float32)

def save_img(path, arr):
    _ensure_dir(os.path.dirname(path))
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    plt.imsave(path, arr, cmap="gray")

def save_fft(path, img, title="FFT (log |F|)"):
    """Save log-magnitude FFT visualization."""
    eps = 1e-8
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(F) + eps)
    plt.figure()
    plt.title(title)
    plt.imshow(mag, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def center_crop(img, ratio=0.75):
    """Take a centered crop that keeps ratio% of H/W."""
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"center_crop: ratio must be in (0,1], got {ratio}")
    H, W = img.shape[:2]
    h = int(round(H * ratio))
    w = int(round(W * ratio))
    r0 = (H - h) // 2
    c0 = (W - w) // 2
    return img[r0:r0+h, c0:c0+w, ...]


# ----------------- Filtering primitives ----------------
def gaussian_kernel_2d(sigma, ksize=None):
    if ksize is None:
        ksize = int(round(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1
    r = (ksize - 1) // 2
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(xx*xx + yy*yy) / (2.0 * sigma * sigma))
    g /= np.sum(g)
    return g.astype(np.float32)

def blur_gaussian(img, sigma, ksize=None):
    k = gaussian_kernel_2d(sigma, ksize)
    out = convolve2d(img, k, mode="same", boundary="symm")
    return out.astype(np.float32)


# ---------------- Hybrid construction -------------------
def low_pass(img, sigma):
    return blur_gaussian(img, sigma)

def high_pass(img, sigma):
    return img - blur_gaussian(img, sigma)

def hybrid_image(im_high_src, im_low_src,
                 sigma_high, sigma_low,
                 alpha_high=1.0, beta_low=1.0):
    """
    Hybrid = beta * Low(im_low_src) + alpha * High(im_high_src)
    """
    low  = beta_low   * low_pass(im_low_src,  sigma_low)
    high = alpha_high * high_pass(im_high_src, sigma_high)
    hyb  = np.clip(low + high, 0.0, 1.0).astype(np.float32)
    return low, high, hyb


# ------------------------- Main -------------------------
def main():
    # Resolve run name + directory
    stem1 = Path(IMG1_PATH).stem
    stem2 = Path(IMG2_PATH).stem
    run_name = RUN_NAME.strip() if RUN_NAME.strip() else f"{stem1}_{stem2}"
    OUTDIR = os.path.join(OUT_ROOT, run_name)
    _ensure_dir(OUTDIR)

    # Load & align
    im1 = plt.imread(IMG1_PATH)
    im2 = plt.imread(IMG2_PATH)
    im1_aligned, im2_aligned = align_images(im1, im2)

    # Save aligned originals
    save_img(f"{OUTDIR}/{run_name}_orig_high.png", to_gray01(im1_aligned))
    save_img(f"{OUTDIR}/{run_name}_orig_low.png", to_gray01(im2_aligned))

    # Center crop
    im1_aligned = center_crop(im1_aligned, ratio=0.83)
    im2_aligned = center_crop(im2_aligned, ratio=0.83)

    im1g = to_gray01(im1_aligned)
    im2g = to_gray01(im2_aligned)

    # Save FFTs of originals
    save_fft(f"{OUTDIR}/{run_name}_fft_orig_high.png", im1g, "FFT of High Source")
    save_fft(f"{OUTDIR}/{run_name}_fft_orig_low.png", im2g, "FFT of Low Source")

    # Build hybrid
    low, high, hyb = hybrid_image(
        im_high_src=im1g, im_low_src=im2g,
        sigma_high=SIGMA_HIGH, sigma_low=SIGMA_LOW,
        alpha_high=ALPHA_HIGH, beta_low=BETA_LOW
    )

    # Visualization scaling for high-pass
    high_vis = (high - high.min()) / (high.max() - high.min() + 1e-8)

    # Save components
    save_img(f"{OUTDIR}/{run_name}_lowpass.png", low)
    save_img(f"{OUTDIR}/{run_name}_highpass_vis.png", high_vis)
    save_img(f"{OUTDIR}/{run_name}_hybrid.png", hyb)

    # Save FFTs of each
    save_fft(f"{OUTDIR}/{run_name}_fft_lowpass.png", low, "FFT of Low-Pass")
    save_fft(f"{OUTDIR}/{run_name}_fft_highpass.png", high_vis, "FFT of High-Pass")
    save_fft(f"{OUTDIR}/{run_name}_fft_hybrid.png", hyb, "FFT of Hybrid")

    print(f"Saved results to {OUTDIR}/")
    print("Files saved:")
    print(f" - Originals: {run_name}_orig_high.png, {run_name}_orig_low.png")
    print(f" - FFTs of originals: {run_name}_fft_orig_high.png, {run_name}_fft_orig_low.png")
    print(f" - Low-pass & FFT: {run_name}_lowpass.png, {run_name}_fft_lowpass.png")
    print(f" - High-pass & FFT: {run_name}_highpass_vis.png, {run_name}_fft_highpass.png")
    print(f" - Hybrid & FFT: {run_name}_hybrid.png, {run_name}_fft_hybrid.png")


if __name__ == "__main__":
    main()