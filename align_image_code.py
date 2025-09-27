import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr

def _prompt(title, im):
    plt.figure()
    plt.title(title)
    plt.imshow(im)
    plt.axis("off")

def get_points(im1, im2):
    print("Click TWO corresponding points on IMAGE 1 (press Enter when done)")
    _prompt("IMAGE 1: click two points", im1)
    p1, p2 = plt.ginput(2)
    plt.close()

    print("Click TWO corresponding points on IMAGE 2 (press Enter when done)")
    _prompt("IMAGE 2: click two points", im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C = im.shape[:2]
    rpad = int(abs(2*r + 1 - R))
    cpad = int(abs(2*c + 1 - C))
    return np.pad(
        im,
        [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
         (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad)]
        + ([(0, 0)] if im.ndim == 3 else []),
        mode="constant"
    )

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]])).astype(int)
    cy = np.round(np.mean([p1[1], p2[1]])).astype(int)
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)
    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.hypot(p2[1] - p1[1], p2[0] - p1[0])
    len2 = np.hypot(p4[1] - p3[1], p4[0] - p3[0])
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=-1 if im1.ndim == 3 else None, mode="reflect", anti_aliasing=True)
    else:
        im2 = sktr.rescale(im2, 1.0/dscale, channel_axis=-1 if im2.ndim == 3 else None, mode="reflect", anti_aliasing=True)
    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi, mode="edge", resize=False)
    return im1, dtheta

def match_img_size(im1, im2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    if h1 < h2:
        im2 = im2[int(np.floor((h2-h1)/2.)): -int(np.ceil((h2-h1)/2.)), :, ...]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1-h2)/2.)): -int(np.ceil((h1-h2)/2.)), :, ...]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2-w1)/2.)): -int(np.ceil((w2-w1)/2.)), ...]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1-w2)/2.)): -int(np.ceil((w1-w2)/2.)), ...]
    assert im1.shape == im2.shape
    return im1, im2

def _crop_common_valid_region(im1, im2, eps=1e-6):
    if im1.ndim == 3:
        m1 = (im1.sum(axis=2) > eps)
        m2 = (im2.sum(axis=2) > eps)
    else:
        m1 = (im1 > eps)
        m2 = (im2 > eps)

    m = m1 & m2
    # If alignment failed badly, fall back to central crop
    if not m.any():
        return im1, im2

    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return im1[r0:r1, c0:c1, ...], im2[r0:r1, c0:c1, ...]
    

def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    # >>> NEW: remove black triangular corners from rotation <<<
    im1, im2 = _crop_common_valid_region(im1, im2)
    return im1, im2
