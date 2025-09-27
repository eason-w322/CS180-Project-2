import numpy as np
from scipy.signal import convolve2d as sp_convolve2d

# 1.1
def _flip_kernel(k: np.ndarray) -> np.ndarray:
    return np.flip((np.flip(k, axis = 0)), axis = 1)

def conv2d_four_forloops(img: np.ndarray, kernel: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    H, W = img.shape
    kh, kw = kernel.shape
    ch, cw = kh // 2, kw // 2
    k   = _flip_kernel(kernel).astype(np.float32, copy=False)
    img = img.astype(np.float32, copy=False)
    out = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            s = 0.0
            for u in range(-ch, ch + 1):
                for v in range(-cw, cw + 1):
                    ii = i + u
                    jj = j + v
                    ku = ch + u  
                    kv = cw + v   
                    if 0 <= ii < H and 0 <= jj < W:
                        s += img[ii, jj] * k[ku, kv]
            out[i, j] = s
    return out



def conv2d_two_forloops(img: np.ndarray, kernel: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    H, W = img.shape
    kh, kw = kernel.shape
    k = _flip_kernel(kernel)
    ch, cw = kh // 2, kw // 2
    out = np.zeros((H,W), dtype = np.float32)
    padded = np.pad(img, ((ch, ch),(cw, cw)), mode = 'constant', constant_values = pad_value)
    for i in range(H):
        for j in range(W):
            window = padded[i:i+ kh, j:j + kw]
            out[i,j] = np.sum(window * k, dtype=np.float32)
    return out
    

def conv2d_scipy_same(img: np.ndarray, kernel: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    return sp_convolve2d(img, kernel, mode='same', boundary='fill', fillvalue=pad_value).astype(np.float32)

def box_kernel(size: int) -> np.ndarray:
    assert size % 2 == 1
    k = np.ones((size,size), dtype = np.float32)
    k = k/(size**2)
    return k

def finite_differences():
    Dx = np.array([[1, 0, -1]], dtype = np.float32)      # (1, 3)
    Dy = np.array([[1], [0], [-1]], dtype = np.float32)  # (3, 1)
    return Dx, Dy

# 1.2
def rescale01(x: np.ndarray) -> np.ndarray:
    """Linearly rescale array to [0,1] for nicer visualization."""
    x = x.astype(np.float32, copy=False)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def gradients(img: np.ndarray, Dx: np.ndarray, Dy: np.ndarray, conv2d):
    Ix = conv2d(img, Dx).astype(np.float32, copy = False)
    Iy = conv2d(img, Dy).astype(np.float32, copy = False)
    G = np.sqrt(Ix**2 + Iy**2, dtype = np.float32)
    return Ix, Iy, G


def binarize_edges(G, thresh):
    G = G.astype(np.float32, copy=False)
    mask = (G >= np.float32(thresh))
    edges = mask.astype(np.float32)
    return edges

# 1.3
import cv2
from scipy.signal import convolve2d as sp_convolve2d

def gaussian1d_cv(sigma, ksize=None):
    if ksize is None:
        ksize = int(round(6*sigma + 1))
    elif ksize % 2 == 0:
        ksize += 1
    g_kernel1d = cv2.getGaussianKernel(ksize, sigma)
    return g_kernel1d

def gaussian2d_cv(sigma, ksize=None):
    g_kernel = gaussian1d_cv(sigma, ksize)
    g_kernel2d = (g_kernel @ g_kernel.T).astype(np.float32, copy = False)
    return g_kernel2d

def gaussian_blur_cv_2d(img, sigma, ksize=None, conv2d=conv2d_two_forloops):
    G = gaussian2d_cv(sigma, ksize)  
    out = conv2d(img, G, 0)
    return out.astype(np.float32, copy=False)

def dog_kernels_cv(sigma, ksize=None):
    G = gaussian2d_cv(sigma, ksize)       
    Dx, Dy = finite_differences()
    convx = conv2d_two_forloops(G, Dx)
    convy = conv2d_two_forloops(G, Dy)
    return convx, convy

# 2.1
def _ensure_odd_ksize(sigma, ksize):
    if ksize is None:
        ksize = int(round(6 * float(sigma) + 1))
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    return ksize

def _impulse_like(shape):
    h, w = shape
    assert h == w and (h % 2 == 1)
    k = np.zeros((h, w), dtype=np.float32)
    k[h // 2, w // 2] = 1.0
    return k

def gaussian_blur(img, sigma=1.5, ksize=None):
    G = gaussian2d_cv(sigma, ksize)
    img = img.astype(np.float32, copy = False)
    if img.ndim == 2:
        return conv2d_two_forloops(img, G)
    elif img.ndim ==3:
        conv = [conv2d_two_forloops(img[..., c], G) for c in range(img.shape[2])]
        out = np.stack(conv, axis = 2)
        return out


def unsharp_mask(img, sigma=1.5, amount=1.0, ksize=None):
    low  = gaussian_blur(img, sigma, ksize)
    high = img - low
    sharp = img + amount * high
    return low.astype(np.float32), high.astype(np.float32), sharp.astype(np.float32)

def unsharp_kernel(sigma=1.5, amount=1.0, ksize=None):
    ksize = _ensure_odd_ksize(sigma, ksize)
    G = gaussian2d_cv(sigma, ksize)
    K = (1.0 + float(amount)) * _impulse_like(G.shape) - float(amount) * G
    return K.astype(np.float32)

def apply_kernel(img, kernel):
    if img.ndim == 2:
        out = conv2d_two_forloops(img, kernel)
    else:
        chans = [conv2d_two_forloops(img[..., c], kernel) for c in range(img.shape[2])]
        out = np.stack(chans, axis=2)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

# 2.2 see align_image_code.py and run_part2_2.py

# 2.3
def gaussian_stack(img: np.ndarray, levels: int = 5, sigma: float = 2.0) -> list[np.ndarray]:
    """
    Build a Gaussian *stack* (not pyramid). Each level is the same size.
    G0 = img
    Gk = gaussian_blur_cv_2d(G{k-1}, sigma),  for k=1..levels-1
    Returns a list [G0, G1, ..., G{L-1}] in float32.
    """
    assert levels >= 1
    g = img.astype(np.float32, copy=False)
    stack = [g]
    for _ in range(1, levels):
        g = gaussian_blur_cv_2d(g, sigma=sigma, ksize=None, conv2d=conv2d_two_forloops)
        stack.append(g.astype(np.float32, copy=False))
    return stack


def laplacian_stack(img: np.ndarray, levels: int = 5, sigma: float = 2.0) -> list[np.ndarray]:
    """
    Laplacian stack derived from the Gaussian stack:
      Lk = Gk - G{k+1} for k=0..L-2
      L{L-1} = G{L-1}  (residual low-frequency)
    Returns [L0, L1, ..., L{L-1}] in float32.
    """
    G = gaussian_stack(img, levels=levels, sigma=sigma)
    L = []
    for k in range(levels - 1):
        L.append((G[k] - G[k + 1]).astype(np.float32, copy=False))
    L.append(G[-1])  # residual
    return L


def stack_for_display(stack: list[np.ndarray], mode: str = "symm") -> list[np.ndarray]:
    """
    Prepare a stack for saving/viewing.
      mode = "symm": symmetric scaling to [-1,1] -> [0,1] (good for Laplacians)
      mode = "01"  : linear rescale to [0,1] (good for Gaussians)
    """
    vis = []
    for arr in stack:
        a = arr.astype(np.float32, copy=False)
        if mode == "symm":
            m = float(np.max(np.abs(a)))
            v = np.zeros_like(a) if m == 0 else 0.5 + 0.5 * (a / m)
        else:  # "01"
            mn, mx = float(a.min()), float(a.max())
            v = np.zeros_like(a) if mx <= mn else (a - mn) / (mx - mn)
        vis.append(v.astype(np.float32, copy=False))
    return vis


# 2.4
def recon_from_laplacian(L: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct an image from a Laplacian stack (no downsampling case):
      sum of all levels (last level is low-frequency residual).
    """
    acc = np.zeros_like(L[0], dtype=np.float32)
    for lvl in L:
        acc = acc + lvl.astype(np.float32, copy=False)
    return np.clip(acc, 0.0, 1.0).astype(np.float32)


def blend_pyramids(A: np.ndarray, B: np.ndarray, M: np.ndarray,
                   levels: int = 5, sigma: float = 2.0) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Multi-resolution blend (no downsampling):
      - L_A, L_B : Laplacian stacks for A and B
      - G_M      : Gaussian stack for mask M in [0,1]
      - L_blend[k] = G_M[k]*L_A[k] + (1-G_M[k])*L_B[k]
      - recon = sum_k L_blend[k]
    Returns (L_A, L_B, G_M, L_blend, recon)
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    M = np.clip(M.astype(np.float32, copy=False), 0.0, 1.0)

    L_A = laplacian_stack(A, levels=levels, sigma=sigma)
    L_B = laplacian_stack(B, levels=levels, sigma=sigma)
    G_M = gaussian_stack(M, levels=levels, sigma=sigma)

    L_blend = []
    for k in range(levels):
        Mk = G_M[k]
        Lk = Mk * L_A[k] + (1.0 - Mk) * L_B[k]
        L_blend.append(Lk.astype(np.float32, copy=False))

    recon = recon_from_laplacian(L_blend)
    return L_A, L_B, G_M, L_blend, recon
