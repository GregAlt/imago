"""Python/numpy translation of the original `pcf.c` helper functions.

Provides: combine(im_bg, im_fg), edge((x,y), image), hough((x,y), image, init_angle, dt)
All image buffers are expected as bytes-like objects (e.g. `bytes` or `bytearray`).
"""
from math import sin, cos
import numpy as np

def combine(im_bg, im_fg):
    """Return average background pixel value where foreground mask is set.

    Args:
        im_bg: bytes-like background image
        im_fg: bytes-like foreground mask (non-zero => include)

    Returns:
        float average value (0.0 if no foreground pixels)
    """
    bg = np.frombuffer(im_bg, dtype=np.uint8)
    fg = np.frombuffer(im_fg, dtype=np.uint8)

    if fg.size == 0:
        return 0.0

    mask = fg != 0
    if not np.any(mask):
        return 0.0

    return float(bg[mask].mean())


def edge(dim, image):
    """Simple edge detector translated from `py_edge` in `pcf.c`.

    Args:
        dim: tuple (x, y) width and height
        image: bytes-like input image (grayscale)

    Returns:
        bytes of same size with detected edges (clamped 0..255)
    """
    x, y = dim
    size = x * y
    img = np.frombuffer(image, dtype=np.uint8).reshape(y, x)
    out = np.zeros((y, x), dtype=np.uint8)

    k = 5 # kernel size
    k2 = k // 2
    if x >= k and y >= k:
        padded = np.pad(img.astype(np.int32), ((k2, k2), (k2, k2)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
        window_sums = windows.sum(axis=(-2, -1))
        center = img[k2:y - k2, k2:x - k2].astype(np.int32)
        inner = window_sums[k2:y - k2, k2:x - k2] - (k**2 * center)
        out[k2:y - k2, k2:x - k2] = np.clip(inner, 0, 255).astype(np.uint8)

    return bytes(out.tobytes())


def hough(dim, image, init_angle, dt):
    """Hough transform translation from `py_hough` in `pcf.c`.

    Args:
        dim: tuple (x, y) width and height
        image: bytes-like input image (binary/edge image)
        init_angle: initial angle in radians
        dt: angle step in radians

    Returns:
        bytes of size x*y containing the normalized Hough accumulator
    """
    x, y = dim
    size = x * y
    img = np.frombuffer(image, dtype=np.uint8).reshape(y, x)

    cx = x / 2.0
    cy = y / 2.0
    angles = np.arange(y, dtype=np.float64) * dt + init_angle
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    coords = np.argwhere(img != 0)
    if coords.size == 0:
        return bytes(size)

    # Separate coordinates and expand dimensions to shape (N, 1) for broadcasting
    i = coords[:, 0][:, None]  # Rows
    j = coords[:, 1][:, None]  # Columns

    # Compute distances using broadcasting -> Resulting shape is (N, y)
    distances = ((j - cx) * sin_a) - ((i - cy) * cos_a) + cx

    columns = np.rint(distances).astype(np.int32)
    valid = (columns >= 0) & (columns < x)

    # Generate the corresponding row indices 'a' (0 to y-1) for every non-zero coordinate
    a_indices = np.broadcast_to(np.arange(y)[None, :], columns.shape)

    rows_to_add = a_indices[valid]
    cols_to_add = columns[valid]

    # Unbuffered in-place accumulation to handle duplicate indices correctly
    matrix = np.zeros((y, x), dtype=np.int32)
    np.add.at(matrix, (rows_to_add, cols_to_add), 1)

    minv = matrix.min()
    maxv = matrix.max()
    denom = maxv - minv + 1
    out = np.zeros(size, dtype=np.uint8)
    if denom > 0:
        vals = ((matrix.ravel() - minv) / float(denom)) * 256.0
        vals = np.clip(np.floor(vals).astype(np.int32), 0, 255)
        out[:] = vals

    return bytes(out.tobytes())


__all__ = ["combine", "edge", "hough"]
