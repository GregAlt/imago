"""Python translation of the original `pcf.c` helper functions.

Provides: combine(im_bg, im_fg), edge((x,y), image), hough((x,y), image, init_angle, dt)
All image buffers are expected as bytes-like objects (e.g. `bytes` or `bytearray`).
"""
from math import sin, cos

def combine(im_bg, im_fg):
    """Return average background pixel value where foreground mask is set.

    Args:
        im_bg: bytes-like background image
        im_fg: bytes-like foreground mask (non-zero => include)

    Returns:
        float average value (0.0 if no foreground pixels)
    """
    # follow original C behavior where size comes from the second buffer
    size = len(im_fg)
    s = 0
    area = 0
    for i in range(size):
        if im_fg[i]:
            s += im_bg[i]
            area += 1
    return float(s) / area if area else 0.0


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
    img = image
    out = bytearray(size)

    # zero borders as in original C code
    for i in range(2 * x):
        if i < size:
            out[i] = 0
    for i in range(2 * x):
        idx = (y - 2) * x + i
        if 0 <= idx < size:
            out[idx] = 0
    for j in range(y):
        if x * j < size:
            out[x * j] = 0
        if x * j + 1 < size:
            out[x * j + 1] = 0
        if x * j + x - 2 < size:
            out[x * j + x - 2] = 0
        if x * j + x - 1 < size:
            out[x * j + x - 1] = 0

    # inner pixels: implement same neighbor sum as original
    for i in range(2, x - 2):
        for j in range(2, y - 2):
            idx = x * j + i
            # sum neighbors (same order as C code)
            s = (
                img[x * j + i - 2] + img[x * j + i - 1] + img[x * j + i + 1] + img[x * j + i + 2]
                + img[x * (j - 2) + i - 2] + img[x * (j - 2) + i - 1] + img[x * (j - 2) + i] + img[x * (j - 2) + i + 1] + img[x * (j - 2) + i + 2]
                + img[x * (j - 1) + i - 2] + img[x * (j - 1) + i - 1] + img[x * (j - 1) + i] + img[x * (j - 1) + i + 1] + img[x * (j - 1) + i + 2]
                + img[x * (j + 2) + i - 2] + img[x * (j + 2) + i - 1] + img[x * (j + 2) + i] + img[x * (j + 2) + i + 1] + img[x * (j + 2) + i + 2]
                + img[x * (j + 1) + i - 2] + img[x * (j + 1) + i - 1] + img[x * (j + 1) + i] + img[x * (j + 1) + i + 1] + img[x * (j + 1) + i + 2]
                - (24 * img[idx])
            )
            if s < 0:
                s = 0
            if s > 255:
                s = 255
            out[idx] = int(s)

    return bytes(out)


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
    img = image

    # accumulator matrix as flat list [angle_index * x + column]
    matrix = [0] * size

    cx = x / 2.0
    cy = y / 2.0

    for i in range(x):
        for j in range(y):
            if img[j * x + i]:
                for a in range(y):
                    ang = (dt * a) + init_angle
                    distance = ((i - cx) * sin(ang)) + ((j - cy) * -cos(ang)) + cx
                    column = int(round(distance))
                    if 0 <= column < x:
                        matrix[a * x + column] += 1

    # normalize matrix to 0..255
    minv = min(matrix)
    maxv = max(matrix)
    denom = (maxv - minv + 1)
    out = bytearray(size)
    for i in range(size):
        val = int(((matrix[i] - minv) / float(denom)) * 256.0)
        if val < 0:
            val = 0
        if val > 255:
            val = 255
        out[i] = val

    return bytes(out)


__all__ = ["combine", "edge", "hough"]
