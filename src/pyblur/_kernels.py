"""Pure kernel factories — no I/O, no PIL, no backend dependencies.

All functions return a normalised float32 ndarray suitable for convolution.
"""

import math
import os.path
from typing import Literal

import numpy as np
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# PSF data (loaded lazily from the bundled psf.npz)
# ---------------------------------------------------------------------------

_psf_data: NpzFile | None = None


def _load_psf() -> NpzFile:
    global _psf_data
    if _psf_data is None:
        npz_path = os.path.join(os.path.dirname(__file__), "psf.npz")
        _psf_data = np.load(npz_path, allow_pickle=False)
    return _psf_data


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def box_kernel(dim: int) -> NDArray[np.float32]:
    """Return a normalised *dim*×*dim* box (mean) kernel."""
    kernel = np.ones((dim, dim), dtype=np.float32)
    kernel /= np.count_nonzero(kernel)
    return kernel


def _disk_adjust(kernel: NDArray[np.float32], kernelwidth: int) -> NDArray[np.float32]:
    """Zero the four corners of a disk kernel (used for dim 3 and 5)."""
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


def disk_kernel(dim: int) -> NDArray[np.float32]:
    """Return a normalised *dim*×*dim* circular disk (defocus) kernel."""
    kernel = np.zeros((dim, dim), dtype=np.float32)
    center = dim // 2
    y, x = np.ogrid[:dim, :dim]
    mask = (y - center) ** 2 + (x - center) ** 2 < (center + 1) ** 2
    kernel[mask] = 1
    if dim == 3 or dim == 5:
        kernel = _disk_adjust(kernel, dim)
    kernel /= np.count_nonzero(kernel)
    return kernel


def line_endpoints(dim: int, angle: float) -> tuple[int, int, int, int]:
    """Compute boundary-crossing line endpoints for a *dim*×*dim* kernel at *angle* degrees.

    Angles outside ``[0°, 180°)`` are wrapped modulo 180°.  Returns
    ``(r0, c0, r1, c1)`` clamped to ``[0, dim-1]``.
    """
    c = dim // 2
    rad = math.radians(math.fmod(angle, 180.0))
    dr = -math.sin(rad)  # row delta (row 0 is top, so sin is negated)
    dc = math.cos(rad)   # col delta
    t_row = c / abs(dr) if abs(dr) > 1e-9 else float("inf")
    t_col = c / abs(dc) if abs(dc) > 1e-9 else float("inf")
    t = min(t_row, t_col)
    edge = dim - 1
    r0 = max(0, min(edge, int(round(c - t * dr))))
    c0 = max(0, min(edge, int(round(c - t * dc))))
    r1 = max(0, min(edge, int(round(c + t * dr))))
    c1 = max(0, min(edge, int(round(c + t * dc))))
    return r0, c0, r1, c1


def _bresenham_line(
    r0: int, c0: int, r1: int, c1: int
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return row/col index arrays for a Bresenham line from (r0,c0) to (r1,c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    rows: list[int] = [r0]
    cols: list[int] = [c0]
    while r0 != r1 or c0 != c1:
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
        rows.append(r0)
        cols.append(c0)
    return np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp)


def line_kernel(
    dim: int, angle: float, linetype: Literal["full", "right", "left"]
) -> NDArray[np.float32]:
    """Return a normalised *dim*×*dim* motion-line kernel."""
    kernel_center = dim // 2
    r0, c0, r1, c1 = line_endpoints(dim, angle)
    if linetype == "right":
        r0, c0 = kernel_center, kernel_center
    elif linetype == "left":
        r1, c1 = kernel_center, kernel_center
    rr, cc = _bresenham_line(r0, c0, r1, c1)
    kernel = np.zeros((dim, dim), dtype=np.float32)
    kernel[rr, cc] = 1
    kernel /= np.count_nonzero(kernel)
    return kernel


def psf_kernel(psfid: int) -> NDArray[np.float32]:
    """Return the pre-computed PSF kernel for the given *psfid*."""
    return _load_psf()[str(psfid)]
