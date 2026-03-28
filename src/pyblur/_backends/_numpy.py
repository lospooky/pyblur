"""Pure numpy convolution backend (no scipy or scikit-image required)."""
import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _convolve2d(
    channel: NDArray[np.float32], kernel: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """Convolve a 2-D float32 channel with *kernel* (same-size output, fill=255)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(
        channel,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=255.0,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    result = (windows * kernel).sum(axis=(-2, -1))
    return result.clip(0, 255).astype("uint8")


def _gaussian_kernel2d(sigma: float) -> NDArray[np.float32]:
    """Build a 2-D Gaussian kernel for the given *sigma*."""
    radius = max(1, math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k1d = np.exp(-0.5 * (x / sigma) ** 2)
    k2d = np.outer(k1d, k1d).astype(np.float32)
    k2d /= k2d.sum()
    return k2d


class PilNumpyBackend:
    """Backend that uses PIL for I/O and pure numpy for convolution."""

    name = "numpy"

    def apply_kernel(
        self, img: Image.Image, kernel: NDArray[np.float32]
    ) -> Image.Image:
        """Convolve *img* with *kernel* and return the result as a PIL image."""
        imgarray = np.array(img, dtype="float32")
        if imgarray.ndim == 3:
            convolved = np.stack(
                [
                    _convolve2d(imgarray[..., c], kernel)
                    for c in range(imgarray.shape[2])
                ],
                axis=2,
            )
        else:
            convolved = _convolve2d(imgarray, kernel)
        return Image.fromarray(convolved)

    def gaussian_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        """Apply a Gaussian blur by building a kernel and convolving."""
        return self.apply_kernel(img, _gaussian_kernel2d(sigma))
