"""OpenCV convolution backend (optional; requires opencv-python)."""

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class PilOpenCVBackend:
    """Backend that uses PIL for I/O and OpenCV for convolution and Gaussian blur."""

    name = "opencv"

    def apply_kernel(
        self, img: Image.Image, kernel: NDArray[np.float32]
    ) -> Image.Image:
        """Convolve *img* with *kernel* using cv2.filter2D.

        Borders are padded with 255 to match the scipy/numpy backend behaviour.
        """
        imgarray = np.array(img, dtype=np.float32)
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        padding = (
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
            if imgarray.ndim == 3
            else ((pad_h, pad_h), (pad_w, pad_w))
        )
        padded = np.pad(imgarray, padding, mode="constant", constant_values=255.0)
        result = cv2.filter2D(
            padded, ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_ISOLATED
        )
        h, w = imgarray.shape[:2]
        cropped = result[pad_h : pad_h + h, pad_w : pad_w + w]
        return Image.fromarray(cropped.clip(0, 255).astype(np.uint8))

    def gaussian_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        """Apply a Gaussian blur using cv2.GaussianBlur."""
        imgarray = np.array(img)
        result = cv2.GaussianBlur(imgarray, (0, 0), sigma)
        return Image.fromarray(result)
