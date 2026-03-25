from typing import Literal

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from pyblur._kernels import line_kernel
from pyblur._validation import (
    _KERNEL_DIMS,
    _SUPPORTED_MODES,
    validate_image,
    validate_mode,
    validate_odd_dim,
)

_LINE_TYPES: list[Literal["full", "right", "left"]] = ["full", "right", "left"]


def _linear_motion_blur_impl(
    img: Image.Image,
    dim: int,
    angle: float,
    linetype: Literal["full", "right", "left"],
) -> Image.Image:
    imgarray = np.array(img, dtype="float32")
    kernel = line_kernel(dim, angle, linetype)
    if imgarray.ndim == 3:
        convolved = np.stack(
            [convolve2d(imgarray[..., c], kernel, mode='same', fillvalue=255.0).astype("uint8")
             for c in range(imgarray.shape[2])],
            axis=2,
        )
    else:
        convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return Image.fromarray(convolved)


@validate_image
@validate_mode(_SUPPORTED_MODES)
def linear_motion_blur_random(img: Image.Image) -> Image.Image:
    """Apply a linear motion blur with randomly chosen parameters.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    line_length = _KERNEL_DIMS[np.random.randint(0, len(_KERNEL_DIMS))]
    line_type = _LINE_TYPES[np.random.randint(0, len(_LINE_TYPES))]
    angle = _random_angle()
    return _linear_motion_blur_impl(img, line_length, angle, line_type)


@validate_image
@validate_mode(_SUPPORTED_MODES)
@validate_odd_dim
def linear_motion_blur(
    img: Image.Image, dim: int, angle: float, linetype: Literal["full", "right", "left"]
) -> Image.Image:
    """Apply a linear motion blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.
    dim : int
        Kernel size. Must be an odd integer >= 3.
    angle : float
        Motion direction in degrees. Any float is accepted; values outside
        ``[0°, 180°)`` are wrapped modulo 180°.
    linetype : Literal["full", "right", "left"]
        ``'full'`` spans the entire kernel; ``'right'`` and ``'left'`` use
        only half the line.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    if linetype not in _LINE_TYPES:
        raise ValueError(
            f"linear_motion_blur() linetype must be one of {_LINE_TYPES}, got {linetype!r}"
        )
    return _linear_motion_blur_impl(img, dim, angle, linetype)





def _random_angle() -> float:
    return float(np.random.uniform(0, 180))
