import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import line

from pyblur._validation import _KERNEL_DIMS, validate_dim, validate_image
from pyblur.line_dictionary import LineDictionary

_LINE_TYPES: list[Literal["full", "right", "left"]] = ["full", "right", "left"]

_line_dict = LineDictionary()


def _linear_motion_blur_impl(
    img: Image.Image,
    dim: int,
    angle: float,
    linetype: Literal["full", "right", "left"],
) -> Image.Image:
    imgarray = np.array(img, dtype="float32")
    kernel = _line_kernel(dim, angle, linetype)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return Image.fromarray(convolved)


@validate_image
def linear_motion_blur_random(img: Image.Image) -> Image.Image:
    """Apply a linear motion blur with randomly chosen parameters.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    line_length = _KERNEL_DIMS[np.random.randint(0, len(_KERNEL_DIMS))]
    line_type = _LINE_TYPES[np.random.randint(0, len(_LINE_TYPES))]
    angle = _random_angle(line_length)
    return _linear_motion_blur_impl(img, line_length, angle, line_type)


@validate_image
@validate_dim(_KERNEL_DIMS)
def linear_motion_blur(
    img: Image.Image, dim: int, angle: float, linetype: Literal["full", "right", "left"]
) -> Image.Image:
    """Apply a linear motion blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.
    dim : int
        Kernel size. Must be one of 3, 5, 7, 9.
    angle : float
        Motion angle in degrees. Snapped to the nearest valid angle for the
        chosen kernel size.
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


def _line_kernel(
    dim: int, angle: float, linetype: Literal["full", "right", "left"]
) -> NDArray[np.float32]:
    kernel_center = dim // 2
    angle = _sanitize_angle(kernel_center, angle)
    kernel = np.zeros((dim, dim), dtype=np.float32)
    anchors = list(_line_dict.lines[dim][angle])  # copy — never mutate shared dict data
    if linetype == 'right':
        anchors[0] = kernel_center
        anchors[1] = kernel_center
    if linetype == 'left':
        anchors[2] = kernel_center
        anchors[3] = kernel_center
    rr, cc = line(anchors[0], anchors[1], anchors[2], anchors[3])
    kernel[rr, cc] = 1
    kernel /= np.count_nonzero(kernel)
    return kernel


def _sanitize_angle(kernel_center: int, angle: float) -> float:
    num_lines = kernel_center * 4
    angle = math.fmod(angle, 180.0)
    valid_angles = np.linspace(0, 180, num_lines, endpoint=False)
    return float(_nearest_value(angle, valid_angles))


def _nearest_value(theta: float, valid_angles: NDArray[np.float64]) -> np.float64:
    idx = (np.abs(valid_angles - theta)).argmin()
    return valid_angles[idx]


def _random_angle(kerneldim: int) -> int:
    kernel_center = kerneldim // 2
    num_lines = kernel_center * 4
    valid_angles = np.linspace(0, 180, num_lines, endpoint=False)
    return int(valid_angles[np.random.randint(0, len(valid_angles))])
