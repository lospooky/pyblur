import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from pyblur._kernels import box_kernel
from pyblur._validation import (
    _KERNEL_DIMS,
    _SUPPORTED_MODES,
    validate_dim,
    validate_image,
    validate_mode,
)


def _box_blur_impl(img: Image.Image, dim: int) -> Image.Image:
    imgarray = np.array(img, dtype="float32")
    kernel = box_kernel(dim)
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
def box_blur_random(img: Image.Image) -> Image.Image:
    """Apply a box blur with a randomly chosen kernel size.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    kerneldim = _KERNEL_DIMS[np.random.randint(0, len(_KERNEL_DIMS))]
    return _box_blur_impl(img, kerneldim)


@validate_image
@validate_mode(_SUPPORTED_MODES)
@validate_dim(_KERNEL_DIMS)
def box_blur(img: Image.Image, dim: int) -> Image.Image:
    """Apply a box (mean) blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.
    dim : int
        Kernel size. Must be one of 3, 5, 7, 9.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    return _box_blur_impl(img, dim)



