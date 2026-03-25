import numpy as np
from PIL import Image

from pyblur._backends import Backend, get_backend
from pyblur._kernels import box_kernel
from pyblur._validation import (
    _KERNEL_DIMS,
    _SUPPORTED_MODES,
    validate_dim,
    validate_image,
    validate_mode,
)


def _box_blur_impl(img: Image.Image, dim: int, backend: Backend) -> Image.Image:
    return backend.apply_kernel(img, box_kernel(dim))


@validate_image
@validate_mode(_SUPPORTED_MODES)
def box_blur_random(img: Image.Image, *, backend: str | Backend | None = None) -> Image.Image:
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
    return _box_blur_impl(img, kerneldim, get_backend(backend))


@validate_image
@validate_mode(_SUPPORTED_MODES)
@validate_dim(_KERNEL_DIMS)
def box_blur(img: Image.Image, dim: int, *, backend: str | Backend | None = None) -> Image.Image:
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
    return _box_blur_impl(img, dim, get_backend(backend))



