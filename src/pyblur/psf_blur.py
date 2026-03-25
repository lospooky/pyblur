import numpy as np
from PIL import Image

from pyblur._backends import Backend, get_backend
from pyblur._kernels import psf_kernel
from pyblur._validation import _SUPPORTED_MODES, validate_image, validate_mode

_PSF_COUNT = 100


def _psf_blur_impl(img: Image.Image, psfid: int, backend: Backend) -> Image.Image:
    return backend.apply_kernel(img, psf_kernel(psfid))


@validate_image
@validate_mode(_SUPPORTED_MODES)
def psf_blur(img: Image.Image, psfid: int, *, backend: str | Backend | None = None) -> Image.Image:
    """Apply a point-spread-function blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.
    psfid : int
        Index of the PSF kernel. Must be an integer in ``[0, 99]``.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    if not isinstance(psfid, int) or not (0 <= psfid < _PSF_COUNT):
        raise ValueError(
            f"psf_blur() psfid must be an integer in [0, {_PSF_COUNT - 1}], got {psfid!r}"
        )
    return _psf_blur_impl(img, psfid, get_backend(backend))


@validate_image
@validate_mode(_SUPPORTED_MODES)
def psf_blur_random(img: Image.Image, *, backend: str | Backend | None = None) -> Image.Image:
    """Apply a point-spread-function blur with a randomly chosen kernel.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale (``'L'``) or RGB (``'RGB'``) input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    psfid = np.random.randint(0, _PSF_COUNT)
    return _psf_blur_impl(img, psfid, get_backend(backend))

