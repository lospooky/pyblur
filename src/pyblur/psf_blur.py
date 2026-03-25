import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from pyblur._kernels import psf_kernel
from pyblur._validation import _SUPPORTED_MODES, validate_image, validate_mode

_PSF_COUNT = 100


def _psf_blur_impl(img: Image.Image, psfid: int) -> Image.Image:
    kernel = psf_kernel(psfid)
    imgarray = np.array(img, dtype="float32")
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
def psf_blur(img: Image.Image, psfid: int) -> Image.Image:
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
    return _psf_blur_impl(img, psfid)


@validate_image
@validate_mode(_SUPPORTED_MODES)
def psf_blur_random(img: Image.Image) -> Image.Image:
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
    return _psf_blur_impl(img, psfid)

