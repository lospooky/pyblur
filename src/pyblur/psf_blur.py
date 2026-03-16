import os.path

import numpy as np
from numpy.lib.npyio import NpzFile
from PIL import Image
from scipy.signal import convolve2d

from pyblur._validation import validate_image

_PSF_COUNT = 100
_psf_data: NpzFile | None = None


def _load_psf() -> NpzFile:
    global _psf_data
    if _psf_data is None:
        npz_path = os.path.join(os.path.dirname(__file__), "psf.npz")
        _psf_data = np.load(npz_path, allow_pickle=False)
    return _psf_data


def _psf_blur_impl(img: Image.Image, psfid: int) -> Image.Image:
    kernel = _load_psf()[str(psfid)]
    imgarray = np.array(img, dtype="float32")
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return Image.fromarray(convolved)


@validate_image
def psf_blur(img: Image.Image, psfid: int) -> Image.Image:
    """Apply a point-spread-function blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.
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
def psf_blur_random(img: Image.Image) -> Image.Image:
    """Apply a point-spread-function blur with a randomly chosen kernel.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    psf = _load_psf()
    psfid = np.random.randint(0, len(psf))
    return _psf_blur_impl(img, psfid)

