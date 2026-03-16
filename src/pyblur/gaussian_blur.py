import numpy as np
from PIL import Image, ImageFilter

from pyblur._validation import validate_image

_GAUSSIAN_BANDWIDTHS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]


def _gaussian_blur_impl(img: Image.Image, bandwidth: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(bandwidth))


@validate_image
def gaussian_blur_random(img: Image.Image) -> Image.Image:
    """Apply a Gaussian blur with a randomly chosen bandwidth.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any mode).

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    bandwidth = _GAUSSIAN_BANDWIDTHS[np.random.randint(0, len(_GAUSSIAN_BANDWIDTHS))]
    return _gaussian_blur_impl(img, bandwidth)


@validate_image
def gaussian_blur(img: Image.Image, bandwidth: float) -> Image.Image:
    """Apply a Gaussian blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any mode).
    bandwidth : float
        Standard deviation of the Gaussian kernel. Must be positive.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
        raise ValueError(f"gaussian_blur() bandwidth must be a positive number, got {bandwidth!r}")
    return _gaussian_blur_impl(img, bandwidth)
