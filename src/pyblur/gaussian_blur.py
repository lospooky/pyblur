import numpy as np
from PIL import Image

from pyblur._backends import Backend, get_backend
from pyblur._validation import validate_image

_GAUSSIAN_BANDWIDTHS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]


def _gaussian_blur_impl(img: Image.Image, bandwidth: float, backend: Backend) -> Image.Image:
    return backend.gaussian_blur(img, bandwidth)


@validate_image
def gaussian_blur_random(img: Image.Image, *, backend: str | Backend | None = None) -> Image.Image:
    """Apply a Gaussian blur with a randomly chosen bandwidth.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image. Grayscale (``'L'``) and RGB (``'RGB'``) are well-tested;
        PIL handles this natively for other modes too.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    bandwidth = _GAUSSIAN_BANDWIDTHS[np.random.randint(0, len(_GAUSSIAN_BANDWIDTHS))]
    return _gaussian_blur_impl(img, bandwidth, get_backend(backend))


@validate_image
def gaussian_blur(
    img: Image.Image, bandwidth: float, *, backend: str | Backend | None = None
) -> Image.Image:
    """Apply a Gaussian blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image. Grayscale (``'L'``) and RGB (``'RGB'``) are well-tested;
        PIL handles this natively for other modes too.
    bandwidth : float
        Standard deviation of the Gaussian kernel. Must be positive.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
        raise ValueError(f"gaussian_blur() bandwidth must be a positive number, got {bandwidth!r}")
    return _gaussian_blur_impl(img, bandwidth, get_backend(backend))
