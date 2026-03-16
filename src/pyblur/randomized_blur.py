import numpy as np
from PIL import Image

from pyblur._validation import validate_image
from pyblur.box_blur import box_blur_random
from pyblur.defocus_blur import defocus_blur_random
from pyblur.gaussian_blur import gaussian_blur_random
from pyblur.linear_motion_blur import linear_motion_blur_random
from pyblur.psf_blur import psf_blur_random

_BLUR_FUNCTIONS = [
    box_blur_random,
    defocus_blur_random,
    gaussian_blur_random,
    linear_motion_blur_random,
    psf_blur_random,
]


@validate_image
def randomized_blur(img: Image.Image) -> Image.Image:
    """Apply a randomly chosen blur to an image.

    One of box, defocus, Gaussian, linear motion, or PSF blur is selected
    uniformly at random and applied with its own random parameters.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    return _BLUR_FUNCTIONS[np.random.randint(0, len(_BLUR_FUNCTIONS))](img)
