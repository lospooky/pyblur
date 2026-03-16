import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.signal import convolve2d
from skimage.draw import disk

from pyblur._validation import _KERNEL_DIMS, validate_dim, validate_image


def _defocus_blur_impl(img: Image.Image, dim: int) -> Image.Image:
    imgarray = np.array(img, dtype="float32")
    kernel = _disk_kernel(dim)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    return Image.fromarray(convolved)


@validate_image
def defocus_blur_random(img: Image.Image) -> Image.Image:
    """Apply a defocus blur with a randomly chosen kernel size.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    kerneldim = _KERNEL_DIMS[np.random.randint(0, len(_KERNEL_DIMS))]
    return _defocus_blur_impl(img, kerneldim)


@validate_image
@validate_dim(_KERNEL_DIMS)
def defocus_blur(img: Image.Image, dim: int) -> Image.Image:
    """Apply a defocus (disk) blur to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Grayscale input image.
    dim : int
        Kernel size. Must be one of 3, 5, 7, 9.

    Returns
    -------
    PIL.Image.Image
        Blurred image with the same dimensions as the input.
    """
    return _defocus_blur_impl(img, dim)


def _disk_kernel(dim: int) -> NDArray[np.float32]:
    kernel = np.zeros((dim, dim), dtype=np.float32)
    center = dim // 2
    rr, cc = disk((center, center), center + 1, shape=kernel.shape)
    kernel[rr, cc] = 1
    if dim == 3 or dim == 5:
        kernel = _adjust(kernel, dim)
    kernel /= np.count_nonzero(kernel)
    return kernel


def _adjust(kernel: NDArray[np.float32], kernelwidth: int) -> NDArray[np.float32]:
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel
