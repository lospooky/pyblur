"""PIL + scipy convolution backend (default)."""
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from scipy.signal import convolve2d


class PilScipyBackend:
    """Backend that uses PIL for I/O and scipy for convolution."""

    name = "scipy"

    def apply_kernel(
        self, img: Image.Image, kernel: NDArray[np.float32]
    ) -> Image.Image:
        """Convolve *img* with *kernel* and return the result as a PIL image."""
        imgarray = np.array(img, dtype="float32")
        if imgarray.ndim == 3:
            convolved = np.stack(
                [
                    convolve2d(
                        imgarray[..., c], kernel, mode="same", fillvalue=255.0
                    ).astype("uint8")
                    for c in range(imgarray.shape[2])
                ],
                axis=2,
            )
        else:
            convolved = convolve2d(
                imgarray, kernel, mode="same", fillvalue=255.0
            ).astype("uint8")
        return Image.fromarray(convolved)

    def gaussian_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        """Apply a Gaussian blur using PIL's built-in fast path."""
        return img.filter(ImageFilter.GaussianBlur(sigma))
