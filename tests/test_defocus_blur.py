"""Tests for pyblur.defocus_blur."""
import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur
from pyblur._kernels import disk_kernel
from pyblur._validation import _KERNEL_DIMS


class TestDiskKernel:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_shape(self, dim: int) -> None:
        k = disk_kernel(dim)
        assert k.shape == (dim, dim)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_sums_to_one(self, dim: int) -> None:
        k = disk_kernel(dim)
        assert pytest.approx(k.sum(), abs=1e-6) == 1.0

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_non_negative(self, dim: int) -> None:
        k = disk_kernel(dim)
        assert (k >= 0).all()


class TestDefocusBlur:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_returns_pil_image(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(gray_img, dim)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_size(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(gray_img, dim)
        assert_same_size(out, gray_img)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_output_mode(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(gray_img, dim)
        assert out.mode == gray_img.mode

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="defocus_blur\\(\\)"):
            pyblur.defocus_blur([1, 2, 3], 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_dim", [0, 2, 4, 6, 11, -3, 5.0, "5", None])
    def test_invalid_dim(self, gray_img: Image.Image, bad_dim) -> None:
        with pytest.raises(ValueError, match="defocus_blur\\(\\)"):
            pyblur.defocus_blur(gray_img, bad_dim)

    def test_flat_image_unchanged(self) -> None:
        """Interior pixels of a constant image are near-unchanged by a defocus blur.

        Border pixels are affected by the convolve2d fillvalue. The disk kernel
        has corner pixels zeroed, so the sum may not be exactly 1.0, causing
        ±1 uint8 rounding in the interior — tolerate that.
        """
        flat = Image.fromarray(np.full((32, 32), 200, dtype=np.uint8))
        out = pyblur.defocus_blur(flat, 5)
        interior = np.array(out)[2:-2, 2:-2].astype(int)
        assert np.all(np.abs(interior - 200) <= 1)


class TestDefocusBlurRandom:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.defocus_blur_random(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.defocus_blur_random(gray_img)
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="defocus_blur_random\\(\\)"):
            pyblur.defocus_blur_random(None)  # type: ignore[arg-type]


class TestDefocusBlurRGBSupport:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_returns_pil_image(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(rgb_img, dim)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_size(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(rgb_img, dim)
        assert_same_size(out, rgb_img)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_mode(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(rgb_img, dim)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_channels(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.defocus_blur(rgb_img, dim)
        assert np.array(out).shape[2] == 3

    def test_random_returns_pil_image(self, rgb_img: Image.Image) -> None:
        out = pyblur.defocus_blur_random(rgb_img)
        assert isinstance(out, Image.Image)

    def test_random_preserves_mode(self, rgb_img: Image.Image) -> None:
        out = pyblur.defocus_blur_random(rgb_img)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.defocus_blur(img, 5)

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_random_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.defocus_blur_random(img)
