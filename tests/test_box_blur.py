"""Tests for pyblur.box_blur."""
import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur
from pyblur._validation import _KERNEL_DIMS
from pyblur.box_blur import _box_kernel


class TestBoxKernel:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_shape(self, dim: int) -> None:
        k = _box_kernel(dim)
        assert k.shape == (dim, dim)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_sums_to_one(self, dim: int) -> None:
        k = _box_kernel(dim)
        assert pytest.approx(k.sum(), abs=1e-6) == 1.0

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_uniform(self, dim: int) -> None:
        k = _box_kernel(dim)
        expected = 1.0 / (dim * dim)
        assert np.allclose(k, expected)


class TestBoxBlur:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_returns_pil_image(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(gray_img, dim)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_size(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(gray_img, dim)
        assert_same_size(out, gray_img)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_output_mode(self, gray_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(gray_img, dim)
        assert out.mode == gray_img.mode

    def test_invalid_img_type(self, gray_img: Image.Image) -> None:
        with pytest.raises(TypeError, match="box_blur\\(\\)"):
            pyblur.box_blur("not_an_image", 5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_dim", [0, 1, 4, 6, 10, -1, 3.0, "3", None])
    def test_invalid_dim(self, gray_img: Image.Image, bad_dim) -> None:
        with pytest.raises(ValueError, match="box_blur\\(\\)"):
            pyblur.box_blur(gray_img, bad_dim)

    def test_flat_image_unchanged(self) -> None:
        """Interior pixels of a constant image are unchanged by a box blur.

        Border pixels are affected by the convolve2d fillvalue, so only the
        interior (beyond the kernel radius of 2) is checked.
        """
        flat = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8))
        out = pyblur.box_blur(flat, 5)
        interior = np.array(out)[2:-2, 2:-2]
        assert np.array_equal(interior, np.full(interior.shape, 128, dtype=np.uint8))


class TestBoxBlurRandom:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.box_blur_random(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.box_blur_random(gray_img)
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="box_blur_random\\(\\)"):
            pyblur.box_blur_random(42)  # type: ignore[arg-type]


class TestBoxBlurRGBSupport:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_returns_pil_image(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(rgb_img, dim)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_size(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(rgb_img, dim)
        assert_same_size(out, rgb_img)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_mode(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(rgb_img, dim)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_channels(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.box_blur(rgb_img, dim)
        assert np.array(out).shape[2] == 3

    def test_random_returns_pil_image(self, rgb_img: Image.Image) -> None:
        out = pyblur.box_blur_random(rgb_img)
        assert isinstance(out, Image.Image)

    def test_random_preserves_mode(self, rgb_img: Image.Image) -> None:
        out = pyblur.box_blur_random(rgb_img)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.box_blur(img, 5)

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_random_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.box_blur_random(img)
