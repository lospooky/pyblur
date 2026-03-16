"""Tests for pyblur.gaussian_blur."""
import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur


class TestGaussianBlur:
    @pytest.mark.parametrize("bw", [0.5, 1.0, 1.5, 2.0, 3.5])
    def test_returns_pil_image(self, gray_img: Image.Image, bw: float) -> None:
        out = pyblur.gaussian_blur(gray_img, bw)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("bw", [0.5, 1.0, 1.5, 2.0, 3.5])
    def test_preserves_size(self, gray_img: Image.Image, bw: float) -> None:
        out = pyblur.gaussian_blur(gray_img, bw)
        assert_same_size(out, gray_img)

    @pytest.mark.parametrize("bw", [0.5, 1.0, 1.5, 2.0, 3.5])
    def test_output_mode_gray(self, gray_img: Image.Image, bw: float) -> None:
        out = pyblur.gaussian_blur(gray_img, bw)
        assert out.mode == gray_img.mode

    def test_rgb_image(self, rgb_img: Image.Image) -> None:
        """Gaussian blur delegates to PIL so it supports any mode."""
        out = pyblur.gaussian_blur(rgb_img, 1.0)
        assert isinstance(out, Image.Image)
        assert_same_size(out, rgb_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="gaussian_blur\\(\\)"):
            pyblur.gaussian_blur(np.zeros((10, 10)), 1.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_bw", [0, -1, -0.5, 0.0, "1.0", None])
    def test_invalid_bandwidth(self, gray_img: Image.Image, bad_bw) -> None:
        with pytest.raises(ValueError, match="gaussian_blur\\(\\)"):
            pyblur.gaussian_blur(gray_img, bad_bw)

    def test_flat_image_unchanged(self) -> None:
        flat = Image.fromarray(np.full((32, 32), 100, dtype=np.uint8))
        out = pyblur.gaussian_blur(flat, 2.0)
        assert np.array_equal(np.array(out), np.full((32, 32), 100, dtype=np.uint8))


class TestGaussianBlurRandom:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.gaussian_blur_random(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.gaussian_blur_random(gray_img)
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="gaussian_blur_random\\(\\)"):
            pyblur.gaussian_blur_random("oops")  # type: ignore[arg-type]
