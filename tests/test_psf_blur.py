"""Tests for pyblur.psf_blur."""
import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur
from pyblur.psf_blur import _PSF_COUNT


class TestPsfBlur:
    @pytest.mark.parametrize("psfid", [0, 1, 50, 98, 99])
    def test_returns_pil_image(self, gray_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(gray_img, psfid)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("psfid", [0, 1, 50, 98, 99])
    def test_preserves_size(self, gray_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(gray_img, psfid)
        assert_same_size(out, gray_img)

    @pytest.mark.parametrize("psfid", [0, 1, 50, 98, 99])
    def test_output_mode(self, gray_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(gray_img, psfid)
        assert out.mode == gray_img.mode

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="psf_blur\\(\\)"):
            pyblur.psf_blur({"img": "dict"}, 0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_id", [-1, 100, _PSF_COUNT, 0.5, "0", None])
    def test_invalid_psfid(self, gray_img: Image.Image, bad_id) -> None:
        with pytest.raises(ValueError, match="psf_blur\\(\\)"):
            pyblur.psf_blur(gray_img, bad_id)

    def test_boundary_ids(self, gray_img: Image.Image) -> None:
        """Edge IDs 0 and 99 must both work."""
        pyblur.psf_blur(gray_img, 0)
        pyblur.psf_blur(gray_img, _PSF_COUNT - 1)


class TestPsfBlurRandom:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.psf_blur_random(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.psf_blur_random(gray_img)
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="psf_blur_random\\(\\)"):
            pyblur.psf_blur_random(3.14)  # type: ignore[arg-type]


class TestPsfBlurRGBSupport:
    @pytest.mark.parametrize("psfid", [0, 50, 99])
    def test_returns_pil_image(self, rgb_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(rgb_img, psfid)
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("psfid", [0, 50, 99])
    def test_preserves_size(self, rgb_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(rgb_img, psfid)
        assert_same_size(out, rgb_img)

    @pytest.mark.parametrize("psfid", [0, 50, 99])
    def test_preserves_mode(self, rgb_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(rgb_img, psfid)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("psfid", [0, 50, 99])
    def test_preserves_channels(self, rgb_img: Image.Image, psfid: int) -> None:
        out = pyblur.psf_blur(rgb_img, psfid)
        assert np.array(out).shape[2] == 3

    def test_random_returns_pil_image(self, rgb_img: Image.Image) -> None:
        out = pyblur.psf_blur_random(rgb_img)
        assert isinstance(out, Image.Image)

    def test_random_preserves_mode(self, rgb_img: Image.Image) -> None:
        out = pyblur.psf_blur_random(rgb_img)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.psf_blur(img, 0)

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_random_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.psf_blur_random(img)
