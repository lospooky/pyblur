"""Tests for pyblur.randomized_blur."""
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur


class TestRandomizedBlur:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.randomized_blur(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.randomized_blur(gray_img)
        assert_same_size(out, gray_img)

    def test_output_mode(self, gray_img: Image.Image) -> None:
        out = pyblur.randomized_blur(gray_img)
        assert out.mode == gray_img.mode

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="randomized_blur\\(\\)"):
            pyblur.randomized_blur("not_an_image")  # type: ignore[arg-type]

    def test_deterministic_with_seed(self, gray_img: Image.Image) -> None:
        """Same RNG seed should produce the same output."""
        import numpy as np

        np.random.seed(42)
        out1 = pyblur.randomized_blur(gray_img)
        np.random.seed(42)
        out2 = pyblur.randomized_blur(gray_img)
        assert np.array_equal(np.array(out1), np.array(out2))
