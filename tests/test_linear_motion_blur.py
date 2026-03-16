"""Tests for pyblur.linear_motion_blur."""
import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur
from pyblur._validation import _KERNEL_DIMS
from pyblur.linear_motion_blur import _line_kernel, _sanitize_angle

_LINE_TYPES = ["full", "right", "left"]


class TestLineKernel:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_shape(self, dim: int, linetype: str) -> None:
        k = _line_kernel(dim, 0.0, linetype)  # type: ignore[arg-type]
        assert k.shape == (dim, dim)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_sums_to_one(self, dim: int, linetype: str) -> None:
        k = _line_kernel(dim, 45.0, linetype)  # type: ignore[arg-type]
        assert pytest.approx(k.sum(), abs=1e-6) == 1.0

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_non_negative(self, dim: int, linetype: str) -> None:
        k = _line_kernel(dim, 90.0, linetype)  # type: ignore[arg-type]
        assert (k >= 0).all()


class TestSanitizeAngle:
    @pytest.mark.parametrize("angle,kernel_center", [(0.0, 1), (45.0, 2), (200.0, 4)])
    def test_result_in_valid_set(self, angle: float, kernel_center: int) -> None:
        result = _sanitize_angle(kernel_center, angle)
        num_lines = kernel_center * 4
        valid = np.linspace(0, 180, num_lines, endpoint=False)
        assert any(np.isclose(result, v) for v in valid)

    def test_angle_wraps_at_180(self) -> None:
        # 180° should map to 0° (endpoint=False means 180 is not valid)
        result = _sanitize_angle(2, 180.0)
        assert 0.0 <= result < 180.0


class TestLinearMotionBlur:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_returns_pil_image(self, gray_img: Image.Image, dim: int, linetype: str) -> None:
        out = pyblur.linear_motion_blur(gray_img, dim, 45.0, linetype)  # type: ignore[arg-type]
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_preserves_size(self, gray_img: Image.Image, dim: int, linetype: str) -> None:
        out = pyblur.linear_motion_blur(gray_img, dim, 45.0, linetype)  # type: ignore[arg-type]
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="linear_motion_blur\\(\\)"):
            pyblur.linear_motion_blur(b"bytes", 5, 0.0, "full")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_dim", [0, 2, 6, -1, 3.0, None])
    def test_invalid_dim(self, gray_img: Image.Image, bad_dim) -> None:
        with pytest.raises(ValueError, match="linear_motion_blur\\(\\)"):
            pyblur.linear_motion_blur(gray_img, bad_dim, 0.0, "full")

    @pytest.mark.parametrize("bad_linetype", ["diagonal", "", "FULL", 0, None])
    def test_invalid_linetype(self, gray_img: Image.Image, bad_linetype) -> None:
        with pytest.raises(ValueError, match="linear_motion_blur\\(\\)"):
            pyblur.linear_motion_blur(gray_img, 5, 0.0, bad_linetype)

    def test_angle_snapping(self, gray_img: Image.Image) -> None:
        """Arbitrary angles should not crash — they are snapped internally."""
        for angle in [0.0, 37.3, 90.0, 179.9, 360.0]:
            out = pyblur.linear_motion_blur(gray_img, 5, angle, "full")
            assert_same_size(out, gray_img)


class TestLinearMotionBlurRandom:
    def test_returns_pil_image(self, gray_img: Image.Image) -> None:
        out = pyblur.linear_motion_blur_random(gray_img)
        assert isinstance(out, Image.Image)

    def test_preserves_size(self, gray_img: Image.Image) -> None:
        out = pyblur.linear_motion_blur_random(gray_img)
        assert_same_size(out, gray_img)

    def test_invalid_img_type(self) -> None:
        with pytest.raises(TypeError, match="linear_motion_blur_random\\(\\)"):
            pyblur.linear_motion_blur_random(123)  # type: ignore[arg-type]
