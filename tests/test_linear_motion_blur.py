"""Tests for pyblur.linear_motion_blur."""
from typing import Literal, cast

import numpy as np
import pytest
from conftest import assert_same_size
from PIL import Image

import pyblur
from pyblur._kernels import line_endpoints, line_kernel
from pyblur._validation import _KERNEL_DIMS

_LINE_TYPES = ["full", "right", "left"]


class TestLineKernel:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_shape(self, dim: int, linetype: str) -> None:
        k = line_kernel(dim, 0.0, linetype)  # type: ignore[arg-type]
        assert k.shape == (dim, dim)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_sums_to_one(self, dim: int, linetype: str) -> None:
        k = line_kernel(dim, 45.0, linetype)  # type: ignore[arg-type]
        assert pytest.approx(k.sum(), abs=1e-6) == 1.0

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_non_negative(self, dim: int, linetype: str) -> None:
        k = line_kernel(dim, 90.0, linetype)  # type: ignore[arg-type]
        assert (k >= 0).all()



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

    @pytest.mark.parametrize("bad_dim", [0, 1, 2, 4, 6, -1, 3.0, "5", None])
    def test_invalid_dim(self, gray_img: Image.Image, bad_dim) -> None:
        with pytest.raises(ValueError, match="linear_motion_blur\\(\\)"):
            pyblur.linear_motion_blur(gray_img, bad_dim, 0.0, "full")

    @pytest.mark.parametrize("dim", [11, 13, 15])
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_large_kernel(self, gray_img: Image.Image, dim: int, linetype: str) -> None:
        lt = cast(Literal["full", "right", "left"], linetype)
        out = pyblur.linear_motion_blur(gray_img, dim, 45.0, lt)
        assert isinstance(out, Image.Image)
        assert_same_size(out, gray_img)

    @pytest.mark.parametrize("bad_linetype", ["diagonal", "", "FULL", 0, None])
    def test_invalid_linetype(self, gray_img: Image.Image, bad_linetype) -> None:
        with pytest.raises(ValueError, match="linear_motion_blur\\(\\)"):
            pyblur.linear_motion_blur(gray_img, 5, 0.0, bad_linetype)

    def test_continuous_angle_support(self, gray_img: Image.Image) -> None:
        """Any float angle is accepted without snapping."""
        for angle in [0.0, 37.3, 90.0, 123.456, 179.9, 360.0]:
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


class TestLinearMotionBlurRGBSupport:
    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    @pytest.mark.parametrize("linetype", _LINE_TYPES)
    def test_returns_pil_image(self, rgb_img: Image.Image, dim: int, linetype: str) -> None:
        out = pyblur.linear_motion_blur(rgb_img, dim, 45.0, linetype)  # type: ignore[arg-type]
        assert isinstance(out, Image.Image)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_size(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.linear_motion_blur(rgb_img, dim, 45.0, "full")
        assert_same_size(out, rgb_img)

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_mode(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.linear_motion_blur(rgb_img, dim, 45.0, "full")
        assert out.mode == "RGB"

    @pytest.mark.parametrize("dim", _KERNEL_DIMS)
    def test_preserves_channels(self, rgb_img: Image.Image, dim: int) -> None:
        out = pyblur.linear_motion_blur(rgb_img, dim, 45.0, "full")
        assert np.array(out).shape[2] == 3

    def test_random_returns_pil_image(self, rgb_img: Image.Image) -> None:
        out = pyblur.linear_motion_blur_random(rgb_img)
        assert isinstance(out, Image.Image)

    def test_random_preserves_mode(self, rgb_img: Image.Image) -> None:
        out = pyblur.linear_motion_blur_random(rgb_img)
        assert out.mode == "RGB"

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.linear_motion_blur(img, 5, 0.0, "full")

    @pytest.mark.parametrize("mode", ["RGBA", "P"])
    def test_random_rejects_unsupported_mode(self, mode: str) -> None:
        img = Image.new(mode, (16, 16))
        with pytest.raises(ValueError, match="image mode"):
            pyblur.linear_motion_blur_random(img)


class TestLineEndpoints:
    @pytest.mark.parametrize("dim,angle,expected", [
        (9, 0.0,   (4, 0, 4, 8)),    # horizontal
        (9, 45.0,  (8, 0, 0, 8)),    # diagonal
        (9, 90.0,  (8, 4, 0, 4)),    # vertical
        (9, 135.0, (8, 8, 0, 0)),    # anti-diagonal (reversed from lookup but same line)
        (5, 0.0,   (2, 0, 2, 4)),
        (5, 90.0,  (4, 2, 0, 2)),
        (3, 0.0,   (1, 0, 1, 2)),
        (3, 90.0,  (2, 1, 0, 1)),
    ])
    def test_canonical_values(
        self, dim: int, angle: float, expected: tuple[int, int, int, int]
    ) -> None:
        result = line_endpoints(dim, angle)
        rev = (expected[2], expected[3], expected[0], expected[1])
        assert result == expected or result == rev, (
            f"line_endpoints({dim}, {angle}) = {result}, expected {expected} or {rev}"
        )

    @pytest.mark.parametrize("dim", [3, 5, 7, 9, 11, 13])
    @pytest.mark.parametrize("angle", [0.0, 22.5, 45.0, 90.0, 135.0, 157.5, 177.3])
    def test_endpoints_within_bounds(self, dim: int, angle: float) -> None:
        r0, c0, r1, c1 = line_endpoints(dim, angle)
        assert 0 <= r0 < dim and 0 <= c0 < dim
        assert 0 <= r1 < dim and 0 <= c1 < dim

    @pytest.mark.parametrize("dim", [5, 7, 9, 11])
    @pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0])
    def test_endpoints_symmetric_around_center(self, dim: int, angle: float) -> None:
        c = dim // 2
        r0, c0, r1, c1 = line_endpoints(dim, angle)
        assert (r0 + r1) // 2 == c or abs((r0 + r1) / 2 - c) < 1.0
        assert (c0 + c1) // 2 == c or abs((c0 + c1) / 2 - c) < 1.0

    def test_angle_wrap_360(self) -> None:
        assert line_endpoints(9, 0.0) == line_endpoints(9, 180.0)

    def test_large_dim_shape(self) -> None:
        for dim in [11, 13, 15]:
            r0, c0, r1, c1 = line_endpoints(dim, 45.0)
            assert r0 == dim - 1 and c0 == 0 and r1 == 0 and c1 == dim - 1


class TestLineKernelParity:
    """Verify dynamic computation produces the geometrically correct kernel."""

    def test_horizontal_kernel(self) -> None:
        k = line_kernel(5, 0.0, "full")
        assert np.allclose(k[2], np.full(5, 0.2))
        assert k[:2].sum() == 0 and k[3:].sum() == 0

    def test_vertical_kernel(self) -> None:
        k = line_kernel(5, 90.0, "full")
        assert np.allclose(k[:, 2], np.full(5, 0.2))
        assert k[:, :2].sum() == 0 and k[:, 3:].sum() == 0

    @pytest.mark.parametrize("dim", [11, 13, 15])
    def test_large_dim_sums_to_one(self, dim: int) -> None:
        for angle in [0.0, 45.0, 90.0, 135.0]:
            k = line_kernel(dim, angle, "full")
            assert pytest.approx(k.sum(), abs=1e-5) == 1.0
