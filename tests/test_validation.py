"""Tests for pyblur._validation decorators."""

import pytest
from PIL import Image

from pyblur._validation import (
    _KERNEL_DIMS,
    validate_dim,
    validate_image,
    validate_mode,
    validate_odd_dim,
)


class TestValidateImage:
    def test_passes_pil_image(self) -> None:
        @validate_image
        def fn(img: Image.Image) -> str:
            return "ok"

        img = Image.new("L", (10, 10))
        assert fn(img) == "ok"

    @pytest.mark.parametrize("bad", [None, 42, "string", b"bytes", [1, 2], object()])
    def test_rejects_non_image(self, bad) -> None:
        @validate_image
        def fn(img):
            return "ok"

        with pytest.raises(TypeError, match="fn\\(\\)"):
            fn(bad)

    def test_error_names_function(self) -> None:
        @validate_image
        def my_special_blur(img):
            return img

        with pytest.raises(TypeError, match="my_special_blur\\(\\)"):
            my_special_blur("bad")

    def test_preserves_function_name(self) -> None:
        @validate_image
        def my_func(img):
            return img

        assert my_func.__name__ == "my_func"

    def test_preserves_function_doc(self) -> None:
        @validate_image
        def my_func(img):
            """Original docstring."""
            return img

        assert my_func.__doc__ == "Original docstring."

    def test_passes_extra_args(self) -> None:
        @validate_image
        def fn(img, x, y=10):
            return x + y

        img = Image.new("L", (4, 4))
        assert fn(img, 5, y=20) == 25


class TestValidateDim:
    def test_passes_valid_dim(self) -> None:
        @validate_image
        @validate_dim(_KERNEL_DIMS)
        def fn(img, dim):
            return dim * 2

        img = Image.new("L", (10, 10))
        for d in _KERNEL_DIMS:
            assert fn(img, d) == d * 2

    @pytest.mark.parametrize("bad", [0, 1, 2, 4, 6, 8, 10, -1, 3.0, "3", None])
    def test_rejects_invalid_dim(self, bad) -> None:
        @validate_image
        @validate_dim(_KERNEL_DIMS)
        def fn(img, dim):
            return dim

        img = Image.new("L", (10, 10))
        with pytest.raises(ValueError, match="fn\\(\\)"):
            fn(img, bad)

    def test_error_names_function(self) -> None:
        @validate_image
        @validate_dim([3, 5])
        def cool_blur(img, dim):
            return img

        img = Image.new("L", (4, 4))
        with pytest.raises(ValueError, match="cool_blur\\(\\)"):
            cool_blur(img, 7)

    def test_preserves_function_name(self) -> None:
        @validate_image
        @validate_dim(_KERNEL_DIMS)
        def my_blur(img, dim):
            return img

        assert my_blur.__name__ == "my_blur"

    def test_custom_valid_list(self) -> None:
        @validate_image
        @validate_dim([1, 2, 4, 8])
        def fn(img, dim):
            return dim

        img = Image.new("L", (4, 4))
        assert fn(img, 8) == 8
        with pytest.raises(ValueError):
            fn(img, 3)


class TestValidateMode:
    def test_passes_supported_modes(self) -> None:
        @validate_image
        @validate_mode(("L", "RGB"))
        def fn(img: Image.Image) -> str:
            return img.mode

        for mode in ("L", "RGB"):
            img = Image.new(mode, (8, 8))
            assert fn(img) == mode

    @pytest.mark.parametrize("bad_mode", ["RGBA", "P"])
    def test_rejects_unsupported_modes(self, bad_mode: str) -> None:
        @validate_image
        @validate_mode(("L", "RGB"))
        def fn(img: Image.Image) -> str:
            return img.mode

        img = Image.new(bad_mode, (8, 8))
        with pytest.raises(ValueError, match="image mode"):
            fn(img)

    def test_error_names_function(self) -> None:
        @validate_image
        @validate_mode(("L",))
        def my_special_blur(img: Image.Image) -> Image.Image:
            return img

        img = Image.new("RGB", (8, 8))
        with pytest.raises(ValueError, match="my_special_blur\\(\\)"):
            my_special_blur(img)

    def test_non_image_raises_type_error_not_value_error(self) -> None:
        """validate_image fires first, so non-images get TypeError not ValueError."""
        @validate_image
        @validate_mode(("L",))
        def fn(img):
            return img

        with pytest.raises(TypeError):
            fn("not an image")

    def test_preserves_function_name(self) -> None:
        @validate_image
        @validate_mode(("L", "RGB"))
        def my_func(img):
            return img

        assert my_func.__name__ == "my_func"


class TestKernelDims:
    def test_kernel_dims_content(self) -> None:
        assert _KERNEL_DIMS == [3, 5, 7, 9]

    def test_kernel_dims_all_odd(self) -> None:
        assert all(d % 2 == 1 for d in _KERNEL_DIMS)


class TestValidateOddDim:
    def test_passes_valid_dims(self) -> None:
        @validate_image
        @validate_odd_dim
        def fn(img: Image.Image, dim: int) -> int:
            return dim

        img = Image.new("L", (10, 10))
        for d in [3, 5, 7, 9, 11, 13]:
            assert fn(img, d) == d

    @pytest.mark.parametrize("bad", [0, 1, 2, 4, -1, -3, 3.0, "3", None])
    def test_rejects_invalid_dims(self, bad) -> None:
        @validate_image
        @validate_odd_dim
        def fn(img: Image.Image, dim):
            return dim

        img = Image.new("L", (10, 10))
        with pytest.raises(ValueError, match="fn\\(\\)"):
            fn(img, bad)

    def test_error_names_function(self) -> None:
        @validate_image
        @validate_odd_dim
        def my_motion_blur(img: Image.Image, dim: int) -> Image.Image:
            return img

        img = Image.new("L", (4, 4))
        with pytest.raises(ValueError, match="my_motion_blur\\(\\)"):
            my_motion_blur(img, 4)

    def test_preserves_function_name(self) -> None:
        @validate_odd_dim
        def my_blur(img, dim):
            return img

        assert my_blur.__name__ == "my_blur"

    def test_passes_extra_args(self) -> None:
        @validate_image
        @validate_odd_dim
        def fn(img: Image.Image, dim: int, angle: float = 0.0) -> float:
            return angle

        img = Image.new("L", (4, 4))
        assert fn(img, 5, 45.0) == 45.0
