"""Tests for the backend registry and PilScipyBackend."""
import sys

import numpy as np
import pytest
from PIL import Image

import pyblur
from pyblur._backends import Backend, get_backend, set_default
from pyblur._backends._numpy import PilNumpyBackend
from pyblur._backends._opencv import PilOpenCVBackend
from pyblur._backends._pil_scipy import PilScipyBackend

# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_default_is_scipy(self) -> None:
        b = get_backend(None)
        assert b.name == "scipy"

    def test_get_by_name(self) -> None:
        b = get_backend("scipy")
        assert b.name == "scipy"

    def test_numpy_always_registered(self) -> None:
        b = get_backend("numpy")
        assert b.name == "numpy"

    def test_opencv_registered(self) -> None:
        b = get_backend("opencv")
        assert b.name == "opencv"

    def test_get_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_set_default_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_default("totally_unknown_xyz")

    def test_get_backend_instance_passthrough(self) -> None:
        b = PilScipyBackend()
        result = get_backend(b)
        assert result is b

    def test_pil_scipy_satisfies_protocol(self) -> None:
        b = PilScipyBackend()
        assert isinstance(b, Backend)

    def test_pil_numpy_satisfies_protocol(self) -> None:
        b = PilNumpyBackend()
        assert isinstance(b, Backend)

    def test_pil_opencv_satisfies_protocol(self) -> None:
        b = PilOpenCVBackend()
        assert isinstance(b, Backend)

    def test_fallback_to_numpy_when_scipy_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _pil_scipy cannot be imported, default falls back to numpy."""
        import pyblur._backends as bmod

        saved_registry = dict(bmod._registry)
        saved_default = bmod._default
        monkeypatch.setitem(sys.modules, "pyblur._backends._pil_scipy", None)
        bmod._registry.clear()
        bmod._default = None
        try:
            bmod._init_backends()
            assert bmod._default is not None
            assert bmod._default.name == "numpy"
            assert "numpy" in bmod._registry
            assert "scipy" not in bmod._registry
        finally:
            bmod._registry.clear()
            bmod._registry.update(saved_registry)
            bmod._default = saved_default

    def test_fallback_when_opencv_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _opencv cannot be imported, opencv backend is simply absent."""
        import pyblur._backends as bmod

        saved_registry = dict(bmod._registry)
        saved_default = bmod._default
        monkeypatch.setitem(sys.modules, "pyblur._backends._opencv", None)
        bmod._registry.clear()
        bmod._default = None
        try:
            bmod._init_backends()
            assert "opencv" not in bmod._registry
            assert bmod._default is not None
            assert bmod._default.name == "scipy"
        finally:
            bmod._registry.clear()
            bmod._registry.update(saved_registry)
            bmod._default = saved_default


# ---------------------------------------------------------------------------
# PilScipyBackend unit tests
# ---------------------------------------------------------------------------


class TestPilScipyBackend:
    @pytest.fixture()
    def backend(self) -> PilScipyBackend:
        return PilScipyBackend()

    @pytest.fixture()
    def gray(self) -> Image.Image:
        return Image.new("L", (16, 16), color=128)

    @pytest.fixture()
    def rgb(self) -> Image.Image:
        return Image.new("RGB", (16, 16), color=(100, 150, 200))

    @pytest.fixture()
    def kernel(self) -> np.ndarray:
        return np.ones((3, 3), dtype=np.float32) / 9.0

    def test_apply_kernel_grayscale(
        self, backend: PilScipyBackend, gray: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(gray, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "L"
        assert out.size == gray.size

    def test_apply_kernel_rgb(
        self, backend: PilScipyBackend, rgb: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(rgb, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == rgb.size

    def test_gaussian_blur(
        self, backend: PilScipyBackend, gray: Image.Image
    ) -> None:
        out = backend.gaussian_blur(gray, sigma=1.5)
        assert isinstance(out, Image.Image)
        assert out.size == gray.size


# ---------------------------------------------------------------------------
# Integration: backend= kwarg accepted by every public function
# ---------------------------------------------------------------------------


@pytest.fixture()
def gray() -> Image.Image:
    return Image.new("L", (16, 16), color=128)


@pytest.mark.parametrize(
    "fn,extra_args",
    [
        (pyblur.box_blur, (3,)),
        (pyblur.defocus_blur, (3,)),
        (pyblur.gaussian_blur, (1.5,)),
        (pyblur.linear_motion_blur, (5, 45.0, "full")),
        (pyblur.psf_blur, (0,)),
    ],
)
def test_explicit_backend_string(
    gray: Image.Image, fn: object, extra_args: tuple[object, ...]
) -> None:
    """Passing backend='scipy' explicitly must return a valid PIL image."""
    assert callable(fn)
    result = fn(gray, *extra_args, backend="scipy")  # type: ignore[operator]
    assert isinstance(result, Image.Image)


@pytest.mark.parametrize(
    "fn",
    [
        pyblur.box_blur_random,
        pyblur.defocus_blur_random,
        pyblur.gaussian_blur_random,
        pyblur.linear_motion_blur_random,
        pyblur.psf_blur_random,
        pyblur.randomized_blur,
    ],
)
def test_explicit_backend_string_random(
    gray: Image.Image, fn: object
) -> None:
    """Random variants also accept backend= and return a valid PIL image."""
    assert callable(fn)
    result = fn(gray, backend="scipy")  # type: ignore[operator]
    assert isinstance(result, Image.Image)


def test_backend_instance_passthrough(gray: Image.Image) -> None:
    """A Backend instance may be passed directly instead of a string name."""
    b = PilScipyBackend()
    result = pyblur.box_blur(gray, 3, backend=b)
    assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# PilNumpyBackend unit tests
# ---------------------------------------------------------------------------


class TestPilNumpyBackend:
    @pytest.fixture()
    def backend(self) -> PilNumpyBackend:
        return PilNumpyBackend()

    @pytest.fixture()
    def gray(self) -> Image.Image:
        return Image.new("L", (16, 16), color=128)

    @pytest.fixture()
    def rgb(self) -> Image.Image:
        return Image.new("RGB", (16, 16), color=(100, 150, 200))

    @pytest.fixture()
    def kernel(self) -> np.ndarray:
        return np.ones((3, 3), dtype=np.float32) / 9.0

    def test_apply_kernel_grayscale(
        self, backend: PilNumpyBackend, gray: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(gray, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "L"
        assert out.size == gray.size

    def test_apply_kernel_rgb(
        self, backend: PilNumpyBackend, rgb: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(rgb, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == rgb.size

    def test_gaussian_blur(
        self, backend: PilNumpyBackend, gray: Image.Image
    ) -> None:
        out = backend.gaussian_blur(gray, sigma=1.5)
        assert isinstance(out, Image.Image)
        assert out.size == gray.size


# ---------------------------------------------------------------------------
# Integration: numpy backend accepted by every public function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,extra_args",
    [
        (pyblur.box_blur, (3,)),
        (pyblur.defocus_blur, (3,)),
        (pyblur.gaussian_blur, (1.5,)),
        (pyblur.linear_motion_blur, (5, 45.0, "full")),
        (pyblur.psf_blur, (0,)),
    ],
)
def test_numpy_backend_explicit(
    gray: Image.Image, fn: object, extra_args: tuple[object, ...]
) -> None:
    """backend='numpy' must return a valid PIL image for every deterministic function."""
    assert callable(fn)
    result = fn(gray, *extra_args, backend="numpy")  # type: ignore[operator]
    assert isinstance(result, Image.Image)


@pytest.mark.parametrize(
    "fn",
    [
        pyblur.box_blur_random,
        pyblur.defocus_blur_random,
        pyblur.gaussian_blur_random,
        pyblur.linear_motion_blur_random,
        pyblur.psf_blur_random,
        pyblur.randomized_blur,
    ],
)
def test_numpy_backend_explicit_random(
    gray: Image.Image, fn: object
) -> None:
    """backend='numpy' must return a valid PIL image for every random function."""
    assert callable(fn)
    result = fn(gray, backend="numpy")  # type: ignore[operator]
    assert isinstance(result, Image.Image)


# ---------------------------------------------------------------------------
# Unit tests: PilOpenCVBackend
# ---------------------------------------------------------------------------


class TestPilOpenCVBackend:
    @pytest.fixture()
    def backend(self) -> PilOpenCVBackend:
        return PilOpenCVBackend()

    @pytest.fixture()
    def gray(self) -> Image.Image:
        return Image.new("L", (16, 16), color=128)

    @pytest.fixture()
    def rgb(self) -> Image.Image:
        return Image.new("RGB", (16, 16), color=(100, 150, 200))

    @pytest.fixture()
    def kernel(self) -> np.ndarray:
        return np.ones((3, 3), dtype=np.float32) / 9.0

    def test_apply_kernel_grayscale(
        self, backend: PilOpenCVBackend, gray: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(gray, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "L"
        assert out.size == gray.size

    def test_apply_kernel_rgb(
        self, backend: PilOpenCVBackend, rgb: Image.Image, kernel: np.ndarray
    ) -> None:
        out = backend.apply_kernel(rgb, kernel)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        assert out.size == rgb.size

    def test_gaussian_blur(
        self, backend: PilOpenCVBackend, gray: Image.Image
    ) -> None:
        out = backend.gaussian_blur(gray, sigma=1.5)
        assert isinstance(out, Image.Image)
        assert out.size == gray.size


# ---------------------------------------------------------------------------
# Integration: opencv backend accepted by every public function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,extra_args",
    [
        (pyblur.box_blur, (3,)),
        (pyblur.defocus_blur, (3,)),
        (pyblur.gaussian_blur, (1.5,)),
        (pyblur.linear_motion_blur, (5, 45.0, "full")),
        (pyblur.psf_blur, (0,)),
    ],
)
def test_opencv_backend_explicit(
    gray: Image.Image, fn: object, extra_args: tuple[object, ...]
) -> None:
    """backend='opencv' must return a valid PIL image for every deterministic function."""
    assert callable(fn)
    result = fn(gray, *extra_args, backend="opencv")  # type: ignore[operator]
    assert isinstance(result, Image.Image)


@pytest.mark.parametrize(
    "fn",
    [
        pyblur.box_blur_random,
        pyblur.defocus_blur_random,
        pyblur.gaussian_blur_random,
        pyblur.linear_motion_blur_random,
        pyblur.psf_blur_random,
        pyblur.randomized_blur,
    ],
)
def test_opencv_backend_explicit_random(
    gray: Image.Image, fn: object
) -> None:
    """backend='opencv' must return a valid PIL image for every random function."""
    assert callable(fn)
    result = fn(gray, backend="opencv")  # type: ignore[operator]
    assert isinstance(result, Image.Image)
