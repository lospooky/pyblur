"""Backend registry and protocol for pyblur convolution backends."""
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from PIL import Image


@runtime_checkable
class Backend(Protocol):  # pragma: no cover
    """Protocol all pyblur backends must satisfy.

    A backend is responsible for applying a convolution kernel or a named
    blur (e.g. Gaussian) to a PIL image.  The default implementation is
    :class:`~pyblur._backends._pil_scipy.PilScipyBackend`.
    """

    name: str

    def apply_kernel(
        self, img: Image.Image, kernel: NDArray[np.float32]
    ) -> Image.Image:
        """Convolve *img* with *kernel*; return the same image type."""
        ...

    def gaussian_blur(self, img: Image.Image, sigma: float) -> Image.Image:
        """Apply a Gaussian blur with *sigma*; may use a native fast path."""
        ...


_registry: dict[str, Backend] = {}
_default: Backend | None = None


def register(backend: Backend) -> None:
    """Register *backend* under its :attr:`~Backend.name`."""
    _registry[backend.name] = backend


def set_default(name: str) -> None:
    """Set the default backend used when ``backend=None`` is passed.

    Parameters
    ----------
    name : str
        Must be the :attr:`~Backend.name` of a previously registered backend.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    global _default
    if name not in _registry:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(_registry)}"
        )
    _default = _registry[name]


def get_backend(backend: "str | Backend | None" = None) -> Backend:
    """Resolve *backend* to a concrete :class:`Backend` instance.

    Parameters
    ----------
    backend : str | Backend | None
        * ``None`` — return the registered default.
        * ``str`` — look up by name in the registry.
        * :class:`Backend` instance — return unchanged.

    Raises
    ------
    ValueError
        If a string name is not found in the registry.
    RuntimeError
        If ``None`` is passed and no default has been set (should never
        happen after normal import, which registers the scipy backend).
    """
    if backend is None:
        if _default is None:  # pragma: no cover
            raise RuntimeError("No backend registered.")
        return _default
    if isinstance(backend, str):
        if backend not in _registry:
            raise ValueError(
                f"Unknown backend {backend!r}. Available: {sorted(_registry)}"
            )
        return _registry[backend]
    return backend  # Backend instance passed directly


# ---------------------------------------------------------------------------
# Register built-in backends and set the default at import time.
# ---------------------------------------------------------------------------
from pyblur._backends._pil_scipy import PilScipyBackend as _PilScipyBackend  # noqa: E402

_scipy_backend = _PilScipyBackend()
register(_scipy_backend)
set_default("scipy")
