import functools
from collections.abc import Callable
from typing import Any, TypeVar

from PIL import Image

# Kernel sizes shared across box, defocus, and linear-motion blur modules.
_KERNEL_DIMS: list[int] = [3, 5, 7, 9]

_F = TypeVar("_F", bound=Callable[..., Any])


def validate_image(func: _F) -> _F:
    """Decorator: raise TypeError if the first argument is not a PIL.Image.Image."""
    @functools.wraps(func)
    def wrapper(img: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(img, Image.Image):
            name = getattr(func, "__name__", repr(func))
            raise TypeError(
                f"{name}() requires a PIL.Image.Image, got {type(img).__name__!r}"
            )
        return func(img, *args, **kwargs)
    return wrapper  # type: ignore[return-value]


def validate_dim(valid: list[int]) -> Callable[[_F], _F]:
    """Decorator factory: raise ValueError if the second argument (dim) is not in *valid*."""
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(img: Any, dim: Any, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(dim, int) or dim not in valid:
                name = getattr(func, "__name__", repr(func))
                raise ValueError(
                    f"{name}() dim must be one of {valid}, got {dim!r}"
                )
            return func(img, dim, *args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator
