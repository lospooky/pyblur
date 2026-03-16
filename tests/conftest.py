"""Shared pytest fixtures for pyblur tests."""
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def gray_img() -> Image.Image:
    """64×64 grayscale image with a reproducible random pattern."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def rgb_img() -> Image.Image:
    """64×64 RGB image with a reproducible random pattern."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def assert_same_size(result: Image.Image, source: Image.Image) -> None:
    """Assert that *result* has the same (width, height) as *source*."""
    assert result.size == source.size, (
        f"Output size {result.size} differs from input size {source.size}"
    )
