# pyblur

[![CI](https://github.com/lospooky/pyblur/actions/workflows/pull_request.yml/badge.svg)](https://github.com/lospooky/pyblur/actions/workflows/pull_request.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyblur)](https://pypi.org/project/pyblur/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyblur)](https://pypi.org/project/pyblur/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Image blurring library for Python. Provides Gaussian, defocus (disk), box, linear motion, and point-spread-function (PSF) blur kernels, plus a randomized dispatcher that picks one at random.

All functions accept a `PIL.Image.Image` and return a new `PIL.Image.Image` of the same size. Both grayscale (`L`) and RGB images are supported. Every blur type exposes a deterministic variant (explicit parameters) and a random variant (parameters sampled automatically).

PSF kernels are taken from [Convolutional Neural Networks for Direct Text Deblurring](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/).

---

## Installation

```bash
pip install pyblur
```

**Requirements:** Python ≥ 3.10, numpy, pillow.

Optional backends ship as extras:

```bash
pip install pyblur[scipy]    # scipy + scikit-image (default backend when installed)
pip install pyblur[opencv]   # opencv-python
```

---

## Quick start

```python
from PIL import Image
import pyblur

img = Image.open("photo.png")   # L or RGB

# Pick a specific blur
blurred = pyblur.gaussian_blur(img, bandwidth=1.5)

# Or let pyblur choose everything at random
blurred = pyblur.randomized_blur(img)

# Explicitly choose a backend
blurred = pyblur.box_blur(img, dim=5, backend="opencv")
```

---

## Backends

Every public function accepts an optional `backend=` keyword argument that controls which convolution engine is used.

| Backend | Extra | Notes |
|---------|-------|-------|
| `"scipy"` | `pyblur[scipy]` | Default when scipy is installed. Identical output to v1.2 and earlier. |
| `"numpy"` | _(none)_ | Always available. Default when scipy is not installed. |
| `"opencv"` | `pyblur[opencv]` | Opt-in only; never set as the automatic default. |

The default is selected automatically at import time: `"scipy"` if scipy is importable, `"numpy"` otherwise. Passing a backend name overrides this for that call only.

```python
# Override per-call
blurred = pyblur.defocus_blur(img, dim=7, backend="numpy")
blurred = pyblur.linear_motion_blur(img, dim=5, angle=30.0, linetype="full", backend="opencv")

# Or pass a Backend instance directly
from pyblur._backends._numpy import PilNumpyBackend
blurred = pyblur.gaussian_blur(img, bandwidth=2.0, backend=PilNumpyBackend())
```

---

## API reference

### `gaussian_blur(img, bandwidth)`

Supports any PIL image mode (delegates to PIL internally).

| Parameter | Type | Description |
|-----------|------|-------------|
| `bandwidth` | `float > 0` | Standard deviation of the Gaussian kernel |

```python
blurred = pyblur.gaussian_blur(img, bandwidth=1.5)
blurred = pyblur.gaussian_blur_random(img)   # bandwidth ∈ {0.5, 1.0, …, 3.5}
```

---

### `defocus_blur(img, dim)`

Simulates a circular (disk) aperture blur. Supports `L` and `RGB` images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — one of `3`, `5`, `7`, `9` |

```python
blurred = pyblur.defocus_blur(img, dim=5)
blurred = pyblur.defocus_blur_random(img)
```

---

### `box_blur(img, dim)`

Uniform box (average) blur. Supports `L` and `RGB` images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — one of `3`, `5`, `7`, `9` |

```python
blurred = pyblur.box_blur(img, dim=7)
blurred = pyblur.box_blur_random(img)
```

---

### `linear_motion_blur(img, dim, angle, linetype)`

Simulates camera or subject motion along a straight line. Supports `L` and `RGB` images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — any odd integer ≥ 3 (e.g. `3`, `5`, `7`, `9`, `11`, …) |
| `angle` | `float` | Motion direction in degrees; any value accepted, wrapped modulo 180° |
| `linetype` | `str` | `"full"` — symmetric; `"right"` / `"left"` — half-kernel |

```python
blurred = pyblur.linear_motion_blur(img, dim=5, angle=45.0, linetype="full")
blurred = pyblur.linear_motion_blur_random(img)
```

---

### `psf_blur(img, psfid)`

Applies one of 100 real-world point-spread-function kernels captured from camera hardware. Supports `L` and `RGB` images.

| Parameter | Type | Description |
|-----------|------|-------------|
| `psfid` | `int` | Kernel index — `0` to `99` |

```python
blurred = pyblur.psf_blur(img, psfid=42)
blurred = pyblur.psf_blur_random(img)
```

---

### `randomized_blur(img)`

Randomly selects one of the five blur types above and samples its parameters uniformly. Useful for data augmentation pipelines where you want diverse blur without manual configuration.

```python
blurred = pyblur.randomized_blur(img)
```

---

## Maintenance

This project is maintained on a best-effort, irregular basis. Issues and PRs are welcome but response times are not guaranteed.

---

## Migrating from v0.2

All public functions were renamed to `snake_case` in v1.0.0 The old `PascalCase` names (`GaussianBlur`, `BoxBlur`, etc.) were removed in v1.0.

| v0.2 | v1.0+ |
|---|---|
| `GaussianBlur(img, bw)` | `gaussian_blur(img, bandwidth=bw)` |
| `GaussianBlur_random(img)` | `gaussian_blur_random(img)` |
| `DefocusBlur(img, dim)` | `defocus_blur(img, dim)` |
| `DefocusBlur_random(img)` | `defocus_blur_random(img)` |
| `BoxBlur(img, dim)` | `box_blur(img, dim)` |
| `BoxBlur_random(img)` | `box_blur_random(img)` |
| `LinearMotionBlur(img, dim, angle, linetype)` | `linear_motion_blur(img, dim, angle, linetype)` |
| `LinearMotionBlur_random(img)` | `linear_motion_blur_random(img)` |
| `PsfBlur(img, psfid)` | `psf_blur(img, psfid)` |
| `PsfBlur_random(img)` | `psf_blur_random(img)` |
| `RandomizedBlur(img)` | `randomized_blur(img)` |