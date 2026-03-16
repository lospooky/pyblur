# pyblur

[![CI](https://github.com/lospooky/pyblur/actions/workflows/ci.yml/badge.svg)](https://github.com/lospooky/pyblur/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyblur)](https://pypi.org/project/pyblur/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyblur)](https://pypi.org/project/pyblur/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Image blurring library for Python. Provides Gaussian, defocus (disk), box, linear motion, and point-spread-function (PSF) blur kernels, plus a randomized dispatcher that picks one at random.

All functions accept a `PIL.Image.Image` and return a new `PIL.Image.Image` of the same size. Every blur type exposes a deterministic variant (explicit parameters) and a random variant (parameters sampled automatically).

PSF kernels are taken from [Convolutional Neural Networks for Direct Text Deblurring](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/).

---

## Installation

```bash
pip install pyblur
```

**Requirements:** Python ≥ 3.10, numpy, pillow, scikit-image, scipy.

---

## Quick start

```python
from PIL import Image
import pyblur

img = Image.open("photo.png")

# Pick a specific blur
blurred = pyblur.gaussian_blur(img, bandwidth=1.5)

# Or let pyblur choose everything at random
blurred = pyblur.randomized_blur(img)
```

---

## API reference

### `gaussian_blur(img, bandwidth)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `bandwidth` | `float > 0` | Standard deviation of the Gaussian kernel |

```python
blurred = pyblur.gaussian_blur(img, bandwidth=1.5)
blurred = pyblur.gaussian_blur_random(img)   # bandwidth ∈ {0.5, 1.0, …, 3.5}
```

---

### `defocus_blur(img, dim)`

Simulates a circular (disk) aperture blur.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — one of `3`, `5`, `7`, `9` |

```python
blurred = pyblur.defocus_blur(img, dim=5)
blurred = pyblur.defocus_blur_random(img)
```

---

### `box_blur(img, dim)`

Uniform box (average) blur.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — one of `3`, `5`, `7`, `9` |

```python
blurred = pyblur.box_blur(img, dim=7)
blurred = pyblur.box_blur_random(img)
```

---

### `linear_motion_blur(img, dim, angle, linetype)`

Simulates camera or subject motion along a straight line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `int` | Kernel size — one of `3`, `5`, `7`, `9` |
| `angle` | `float` | Motion direction in degrees; snapped to the nearest valid angle for the kernel size |
| `linetype` | `str` | `"full"` — symmetric; `"right"` / `"left"` — half-kernel |

```python
blurred = pyblur.linear_motion_blur(img, dim=5, angle=45.0, linetype="full")
blurred = pyblur.linear_motion_blur_random(img)
```

---

### `psf_blur(img, psfid)`

Applies one of 100 real-world point-spread-function kernels captured from camera hardware.

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

All public functions were renamed to `snake_case` in v0.3. The old `PascalCase` names (`GaussianBlur`, `BoxBlur`, etc.) were removed in v1.0.

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