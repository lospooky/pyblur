# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 22-03-2026

### Changed
- `linear_motion_blur`: `dim` now accepts **any odd integer ≥ 3** (previously restricted to `3`, `5`, `7`, `9`).
- `linear_motion_blur`: `angle` is now accepted as-is — any `float`, wrapped modulo 180°. Discrete angle snapping to a kernel-size-dependent set of valid angles has been removed.
- Replaced the hardcoded `LineDictionary` lookup table with a dynamic geometric computation using trigonometry (`_line_endpoints`). Behaviour is identical for the previously supported dims and canonical angles.

---

## [1.1.0] — 21-03-2026

### Added
- RGB (`"RGB"`) image support for all blur functions. Grayscale (`"L"`) remains supported; other modes (e.g. `"RGBA"`, `"P"`) raise `ValueError`.

---

## [1.0.0] — 16-03-2026

Full modernization of a Python 2-era codebase. Requires Python ≥ 3.10.

### Breaking
- `PascalCase` API removed (`BoxBlur`, `GaussianBlur`, etc.) — use `snake_case` equivalents.
- Package moved to `src/` layout; module files renamed to `snake_case`.
- `psf.pkl` replaced by `psf.npz`.

### Added
- `snake_case` public API with explicit and random variants for all five blur types.
- Input validation at all public boundaries (`TypeError` / `ValueError`).
- Full type annotations and `py.typed` marker (PEP 561).
- `pyproject.toml` build config (PEP 517/621), replacing `setup.py` / `setup.cfg` / `MANIFEST.in`.
- 242-test pytest suite with 100% branch coverage.

### Fixed
- **Security:** `psf.pkl` → `psf.npz` with `allow_pickle=False`, closing an arbitrary code-execution vector.
- Python 3 import errors in `RandomizedBlur` and `LinearMotionBlur`.
- Integer division (`/` → `//`) in defocus and linear motion kernels.
- `skimage.draw.circle` → `skimage.draw.disk` (removed upstream API).
- `linear_motion_blur` mutating shared kernel state on repeated calls.

---

## [0.2.3] — 2016

Initial public release. Python 2.7 only.

[1.2.0]: https://github.com/lospooky/pyblur/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/lospooky/pyblur/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lospooky/pyblur/compare/v0.2.3...v1.0.0
[0.2.3]: https://github.com/lospooky/pyblur/releases/tag/v0.2.3
