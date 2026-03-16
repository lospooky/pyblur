# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — unreleased

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

[1.0.0]: https://github.com/lospooky/pyblur/compare/v0.2.3...v1.0.0
[0.2.3]: https://github.com/lospooky/pyblur/releases/tag/v0.2.3
