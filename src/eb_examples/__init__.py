"""
Public API for eb-examples.

This package provides small, reusable utilities that support Electric Barometer
example workflows (for example, canonical artifact paths and base directory
resolution). It is intentionally lightweight and primarily supports the demo
and example scripts in this repository.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .paths import GoldenV1Artifacts, default_base_dir, resolve_base_dir


def _resolve_version() -> str:
    """Return the installed version of the eb-examples distribution."""
    try:
        return version("eb-examples")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()

__all__ = [
    "GoldenV1Artifacts",
    "__version__",
    "default_base_dir",
    "resolve_base_dir",
]
