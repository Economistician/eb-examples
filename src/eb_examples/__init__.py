"""
Public API for eb-examples.

This package provides small, reusable utilities that support Electric Barometer
example workflows (for example, canonical artifact paths and base directory
resolution). It is intentionally lightweight and primarily supports the demo
and example scripts in this repository.
"""

from __future__ import annotations

from .paths import GoldenV1Artifacts, default_base_dir, resolve_base_dir

__all__ = [
    "GoldenV1Artifacts",
    "default_base_dir",
    "resolve_base_dir",
]
