# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `__version__` on the package root.

### Changed

- Rephrased demo script headers to imperative voice.
- Changelog version header now matches `pyproject.toml` (`0.2.0`).

### Fixed

- Resolved Pyright import resolution for the CLI entrypoint and smoke tests.

## [0.2.0] - 2026-08-22

### Fixed

- Aligned golden path pipeline script to include explicit FRS evaluation with required `cwsl_max`.
- Updated all metric scripts to import exclusively from public package roots.

### Added

- Added `py.typed` marker and `pyarrow` dependency for parquet handling.
