# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-23

### Changed

- Golden-path scripts import exclusively from public package roots.
- Golden-path governance and RAL scripts call `electric_barometer.run_governance_workflow_df` and `electric_barometer.apply_ral`.
- FAS review is mandatory; `--no-fas` is removed.
- Runtime pin includes `electric-barometer==0.2.9`.
- Rephrased demo script headers to imperative voice.
- Changelog version header now matches `pyproject.toml` (`0.2.0`).
- Pinned sibling Electric Barometer packages to exact System Release 0.2.9 versions.

### Fixed

- Resolved Pyright import resolution for the CLI entrypoint and smoke tests.
- Aligned golden path pipeline script to include explicit FRS evaluation with required `cwsl_max`.
- Updated all metric scripts to import exclusively from public package roots.

### Added

- `__version__` on the package root.
- Added `py.typed` marker and `pyarrow` dependency for parquet handling.
