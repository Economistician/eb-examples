# Electric Barometer — Benchmark Notebooks (`eb-examples`)

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Docs](https://img.shields.io/badge/docs-electric--barometer-blue)](https://economistician.github.io/eb-docs/)
![Project Status](https://img.shields.io/badge/Status-Alpha-yellow)

This folder contains **benchmark notebooks** used to evaluate Electric Barometer
behavior on **standardized, publicly available datasets**.

Benchmark notebooks are designed to establish *external credibility* by demonstrating
how Electric Barometer metrics behave relative to classical evaluation approaches
(e.g., RMSE) under controlled, reproducible conditions.

---

## Purpose of Benchmarks

Benchmark notebooks exist to answer skeptical, externally oriented questions such as:

- Does Electric Barometer behave sensibly on non-synthetic data?
- Are observed behaviors consistent across many independent time series?
- How does asymmetric cost alter model selection relative to symmetric metrics?
- Are results reproducible on well-known public datasets?

These notebooks emphasize **comparative behavior**, not performance optimization.

---

## Benchmark Set

The current benchmark suite consists of the following notebooks:

1. **01_m4_subset_comparison.ipynb**  
   Compares Electric Barometer metrics against classical regression metrics
   (e.g., RMSE) on a curated subset of the M4 forecasting dataset, highlighting
   rank inversions and decision differences under asymmetric cost.

2. **02_cost_sensitivity_open_retail.ipynb**  
   Evaluates forecast selection stability across a range of cost ratios using
   an open, retail-style demand dataset, revealing decision boundaries and
   robustness tradeoffs.

3. **03_cross_series_robustness.ipynb**  
   Examines Electric Barometer behavior across many independent time series,
   focusing on aggregate behavior, selection consistency, and distributional
   properties rather than single-series outcomes.

Each notebook is scoped to answer a **single evaluation question**.

---

## Design Principles

All benchmark notebooks adhere to the following principles:

- Use **public, open datasets** only
- Operate on **frozen or cached data subsets**
- Fix random seeds and sampling logic
- Avoid hyperparameter tuning or model optimization
- Prefer clarity and interpretability over exhaustive coverage

Benchmarks are intentionally conservative and reproducible.

---

## Non-Goals

Benchmark notebooks are **not** intended to:

- Serve as instructional tutorials
- Explore experimental or speculative ideas
- Optimize forecasting models
- Connect to live or proprietary data sources
- Function as production pipelines

Those use cases are handled in other notebook categories.

---

## Relationship to Other Notebook Types

Within `eb-examples`, notebooks are organized by intent:

- **Tutorials**  
  Narrative-driven, instructional notebooks introducing core concepts.

- **Benchmarks**  
  Comparative, externally oriented evaluations (this folder).

- **Experiments**  
  Exploratory or research-oriented analysis.

- **Figures**  
  Deterministic notebooks used to generate paper-ready plots and tables.

Benchmarks should not depend on tutorial execution state or experimental artifacts.

---

## Relationship to Other EB Repositories

- `eb-papers`  
  Source of truth for conceptual definitions, evaluation philosophy, and formal claims.

- `eb-metrics`  
  Provides the metric implementations evaluated in benchmarks.

- `eb-evaluation`  
  Supplies evaluation orchestration utilities used in multi-series benchmarks.

- `eb-adapters`  
  Defines model interfaces where heterogeneous forecasts are compared.

When discrepancies arise, conceptual intent in `eb-papers` should be treated as authoritative.

---

## Status

Benchmark notebooks are under active development.
Results may evolve as datasets, metrics, and evaluation conventions mature.