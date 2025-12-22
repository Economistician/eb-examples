# Electric Barometer — Tutorial Notebooks (`eb-examples`)

![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Docs](https://img.shields.io/badge/docs-electric--barometer-blue)](https://economistician.github.io/eb-docs/)
![Project Status](https://img.shields.io/badge/Status-Alpha-yellow)

This folder contains the **canonical tutorial notebooks** for the *Electric Barometer*
ecosystem.

The tutorials are designed to introduce Electric Barometer concepts incrementally,
demonstrating how forecast evaluation and selection change under asymmetric cost,
service constraints, and operational readiness considerations.

These notebooks are intentionally **narrative-driven, reproducible, and opinionated**.
They represent the recommended learning path for new users.

---

## What These Tutorials Are

The tutorial notebooks provide:

- Conceptual introductions to Electric Barometer evaluation philosophy
- End-to-end, runnable examples using synthetic or open datasets
- Visual intuition for asymmetric cost tradeoffs and decision boundaries
- Operationally motivated forecast selection scenarios

Each tutorial builds on the previous one, gradually increasing realism and complexity.

---

## What These Tutorials Are Not

Tutorial notebooks are **not** intended to:

- Serve as exploratory research scratchpads
- Benchmark forecasting models exhaustively
- Connect to live, credentialed, or proprietary data sources
- Replace formal theoretical documentation

Those use cases are handled elsewhere in the ecosystem.

---

## Tutorial Sequence

The recommended tutorial sequence is:

1. **01_basic_forecast_selection.ipynb**  
   Introduces Electric Barometer as a forecast *evaluation and selection* framework,
   illustrating how asymmetric service costs change model preference.

2. **02_cost_asymmetry_sweep.ipynb**  
   Explores how forecast selection evolves as cost ratios vary, revealing decision
   boundaries and robustness tradeoffs.

3. **03_readiness_adjustment.ipynb**  
   Demonstrates how operational readiness penalties affect forecast selection,
   moving beyond accuracy-only metrics.

4. **04_hierarchical_evaluation.ipynb** *(advanced)*  
   Applies Electric Barometer across multiple entities and hierarchical groupings,
   illustrating system-level selection behavior.

Users are encouraged to follow the tutorials in order.

---

## Design Principles

All tutorial notebooks adhere to the following principles:

- Use **synthetic or open datasets** only
- Be fully **reproducible** when run top-to-bottom
- Avoid hidden state or reliance on cell execution order
- Prioritize clarity and explanation over performance optimization
- Emphasize *why* decisions change, not just *what* changes

These constraints are intentional and enforced to maintain tutorial quality.

---

## Relationship to Other Notebook Types

Within `eb-examples`, notebooks are organized by intent:

- **Tutorials**  
  Stable, public-facing, instructional notebooks (this folder)

- **Benchmarks**  
  Comparative evaluations using open datasets

- **Experiments**  
  Exploratory or research-oriented notebooks

- **Figures**  
  Deterministic notebooks used to generate paper-ready outputs

Only tutorial notebooks are considered part of the public learning contract.

---

## Relationship to Other EB Repositories

- `eb-papers`  
  Source of truth for conceptual definitions and evaluation philosophy.

- `eb-metrics`  
  Provides the metric implementations demonstrated in tutorials.

- `eb-evaluation`  
  Orchestrates evaluation workflows shown in multi-entity examples.

- `eb-adapters`  
  Supplies adapter interfaces used in end-to-end demonstrations.

When discrepancies arise, conceptual intent in `eb-papers` should be treated as authoritative.

---

## Status

These tutorials are under active development.
Content and examples may evolve as the Electric Barometer ecosystem matures.