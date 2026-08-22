# Electric Barometer · Examples (`eb-examples`)

[![CI](https://github.com/Economistician/eb-examples/actions/workflows/ci.yml/badge.svg)](https://github.com/Economistician/eb-examples/actions/workflows/ci.yml)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/eb-examples)
![PyPI](https://img.shields.io/pypi/v/eb-examples)

Runnable golden-path pipelines that compose Electric Barometer packages into one auditable forecast workflow.

---

## Overview

`eb-examples` ships the end-to-end demonstration for the Electric Barometer ecosystem. The golden path takes raw demand through contract validation, a baseline forecast, cost-sensitive evaluation (including Forecast Readiness Score), and governance gating. Scripts import metric and evaluation primitives from public package roots only.

---

## Installation

```bash
pip install eb-examples
```

For a local checkout, install this repository in editable mode together with the sibling Electric Barometer packages (`eb-metrics`, `eb-evaluation`, `eb-contracts`, `eb-adapters`).

The package supports Python 3.11 and later.

---

## Golden-path pipeline

The institutional path is five stages. The shipped `eb_golden_v1` runner implements each stage as one or more independently runnable scripts.

### 1. Data quality check

Raw demand is generated, then adapted into a versioned demand contract (`PanelDemandV1`). Structural diagnostics (DQC, later FPC) record whether the panel is compatible with evaluation and gating. Invalid grain, missing identity keys, or non-finite demand fail before a forecast is treated as decision-ready.

Scripts: `make_demo_eb_golden_v1.py`, `contractify_demo_eb_golden_v1.py`, `eval_dqc_demo_eb_golden_v1.py`.

### 2. Feature generation

Panel features are constructed with `eb-features` using explicit entity and target columns. The golden baseline is a mean forecast and does not require a feature table; when you replace that baseline with a supervised adapter, use the public root:

```python
from eb_features import add_lag_features, add_rolling_features

df, lag_cols = add_lag_features(
    df,
    entity_col="entity_id",
    target_col="y",
    lag_steps=[1, 2],
)
df, roll_cols = add_rolling_features(
    df,
    entity_col="entity_id",
    target_col="y",
    rolling_windows=[3],
    rolling_stats=["mean"],
)
```

### 3. Model fitting

`baseline_forecast_demo_eb_golden_v1.py` fits a per-entity mean on observable history and writes `panel_point_forecast_v1.parquet` (`entity_id`, `interval_start`, `y_true`, `y_pred`). Substitute an adapter from `eb-adapters` (`ArimaAdapter`, `SarimaxAdapter`, `XGBoostRegressorAdapter`) with the same `fit(X, y)` / `predict(X)` contract.

### 4. Evaluation

Each metric script reads the point-forecast artifact and writes a diagnostic parquet. Imports use public roots:

```python
from eb_metrics import cwsl, nsl, ud, frs, hr_at_tau
from eb_evaluation import evaluate_groups_df
```

Cost coefficients are explicit (`cu=2.0`, `co=1.0`). Forecast Readiness Score is a required golden-path step and **requires** a finite `cwsl_max > 0` (demo value `0.30`):

```python
from eb_metrics import frs
from eb_evaluation import evaluate_groups_df

overall = frs(y_true, y_pred, cu=2.0, co=1.0, cwsl_max=0.30)
grouped = evaluate_groups_df(
    df,
    group_cols=["entity_id"],
    actual_col="y_true",
    forecast_col="y_pred",
    cu=2.0,
    co=1.0,
    cwsl_max=0.30,
)
```

Scripts: `eval_cwsl_demo_eb_golden_v1.py`, `eval_hr_tau_demo_eb_golden_v1.py`, `eval_nsl_ud_demo_eb_golden_v1.py`, `eval_frs_demo_eb_golden_v1.py`, plus optional FAS.

### 5. Governance gating

`govern_demo_eb_golden_v1.py` composes DQC, FPC, HR@τ, and related diagnostics into a binding permission artifact. RAL and serving run only after that decision file is written. This demo permits identity (no-op) RAL when structural checks pass.

Scripts: `eval_fpc_demo_eb_golden_v1.py`, `govern_demo_eb_golden_v1.py`, `ral_demo_eb_golden_v1.py`, `serve_demo_eb_golden_v1.py`.

---

## Run the demo

From the `eb-examples` repository root:

```bash
python scripts/run_demo_eb_golden_v1.py
python scripts/run_demo_eb_golden_v1.py --base-dir data/demo/eb_golden_v1_run2
python -m eb_examples demo golden-v1 --steps
eb-demo golden-v1 --no-fas
```

Canonical runner order:

1. Generate demo raw data
2. Contractify to `PanelDemandV1`
3. Baseline point forecast
4. Evaluate CWSL, HR@τ, NSL/UD, then FRS (`cwsl_max` required)
5. Optional FAS
6. DQC and FPC diagnostics
7. Governance composition
8. Identity RAL (when permitted)
9. Serving artifact

Key outputs land under `--base-dir` (default `data/demo/eb_golden_v1`), including `diagnostics/frs_v1.parquet` and `governance/governance_v1.parquet`.

---

## License

BSD 3-Clause License.
© 2026 Kyle Corrie.
