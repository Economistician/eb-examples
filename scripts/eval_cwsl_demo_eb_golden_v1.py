"""
Compute CWSL (Cost-Weighted Service Loss) for the baseline forecast on eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet

Writes:
- <base-dir>/diagnostics/cwsl_v1.parquet

Uses:
- eb_metrics.metrics.loss.cwsl(y_true, y_pred, cu, co, sample_weight=None)

Notes:
- Diagnostics only (no gating, no adjustment).
- Evaluates only where y_true is known (history).
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir
from eb_metrics.metrics.loss import cwsl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CWSL for demo golden-v1.")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for dataset artifacts (relative to repo root unless absolute). "
        "Default: data/demo/eb_golden_v1",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    base_dir = resolve_base_dir(args.base_dir)
    artifacts = GoldenV1Artifacts(base=base_dir)

    in_path = artifacts.panel_point_forecast_v1
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing baseline forecast at {in_path}. "
            "Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir <base-dir>"
        )

    fcst = pd.read_parquet(in_path)

    required = {"entity_id", "interval_start", "y_true", "y_pred"}
    missing = sorted(required - set(fcst.columns))
    if missing:
        raise ValueError(
            f"panel_point_forecast_v1 missing required columns: {missing}. Got: {list(fcst.columns)}"
        )

    # Evaluate only where truth is known
    eval_df = fcst[fcst["y_true"].notna()].copy()
    if eval_df.empty:
        raise ValueError("No rows with known y_true found. Nothing to evaluate.")

    # Demo calibration (evaluation context only)
    co = 1.0
    cu = 2.0  # shortfall costs 2x overbuild

    # Compute overall CWSL (scalar)
    overall = cwsl(
        y_true=eval_df["y_true"].astype(float).to_numpy(),
        y_pred=eval_df["y_pred"].astype(float).to_numpy(),
        cu=cu,
        co=co,
        sample_weight=None,
    )

    # Compute per-entity CWSL (one scalar per entity_id)
    per_entity: list[dict[str, object]] = []
    for entity_id, g in eval_df.groupby("entity_id", sort=True):
        v = cwsl(
            y_true=g["y_true"].astype(float).to_numpy(),
            y_pred=g["y_pred"].astype(float).to_numpy(),
            cu=cu,
            co=co,
            sample_weight=None,
        )
        per_entity.append({"entity_id": entity_id, "cwsl": float(v), "cu": cu, "co": co})

    out = pd.DataFrame(per_entity).sort_values("cwsl", ascending=False, kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.diagnostics_dir / "cwsl_v1.parquet"
    out.to_parquet(out_path, index=False)

    print("CWSL OK")
    print(f"- input:   {in_path}")
    print(f"- output:  {out_path}")
    print(f"- cu/co:   {cu}/{co}")
    print(f"- overall: {overall:.6f}")
    print(f"- entities: {out.shape[0]}")
    print(f"- base-dir:{artifacts.base}")


if __name__ == "__main__":
    main()
