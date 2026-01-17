"""
Compute HR@τ (Hit Rate within Tolerance) for the baseline forecast on eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet

Writes:
- <base-dir>/diagnostics/hr_tau_v1.parquet

Uses:
- eb_metrics.metrics.service.hr_at_tau(y_true, y_pred, tau, sample_weight=None)

Notes:
- Diagnostics only (no gating, no adjustment).
- Evaluates only where y_true is known (history).
- tau is an absolute tolerance in demand units for this demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir
from eb_metrics.metrics.service import hr_at_tau


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HR@τ for eb_golden_v1 demo baseline forecast")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). "
        "Default: data/demo/eb_golden_v1",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
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
            f"panel_point_forecast_v1 missing required columns: {missing}. "
            f"Got: {list(fcst.columns)}"
        )

    # Evaluate only where truth is known
    eval_df = fcst[fcst["y_true"].notna()].copy()
    if eval_df.empty:
        raise ValueError("No rows with known y_true found. Nothing to evaluate.")

    # Demo calibration: absolute tolerance
    tau = 2.0

    # Overall HR@τ
    overall = hr_at_tau(
        y_true=eval_df["y_true"].astype(float).to_numpy(),
        y_pred=eval_df["y_pred"].astype(float).to_numpy(),
        tau=tau,
        sample_weight=None,
    )

    # Per-entity HR@τ
    rows: list[dict[str, object]] = []
    for entity_id, g in eval_df.groupby("entity_id", sort=True):
        v = hr_at_tau(
            y_true=g["y_true"].astype(float).to_numpy(),
            y_pred=g["y_pred"].astype(float).to_numpy(),
            tau=tau,
            sample_weight=None,
        )
        rows.append({"entity_id": entity_id, "hr_tau": float(v), "tau": tau})

    out = pd.DataFrame(rows).sort_values("hr_tau", ascending=True, kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.hr_tau_v1
    out.to_parquet(out_path, index=False)

    print("HR@τ OK")
    print(f"- input:    {in_path}")
    print(f"- output:   {out_path}")
    print(f"- tau:      {tau}")
    print(f"- overall:  {overall:.6f}")
    print(f"- entities: {out.shape[0]}")
    print(f"- base-dir: {artifacts.base}")


if __name__ == "__main__":
    main()
