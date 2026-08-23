"""
Compute Forecast Readiness Score (FRS) for the baseline forecast on eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet

Writes:
- <base-dir>/diagnostics/frs_v1.parquet

Uses public roots:
- eb_metrics.frs (requires cwsl_max)
- eb_evaluation.evaluate_groups_df
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_evaluation import evaluate_groups_df
from eb_examples import GoldenV1Artifacts, resolve_base_dir
from eb_metrics import frs

CWSL_MAX = 0.30
CU = 2.0
CO = 1.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FRS for eb_golden_v1 demo baseline forecast")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
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

    eval_df = fcst[fcst["y_true"].notna()].copy()
    if eval_df.empty:
        raise ValueError("No rows with known y_true found. Nothing to evaluate.")

    overall = frs(
        y_true=eval_df["y_true"].astype(float).to_numpy(),
        y_pred=eval_df["y_pred"].astype(float).to_numpy(),
        cu=CU,
        co=CO,
        cwsl_max=CWSL_MAX,
    )

    grouped = evaluate_groups_df(
        eval_df,
        group_cols=["entity_id"],
        actual_col="y_true",
        forecast_col="y_pred",
        cu=CU,
        co=CO,
        cwsl_max=CWSL_MAX,
    )
    out = grouped.loc[:, ["entity_id", "CWSL", "NSL", "FRS"]].copy()
    out = out.sort_values("FRS", ascending=True, kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.frs_v1
    out.to_parquet(out_path, index=False)

    print("FRS OK")
    print(f"- input:     {in_path}")
    print(f"- output:    {out_path}")
    print(f"- cu/co:     {CU}/{CO}")
    print(f"- cwsl_max:  {CWSL_MAX}")
    print(f"- overall:   {overall:.6f}")
    print(f"- entities:  {out.shape[0]}")
    print(f"- base-dir:  {artifacts.base}")


if __name__ == "__main__":
    main()
