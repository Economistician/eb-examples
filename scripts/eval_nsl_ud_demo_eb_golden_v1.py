"""
Compute NSL and UD for the baseline forecast on eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet

Writes:
- <base-dir>/diagnostics/nsl_ud_v1.parquet

Uses:
- eb_metrics.metrics.service.nsl
- eb_metrics.metrics.service.ud

Notes:
- Diagnostics only (no gating, no adjustment).
- Evaluates only where y_true is known (history).
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir
from eb_metrics.metrics.service import nsl, ud


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NSL/UD for eb_golden_v1 demo baseline forecast")
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

    eval_df = fcst[fcst["y_true"].notna()].copy()
    if eval_df.empty:
        raise ValueError("No rows with known y_true found. Nothing to evaluate.")

    y_true_all = eval_df["y_true"].astype(float).to_numpy()
    y_pred_all = eval_df["y_pred"].astype(float).to_numpy()

    overall_nsl = nsl(y_true=y_true_all, y_pred=y_pred_all, sample_weight=None)
    overall_ud = ud(y_true=y_true_all, y_pred=y_pred_all, sample_weight=None)

    rows: list[dict[str, object]] = []
    for entity_id, g in eval_df.groupby("entity_id", sort=True):
        y_true = g["y_true"].astype(float).to_numpy()
        y_pred = g["y_pred"].astype(float).to_numpy()

        rows.append(
            {
                "entity_id": entity_id,
                "nsl": float(nsl(y_true=y_true, y_pred=y_pred, sample_weight=None)),
                "ud": float(ud(y_true=y_true, y_pred=y_pred, sample_weight=None)),
            }
        )

    out = pd.DataFrame(rows).sort_values("ud", ascending=False, kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.nsl_ud_v1
    out.to_parquet(out_path, index=False)

    print("NSL/UD OK")
    print(f"- input:       {in_path}")
    print(f"- output:      {out_path}")
    print(f"- overall_nsl: {overall_nsl:.6f}")
    print(f"- overall_ud:  {overall_ud:.6f}")
    print(f"- entities:    {out.shape[0]}")
    print(f"- base-dir:    {artifacts.base}")


if __name__ == "__main__":
    main()
