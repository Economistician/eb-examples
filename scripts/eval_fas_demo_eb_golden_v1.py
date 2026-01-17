"""
Compute FAS (Forecast Admissibility Surface) for the baseline forecast on eb_golden_v1.

Reads:
- <base-dir>/panel_demand_v1.parquet
- <base-dir>/panel_point_forecast_v1.parquet

Writes:
- <base-dir>/diagnostics/fas_v1.parquet

Uses:
- eb_evaluation.diagnostics.fas.slice_keys
- eb_evaluation.diagnostics.fas.compute_error_anatomy
- eb_evaluation.diagnostics.fas.build_fas_surface

Notes:
- FAS is informative-only (no gating, no adjustment).
- We compute anatomy only on rows where truth is known (y not NA) and baseline is present.
- For this demo, we classify at mode="entity" to ensure sufficient support per slice.
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_evaluation.diagnostics.fas import (
    FASThresholds,
    build_fas_surface,
    compute_error_anatomy,
    slice_keys,
)
from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FAS for eb_golden_v1 demo baseline forecast")
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

    demand_path = artifacts.panel_demand_v1
    fcst_path = artifacts.panel_point_forecast_v1

    if not demand_path.exists():
        raise FileNotFoundError(
            f"Missing demand panel at {demand_path}. "
            "Run: python scripts/contractify_demo_eb_golden_v1.py --base-dir <base-dir>"
        )
    if not fcst_path.exists():
        raise FileNotFoundError(
            f"Missing forecast panel at {fcst_path}. "
            "Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir <base-dir>"
        )

    demand = pd.read_parquet(demand_path)
    fcst = pd.read_parquet(fcst_path)

    # Demand must have these (from your adapter output)
    demand_required = {
        "site_id",
        "forecast_entity_id",
        "y",
        "is_observable",
        "INTERVAL_START_TS",
        "INTERVAL_30_INDEX",
    }
    missing = sorted(demand_required - set(demand.columns))
    if missing:
        raise ValueError(f"panel_demand_v1 missing columns: {missing}. Got: {list(demand.columns)}")

    # Forecast contract columns (validated via eb_contracts)
    fcst_required = {"entity_id", "interval_start", "y_true", "y_pred"}
    missing = sorted(fcst_required - set(fcst.columns))
    if missing:
        raise ValueError(f"panel_point_forecast_v1 missing columns: {missing}. Got: {list(fcst.columns)}")

    # Join key from demand -> forecast:
    # baseline builder used: entity_id = f"{site_id}::{forecast_entity_id}"
    work = demand.copy()
    work["entity_id"] = work["site_id"].astype(str) + "::" + work["forecast_entity_id"].astype(str)

    work["interval_start"] = pd.to_datetime(work["INTERVAL_START_TS"], errors="raise")
    fcst2 = fcst.copy()
    fcst2["interval_start"] = pd.to_datetime(fcst2["interval_start"], errors="raise")

    merged = work.merge(
        fcst2[["entity_id", "interval_start", "y_pred"]],
        on=["entity_id", "interval_start"],
        how="left",
        validate="many_to_one",
    )

    # Compute anatomy only where y and y_pred are both known (compute_error_anatomy also drops NA)
    # Focus on observable rows (learnable / realized history)
    merged = merged[merged["is_observable"] == True].copy()  # noqa: E712

    # Slice mode for demo: "entity" provides enough support per slice.
    mode = "entity"
    keys = slice_keys(
        mode,
        site_col="site_id",
        entity_col="forecast_entity_id",
        interval_col="INTERVAL_30_INDEX",
    )

    anatomy = compute_error_anatomy(
        merged,
        y_col="y",
        yhat_col="y_pred",
        keys=keys,
        spike_ge=10.0,
    )

    fas = build_fas_surface(
        anatomy=anatomy,
        keys=keys,
        thr=FASThresholds(),
    )

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.fas_v1
    fas.to_parquet(out_path, index=False)

    print("FAS OK")
    print(f"- input demand:  {demand_path}")
    print(f"- input forecast:{fcst_path}")
    print(f"- output:        {out_path}")
    print(f"- mode:          {mode}")
    print(f"- slices:        {fas.shape[0]}")
    print(f"- base-dir:      {artifacts.base}")


if __name__ == "__main__":
    main()
