"""
Apply RAL for eb_golden_v1 through ``electric_barometer.apply_ral``.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet
- <base-dir>/governance/governance_v1.parquet

Writes:
- <base-dir>/ral/panel_point_forecast_v1_ral.parquet
- <base-dir>/ral/ral_trace_v1.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples import GoldenV1Artifacts, resolve_base_dir
from electric_barometer import apply_ral


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply official RAL under governance decisions (eb_golden_v1 demo)"
    )
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    return p.parse_args()


def _parse_forecast_entity_id(entity_id: str) -> str:
    parts = str(entity_id).split("::", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Unexpected entity_id format: {entity_id!r} (expected 'site_id::forecast_entity_id')"
        )
    return parts[1]


def main() -> None:
    args = _parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    artifacts = GoldenV1Artifacts(base=base_dir)

    fcst_path = artifacts.panel_point_forecast_v1
    gov_path = artifacts.governance_v1
    if not fcst_path.exists():
        raise FileNotFoundError(
            f"Missing forecast panel at {fcst_path}. "
            f"Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir {base_dir}"
        )
    if not gov_path.exists():
        raise FileNotFoundError(
            f"Missing governance artifact at {gov_path}. "
            f"Run: python scripts/govern_demo_eb_golden_v1.py --base-dir {base_dir}"
        )

    fcst = pd.read_parquet(fcst_path)
    gov = pd.read_parquet(gov_path)
    required_fcst = {"entity_id", "interval_start", "y_pred"}
    missing = sorted(required_fcst - set(fcst.columns))
    if missing:
        raise ValueError(
            f"panel_point_forecast_v1 missing required columns: {missing}. Got: {list(fcst.columns)}"
        )
    required_gov = {
        "forecast_entity_id",
        "ral_policy",
        "status",
        "fas_class",
        "dqc_class",
        "snap_required",
    }
    missing = sorted(required_gov - set(gov.columns))
    if missing:
        raise ValueError(
            f"governance_v1 missing required official columns: {missing}. Got: {list(gov.columns)}"
        )

    work = fcst.copy()
    work["forecast_entity_id"] = work["entity_id"].map(_parse_forecast_entity_id)
    work["yhat_base"] = work["y_pred"]
    work["yhat_ral"] = work["y_pred"]
    applied = apply_ral(
        df=work,
        decisions=gov,
        key_cols=["forecast_entity_id"],
        yhat_base_col="yhat_base",
        yhat_ral_col="yhat_ral",
        snap_mode="ceil",
    )
    out = applied.copy()
    out["y_pred_ral"] = out["yhat_ral_governed"]
    applied_col = "ral_apply_ral_applied" if "ral_apply_ral_applied" in out.columns else None
    if applied_col is None:
        raise ValueError("apply_ral did not emit ral_apply_ral_applied audit column.")
    out["ral_applied"] = out[applied_col].astype(bool)
    out["ral_mode"] = out["ral_applied"].map(lambda ok: "governed" if bool(ok) else "none")

    trace_cols = ["entity_id", "forecast_entity_id", "ral_applied", "ral_mode"]
    trace = (
        out[trace_cols]
        .drop_duplicates()
        .sort_values(by=["forecast_entity_id", "entity_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    artifacts.ral_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(artifacts.panel_point_forecast_v1_ral, index=False)
    trace.to_parquet(artifacts.ral_trace_v1, index=False)

    print("RAL OK")
    print(f"- input forecast: {fcst_path}")
    print(f"- input gov:      {gov_path}")
    print(f"- output:         {artifacts.panel_point_forecast_v1_ral}")
    print(f"- trace:          {artifacts.ral_trace_v1}")
    print(f"- adjusted rows:  {int(out['ral_applied'].sum())} / {out.shape[0]}")
    print(f"- base-dir:       {artifacts.base}")


if __name__ == "__main__":
    main()
