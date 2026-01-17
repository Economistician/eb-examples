"""
Apply RAL (Readiness Adjustment Layer) for eb_golden_v1, strictly under Governance permission.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet
- <base-dir>/governance/governance_v1.parquet

Writes:
- <base-dir>/ral/panel_point_forecast_v1_ral.parquet
- <base-dir>/ral/ral_trace_v1.parquet

RAL mode (demo):
- identity (no-op): y_pred_ral == y_pred
- Applied only when governance.allow_adjustment is True for the forecast_entity_id.

Notes:
- This produces an adjusted forecast artifact + traceability.
- No fallback is performed here.
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply RAL under governance permissions (eb_golden_v1 demo)")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    return p.parse_args()


def _parse_forecast_entity_id(entity_id: str) -> str:
    parts = str(entity_id).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected entity_id format: {entity_id!r} (expected 'site_id::forecast_entity_id')")
    return parts[1]


def main() -> None:
    args = _parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    artifacts = GoldenV1Artifacts(base=base_dir)

    fcst_path = artifacts.panel_point_forecast_v1
    gov_path = artifacts.governance_v1

    if not fcst_path.exists():
        raise FileNotFoundError(
            f"Missing forecast panel at {fcst_path}. Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir {base_dir}"
        )
    if not gov_path.exists():
        raise FileNotFoundError(
            f"Missing governance artifact at {gov_path}. Run: python scripts/govern_demo_eb_golden_v1.py --base-dir {base_dir}"
        )

    fcst = pd.read_parquet(fcst_path)
    gov = pd.read_parquet(gov_path)

    required_fcst = {"entity_id", "interval_start", "y_pred"}
    missing = sorted(required_fcst - set(fcst.columns))
    if missing:
        raise ValueError(f"panel_point_forecast_v1 missing required columns: {missing}. Got: {list(fcst.columns)}")

    required_gov = {"forecast_entity_id", "allow_adjustment"}
    missing = sorted(required_gov - set(gov.columns))
    if missing:
        raise ValueError(f"governance_v1 missing required columns: {missing}. Got: {list(gov.columns)}")

    # Determine permission per forecast_entity_id
    gov_map: dict[str, bool] = (
        gov.assign(forecast_entity_id=gov["forecast_entity_id"].astype(str))
        .set_index("forecast_entity_id")["allow_adjustment"]
        .map(bool)
        .to_dict()
    )

    out = fcst.copy()
    out["forecast_entity_id"] = out["entity_id"].map(_parse_forecast_entity_id)

    # Permission lookup (missing => False)
    out["allow_adjustment"] = out["forecast_entity_id"].map(lambda x: bool(gov_map.get(str(x), False)))

    # Identity RAL (no-op). In demo: we keep y_pred unchanged, but record whether adjustment was permitted.
    out["y_pred_ral"] = out["y_pred"]
    out["ral_applied"] = out["allow_adjustment"]
    out["ral_mode"] = out["allow_adjustment"].map(lambda ok: "identity" if ok else "none")

    # Trace: one row per entity_id indicating whether RAL applied (and why)
    trace_cols = ["entity_id", "forecast_entity_id", "ral_applied", "ral_mode"]
    trace = (
        out[trace_cols]
        .drop_duplicates()
        .sort_values(by=["forecast_entity_id", "entity_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    artifacts.ral_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.panel_point_forecast_v1_ral
    trace_path = artifacts.ral_trace_v1

    out.to_parquet(out_path, index=False)
    trace.to_parquet(trace_path, index=False)

    print("RAL OK")
    print(f"- input forecast: {fcst_path}")
    print(f"- input gov:      {gov_path}")
    print(f"- output:         {out_path}")
    print(f"- trace:          {trace_path}")
    print(f"- adjusted rows:  {int(out['ral_applied'].sum())} / {out.shape[0]}")
    print(f"- base-dir:       {artifacts.base}")


if __name__ == "__main__":
    main()
