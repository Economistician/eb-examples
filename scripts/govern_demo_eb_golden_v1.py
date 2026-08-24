"""
Governance composition for the EB golden demo dataset (eb_golden_v1).

Reads:
- <base-dir>/panel_demand_v1.parquet
- <base-dir>/panel_point_forecast_v1.parquet
- <base-dir>/diagnostics/fas_v1.parquet  (mandatory)

Writes:
- <base-dir>/governance/governance_v1.parquet
- <base-dir>/governance/governance_v1_policy.json

This script calls ``electric_barometer.run_governance_workflow_df`` (which applies
``apply_ral``). FAS review is mandatory; a missing FAS artifact fails closed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from eb_examples import GoldenV1Artifacts, resolve_base_dir
from electric_barometer import run_governance_workflow_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run official governance workflow for eb_golden_v1 demo"
    )
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Governance tau in raw units (default: 2.0).",
    )
    return p.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _allow_adjustment(decisions: pd.DataFrame) -> pd.Series:
    policy = decisions["ral_policy"].astype(str).str.strip().str.lower()
    status = decisions["status"].astype(str).str.strip().str.lower()
    return policy.isin(("allow", "allow_after_snap")) & status.ne("red")


def main() -> None:
    args = _parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    artifacts = GoldenV1Artifacts(base=base_dir)

    demand_path = artifacts.panel_demand_v1
    fcst_path = artifacts.panel_point_forecast_v1
    fas_path = artifacts.fas_v1
    for path, hint in (
        (demand_path, "contractify_demo_eb_golden_v1.py"),
        (fcst_path, "baseline_forecast_demo_eb_golden_v1.py"),
        (fas_path, "eval_fas_demo_eb_golden_v1.py"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}. "
                f"FAS review is mandatory. Run: python scripts/{hint} --base-dir {base_dir}"
            )

    demand = pd.read_parquet(demand_path)
    fcst = pd.read_parquet(fcst_path)
    fas = pd.read_parquet(fas_path)

    demand_required = {"site_id", "forecast_entity_id", "y", "INTERVAL_INDEX_START_TIME"}
    missing = sorted(demand_required - set(demand.columns))
    if missing:
        raise ValueError(f"panel_demand_v1 missing columns: {missing}. Got: {list(demand.columns)}")
    fcst_required = {"entity_id", "interval_start", "y_pred"}
    missing = sorted(fcst_required - set(fcst.columns))
    if missing:
        raise ValueError(
            f"panel_point_forecast_v1 missing columns: {missing}. Got: {list(fcst.columns)}"
        )

    work = demand.copy()
    work["entity_id"] = work["site_id"].astype(str) + "::" + work["forecast_entity_id"].astype(str)
    work["interval_start"] = pd.to_datetime(work["INTERVAL_INDEX_START_TIME"], errors="raise")
    fcst2 = fcst.copy()
    fcst2["interval_start"] = pd.to_datetime(fcst2["interval_start"], errors="raise")
    panel = work.merge(
        fcst2[["entity_id", "interval_start", "y_pred"]],
        on=["entity_id", "interval_start"],
        how="inner",
        validate="many_to_one",
    )
    if "is_observable" in panel.columns:
        panel = panel[panel["is_observable"] == True].copy()  # noqa: E712

    fas_key = _pick_col(fas, ["forecast_entity_id", "FORECAST_ENTITY_KEY", "entity_id", "id"])
    if fas_key is None:
        raise ValueError(
            f"fas_v1.parquet missing a recognizable key column. Got: {list(fas.columns)}"
        )
    fas_class_col = _pick_col(fas, ["fas_class", "FAS_CLASS", "class"])
    if fas_class_col is None:
        raise ValueError(
            "fas_v1.parquet missing fas_class. FAS review is mandatory for governance."
        )
    fas_join = fas[[fas_key, fas_class_col]].rename(
        columns={fas_key: "forecast_entity_id", fas_class_col: "fas_class"}
    )
    panel["forecast_entity_id"] = panel["forecast_entity_id"].astype(str)
    fas_join["forecast_entity_id"] = fas_join["forecast_entity_id"].astype(str)
    panel = panel.merge(fas_join, on="forecast_entity_id", how="left")
    panel["yhat_base"] = panel["y_pred"]
    panel["yhat_ral"] = panel["y_pred"]

    governed, decisions = run_governance_workflow_df(
        df=panel,
        keys=["forecast_entity_id"],
        actual_col="y",
        base_forecast_col="yhat_base",
        ral_forecast_col="yhat_ral",
        tau=float(args.tau),
        fas_class_col="fas_class",
        snap_mode="ceil",
    )
    decisions = decisions.copy()
    decisions["allow_adjustment"] = _allow_adjustment(decisions)

    artifacts.governance_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.governance_v1
    decisions.to_parquet(out_path, index=False)

    policy_payload: dict[str, Any] = {
        "version": "v1",
        "api": "electric_barometer.run_governance_workflow_df",
        "apply": "electric_barometer.apply_ral",
        "fas_review": "mandatory",
        "tau": float(args.tau),
        "snap_mode": "ceil",
        "n_streams": len(decisions),
        "n_governed_rows": len(governed),
    }
    artifacts.governance_v1_policy_json.write_text(
        json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8"
    )

    print("Governance OK")
    print(f"- output:   {out_path}")
    print(f"- policy:   {artifacts.governance_v1_policy_json}")
    print(f"- slices:   {decisions.shape[0]}")
    print(f"- base-dir: {artifacts.base}")


if __name__ == "__main__":
    main()
