"""
Contractify the EB golden demo dataset (eb_golden_v1) into PanelDemandV1.

Inputs (via loader):
- <base-dir>/raw_demand.csv.gz

Outputs:
- <base-dir>/panel_demand_v1.parquet

Implementation notes:
- Uses the canonical QSR interval adapter from eb-adapters:
    eb_adapters.contracts.demand_panel.v1.qsr.entity_usage_interval_panel.to_panel_demand_v1
- Normalizes demo semantics into eb-contracts semantics, notably:
    structural_zero == True => y must be NA
    structural_zero == True => is_observable must not be True
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw demo dtypes without changing semantics.
    Keep IDs as strings and parse time fields.
    """
    out = df.copy()

    # IDs as strings (preserve leading zeros)
    for c in ["STORE_ID", "FORECAST_ENTITY_ID", "FORECAST_ENTITY_NAME"]:
        if c in out.columns:
            out[c] = out[c].astype("string")

    # Dates/timestamps
    if "BUSINESS_DAY" in out.columns:
        out["BUSINESS_DAY"] = pd.to_datetime(out["BUSINESS_DAY"], errors="raise").dt.date

    if "INTERVAL_START_TS" in out.columns:
        out["INTERVAL_START_TS"] = pd.to_datetime(out["INTERVAL_START_TS"], errors="raise")

    # Interval index
    if "INTERVAL_30_INDEX" in out.columns:
        out["INTERVAL_30_INDEX"] = out["INTERVAL_30_INDEX"].astype("int64")

    # Demand qty (nullable numeric)
    if "DEMAND_QTY" in out.columns:
        out["DEMAND_QTY"] = pd.to_numeric(out["DEMAND_QTY"], errors="coerce")

    # Flags -> nullable boolean semantics are handled by adapter; here we keep bool clean.
    for c in [
        "IS_DAY_OBSERVABLE",
        "IS_INTERVAL_OBSERVABLE",
        "IS_STRUCTURAL_ZERO",
        "HAS_DEMAND",
        "IS_FUTURE",
        "IS_VALUE_KNOWN",
    ]:
        if c in out.columns:
            out[c] = out[c].astype("bool")

    return out


def _normalize_to_contract_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize demo dataset semantics into eb-contracts PanelDemandV1 semantics.

    eb-contracts requires:
      - structural_zero True => y is NA
      - structural_zero True => is_observable must not be True

    Your demo generator currently uses:
      - structural_zero True => DEMAND_QTY = 0
      - structural_zero True => IS_INTERVAL_OBSERVABLE = True

    So we apply a deterministic transform here.
    """
    out = df.copy()

    if "IS_STRUCTURAL_ZERO" in out.columns:
        structural = out["IS_STRUCTURAL_ZERO"] == True  # noqa: E712

        # Structural zero => y must be NA (not 0)
        if "DEMAND_QTY" in out.columns:
            out.loc[structural, "DEMAND_QTY"] = pd.NA

        # Structural zero => is_observable must not be True
        if "IS_INTERVAL_OBSERVABLE" in out.columns:
            out.loc[structural, "IS_INTERVAL_OBSERVABLE"] = False

        # Optional: if you ever fall back to day observability, keep it consistent too
        if "IS_DAY_OBSERVABLE" in out.columns:
            out.loc[structural, "IS_DAY_OBSERVABLE"] = False

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contractify demo golden-v1 into PanelDemandV1.")
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

    # Load via stable API (no hardcoded paths)
    # Canonical adapter + spec from eb-adapters
    from eb_adapters.contracts.demand_panel.v1.qsr.entity_usage_interval_panel import (
        QSRIntervalPanelDemandSpecV1,
        to_panel_demand_v1,
    )
    from eb_examples.datasets import load_demo_golden_v1

    df_raw = load_demo_golden_v1()
    df = _coerce_types(df_raw)
    df = _normalize_to_contract_semantics(df)

    # Configure adapter for this demo dataset
    spec = QSRIntervalPanelDemandSpecV1(
        site_col="STORE_ID",
        forecast_entity_col="FORECAST_ENTITY_ID",
        business_day_col="BUSINESS_DAY",
        interval_index_col="INTERVAL_30_INDEX",
        interval_start_ts_col="INTERVAL_START_TS",
        y_source_col="DEMAND_QTY",
        is_interval_observable_col="IS_INTERVAL_OBSERVABLE",
        is_day_observable_col="IS_DAY_OBSERVABLE",
        is_structural_zero_col="IS_STRUCTURAL_ZERO",
        is_possible_col=None,  # derive per adapter default
        impute_zero_when_observable=False,
        interval_minutes=30,
        periods_per_day=48,
        business_day_start_local_minutes=240,  # 4:00 AM (harmless in demo)
    )

    panel = to_panel_demand_v1(df, spec=spec, validate=True)

    artifacts.base.mkdir(parents=True, exist_ok=True)
    panel.frame.to_parquet(artifacts.panel_demand_v1, index=False)

    print("Contractify OK")
    print(f"- input rows: {df_raw.shape[0]}")
    print(f"- base-dir:  {artifacts.base}")
    print(
        f"- output:    {artifacts.panel_demand_v1.relative_to(artifacts.base.parent.parent.parent)}"
    )


if __name__ == "__main__":
    main()
