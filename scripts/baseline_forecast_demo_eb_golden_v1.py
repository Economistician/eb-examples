"""
Baseline forecast for the EB golden demo dataset (eb_golden_v1).

Reads:
- <base-dir>/panel_demand_v1.parquet

Writes:
- <base-dir>/panel_point_forecast_v1.parquet

Strategy:
- For each (site_id, forecast_entity_id, INTERVAL_30_INDEX), compute baseline =
  mean(y) over observable rows with known y.
- Emit that baseline prediction for every row (history + future scaffold).

Validation:
- Uses eb-contracts point forecast contract:
    eb_contracts.api.validate.panel_point_forecast_v1
  which expects columns: entity_id, interval_start, y_true, y_pred
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _make_entity_id(site_id: object, forecast_entity_id: object) -> str:
    """
    Forecast contracts use a single opaque `entity_id`.
    We compose it deterministically from the demand identity keys.
    """
    return f"{site_id}::{forecast_entity_id}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Baseline point forecast for demo golden-v1.")
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

    in_path = artifacts.panel_demand_v1
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing contractified demand at {in_path}. "
            "Run: python scripts/contractify_demo_eb_golden_v1.py --base-dir <base-dir>"
        )

    demand = pd.read_parquet(in_path)

    required = {
        "site_id",
        "forecast_entity_id",
        "y",
        "is_observable",
        "INTERVAL_30_INDEX",
        "INTERVAL_START_TS",
    }
    missing = sorted(required - set(demand.columns))
    if missing:
        raise ValueError(
            f"panel_demand_v1 missing required columns: {missing}. Got: {list(demand.columns)}"
        )

    # Ensure timestamp dtype
    demand = demand.copy()
    demand["INTERVAL_START_TS"] = pd.to_datetime(demand["INTERVAL_START_TS"], errors="raise")

    # Compute baseline mean per (site, entity, interval_index) using observable, known y only
    hist = demand[(demand["is_observable"] == True) & (demand["y"].notna())]  # noqa: E712
    grp_cols = ["site_id", "forecast_entity_id", "INTERVAL_30_INDEX"]
    baseline = hist.groupby(grp_cols, as_index=False)["y"].mean().rename(columns={"y": "y_pred"})

    # Join baseline back to all rows (dense forecast over full scaffold)
    out = demand[
        ["site_id", "forecast_entity_id", "INTERVAL_30_INDEX", "INTERVAL_START_TS", "y"]
    ].merge(
        baseline,
        on=grp_cols,
        how="left",
        validate="many_to_one",
    )

    # If a slice had no observable history (shouldn't happen in demo), fill predictions with 0
    out["y_pred"] = out["y_pred"].fillna(0.0)

    # Build forecast-contract frame
    fcst = pd.DataFrame(
        {
            "entity_id": out.apply(
                lambda r: _make_entity_id(r["site_id"], r["forecast_entity_id"]), axis=1
            ),
            "interval_start": out["INTERVAL_START_TS"],
            "y_true": out["y"],  # may be NA for future scaffold (allowed)
            "y_pred": out["y_pred"],  # baseline prediction
        }
    )

    # Validate + construct contract object (raises on violation)
    from eb_contracts.api.validate import panel_point_forecast_v1

    panel = panel_point_forecast_v1(fcst)

    artifacts.base.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.panel_point_forecast_v1
    panel.frame.to_parquet(out_path, index=False)

    print("Baseline Forecast OK")
    print(f"- input:   {in_path}")
    print(f"- output:  {out_path}")
    print(f"- rows:    {panel.frame.shape[0]}")
    print(f"- base-dir:{artifacts.base}")


if __name__ == "__main__":
    main()
