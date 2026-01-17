"""
Compute DQC (Demand Quantization Compatibility) diagnostics for eb_golden_v1.

Reads:
- <base-dir>/panel_demand_v1.parquet

Writes:
- <base-dir>/diagnostics/dqc_v1.parquet

Uses:
- eb_evaluation.diagnostics.dqc.classify_dqc
- eb_evaluation.diagnostics.dqc.dqc_to_dict

Notes:
- DQC is structural and diagnostic-only (no gating, no adjustment).
- Evaluates realized demand only: is_observable==True and y not NA.
- For this demo we compute DQC at the forecast_entity level (aggregating across sites).
"""

from __future__ import annotations

import argparse

import pandas as pd

from eb_evaluation.diagnostics.dqc import DQCThresholds, classify_dqc, dqc_to_dict
from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DQC for eb_golden_v1 demo demand panel")
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

    in_path = artifacts.panel_demand_v1
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing demand panel at {in_path}. "
            "Run: python scripts/contractify_demo_eb_golden_v1.py --base-dir <base-dir>"
        )

    demand = pd.read_parquet(in_path)

    required = {"forecast_entity_id", "y", "is_observable"}
    missing = sorted(required - set(demand.columns))
    if missing:
        raise ValueError(
            f"panel_demand_v1 missing required columns: {missing}. Got: {list(demand.columns)}"
        )

    # Realized / learnable demand only
    df = demand[(demand["is_observable"] == True) & (demand["y"].notna())].copy()  # noqa: E712
    if df.empty:
        raise ValueError("No realized, observable y values found. Nothing to run DQC on.")

    thr = DQCThresholds()

    rows: list[dict[str, object]] = []
    for fe_id, g in df.groupby("forecast_entity_id", sort=True):
        y = g["y"].astype(float).to_list()
        res = classify_dqc(y, thresholds=thr)
        d = dqc_to_dict(res)
        d["forecast_entity_id"] = str(fe_id)
        rows.append(d)

    out = pd.DataFrame(rows).sort_values(by=["forecast_entity_id"], kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.dqc_v1
    out.to_parquet(out_path, index=False)

    print("DQC OK")
    print(f"- input:    {in_path}")
    print(f"- output:   {out_path}")
    print(f"- slices:   {out.shape[0]}")
    print(f"- base-dir: {artifacts.base}")


if __name__ == "__main__":
    main()
