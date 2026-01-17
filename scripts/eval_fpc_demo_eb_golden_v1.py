"""
Compute FPC (Forecast Primitive Compatibility) diagnostics for eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet
- <base-dir>/diagnostics/dqc_v1.parquet   (required for pipeline ordering/audit; not consumed by FPC API)

Writes:
- <base-dir>/diagnostics/fpc_v1.parquet

Uses (eb-evaluation):
- eb_evaluation.diagnostics.fpc.build_signals_from_series
- eb_evaluation.diagnostics.fpc.classify_fpc
- eb_evaluation.diagnostics.fpc.results_to_dict

Notes:
- DQC MUST precede FPC in the pipeline; we require the DQC artifact exists.
- This script runs FPC with an identity "RAL" (yhat_ral == yhat_base) because we
  have not yet run governance-permitted RAL.
- FPC is diagnostic-only here; no gating/adjustment.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from eb_evaluation.diagnostics.fpc import (
    FPCThresholds,
    build_signals_from_series,
    classify_fpc,
    results_to_dict,
)
from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FPC for eb_golden_v1 demo forecast panel")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=2.0,
        help="Absolute tolerance (same meaning as HR@τ). Default: 2.0",
    )
    p.add_argument(
        "--cost-ratio",
        type=float,
        default=2.0,
        help="Cost ratio (cu/co). Default: 2.0",
    )
    return p.parse_args()


def _parse_forecast_entity_id(entity_id: str) -> str:
    # baseline script used: f"{site_id}::{forecast_entity_id}"
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
    dqc_path = artifacts.dqc_v1

    if not fcst_path.exists():
        raise FileNotFoundError(
            f"Missing forecast panel at {fcst_path}. "
            "Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir <base-dir>"
        )
    if not dqc_path.exists():
        raise FileNotFoundError(
            f"Missing DQC artifact at {dqc_path}. "
            "Run: python scripts/eval_dqc_demo_eb_golden_v1.py --base-dir <base-dir>"
        )

    fcst = pd.read_parquet(fcst_path)

    required = {"entity_id", "interval_start", "y_true", "y_pred"}
    missing = sorted(required - set(fcst.columns))
    if missing:
        raise ValueError(
            f"panel_point_forecast_v1 missing required columns: {missing}. Got: {list(fcst.columns)}"
        )

    # Realized rows only (where truth exists)
    df = fcst[fcst["y_true"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with known y_true found. Nothing to run FPC on.")

    # Attach forecast_entity_id for grouping compatible with DQC slices
    df["forecast_entity_id"] = df["entity_id"].map(_parse_forecast_entity_id)

    thr = FPCThresholds()

    tau = float(args.tau)
    cost_ratio = float(args.cost_ratio)

    rows: list[dict[str, object]] = []
    for fe_id, g in df.groupby("forecast_entity_id", sort=True):
        y = g["y_true"].astype(float).to_list()
        yhat = g["y_pred"].astype(float).to_list()

        # Identity "RAL" for now (no adjustment yet)
        signals = build_signals_from_series(
            y=y,
            yhat_base=yhat,
            yhat_ral=yhat,
            tau=tau,
            cost_ratio=cost_ratio,
        )

        res = classify_fpc(signals=signals, thresholds=thr)
        d = results_to_dict(res)

        # Flatten for Parquet friendliness
        sig = d["signals"]
        rows.append(
            {
                "forecast_entity_id": str(fe_id),
                "fpc_class": d["fpc_class"],
                "reasons_json": json.dumps(d["reasons"], ensure_ascii=False),
                "nsl_base": sig["nsl_base"],
                "nsl_ral": sig["nsl_ral"],
                "delta_nsl": sig["delta_nsl"],
                "hr_base_tau": sig["hr_base_tau"],
                "hr_ral_tau": sig["hr_ral_tau"],
                "delta_hr_tau": sig["delta_hr_tau"],
                "ud": sig["ud"],
                "cwsl_base": sig["cwsl_base"],
                "cwsl_ral": sig["cwsl_ral"],
                "delta_cwsl": sig["delta_cwsl"],
                "intervals": sig["intervals"],
                "shortfall_intervals": sig["shortfall_intervals"],
                "tau": tau,
                "cost_ratio": cost_ratio,
                "ral_mode": "identity",
            }
        )

    out = pd.DataFrame(rows).sort_values(by=["forecast_entity_id"], kind="mergesort")

    artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.fpc_v1
    out.to_parquet(out_path, index=False)

    print("FPC OK")
    print(f"- input forecast: {fcst_path}")
    print(f"- input dqc:      {dqc_path}")
    print(f"- output:         {out_path}")
    print(f"- slices:         {out.shape[0]}")
    print(f"- tau:            {tau}")
    print(f"- cost_ratio:     {cost_ratio}")
    print("- ral_mode:       identity")
    print(f"- base-dir:       {artifacts.base}")


if __name__ == "__main__":
    main()
