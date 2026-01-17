"""
Create the served forecast artifact for eb_golden_v1.

Reads:
- <base-dir>/panel_point_forecast_v1.parquet
- <base-dir>/ral/panel_point_forecast_v1_ral.parquet
- <base-dir>/governance/governance_v1.parquet

Writes:
- <base-dir>/serving/served_forecast_v1.parquet
- <base-dir>/serving/served_forecast_v1_manifest.json

Serving rule (demo):
- If governance.allow_adjustment is True for the forecast_entity_id:
    serve y_pred_ral (from RAL artifact)
  else:
    serve y_pred (baseline)
- Fallback is not implemented in this demo; served_source indicates what happened.

Notes:
- This is the first artifact intended for downstream consumption.
- Includes trace columns so behavior is auditable.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build served forecast artifact (eb_golden_v1 demo)")
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

    base_fcst_path = artifacts.panel_point_forecast_v1
    ral_fcst_path = artifacts.panel_point_forecast_v1_ral
    gov_path = artifacts.governance_v1

    for pth, hint in [
        (
            base_fcst_path,
            f"Run: python scripts/baseline_forecast_demo_eb_golden_v1.py --base-dir {base_dir}",
        ),
        (ral_fcst_path, f"Run: python scripts/ral_demo_eb_golden_v1.py --base-dir {base_dir}"),
        (gov_path, f"Run: python scripts/govern_demo_eb_golden_v1.py --base-dir {base_dir}"),
    ]:
        if not pth.exists():
            raise FileNotFoundError(f"Missing required input: {pth}. {hint}")

    base_fcst = pd.read_parquet(base_fcst_path)
    ral_fcst = pd.read_parquet(ral_fcst_path)
    gov = pd.read_parquet(gov_path)

    # Required columns
    checks = [
        ("baseline", base_fcst, {"entity_id", "interval_start", "y_pred"}),
        ("ral", ral_fcst, {"entity_id", "interval_start", "y_pred_ral", "ral_mode", "ral_applied"}),
        ("governance", gov, {"forecast_entity_id", "allow_adjustment"}),
    ]
    for name, df, cols in checks:
        missing = sorted(cols - set(df.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}. Got: {list(df.columns)}")

    # Permission map
    allow_map: dict[str, bool] = (
        gov.assign(forecast_entity_id=gov["forecast_entity_id"].astype(str))
        .set_index("forecast_entity_id")["allow_adjustment"]
        .map(bool)
        .to_dict()
    )

    # Join baseline + ral (same entity_id/interval_start)
    base_fcst = base_fcst.copy()
    base_fcst["interval_start"] = pd.to_datetime(base_fcst["interval_start"], errors="raise")

    ral_fcst = ral_fcst.copy()
    ral_fcst["interval_start"] = pd.to_datetime(ral_fcst["interval_start"], errors="raise")

    merged = base_fcst.merge(
        ral_fcst[["entity_id", "interval_start", "y_pred_ral", "ral_mode", "ral_applied"]],
        on=["entity_id", "interval_start"],
        how="left",
        validate="one_to_one",
    )

    merged["forecast_entity_id"] = merged["entity_id"].map(_parse_forecast_entity_id)
    merged["allow_adjustment"] = merged["forecast_entity_id"].map(
        lambda x: bool(allow_map.get(str(x), False))
    )

    # Serve RAL only when (a) allowed and (b) present
    use_ral = (merged["allow_adjustment"] == True) & merged["y_pred_ral"].notna()  # noqa: E712
    merged["y_served"] = merged["y_pred"]
    merged.loc[use_ral, "y_served"] = merged.loc[use_ral, "y_pred_ral"]

    merged["served_source"] = "baseline"
    merged.loc[use_ral, "served_source"] = "ral"

    # Consumer-friendly schema + traceability
    cols = [
        "entity_id",
        "forecast_entity_id",
        "interval_start",
        "y_served",
        "served_source",
        # trace columns
        "y_pred",
        "y_pred_ral",
        "ral_mode",
        "ral_applied",
        "allow_adjustment",
    ]
    out = (
        merged[cols]
        .sort_values(by=["entity_id", "interval_start"], kind="mergesort")
        .reset_index(drop=True)
    )

    artifacts.serving_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.served_forecast_v1
    out.to_parquet(out_path, index=False)

    manifest = {
        "dataset_id": "eb_golden_v1",
        "served_artifact": "served_forecast_v1",
        "base_dir": str(artifacts.base),
        "inputs": {
            "baseline": str(base_fcst_path),
            "ral": str(ral_fcst_path),
            "governance": str(gov_path),
        },
        "schema": cols,
        "notes": [
            "y_served is the final value intended for downstream consumption.",
            "served_source indicates whether baseline or RAL provided y_served.",
            "Fallback is not implemented in this demo.",
        ],
    }
    (artifacts.serving_dir / "served_forecast_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SERVING OK")
    print(f"- output: {out_path}")
    print(f"- rows:   {out.shape[0]}")
    print(f"- served_source counts:\n{out['served_source'].value_counts().to_string()}")
    print(f"- base-dir:{artifacts.base}")


if __name__ == "__main__":
    main()
