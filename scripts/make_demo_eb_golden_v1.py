"""
Generate the raw EB golden demo dataset (eb_golden_v1).

Writes:
- raw_demand.csv.gz
- manifest.json

Purpose:
- Provide a deterministic, small-but-complete dataset that exercises the full
  Electric Barometer pipeline (contractify → forecast → metrics → governance → RAL → serving).

Notes:
- This is NOT production logic.
- This is a pedagogical + integration demo dataset.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_base_dir() -> Path:
    return _repo_root() / "data" / "demo" / "eb_golden_v1"


def _resolve_base_dir(base_dir: str | None) -> Path:
    if base_dir is None or base_dir.strip() == "":
        return _default_base_dir()
    p = Path(base_dir)
    return p if p.is_absolute() else (_repo_root() / p)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate eb_golden_v1 demo dataset")
    parser.add_argument("--base-dir", default=None, help="Output base directory")
    args = parser.parse_args()

    base = _resolve_base_dir(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    stores = ["S001", "S002"]
    forecast_entities = [
        ("E_CHICKEN_STRIPS", "Chicken Strips"),
        ("E_CHICKEN_NUGGETS", "Chicken Nuggets"),
    ]

    start_day = date(2024, 1, 1)
    num_days = 7
    intervals_per_day = 48

    rows: list[dict[str, object]] = []

    for store_id in stores:
        for fe_id, fe_name in forecast_entities:
            for d in range(num_days):
                business_day = start_day + timedelta(days=d)
                for idx in range(intervals_per_day):
                    ts = datetime.combine(business_day, datetime.min.time()) + timedelta(minutes=30 * idx)

                    # Demo semantics (intentionally "imperfect"):
                    # - structural zeros represent "closed" periods
                    # - generator emits y=0 for structural zeros
                    # - contractify step will normalize to contract semantics (y -> NA and observable -> False)
                    is_structural_zero = idx < 6  # early morning closed
                    is_observable = not is_structural_zero
                    has_demand = is_observable and (rng.random() > 0.15)

                    demand_qty = int(rng.poisson(lam=6)) if has_demand else 0

                    rows.append(
                        {
                            "STORE_ID": store_id,
                            "FORECAST_ENTITY_ID": fe_id,
                            "FORECAST_ENTITY_NAME": fe_name,
                            "BUSINESS_DAY": business_day.isoformat(),
                            "INTERVAL_30_INDEX": idx,
                            "INTERVAL_START_TS": ts.isoformat(),
                            "DEMAND_QTY": demand_qty,
                            "IS_DAY_OBSERVABLE": True,
                            "IS_INTERVAL_OBSERVABLE": is_observable,
                            "IS_STRUCTURAL_ZERO": is_structural_zero,
                            "HAS_DEMAND": has_demand,
                            "IS_FUTURE": False,
                            "IS_VALUE_KNOWN": True,
                        }
                    )

    df = pd.DataFrame(rows)

    out_csv = base / "raw_demand.csv.gz"
    df.to_csv(out_csv, index=False, compression="gzip")

    manifest = {
        "dataset_id": "eb_golden_v1",
        # timezone-aware; avoids utcnow() deprecation
        "generated_at": datetime.now(UTC).isoformat(),
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "notes": [
            "Synthetic demo dataset for Electric Barometer examples",
            "Includes structural zeros, observable demand, and interval scaffolding",
        ],
    }

    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("DEMO DATASET OK")
    print(f"- base-dir: {base}")
    print(f"- rows:     {df.shape[0]}")
    print(f"- outputs:")
    print(f"  - {out_csv.relative_to(_repo_root())}")
    print(f"  - {manifest_path.relative_to(_repo_root())}")


if __name__ == "__main__":
    main()
