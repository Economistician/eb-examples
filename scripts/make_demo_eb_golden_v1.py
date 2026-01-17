"""
Generate the EB golden demo dataset (v1).

Writes:
- data/demo/eb_golden_v1/raw_demand.csv.gz
- data/demo/eb_golden_v1/manifest.json

Deterministic (seeded) and intentionally includes:
- Structural zeros for closed periods
- Observable history with rare missing interval observability
- Future scaffold with unknown demand and non-observable intervals
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    seed: int = 7

    # Panel dimensions
    stores: tuple[str, ...] = ("0001", "0002")
    entities: tuple[tuple[str, str], ...] = (
        ("100", "BEEF_PATTY"),        # generic (no brand/menu coupling)
        ("200", "CHICKEN_STRIPS"),    # acceptable generic menu category
    )

    # Time window
    start_business_day: str = "2026-01-01"
    history_days: int = 7
    future_days: int = 7

    # Interval definition (30-min)
    intervals_per_day: int = 48

    # “Business hours” used to create structural zeros
    # Example: open 06:00–23:00 => interval indices [12, 46)
    open_start_interval: int = 12
    open_end_interval: int = 46

    # Random missing observability in historical data (kept low)
    p_missing_past_interval_observable: float = 0.01

    # Output
    dataset_id: str = "eb_golden_v1"
    out_rel_dir: str = "data/demo/eb_golden_v1"
    out_filename: str = "raw_demand.csv.gz"


def _interval_start_ts(business_day: pd.Timestamp, interval_30_index: int) -> pd.Timestamp:
    return business_day + pd.Timedelta(minutes=30 * interval_30_index)


def _demand_shape(interval_30_index: int) -> float:
    """
    Smooth daily pattern with lunch and dinner peaks.
    Unitless shape before scaling.
    """
    t = interval_30_index / 48.0  # 0..1
    lunch_peak = np.exp(-((t - 0.50) / 0.08) ** 2)
    dinner_peak = np.exp(-((t - 0.75) / 0.10) ** 2)
    baseline = 0.10
    return float(baseline + 1.0 * lunch_peak + 1.2 * dinner_peak)


def _entity_scale(entity_name: str) -> float:
    """
    Entity-specific scale factors to create realistic relative volumes without
    implying any brand-specific menu item semantics.
    """
    if entity_name == "BEEF_PATTY":
        return 40.0
    if entity_name == "CHICKEN_STRIPS":
        return 14.0
    raise ValueError(f"Unknown entity_name: {entity_name}")


def _store_scale(store_id: str) -> float:
    return 1.00 if store_id == "0001" else 0.85


def build_dataframe(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    start_day = pd.Timestamp(cfg.start_business_day)
    all_days = [start_day + pd.Timedelta(days=i) for i in range(cfg.history_days + cfg.future_days)]
    hist_days = set(all_days[: cfg.history_days])
    fut_days = set(all_days[cfg.history_days :])

    rows: list[dict[str, Any]] = []

    for store_id in cfg.stores:
        for entity_id, entity_name in cfg.entities:
            for business_day in all_days:
                day_is_history = business_day in hist_days
                day_is_future = business_day in fut_days

                # History is observable. Future is unknown -> mark as not day-observable
                is_day_observable = bool(day_is_history)

                for interval_30_index in range(cfg.intervals_per_day):
                    ts = _interval_start_ts(business_day, interval_30_index)

                    is_open = cfg.open_start_interval <= interval_30_index < cfg.open_end_interval
                    is_structural_zero = not is_open

                    # Interval observability rules
                    if is_structural_zero:
                        # Closed intervals are known/observable
                        is_interval_observable = True
                    elif day_is_history:
                        # Rare missing observability in the past
                        is_interval_observable = rng.random() >= cfg.p_missing_past_interval_observable
                    else:
                        # Future scaffold: unknown -> not learnable / not countable yet
                        is_interval_observable = False

                    # Demand rules
                    if is_structural_zero:
                        demand_qty: float | None = 0.0
                    elif day_is_future:
                        demand_qty = None
                    else:
                        shape = _demand_shape(interval_30_index)
                        mu = shape * _entity_scale(entity_name) * _store_scale(store_id)

                        noise = rng.normal(loc=0.0, scale=0.15 * mu)
                        spike = 0.0
                        if entity_name == "CHICKEN_STRIPS" and rng.random() < 0.02:
                            spike = rng.uniform(0.8, 1.6) * mu

                        val = max(0.0, mu + noise + spike)
                        demand_qty = float(int(round(val)))

                        # If not observable, hide the value (even in history)
                        if not is_interval_observable:
                            demand_qty = None

                    has_demand = (demand_qty is not None) and (demand_qty > 0)

                    rows.append(
                        {
                            "STORE_ID": store_id,
                            "FORECAST_ENTITY_ID": entity_id,
                            "FORECAST_ENTITY_NAME": entity_name,
                            "BUSINESS_DAY": business_day.date().isoformat(),
                            "INTERVAL_30_INDEX": interval_30_index,
                            "INTERVAL_START_TS": ts.isoformat(),
                            "DEMAND_QTY": demand_qty,
                            "IS_DAY_OBSERVABLE": is_day_observable,
                            "IS_INTERVAL_OBSERVABLE": is_interval_observable,
                            "IS_STRUCTURAL_ZERO": is_structural_zero,
                            "HAS_DEMAND": has_demand,
                        }
                    )

    df = pd.DataFrame(rows)
    df["IS_FUTURE"] = ~df["IS_DAY_OBSERVABLE"]
    df["IS_VALUE_KNOWN"] = df["DEMAND_QTY"].notna()

    # Stable ordering for diff friendliness
    cols = [
        "STORE_ID",
        "FORECAST_ENTITY_ID",
        "FORECAST_ENTITY_NAME",
        "BUSINESS_DAY",
        "INTERVAL_30_INDEX",
        "INTERVAL_START_TS",
        "DEMAND_QTY",
        "IS_DAY_OBSERVABLE",
        "IS_INTERVAL_OBSERVABLE",
        "IS_STRUCTURAL_ZERO",
        "HAS_DEMAND",
        "IS_FUTURE",
        "IS_VALUE_KNOWN",
    ]
    return df[cols].sort_values(
        by=["STORE_ID", "FORECAST_ENTITY_ID", "BUSINESS_DAY", "INTERVAL_30_INDEX"],
        kind="mergesort",
    )


def write_gz_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    with gzip.open(path, "wb") as f:
        f.write(csv_bytes)


def write_manifest(cfg: Config, df: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_id": cfg.dataset_id,
        "version": "v1",
        "generator": "scripts/make_demo_eb_golden_v1.py",
        "seed": cfg.seed,
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "dimensions": {
            "stores": list(cfg.stores),
            "entities": [{"id": eid, "name": ename} for eid, ename in cfg.entities],
            "interval_minutes": 30,
            "intervals_per_day": cfg.intervals_per_day,
        },
        "time_window": {
            "start_business_day": cfg.start_business_day,
            "history_days": cfg.history_days,
            "future_days": cfg.future_days,
        },
        "structural_zero_policy": {
            "open_start_interval": cfg.open_start_interval,
            "open_end_interval": cfg.open_end_interval,
            "closed_is_structural_zero": True,
        },
        "observability_policy": {
            "history_day_observable": True,
            "future_day_observable": False,
            "future_interval_observable": False,
            "p_missing_past_interval_observable": cfg.p_missing_past_interval_observable,
        },
        "columns": list(df.columns),
        "notes": [
            "Structural zeros use DEMAND_QTY=0 and IS_STRUCTURAL_ZERO=True.",
            "Future scaffold uses DEMAND_QTY missing and IS_INTERVAL_OBSERVABLE=False.",
            "Rare missing observability in history hides DEMAND_QTY (sets to missing).",
            "Entity names are generic; no brand-specific menu items are used.",
        ],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    cfg = Config()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / cfg.out_rel_dir

    df = build_dataframe(cfg)

    out_csv_gz = out_dir / cfg.out_filename
    out_manifest = out_dir / "manifest.json"

    write_gz_csv(df, out_csv_gz)
    write_manifest(cfg, df, out_manifest)

    print("Wrote:")
    print(f"- {out_csv_gz.relative_to(repo_root)}  (rows={df.shape[0]}, cols={df.shape[1]})")
    print(f"- {out_manifest.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
