"""
Dataset loader for the Electric Barometer golden demo dataset (v1).

This loader is the stable public entrypoint. Callers should NOT hardcode paths.

Data location (repo-relative):
- data/demo/eb_golden_v1/raw_demand.csv.gz
- data/demo/eb_golden_v1/manifest.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DemoGoldenV1Paths:
    raw_csv_gz: Path
    manifest_json: Path


def _repo_root() -> Path:
    """
    Resolve repo root from installed package layout.

    Assumes this file lives at:
      <repo>/src/eb_examples/datasets/demo_golden_v1.py
    """
    return Path(__file__).resolve().parents[3]


def demo_golden_v1_paths() -> DemoGoldenV1Paths:
    root = _repo_root()
    base = root / "data" / "demo" / "eb_golden_v1"
    return DemoGoldenV1Paths(
        raw_csv_gz=base / "raw_demand.csv.gz",
        manifest_json=base / "manifest.json",
    )


def load_demo_golden_v1(*, as_dataframe: bool = True) -> pd.DataFrame:
    """
    Load the EB golden demo dataset (v1) as a pandas DataFrame.

    Notes:
    - Identifier columns are forced to string to preserve leading zeros.
    - DEMAND_QTY is nullable and will be float dtype in pandas when missing exists.
    - INTERVAL_30_INDEX loads as int64.
    - Temporal fields load as strings (caller may parse if needed).
    """
    paths = demo_golden_v1_paths()

    if not paths.raw_csv_gz.exists():
        raise FileNotFoundError(
            f"Golden demo dataset not found at {paths.raw_csv_gz}. "
            "Run: python scripts/make_demo_eb_golden_v1.py"
        )

    df = pd.read_csv(
        paths.raw_csv_gz,
        dtype={
            "STORE_ID": "string",
            "FORECAST_ENTITY_ID": "string",
            "FORECAST_ENTITY_NAME": "string",
            "BUSINESS_DAY": "string",
            "INTERVAL_START_TS": "string",
        },
    )

    # Type normalization
    if "INTERVAL_30_INDEX" in df.columns:
        df["INTERVAL_30_INDEX"] = df["INTERVAL_30_INDEX"].astype("int64")

    for c in [
        "IS_DAY_OBSERVABLE",
        "IS_INTERVAL_OBSERVABLE",
        "IS_STRUCTURAL_ZERO",
        "HAS_DEMAND",
        "IS_FUTURE",
        "IS_VALUE_KNOWN",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("bool")

    return df


def load_demo_golden_v1_manifest() -> dict[str, Any]:
    """
    Load the dataset manifest.json for the EB golden demo dataset (v1).
    """
    import json

    paths = demo_golden_v1_paths()
    if not paths.manifest_json.exists():
        raise FileNotFoundError(
            f"Golden demo manifest not found at {paths.manifest_json}. "
            "Run: python scripts/make_demo_eb_golden_v1.py"
        )

    return json.loads(paths.manifest_json.read_text(encoding="utf-8"))
