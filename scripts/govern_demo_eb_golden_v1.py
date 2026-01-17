"""
Governance composition for the EB golden demo dataset (eb_golden_v1).

Reads (diagnostics):
- <base-dir>/diagnostics/cwsl_v1.parquet
- <base-dir>/diagnostics/hr_tau_v1.parquet
- <base-dir>/diagnostics/nsl_ud_v1.parquet
- <base-dir>/diagnostics/fas_v1.parquet
- <base-dir>/diagnostics/dqc_v1.parquet
- <base-dir>/diagnostics/fpc_v1.parquet

Writes:
- <base-dir>/governance/governance_v1.parquet
- <base-dir>/governance/governance_v1_policy.json

Policy (demo, conservative):
- Structural prerequisites first:
  - DQC must not be "BLOCK"/"FAIL"/"INCOMPATIBLE" (token-based conservative check).
  - FPC must not be "BLOCK"/"FAIL"/"INCOMPATIBLE".
- HR@τ must be >= hr_tau_min to permit adjustment.
- If permitted, only identity RAL is allowed in this demo (no-op), unless you later
  introduce real RAL modes under explicit permission.

Notes:
- This script produces a binding governance artifact (a decision file).
- It does NOT execute adjustment or fallback. It only records permissions and reasons.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from eb_examples.paths import GoldenV1Artifacts, resolve_base_dir


@dataclass(frozen=True)
class DemoGovernancePolicyV1:
    version: str = "v1"
    hr_tau: float = 2.0
    hr_tau_min: float = 0.70
    allow_fallback_default: bool = False
    allowed_ral_modes_if_permitted: tuple[str, ...] = ("identity",)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Governance composition for eb_golden_v1 demo")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    p.add_argument(
        "--hr-tau-min",
        type=float,
        default=None,
        help="Override HR@τ threshold for permitting adjustment (default from policy: 0.70).",
    )
    p.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Override policy to allow fallback (default is False).",
    )
    return p.parse_args()


def _load_parquet(path: Any) -> pd.DataFrame:
    p = path if isinstance(path, str) else str(path)
    path_obj = path if hasattr(path, "exists") else None
    if path_obj is not None and not path_obj.exists():
        raise FileNotFoundError(f"Missing required input: {path_obj}")
    return pd.read_parquet(p)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_class(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def _is_structurally_ok(dqc_row: dict[str, Any], fpc_row: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    dqc_df = pd.DataFrame([dqc_row])
    fpc_df = pd.DataFrame([fpc_row])

    dqc_class_col = _pick_col(dqc_df, ["dqc_class", "class", "status", "dqc"])
    fpc_class_col = _pick_col(fpc_df, ["fpc_class", "class", "status", "fpc"])

    dqc_class = _normalize_class(dqc_row.get(dqc_class_col, "")) if dqc_class_col else ""
    fpc_class = _normalize_class(fpc_row.get(fpc_class_col, "")) if fpc_class_col else ""

    bad_tokens = {"BLOCK", "FAIL", "FAILED", "INCOMPATIBLE", "DENY", "NO"}

    ok = True
    if dqc_class and any(tok in dqc_class for tok in bad_tokens):
        ok = False
        reasons.append(f"DQC not admissible: {dqc_class}")
    if fpc_class and any(tok in fpc_class for tok in bad_tokens):
        ok = False
        reasons.append(f"FPC not compatible: {fpc_class}")

    if not dqc_class:
        reasons.append("DQC class not found in artifact (proceeding conservatively).")
    if not fpc_class:
        reasons.append("FPC class not found in artifact (proceeding conservatively).")

    return ok, reasons


def main() -> None:
    args = _parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    artifacts = GoldenV1Artifacts(base=base_dir)

    policy = DemoGovernancePolicyV1()
    hr_tau_min = float(args.hr_tau_min) if args.hr_tau_min is not None else policy.hr_tau_min
    allow_fallback = bool(args.allow_fallback or policy.allow_fallback_default)

    # --- Load required artifacts ---
    cwsl = _load_parquet(artifacts.cwsl_v1)
    hr = _load_parquet(artifacts.hr_tau_v1)
    nslud = _load_parquet(artifacts.nsl_ud_v1)
    fas = _load_parquet(artifacts.fas_v1)
    dqc = _load_parquet(artifacts.dqc_v1)
    fpc = _load_parquet(artifacts.fpc_v1)

    # --- Validate/normalize metric artifacts ---
    # cwsl script writes: entity_id, cwsl, cu, co
    if "entity_id" not in cwsl.columns:
        raise ValueError(f"cwsl missing entity_id. Got: {list(cwsl.columns)}")

    cwsl_val_col = _pick_col(cwsl, ["cwsl", "cwsl_mean"])
    if cwsl_val_col is None:
        raise ValueError(f"cwsl missing cwsl value column. Got: {list(cwsl.columns)}")

    if "entity_id" not in hr.columns or "hr_tau" not in hr.columns:
        raise ValueError(f"hr_tau missing required columns. Got: {list(hr.columns)}")

    if "entity_id" not in nslud.columns or "nsl" not in nslud.columns or "ud" not in nslud.columns:
        raise ValueError(f"nsl_ud missing required columns. Got: {list(nslud.columns)}")

    # --- Keying notes ---
    # - cwsl/hr/nslud are keyed by entity_id (site::forecast_entity_id)
    # - fas/dqc/fpc are keyed by forecast_entity_id
    #
    # Governance for this demo is most coherent at forecast_entity_id level, since
    # DQC/FPC are computed at that level. We'll aggregate entity_id metrics up to
    # forecast_entity_id by mean (across sites).

    def fe_from_entity_id(entity_id: str) -> str:
        parts = str(entity_id).split("::", 1)
        return parts[1] if len(parts) == 2 else str(entity_id)

    m = (
        cwsl[["entity_id", cwsl_val_col]]
        .rename(columns={cwsl_val_col: "cwsl"})
        .merge(hr[["entity_id", "hr_tau"]], on="entity_id", how="inner")
        .merge(nslud[["entity_id", "nsl", "ud"]], on="entity_id", how="inner")
    )
    m["forecast_entity_id"] = m["entity_id"].map(fe_from_entity_id)

    m_agg = m.groupby("forecast_entity_id", as_index=False)[["cwsl", "hr_tau", "nsl", "ud"]].mean()

    # --- Normalize keys for fas/dqc/fpc ---
    if "forecast_entity_id" not in fas.columns:
        key = _pick_col(fas, ["forecast_entity_id", "FORECAST_ENTITY_ID", "entity_id", "id"])
        if key is None:
            raise ValueError(f"fas_v1.parquet missing a recognizable key column. Got: {list(fas.columns)}")
        fas = fas.rename(columns={key: "forecast_entity_id"})

    if "forecast_entity_id" not in dqc.columns:
        raise ValueError(f"dqc_v1.parquet missing forecast_entity_id. Got: {list(dqc.columns)}")
    if "forecast_entity_id" not in fpc.columns:
        raise ValueError(f"fpc_v1.parquet missing forecast_entity_id. Got: {list(fpc.columns)}")

    # Merge all context at forecast_entity_id level
    merged = (
        m_agg.merge(dqc, on="forecast_entity_id", how="left")
        .merge(fpc, on="forecast_entity_id", how="left", suffixes=("", "_fpc"))
        .merge(fas, on="forecast_entity_id", how="left", suffixes=("", "_fas"))
    )

    # Identify class columns (if present)
    dqc_class_col = _pick_col(dqc, ["dqc_class", "class", "status", "dqc"])
    fpc_class_col = _pick_col(fpc, ["fpc_class", "class", "status", "fpc"])
    fas_class_col = _pick_col(fas, ["fas_class", "class", "status", "fas"])

    out_rows: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        row = r.to_dict()
        fe_id = str(row["forecast_entity_id"])

        hr_val = row.get("hr_tau", None)
        hr_val_f = float(hr_val) if hr_val is not None and pd.notna(hr_val) else None

        # Structural checks require presence of DQC/FPC rows
        dqc_row = dqc[dqc["forecast_entity_id"].astype(str) == fe_id]
        fpc_row = fpc[fpc["forecast_entity_id"].astype(str) == fe_id]

        if dqc_row.empty or fpc_row.empty:
            structural_ok = False
            structural_reasons: list[str] = []
            if dqc_row.empty:
                structural_reasons.append("Missing DQC row for this forecast_entity_id.")
            if fpc_row.empty:
                structural_reasons.append("Missing FPC row for this forecast_entity_id.")
        else:
            structural_ok, structural_reasons = _is_structurally_ok(
                dqc_row.iloc[0].to_dict(),
                fpc_row.iloc[0].to_dict(),
            )

        metric_ok = (hr_val_f is not None) and (hr_val_f >= hr_tau_min)

        allow_adjustment = bool(structural_ok and metric_ok)

        reasons: list[str] = []
        reasons.extend(structural_reasons)

        if hr_val_f is None:
            reasons.append("Missing HR@τ; adjustment not permitted under conservative policy.")
        else:
            reasons.append(f"HR@τ={hr_val_f:.6f} vs threshold {hr_tau_min:.6f}")

        reasons.append(
            "Adjustment permitted (structural OK and HR threshold met)."
            if allow_adjustment
            else "Adjustment NOT permitted (conservative policy)."
        )

        out_rows.append(
            {
                "forecast_entity_id": fe_id,
                "allow_adjustment": allow_adjustment,
                "allow_fallback": allow_fallback,
                "allowed_ral_modes": json.dumps(list(policy.allowed_ral_modes_if_permitted)),
                "policy_version": policy.version,
                "hr_tau": hr_val_f,
                "cwsl": float(row["cwsl"]) if pd.notna(row.get("cwsl")) else None,
                "nsl": float(row["nsl"]) if pd.notna(row.get("nsl")) else None,
                "ud": float(row["ud"]) if pd.notna(row.get("ud")) else None,
                "dqc_class": row.get(dqc_class_col, None) if dqc_class_col else None,
                "fpc_class": row.get(fpc_class_col, None) if fpc_class_col else None,
                "fas_class": row.get(fas_class_col, None) if fas_class_col else None,
                "reasons_json": json.dumps(reasons, ensure_ascii=False),
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values(by=["forecast_entity_id"], kind="mergesort")

    artifacts.governance_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts.governance_v1
    out_df.to_parquet(out_path, index=False)

    policy_path = artifacts.governance_v1_policy_json
    policy_payload = asdict(policy)
    policy_payload["hr_tau_min"] = hr_tau_min
    policy_payload["allow_fallback_default"] = allow_fallback
    policy_path.write_text(json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8")

    print("Governance OK")
    print(f"- output:   {out_path}")
    print(f"- policy:   {policy_path}")
    print(f"- slices:   {out_df.shape[0]}")
    print(f"- base-dir: {artifacts.base}")


if __name__ == "__main__":
    main()
