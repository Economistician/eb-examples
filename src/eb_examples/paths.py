from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (pyproject.toml not found).")


def default_base_dir() -> Path:
    return repo_root() / "data" / "demo" / "eb_golden_v1"


def resolve_base_dir(base_dir: str | None) -> Path:
    if base_dir is None or base_dir.strip() == "":
        return default_base_dir()
    p = Path(base_dir)
    return p if p.is_absolute() else (repo_root() / p)


@dataclass(frozen=True)
class GoldenV1Artifacts:
    """
    Canonical file layout for the eb_golden_v1 demo dataset.

    All demo scripts should:
      - accept --base-dir
      - construct GoldenV1Artifacts(base=resolve_base_dir(args.base_dir))
      - read/write ONLY via these properties

    This keeps scripts stable even as the folder layout evolves.
    """

    base: Path

    # ----------------------------
    # Core inputs / contractified artifacts
    # ----------------------------
    @property
    def raw_csv_gz(self) -> Path:
        return self.base / "raw_demand.csv.gz"

    @property
    def manifest_json(self) -> Path:
        return self.base / "manifest.json"

    @property
    def panel_demand_v1(self) -> Path:
        return self.base / "panel_demand_v1.parquet"

    @property
    def panel_point_forecast_v1(self) -> Path:
        return self.base / "panel_point_forecast_v1.parquet"

    # ----------------------------
    # Subdirectories
    # ----------------------------
    @property
    def diagnostics_dir(self) -> Path:
        return self.base / "diagnostics"

    @property
    def governance_dir(self) -> Path:
        return self.base / "governance"

    @property
    def ral_dir(self) -> Path:
        return self.base / "ral"

    @property
    def serving_dir(self) -> Path:
        return self.base / "serving"

    # ----------------------------
    # Diagnostics artifacts
    # ----------------------------
    @property
    def cwsl_v1(self) -> Path:
        return self.diagnostics_dir / "cwsl_v1.parquet"

    @property
    def hr_tau_v1(self) -> Path:
        return self.diagnostics_dir / "hr_tau_v1.parquet"

    @property
    def nsl_ud_v1(self) -> Path:
        return self.diagnostics_dir / "nsl_ud_v1.parquet"

    @property
    def fas_v1(self) -> Path:
        return self.diagnostics_dir / "fas_v1.parquet"

    @property
    def dqc_v1(self) -> Path:
        return self.diagnostics_dir / "dqc_v1.parquet"

    @property
    def fpc_v1(self) -> Path:
        return self.diagnostics_dir / "fpc_v1.parquet"

    # ----------------------------
    # Governance artifacts
    # ----------------------------
    @property
    def governance_v1(self) -> Path:
        return self.governance_dir / "governance_v1.parquet"

    @property
    def governance_v1_policy_json(self) -> Path:
        return self.governance_dir / "governance_v1_policy.json"

    # ----------------------------
    # RAL artifacts
    # ----------------------------
    @property
    def panel_point_forecast_v1_ral(self) -> Path:
        return self.ral_dir / "panel_point_forecast_v1_ral.parquet"

    @property
    def ral_trace_v1(self) -> Path:
        return self.ral_dir / "ral_trace_v1.parquet"

    # ----------------------------
    # Serving artifacts
    # ----------------------------
    @property
    def served_forecast_v1(self) -> Path:
        return self.serving_dir / "served_forecast_v1.parquet"

    @property
    def served_forecast_v1_manifest_json(self) -> Path:
        return self.serving_dir / "served_forecast_v1_manifest.json"
