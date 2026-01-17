"""
Run the full EB golden demo pipeline (eb_golden_v1) end-to-end.

This is the "one command" UX path for eb-examples.

Runs (in canonical order):
1) Generate demo raw data
2) Contractify to PanelDemandV1
3) Baseline point forecast
4) Metrics: CWSL, HR@τ, NSL/UD
5) FAS (optional)
6) DQC
7) FPC (identity RAL signals)
8) Governance
9) RAL (identity/no-op under permission)
10) Serving artifact

Usage:
  python scripts/run_demo_eb_golden_v1.py
  python scripts/run_demo_eb_golden_v1.py --base-dir data/demo/eb_golden_v1_run2
  python scripts/run_demo_eb_golden_v1.py --base-dir data/demo/eb_golden_v1_run3 --no-fas
  python scripts/run_demo_eb_golden_v1.py --steps

Notes:
- This runner uses subprocess to call the existing scripts so each step remains
  independently runnable and debuggable.
- Fails fast on first non-zero exit code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    # Robust: walk up until we find pyproject.toml
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (pyproject.toml not found).")


def _scripts_dir() -> Path:
    # scripts/run_demo_...py lives in <repo>/scripts
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    name: str
    script: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full eb_golden_v1 demo pipeline.")
    p.add_argument(
        "--base-dir",
        default=None,
        help="Artifact base directory (repo-relative or absolute). Default: data/demo/eb_golden_v1",
    )
    p.add_argument("--no-fas", action="store_true", help="Skip optional FAS step")
    p.add_argument("--steps", action="store_true", help="Print the ordered steps and exit")
    return p.parse_args()


def _run_step(step: Step, *, repo_root: Path, scripts_dir: Path, base_dir: str | None) -> None:
    # Resolve scripts relative to THIS directory (most robust)
    script_path = scripts_dir / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print("\n" + "=" * 88)
    print(f"STEP: {step.name}")
    if base_dir:
        print(f"CMD : {step.script} --base-dir {base_dir}")
    else:
        print(f"CMD : {step.script}")
    print("=" * 88)

    cmd = [sys.executable, str(script_path)]
    if base_dir is not None and base_dir.strip() != "":
        cmd.extend(["--base-dir", base_dir])

    proc = subprocess.run(cmd, cwd=str(repo_root))
    if proc.returncode != 0:
        raise SystemExit(
            f"\nFAILED: {step.name} (exit={proc.returncode})\nCommand: {' '.join(cmd)}"
        )


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    scripts_dir = _scripts_dir()

    steps: list[Step] = [
        Step("Generate demo dataset", "make_demo_eb_golden_v1.py"),
        Step("Contractify demand -> PanelDemandV1", "contractify_demo_eb_golden_v1.py"),
        Step("Baseline point forecast", "baseline_forecast_demo_eb_golden_v1.py"),
        Step("Evaluate CWSL", "eval_cwsl_demo_eb_golden_v1.py"),
        Step("Evaluate HR@τ", "eval_hr_tau_demo_eb_golden_v1.py"),
        Step("Evaluate NSL/UD", "eval_nsl_ud_demo_eb_golden_v1.py"),
    ]
    if not args.no_fas:
        steps.append(Step("Evaluate FAS", "eval_fas_demo_eb_golden_v1.py"))
    steps.extend(
        [
            Step("Evaluate DQC", "eval_dqc_demo_eb_golden_v1.py"),
            Step("Evaluate FPC", "eval_fpc_demo_eb_golden_v1.py"),
            Step("Governance composition", "govern_demo_eb_golden_v1.py"),
            Step("RAL (identity under permission)", "ral_demo_eb_golden_v1.py"),
            Step("Serving / execution artifact", "serve_demo_eb_golden_v1.py"),
        ]
    )

    if args.steps:
        for i, s in enumerate(steps, start=1):
            print(f"{i:02d}. {s.script}  —  {s.name}")
        return

    print("EB DEMO RUNNER: eb_golden_v1")
    print(f"- repo:     {repo_root}")
    print(f"- steps:    {len(steps)}")
    print(f"- base-dir: {args.base_dir or 'data/demo/eb_golden_v1'}")
    print(f"- fas:      {'disabled' if args.no_fas else 'enabled'}")

    for step in steps:
        _run_step(step, repo_root=repo_root, scripts_dir=scripts_dir, base_dir=args.base_dir)

    print("\n" + "=" * 88)
    print("ALL STEPS OK ✅")
    print("=" * 88)

    base = (
        Path(args.base_dir)
        if (args.base_dir and args.base_dir.strip() != "")
        else Path("data/demo/eb_golden_v1")
    )
    if not base.is_absolute():
        base = repo_root / base

    outputs = [
        base / "raw_demand.csv.gz",
        base / "manifest.json",
        base / "panel_demand_v1.parquet",
        base / "panel_point_forecast_v1.parquet",
        base / "diagnostics" / "cwsl_v1.parquet",
        base / "diagnostics" / "hr_tau_v1.parquet",
        base / "diagnostics" / "nsl_ud_v1.parquet",
        base / "diagnostics" / "fas_v1.parquet",
        base / "diagnostics" / "dqc_v1.parquet",
        base / "diagnostics" / "fpc_v1.parquet",
        base / "governance" / "governance_v1.parquet",
        base / "governance" / "governance_v1_policy.json",
        base / "ral" / "panel_point_forecast_v1_ral.parquet",
        base / "ral" / "ral_trace_v1.parquet",
        base / "serving" / "served_forecast_v1.parquet",
        base / "serving" / "served_forecast_v1_manifest.json",
    ]

    print("Key outputs:")
    for p in outputs:
        status = "OK" if p.exists() else "MISSING"
        try:
            rel = p.relative_to(repo_root)
            print(f"- {status:7} {rel}")
        except Exception:
            print(f"- {status:7} {p}")


if __name__ == "__main__":
    main()
