"""
eb-examples CLI.

Goal: a user-friendly entrypoint to run demo pipelines without hunting for scripts.

Examples:
  python -m eb_examples demo golden-v1
  python -m eb_examples demo golden-v1 --no-fas
  python -m eb_examples demo golden-v1 --steps
  python -m eb_examples demo golden-v1 --base-dir data/demo/eb_golden_v1_run5

Console script behavior:
  eb-demo golden-v1 --steps        # shorthand for: eb-demo demo golden-v1 --steps
  eb-demo demo golden-v1 --steps   # explicit form
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    """
    Find the eb-examples repo root by walking upward until we find pyproject.toml.

    This avoids fragile parent-depth assumptions and works even inside a mono-repo checkout.
    """
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (pyproject.toml not found when walking upward).")


def _resolve_base_dir(base_dir: str | None, *, repo_root: Path) -> Path:
    """
    Resolve a base-dir argument (absolute or repo-relative), defaulting to the canonical demo dir.
    """
    if base_dir is None or str(base_dir).strip() == "":
        return repo_root / "data" / "demo" / "eb_golden_v1"
    p = Path(base_dir)
    return p if p.is_absolute() else (repo_root / p)


@dataclass(frozen=True)
class Step:
    name: str
    script: str


def _run_step(step: Step, *, repo_root: Path, base_dir: Path | None) -> None:
    script_path = repo_root / "scripts" / step.script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print("\n" + "=" * 88)
    print(f"STEP: {step.name}")
    if base_dir is not None:
        try:
            shown = base_dir.relative_to(repo_root)
        except ValueError:
            shown = base_dir
        print(f"CMD : {step.script} --base-dir {shown}")
    else:
        print(f"CMD : {step.script}")
    print("=" * 88)

    cmd = [sys.executable, str(script_path)]
    if base_dir is not None:
        cmd += ["--base-dir", str(base_dir)]

    proc = subprocess.run(cmd, cwd=str(repo_root))
    if proc.returncode != 0:
        raise SystemExit(f"\nFAILED: {step.name} (exit={proc.returncode})\nCommand: {' '.join(cmd)}")


def _print_outputs(repo_root: Path, *, base_dir: Path) -> None:
    outputs = [
        base_dir / "raw_demand.csv.gz",
        base_dir / "manifest.json",
        base_dir / "panel_demand_v1.parquet",
        base_dir / "panel_point_forecast_v1.parquet",
        base_dir / "diagnostics" / "cwsl_v1.parquet",
        base_dir / "diagnostics" / "hr_tau_v1.parquet",
        base_dir / "diagnostics" / "nsl_ud_v1.parquet",
        base_dir / "diagnostics" / "fas_v1.parquet",
        base_dir / "diagnostics" / "dqc_v1.parquet",
        base_dir / "diagnostics" / "fpc_v1.parquet",
        base_dir / "governance" / "governance_v1.parquet",
        base_dir / "governance" / "governance_v1_policy.json",
        base_dir / "ral" / "panel_point_forecast_v1_ral.parquet",
        base_dir / "ral" / "ral_trace_v1.parquet",
        base_dir / "serving" / "served_forecast_v1.parquet",
        base_dir / "serving" / "served_forecast_v1_manifest.json",
    ]

    print("\n" + "=" * 88)
    print("Key outputs:")
    for p in outputs:
        status = "OK" if p.exists() else "MISSING"
        rel = p.relative_to(repo_root) if str(p).startswith(str(repo_root)) else p
        print(f"- {status:7} {rel}")


def _demo_golden_v1_steps(*, include_fas: bool) -> list[Step]:
    steps: list[Step] = [
        Step("Generate demo dataset", "make_demo_eb_golden_v1.py"),
        Step("Contractify demand -> PanelDemandV1", "contractify_demo_eb_golden_v1.py"),
        Step("Baseline point forecast", "baseline_forecast_demo_eb_golden_v1.py"),
        Step("Evaluate CWSL", "eval_cwsl_demo_eb_golden_v1.py"),
        Step("Evaluate HR@τ", "eval_hr_tau_demo_eb_golden_v1.py"),
        Step("Evaluate NSL/UD", "eval_nsl_ud_demo_eb_golden_v1.py"),
    ]
    if include_fas:
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
    return steps


def _cmd_demo(args: argparse.Namespace) -> int:
    repo_root = _repo_root()

    if args.demo_name != "golden-v1":
        raise SystemExit(f"Unknown demo: {args.demo_name!r}. Supported: golden-v1")

    base_dir = _resolve_base_dir(getattr(args, "base_dir", None), repo_root=repo_root)
    steps = _demo_golden_v1_steps(include_fas=not args.no_fas)

    if args.steps:
        for i, s in enumerate(steps, start=1):
            print(f"{i:02d}. {s.script}  —  {s.name}")
        return 0

    print("EB DEMO CLI")
    print(f"- demo:     {args.demo_name}")
    print(f"- repo:     {repo_root}")
    print(f"- steps:    {len(steps)}")
    print(f"- base-dir: {base_dir.relative_to(repo_root) if str(base_dir).startswith(str(repo_root)) else base_dir}")
    print(f"- fas:      {'enabled' if (not args.no_fas) else 'disabled'}")

    for step in steps:
        _run_step(step, repo_root=repo_root, base_dir=base_dir)

    print("\n" + "=" * 88)
    print("ALL STEPS OK ✅")
    print("=" * 88)

    _print_outputs(repo_root, base_dir=base_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eb_examples", description="Electric Barometer examples CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run a demo pipeline")
    demo.add_argument("demo_name", help="Demo name (e.g., golden-v1)")
    demo.add_argument("--base-dir", default=None, help="Output base directory (absolute or repo-relative)")
    demo.add_argument("--no-fas", action="store_true", help="Skip optional FAS step")
    demo.add_argument("--steps", action="store_true", help="Print the ordered steps and exit")
    demo.set_defaults(func=_cmd_demo)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    # If invoked via console script `eb-demo`, allow shorthand:
    #   eb-demo golden-v1 --steps
    # which becomes:
    #   eb-demo demo golden-v1 --steps
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] != "demo" and argv[0] not in {"-h", "--help"}:
        argv = ["demo", *argv]

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
