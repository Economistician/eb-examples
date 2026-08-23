from __future__ import annotations

import importlib


def test_import_eb_examples() -> None:
    # Ensures the package is importable in editable installs and CI.
    pkg = importlib.import_module("eb_examples")
    version = pkg.__version__
    assert isinstance(version, str) and version


def test_cli_help_runs() -> None:
    # argparse prints help and exits with SystemExit(0); treat that as success.
    from eb_examples.cli import main  # type: ignore[reportMissingImports]

    try:
        rc = main(["--help"])
    except SystemExit as e:  # pragma: no cover
        code = 0 if e.code is None else int(e.code)
        assert code == 0
    else:
        assert rc == 0
