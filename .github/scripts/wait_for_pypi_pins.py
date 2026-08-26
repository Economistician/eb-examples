"""Poll PyPI until exact runtime pins in pyproject.toml resolve.

Exits non-zero after a bounded timeout so CI fails closed on unpublished system-release pins.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request

PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^;\s]+)")
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
TIMEOUT_S = 600
INITIAL_SLEEP_S = 5
MAX_SLEEP_S = 40
HTTP_TIMEOUT_S = 30


def _exact_pins(pyproject: Path) -> list[tuple[str, str]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pins: list[tuple[str, str]] = []
    for dep in data.get("project", {}).get("dependencies", []):
        match = PIN_RE.match(str(dep).strip())
        if match is not None:
            pins.append((match.group(1), match.group(2)))
    return pins


def _exists_on_pypi(name: str, version: str) -> bool:
    url = PYPI_JSON.format(name=name, version=version)
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "eb-examples-ci"},
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            payload = json.load(resp)
        urls = payload.get("urls")
        return isinstance(urls, list) and bool(urls)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        print(
            f"PyPI HTTP {exc.code} for {name}=={version}; treating as unavailable", file=sys.stderr
        )
        return False
    except urllib.error.URLError as exc:
        print(f"PyPI request failed for {name}=={version}: {exc.reason}", file=sys.stderr)
        return False


def main() -> int:
    pyproject = Path("pyproject.toml")
    if not pyproject.is_file():
        print("ERROR: pyproject.toml not found in the working directory.", file=sys.stderr)
        return 1

    pins = _exact_pins(pyproject)
    if not pins:
        print("No exact dependency pins to wait for.")
        return 0

    print("Waiting for exact runtime pins on PyPI:")
    for name, version in pins:
        print(f"  - {name}=={version}")

    deadline = time.monotonic() + TIMEOUT_S
    sleep_s = INITIAL_SLEEP_S
    missing = list(pins)
    while missing and time.monotonic() < deadline:
        still_missing: list[tuple[str, str]] = []
        for name, version in missing:
            if _exists_on_pypi(name, version):
                print(f"found {name}=={version}")
            else:
                print(f"waiting for {name}=={version}")
                still_missing.append((name, version))
        missing = still_missing
        if not missing:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait = min(float(sleep_s), remaining)
        print(f"retrying in {wait:.0f}s")
        time.sleep(wait)
        sleep_s = min(sleep_s * 2, MAX_SLEEP_S)

    if missing:
        formatted = ", ".join(f"{name}=={version}" for name, version in missing)
        print(
            "ERROR: timed out waiting for PyPI to serve exact pins: "
            f"{formatted}. Publish sibling packages before pushing consumer pins, then re-run CI.",
            file=sys.stderr,
        )
        return 1

    print("All exact pins are available on PyPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
