"""Minimal post-installation check for pydrlpdid 0.7.2."""

from __future__ import annotations

import sys

import pydrlpdid
from pydrlpdid import DRLPDID, LPDID

EXPECTED_VERSION = "0.7.2"


def main() -> int:
    version = getattr(pydrlpdid, "__version__", "unknown")
    print(f"pydrlpdid version: {version}")
    print(f"loaded from: {pydrlpdid.__file__}")
    print(f"DRLPDID: {DRLPDID.__module__}.{DRLPDID.__name__}")
    print(f"LPDID: {LPDID.__module__}.{LPDID.__name__}")
    if version != EXPECTED_VERSION:
        print(f"ERROR: expected version {EXPECTED_VERSION}", file=sys.stderr)
        return 1
    print("Installation check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
