"""Compatibility wrapper forwarding legacy board pipeline test invocations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tools" / "evaluate_board_calibration.py"
    argv = [sys.executable, str(script), *sys.argv[1:]]
    completed = subprocess.run(argv, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()