"""Utility script for building the how_many executable via PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BuildConfig:
    name: str = "how_many"
    entry: Path = Path("src/how_many/app.py")


def main() -> None:
    cfg = BuildConfig()
    root = Path(__file__).resolve().parent.parent
    script_path = root / cfg.entry
    cmd: list[str] = [
        "pyinstaller",
        "--noconsole",
        "--onefile",
        "--name",
        cfg.name,
        "--clean",
        str(script_path),
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
