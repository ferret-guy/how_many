import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BuildConfig:
    name: str = "how_many"
    entry: str = "how_many.py"


def main() -> None:
    cfg = BuildConfig()
    root = Path(__file__).parent
    cmd: list[str] = [
        "pyinstaller",
        "--noconsole",
        "--onefile",
        "--name",
        cfg.name,
        "--clean",
        str((root / cfg.entry).as_posix()),
    ]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
