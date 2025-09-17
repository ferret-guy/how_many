"""Utility script for building the how_many executable via PyInstaller."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from textwrap import dedent


def _version_tuple(version: str) -> tuple[int, int, int, int]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 4:
            break
    while len(parts) < 4:
        parts.append(0)
    return tuple(parts[:4])


def _write_version_file(version: str) -> Path:
    """Create a temporary version metadata file for PyInstaller."""
    numbers = _version_tuple(version)
    contents = dedent(
        f"""
        VSVersionInfo(
          ffi=FixedFileInfo(
            filevers={numbers},
            prodvers={numbers},
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0)
          ),
          kids=[
            StringFileInfo(
              [
                StringTable(
                  '040904B0',
                  [
                    StringStruct('CompanyName', 'how_many'),
                    StringStruct('FileDescription', 'how_many overlay'),
                    StringStruct('FileVersion', '{version}'),
                    StringStruct('InternalName', 'how_many'),
                    StringStruct('OriginalFilename', 'how_many.exe'),
                    StringStruct('ProductName', 'how_many'),
                    StringStruct('ProductVersion', '{version}')
                  ]
                )
              ]
            ),
            VarFileInfo([VarStruct('Translation', [1033, 1200])])
          ]
        )
        """
    ).strip()

    with NamedTemporaryFile("w", delete=False, suffix=".txt") as fp:
        fp.write(contents)
        return Path(fp.name)


@dataclass
class BuildConfig:
    """Describe the PyInstaller build inputs."""

    name: str = "how_many"
    entry: Path = Path("src/how_many/app.py")


def main() -> None:
    """Build the application executable using PyInstaller."""
    cfg = BuildConfig()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    try:
        pkg = import_module("how_many")
        version = getattr(pkg, "__version__", "0.0.0")
    except Exception:  # pragma: no cover - build script safety net
        version = "0.0.0"

    version_file: Path | None = None
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
    try:
        version_file = _write_version_file(version)
    except Exception:
        version_file = None

    if version_file is not None:
        cmd.extend(["--version-file", str(version_file)])

    try:
        sys.exit(subprocess.call(cmd))
    finally:
        if version_file is not None and version_file.exists():
            try:
                version_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
