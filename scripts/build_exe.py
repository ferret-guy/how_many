"""Utility script for building the how_many executable via PyInstaller."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
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


def _write_version_file(version: str, original_filename: str) -> Path:
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
                    StringStruct('OriginalFilename', '{original_filename}'),
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


def _write_embedded_version(version: str) -> Path:
    payload = {"version": version}
    with NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as fp:
        json.dump(payload, fp)
        fp.flush()
        return Path(fp.name)


def _scm_version(root: Path) -> str:
    try:
        import setuptools_scm  # type: ignore[import-untyped]
    except Exception:
        return "0.0.0"

    try:
        return str(setuptools_scm.get_version(root=root))
    except Exception:
        return "0.0.0"


_SAFE_PART = re.compile(r"[^A-Za-z0-9.-]+")


def _normalized_version(version: str) -> str:
    cleaned = _SAFE_PART.sub("-", version).strip("-")
    return cleaned or "0.0.0"


@dataclass
class BuildConfig:
    name: str = "how_many"
    entry: Path = Path("src/how_many/app.py")


def main() -> None:
    cfg = BuildConfig()
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    raw_version = _scm_version(root)

    try:
        pkg = import_module("how_many")
        runtime_version = getattr(pkg, "__version__", raw_version)
    except Exception:  # pragma: no cover - build script safety net
        runtime_version = raw_version

    version = runtime_version or raw_version
    normalized = _normalized_version(version)
    executable_name = f"{cfg.name}-v{normalized}"

    version_file: Path | None = None
    embedded_version: Path | None = None
    script_path = root / cfg.entry
    cmd: list[str] = [
        "pyinstaller",
        "--noconsole",
        "--onefile",
        "--name",
        executable_name,
        "--clean",
        str(script_path),
    ]
    try:
        version_file = _write_version_file(normalized, f"{executable_name}.exe")
    except Exception:
        version_file = None

    try:
        embedded_version = _write_embedded_version(version)
    except Exception:
        embedded_version = None

    if version_file is not None:
        cmd.extend(["--version-file", str(version_file)])

    if embedded_version is not None:
        data_sep = ";" if os.name == "nt" else ":"
        cmd.extend(["--add-data", f"{embedded_version}{data_sep}version.json"])

    try:
        sys.exit(subprocess.call(cmd))
    finally:
        if version_file is not None and version_file.exists():
            try:
                version_file.unlink()
            except OSError:
                pass
        if embedded_version is not None and embedded_version.exists():
            try:
                embedded_version.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
