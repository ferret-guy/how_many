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
from tempfile import NamedTemporaryFile, TemporaryDirectory
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


def _normalized_version_tag(version: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", version.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "0.0.0"


@dataclass
class BuildConfig:
    name: str = "how_many"
    entry: Path = Path("src/how_many/app.py")


def main() -> None:
    cfg = BuildConfig()
    root = Path(__file__).resolve().parent.parent
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    version = "0.0.0"
    resolved_version: str | None = None

    try:
        import setuptools_scm  # type: ignore[import-untyped]
    except Exception:
        pass
    else:
        try:
            resolved_version = str(
                setuptools_scm.get_version(
                    root=str(root),
                    fallback_version="0.0.0",
                )
            )
        except Exception:
            resolved_version = None

    if resolved_version is None:
        try:
            version_module = import_module("how_many._version")
        except Exception:
            version_module = None
        if version_module is not None:
            get_version = getattr(version_module, "get_version", None)
            if callable(get_version):
                try:
                    resolved_version = str(get_version())
                except Exception:
                    resolved_version = None

    if resolved_version is None:
        try:
            pkg = import_module("how_many")
        except Exception:
            pkg = None
        if pkg is not None:
            pkg_version = getattr(pkg, "__version__", None)
            if isinstance(pkg_version, str):
                resolved_version = pkg_version

    if resolved_version is not None:
        version = resolved_version

    version_file: Path | None = None
    embedded_dir: TemporaryDirectory[str] | None = None
    version_resource: Path | None = None
    version_tag = _normalized_version_tag(version)
    versioned_name = f"{cfg.name}-v{version_tag}"
    original_filename = f"{versioned_name}.exe"
    script_path = root / cfg.entry
    cmd: list[str] = [
        "pyinstaller",
        "--noconsole",
        "--onefile",
        "--name",
        versioned_name,
        "--clean",
        str(script_path),
    ]
    try:
        version_file = _write_version_file(version, original_filename)
    except Exception:
        version_file = None

    try:
        embedded_dir = TemporaryDirectory()
        version_resource = Path(embedded_dir.name) / "version.json"
        with version_resource.open("w", encoding="utf-8") as fp:
            json.dump({"version": version}, fp, indent=4)
    except Exception:
        version_resource = None

    if version_file is not None:
        cmd.extend(["--version-file", str(version_file)])

    if version_resource is not None:
        data_spec = f"{os.fspath(version_resource)}{os.pathsep}."
        cmd.extend(["--add-data", data_spec])

    try:
        sys.exit(subprocess.call(cmd))
    finally:
        if version_file is not None and version_file.exists():
            try:
                version_file.unlink()
            except OSError:
                pass
        if embedded_dir is not None:
            embedded_dir.cleanup()


if __name__ == "__main__":
    main()
