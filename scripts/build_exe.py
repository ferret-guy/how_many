"""Build the how_many executable with PyInstaller."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent

BASE_DIR = Path(__file__).resolve().parent.parent
ENTRY_POINT = BASE_DIR / "src/how_many/app.py"
APP_NAME = "how_many"

_DIGITS = re.compile(r"\d+")
_SAFE = re.compile(r"[^A-Za-z0-9.-]+")


def _version_numbers(version: str) -> tuple[int, int, int, int]:
    numbers = [min(int(match), 65535) for match in _DIGITS.findall(version)[:4]]
    while len(numbers) < 4:
        numbers.append(0)
    return tuple(numbers)


def _safe_version(version: str) -> str:
    cleaned = _SAFE.sub("-", version).strip("-")
    return cleaned or "unknown"


def main() -> None:
    try:
        import setuptools_scm  # type: ignore[import-untyped]

        version = str(setuptools_scm.get_version(root=BASE_DIR))
    except Exception:  # pragma: no cover - setuptools_scm optional
        version = "Unknown"

    executable_name = f"{APP_NAME}-v{_safe_version(version)}"
    original_filename = f"{executable_name}.exe"
    version_numbers = _version_numbers(version)

    with NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8") as vf:
        vf.write(
            dedent(
                f"""
                VSVersionInfo(
                  ffi=FixedFileInfo(
                    filevers={version_numbers},
                    prodvers={version_numbers},
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
        )
    version_file = Path(vf.name)

    temp_dir = TemporaryDirectory()
    version_json = Path(temp_dir.name) / "version.json"
    version_json.write_text(
        json.dumps({"version": version}, indent=4), encoding="utf-8"
    )
    try:
        os.chmod(version_json, 0o644)
    except OSError:
        pass

    data_sep = ";" if os.name == "nt" else ":"

    cmd = [
        "pyinstaller",
        "--noconsole",
        "--onefile",
        "--name",
        executable_name,
        "--clean",
        "--version-file",
        str(version_file),
        "--add-data",
        f"{version_json}{data_sep}how_many{os.sep}version.json",
        str(ENTRY_POINT),
    ]

    try:
        sys.exit(subprocess.call(cmd))
    finally:
        try:
            version_file.unlink()
        except OSError:
            pass
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
