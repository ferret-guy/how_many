"""Build the how_many executable with PyInstaller."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from textwrap import dedent

import setuptools_scm

BASE_DIR = Path(__file__).resolve().parent.parent
ENTRY_POINT = BASE_DIR / "src/how_many/app.py"
APP_NAME = "how_many"

_DIGITS = re.compile(r"\d+")
_SAFE = re.compile(r"[^A-Za-z0-9.-]+")


def _version_numbers(version: str) -> tuple[int, ...]:
    numbers = [min(int(match), 65535) for match in _DIGITS.findall(version)[:4]]
    while len(numbers) < 4:
        numbers.append(0)
    return tuple(numbers)


def _safe_version(version: str) -> str:
    cleaned = _SAFE.sub("-", version).strip("-")
    return cleaned or "unknown"


def main() -> None:
    """Build the packaged executable using PyInstaller."""
    version = str(setuptools_scm.get_version(root=BASE_DIR))

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

    result = 1
    try:
        with TemporaryDirectory() as temp_dir:
            version_json = Path(temp_dir) / "version.json"
            version_json.write_text(
                json.dumps({"version": version}, indent=4), encoding="utf-8"
            )

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
                f"--add-data={version_json}{data_sep}.",
                str(ENTRY_POINT),
            ]

            result = subprocess.call(cmd)
    finally:
        try:
            version_file.unlink()
        except OSError:
            pass

    sys.exit(result)


if __name__ == "__main__":
    main()
