"""Utilities for retrieving how_many version metadata."""

import json
from os import PathLike
from pathlib import Path
import pkgutil
import sys

PACKAGE_NAME = __package__ or "how_many"
VERSION_FILENAME = "version.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_embedded_path(name: str | PathLike[str]) -> Path:
    """Return the path to an embedded resource shipped with the binary."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / Path(name)


def _load_embedded_version() -> str:
    with get_embedded_path(VERSION_FILENAME).open(
        "r", encoding="utf-8"
    ) as version_file:
        payload = json.load(version_file)
    return str(payload["version"])


def _load_package_version() -> str:
    try:
        data = pkgutil.get_data(PACKAGE_NAME, VERSION_FILENAME)
    except OSError:
        data = None
    if data is not None:
        payload = json.loads(data.decode("utf-8"))
        return str(payload["version"])

    import setuptools_scm  # type: ignore[import-untyped]

    return str(setuptools_scm.get_version(root=PROJECT_ROOT))


def get_version() -> str:
    """Return the current how_many version string."""
    if getattr(sys, "frozen", False):
        return _load_embedded_version()
    return _load_package_version()


__all__ = ["get_version", "get_embedded_path"]
