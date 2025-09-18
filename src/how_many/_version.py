"""Minimal version helper for the how_many application."""

import json
from os import PathLike
from pathlib import Path
import pkgutil
import sys
from typing import Iterable

PACKAGE_NAME = "how_many"
VERSION_FILENAME = "version.json"


def get_embedded_path(name: str | PathLike[str]) -> Path:
    """Return the path to an embedded resource shipped with the binary."""

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / Path(name)


def get_version() -> str:
    """
    Get version for application.

    :return: Version number.
    """
    if getattr(sys, "frozen", False):  # *.exe
        with open(get_embedded_path("version.json"), "r") as f:
            version = str(json.load(f)["version"])
    else:  # dev
        import setuptools_scm  # type: ignore[import-untyped]

        version = setuptools_scm.get_version()
    return version


__all__ = ["get_version", "get_embedded_path"]
