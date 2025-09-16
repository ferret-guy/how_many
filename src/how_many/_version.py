"""Minimal version helper for the how_many application."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def get_embedded_path(name: str) -> Path:
    """Return the path to an embedded resource shipped with the binary."""

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def get_version() -> str:
    """Get version for application."""

    if getattr(sys, "frozen", False):  # *.exe
        with get_embedded_path("version.json").open("r", encoding="utf-8") as handle:
            version = str(json.load(handle)["version"])
    else:  # dev
        import setuptools_scm  # type: ignore[import-untyped]

        version = setuptools_scm.get_version(
            root=Path(__file__).resolve().parent.parent.parent
        )
    return str(version)


__all__ = ["get_version", "get_embedded_path"]
