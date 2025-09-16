"""Minimal version helper for the how_many application."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def get_version() -> str:
    """Return the application version.

    Preference order:
    1. Embedded ``version.json`` generated during PyInstaller builds.
    2. ``setuptools_scm`` metadata for source checkouts.
    3. ``"Unknown"`` when neither source succeeds.
    """

    base_path = Path(__file__).resolve().parent

    if getattr(sys, "frozen", False):  # pragma: no cover - runtime only
        frozen_base = Path(getattr(sys, "_MEIPASS", base_path))
        try:
            with (frozen_base / "version.json").open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "version" in data:
                return str(data["version"])
        except (OSError, json.JSONDecodeError):
            pass

    try:
        import setuptools_scm  # type: ignore[import-untyped]

        return str(setuptools_scm.get_version(root=base_path.parent))
    except Exception:  # pragma: no cover - optional dependency
        return "Unknown"


__all__ = ["get_version"]
