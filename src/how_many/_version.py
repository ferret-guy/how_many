"""Version helpers for the how_many application."""

from __future__ import annotations

import json
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _frozen_base() -> Path:
    if getattr(sys, "frozen", False):  # pragma: no cover - runtime branch
        base = getattr(sys, "_MEIPASS", None)
        if base is not None:
            return Path(base)
    return _package_root()


def get_embedded_path(*parts: str | Path) -> Path:
    """Return a path to an embedded resource bundled with PyInstaller."""

    base = _frozen_base()
    return base.joinpath(*map(str, parts))


def _load_embedded_version() -> str | None:
    try:
        with open(get_embedded_path("version.json"), "r", encoding="utf-8") as fp:
            data: Any = json.load(fp)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if isinstance(version, str) else None


def _scm_version(root: Path) -> str | None:
    try:
        import setuptools_scm  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        return str(setuptools_scm.get_version(root=root))
    except Exception:  # pragma: no cover - setuptools_scm failure
        return None


def get_version() -> str:
    """Resolve the application version for both frozen and dev environments."""

    embedded = _load_embedded_version()
    if embedded:
        return embedded

    try:
        return importlib_metadata.version("how-many")
    except importlib_metadata.PackageNotFoundError:
        pass

    scm = _scm_version(_package_root().parent)
    return scm or "0.0.0"


__all__ = ["get_embedded_path", "get_version"]
