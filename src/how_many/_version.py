"""Minimal version helper for the how_many application."""

from __future__ import annotations

import json
import pkgutil
import sys
from os import PathLike
from pathlib import Path
from typing import Iterable


PACKAGE_NAME = "how_many"
VERSION_FILENAME = "version.json"


def get_embedded_path(name: str | PathLike[str]) -> Path:
    """Return the path to an embedded resource shipped with the binary."""

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / Path(name)


def _load_version_from_blob(blob: bytes | str | None) -> str | None:
    if not blob:
        return None
    try:
        data = json.loads(blob if isinstance(blob, str) else blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version is not None else None


def _load_version_from_pkgutil() -> str | None:
    try:
        blob = pkgutil.get_data(PACKAGE_NAME, VERSION_FILENAME)
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return None
    return _load_version_from_blob(blob)


def _load_version_from_filesystem(paths: Iterable[Path]) -> str | None:
    for candidate in paths:
        try:
            payload = candidate.read_bytes()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        version = _load_version_from_blob(payload)
        if version is not None:
            return version
    return None


def get_version() -> str:
    """Get version for application."""

    version = _load_version_from_pkgutil()
    if version is None and getattr(sys, "frozen", False):
        candidates = (
            get_embedded_path(Path(PACKAGE_NAME) / VERSION_FILENAME),
            get_embedded_path(VERSION_FILENAME),
        )
        version = _load_version_from_filesystem(candidates)

    if version is None:
        import setuptools_scm  # type: ignore[import-untyped]

        version = setuptools_scm.get_version(
            root=Path(__file__).resolve().parent.parent.parent
        )
    return str(version)


__all__ = ["get_version", "get_embedded_path"]
