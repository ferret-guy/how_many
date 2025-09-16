"""Helpers for resolving the application version information."""

from __future__ import annotations

from functools import lru_cache
import json
import sys
from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def get_embedded_path(*parts: str) -> Path:
    """Return the path to an embedded resource within the executable."""

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = _package_root()
    return base.joinpath(*parts)


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the application version string."""

    if getattr(sys, "frozen", False):
        version_path = get_embedded_path("version.json")
        try:
            with version_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (OSError, ValueError):
            return "0.0.0"
        version = data.get("version", "0.0.0")
        return str(version)

    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        importlib_metadata = None
    else:
        try:
            return importlib_metadata.version("how-many")
        except importlib_metadata.PackageNotFoundError:
            pass
        except Exception:
            return "0.0.0"

    package_root = _package_root()
    repo_root = package_root.parent.parent

    try:
        import setuptools_scm  # type: ignore[import-untyped]
    except Exception:
        return "0.0.0"

    version = setuptools_scm.get_version(
        root=str(repo_root),
        fallback_version="0.0.0",
    )
    return str(version)


__all__ = ["get_embedded_path", "get_version"]
