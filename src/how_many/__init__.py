"""how_many package exposing a lazy ``main`` entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0"

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from .app import main as _main_type  # noqa: F401


def main() -> None:
    """Entry point for ``python -m how_many`` and console scripts."""
    from .app import main as _main

    _main()


__all__ = ["main", "__version__"]
