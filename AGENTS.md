# Agent Guidelines

- Running the GUI integration tests requires the system packages `libgl1`, `libegl1`, and `libxkbcommon-x11-0` to satisfy Qt's OpenGL dependencies.
- Install these packages in CI or local environments before executing `uv run pytest` or launching the packaged executable tests.
- Do not add `from __future__ import annotations`; the project targets Python 3.10+ and relies on native postponed evaluation.
