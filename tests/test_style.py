"""Ensure the repository meets the configured style and type expectations."""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Iterable, Tuple

import pytest

Command = Tuple[str, ...]
StyleCheck = Tuple[str, Command]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PATHS: Tuple[str, ...] = ("src", "tests", "scripts")


def _python_module(module: str, *args: str) -> Command:
    return (sys.executable, "-m", module, *args)


MYPY_TARGETS: Tuple[str, ...] = ("src/how_many/utils",)

CHECKS: Tuple[StyleCheck, ...] = (
    (
        "black",
        _python_module("black", "--check", "--diff", *PATHS),
    ),
    (
        "isort",
        _python_module("isort", "--check-only", "--diff", *PATHS),
    ),
    (
        "flake8",
        _python_module("flake8", *PATHS),
    ),
    (
        "mypy",
        _python_module("mypy", *MYPY_TARGETS),
    ),
)


@pytest.mark.parametrize(
    ("name", "command"),
    CHECKS,
    ids=[check[0] for check in CHECKS],
)
def test_code_style(name: str, command: Command) -> None:
    """Run ``name`` and ensure it reports no violations."""
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        details: Iterable[str] = [
            f"command: {' '.join(command)}",
            f"exit code: {result.returncode}",
        ]
        if stdout:
            details.append("stdout:\n" + stdout)
        if stderr:
            details.append("stderr:\n" + stderr)
        pytest.fail("\n\n".join(details))
