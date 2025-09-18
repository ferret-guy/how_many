"""Formatting validation tests.

These tests ensure that the codebase complies with the configured
formatting tools. They provide fast feedback when formatting drifts from
the expected style defined in ``pyproject.toml``.
"""

from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Execute *command* from the repository root.

    Args:
        command: The command line to execute.

    Returns:
        The :class:`~subprocess.CompletedProcess` describing the completed
        command.
    """

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _format_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    """Generate a helpful error message for a failed formatting command."""

    details: Iterable[str] = (
        "Formatting command failed.",
        f"Command: {' '.join(result.args)}",
        "--- stdout ---",
        result.stdout.strip(),
        "--- stderr ---",
        result.stderr.strip(),
    )
    return "\n".join(filter(None, details))


@pytest.mark.parametrize(
    ("description", "command"),
    (
        ("black", ["python", "-m", "black", "--check", "src", "tests", "scripts"]),
        ("isort", ["python", "-m", "isort", "--check-only", "src", "tests", "scripts"]),
    ),
)
def test_formatting_commands(description: str, command: Sequence[str]) -> None:
    """Ensure that the configured formatting commands succeed."""

    result = _run_command(command)
    if result.returncode != 0:
        pytest.fail(_format_failure_message(result))
