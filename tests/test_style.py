"""Ensure the repository meets the configured style and type expectations."""

import pathlib
import subprocess
import sys
from typing import Iterable, Tuple

import pytest
import tomllib

Command = Tuple[str, ...]
StyleCheck = Tuple[str, Command]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PATHS: Tuple[str, ...] = ("src", "tests", "scripts")
FLAKE8_OPTION_KEYS: Tuple[Tuple[str, str], ...] = (
    ("max-line-length", "--max-line-length"),
    ("max-complexity", "--max-complexity"),
    ("select", "--select"),
    ("ignore", "--ignore"),
    ("per-file-ignores", "--per-file-ignores"),
    ("exclude", "--exclude"),
)


def _python_module(module: str, *args: str) -> Command:
    return (sys.executable, "-m", module, *args)


MYPY_TARGETS: Tuple[str, ...] = ("src/how_many/utils",)


def _flake8_cli_args() -> Tuple[str, ...]:
    config_path = REPO_ROOT / "pyproject.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    tool_section = config.get("tool", {})
    flake8_section = tool_section.get("flake8", {})
    args: list[str] = []
    for key, option in FLAKE8_OPTION_KEYS:
        value = flake8_section.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            serialized = ",".join(str(item) for item in value)
        else:
            serialized = str(value)
        args.append(f"{option}={serialized}")
    return tuple(args)


FLAKE8_ARGS = _flake8_cli_args()

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
        _python_module("flake8", *FLAKE8_ARGS, *PATHS),
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
