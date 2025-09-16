"""End-to-end test for the packaged how_many executable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

def test_pyinstaller_executable_handles_analyze(tmp_path: Path) -> None:
    """Build the PyInstaller bundle and ensure the analyze shortcut fires in self-test mode."""

    project_root = Path(__file__).resolve().parents[1]
    build_script = project_root / "scripts" / "build_exe.py"

    subprocess.run([sys.executable, str(build_script)], check=True, cwd=project_root)

    exe_name = "how_many.exe" if sys.platform.startswith("win") else "how_many"
    exe_path = project_root / "dist" / exe_name
    assert exe_path.exists(), f"Expected executable at {exe_path}" 

    marker_path = tmp_path / "selftest-marker.txt"

    env = os.environ.copy()
    env["HOW_MANY_SELFTEST"] = "1"
    env["HOW_MANY_SELFTEST_MARKER"] = str(marker_path)
    env.setdefault("QT_OPENGL", "software")
    if sys.platform.startswith("linux"):
        if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("XDG_RUNTIME_DIR", str(tmp_path))
    elif sys.platform == "darwin":
        if env.get("CI") or env.get("GITHUB_ACTIONS"):
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
    else:
        env.setdefault("QT_QPA_PLATFORM", env.get("QT_QPA_PLATFORM", "windows"))

    try:
        completed = subprocess.run(
            [str(exe_path)],
            cwd=project_root,
            env=env,
            check=False,
            timeout=30,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        pytest.fail(
            "Executable timed out during self-test."
            + (f"\nSTDOUT:\n{stdout}" if stdout else "")
            + (f"\nSTDERR:\n{stderr}" if stderr else "")
        )

    if completed.returncode != 0:
        pytest.fail(
            f"Executable exited with code {completed.returncode}."
            + (f"\nSTDOUT:\n{completed.stdout}" if completed.stdout else "")
            + (f"\nSTDERR:\n{completed.stderr}" if completed.stderr else "")
        )

    assert marker_path.exists(), "Self-test marker file was not created"
    contents = marker_path.read_text(encoding="utf-8")
    assert "analyze-called" in contents, contents
    assert "selftest-ok" in contents, contents
    assert "selftest-error" not in contents, contents
