"""End-to-end test for the packaged how_many executable."""

import ctypes
import ctypes.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pytest


def _library_available(candidates: Iterable[str]) -> bool:
    """Return True if any candidate library can be loaded via dlopen."""
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate)
            return True
        except OSError:
            resolved = ctypes.util.find_library(candidate)
            if resolved and resolved != candidate:
                try:
                    ctypes.CDLL(resolved)
                    return True
                except OSError:
                    continue
    return False


def _missing_linux_packages() -> list[str]:
    """Identify required system packages that are absent on Linux."""
    required = {
        "libegl1": ("EGL", "libEGL.so.1"),
        "libgl1": ("GL", "libGL.so.1"),
        "libxkbcommon-x11-0": ("xkbcommon-x11", "libxkbcommon-x11.so.0"),
    }
    missing: list[str] = []

    for package, sonames in required.items():
        if not _library_available(sonames):
            missing.append(package)

    return missing


def test_pyinstaller_executable_handles_analyze(tmp_path: Path) -> None:
    """Ensure the packaged app triggers analyze during its self-test run."""
    project_root = Path(__file__).resolve().parents[1]
    build_script = project_root / "scripts" / "build_exe.py"

    if sys.platform.startswith("linux"):
        missing_packages = _missing_linux_packages()
        if missing_packages:
            install_cmd = (
                "sudo apt-get update && sudo apt-get install "
                "libegl1 libgl1 libxkbcommon-x11-0"
            )
            pytest.fail(
                "Missing system packages required for Qt: "
                + ", ".join(sorted(missing_packages))
                + f". Install them with `{install_cmd}`."
            )

    subprocess.run([sys.executable, str(build_script)], check=True, cwd=project_root)

    dist_dir = project_root / "dist"
    if sys.platform.startswith("win"):
        candidates = sorted(
            (p for p in dist_dir.glob("how_many*.exe") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    else:
        candidates = sorted(
            (p for p in dist_dir.glob("how_many*") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    assert candidates, f"No built executable found in {dist_dir}"
    exe_path = candidates[0]

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
        env.setdefault(
            "QT_QPA_PLATFORM",
            env.get("QT_QPA_PLATFORM", "windows"),
        )

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
