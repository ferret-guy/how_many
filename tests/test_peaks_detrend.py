import importlib.resources as resources
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import scipy

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from how_many.analysis import peaks_detrend as standalone

EXPECTED_SCIPY_VERSION = standalone.SCIPY_REFERENCE_VERSION

SCIPY_TEST_ROOT = resources.files("scipy.signal.tests")
PEAK_TEST_RESOURCE = SCIPY_TEST_ROOT / "test_peak_finding.py"
DETREND_TEST_RESOURCE = SCIPY_TEST_ROOT / "test_signaltools.py"


@pytest.fixture(scope="session", autouse=True)
def ensure_expected_scipy_version():
    version = scipy.__version__
    if version != EXPECTED_SCIPY_VERSION:
        pytest.fail(
            "SciPy regression tests require the reference version. "
            f"Expected {EXPECTED_SCIPY_VERSION}, found {version}."
        )
    with (ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    configured_version = (
        config.get("tool", {})
        .get("how-many", {})
        .get("scipy-reference-version")
    )
    if configured_version != EXPECTED_SCIPY_VERSION:
        pytest.fail(
            "pyproject.toml and peaks_detrend.SCIPY_REFERENCE_VERSION disagree. "
            f"Configured {configured_version!r} versus constant {EXPECTED_SCIPY_VERSION!r}."
        )
    for resource in (PEAK_TEST_RESOURCE, DETREND_TEST_RESOURCE):
        if not resource.is_file():
            pytest.fail(f"SciPy test file not found: {resource}")


def run_tests_from_copy(resource, test_class_name, tmp_path, test_id):
    """
    Copies a SciPy test file to a temporary directory with a unique name and runs pytest.
    The unique name (via test_id) avoids Python module caching issues during repeated runs.
    """
    dest_file = tmp_path / f"temp_{test_id}_{resource.name}"
    with resources.as_file(resource) as source_path:
        shutil.copy(source_path, dest_file)

    test_target = f"{dest_file}::{test_class_name}"
    return pytest.main(["-v", test_target])


# --- Test Suite for find_peaks ---


def test_find_peaks_scipy_original(tmp_path):
    """Establishes a baseline by running tests against the installed SciPy."""
    print("\n\n--- Running tests for ORIGINAL scipy.signal.find_peaks ---\n")
    result = run_tests_from_copy(
        PEAK_TEST_RESOURCE,
        "TestFindPeaks",
        tmp_path,
        "peaks_original",
    )
    assert (
        result == pytest.ExitCode.OK
    ), "Original SciPy find_peaks failed its own tests!"


def test_find_peaks_standalone(tmp_path):
    """Runs the same tests, but with the SciPy function replaced by yours."""
    print("\n\n--- Running tests for STANDALONE find_peaks (monkey-patched) ---\n")
    with patch("scipy.signal.find_peaks", new=standalone.find_peaks):
        result = run_tests_from_copy(
            PEAK_TEST_RESOURCE,
            "TestFindPeaks",
            tmp_path,
            "peaks_standalone",
        )
    assert result == pytest.ExitCode.OK, "Standalone find_peaks failed SciPy's tests!"


# --- Test Suite for detrend ---


def test_detrend_scipy_original(tmp_path):
    """Establishes a baseline by running tests against the installed SciPy."""
    print("\n\n--- Running tests for ORIGINAL scipy.signal.detrend ---\n")
    result = run_tests_from_copy(
        DETREND_TEST_RESOURCE,
        "TestDetrend",
        tmp_path,
        "detrend_original",
    )
    assert result == pytest.ExitCode.OK, "Original SciPy detrend failed its own tests!"


def test_detrend_standalone(tmp_path):
    """Runs the same tests, but with the SciPy function replaced by yours."""
    print("\n\n--- Running tests for STANDALONE detrend (monkey-patched) ---\n")
    with patch("scipy.signal.detrend", new=standalone.detrend):
        result = run_tests_from_copy(
            DETREND_TEST_RESOURCE,
            "TestDetrend",
            tmp_path,
            "detrend_standalone",
        )
    assert result == pytest.ExitCode.OK, "Standalone detrend failed SciPy's tests!"
