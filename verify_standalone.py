import pytest
from pathlib import Path
from unittest.mock import patch
import shutil

"""
Verifies find_peaks and detrend  match the real scipy tests.

git clone https://github.com/scipy/scipy.git
move the cloned scipy to scipy_source (to avoid import issues)

pip show scipy <- find the current version
Check out the matching tag  This was verified v1.15.3
cd scipy_source
git checkout vX.Y.Z 

Verify
pytest -v -s verify_standalone.py
"""

SCIPY_REPO_PATH = Path("./scipy_source")

import peaks_standalone as standalone

# Define paths to the original test files in the source tree
signal_tests_path = SCIPY_REPO_PATH / 'scipy' / 'signal' / 'tests'
PEAK_TEST_FILE = signal_tests_path / 'test_peak_finding.py'
DETREND_TEST_FILE = signal_tests_path / 'test_signaltools.py'

@pytest.fixture(scope="session", autouse=True)
def check_paths():
    """Validates that the SciPy repository path and test files exist."""
    if not SCIPY_REPO_PATH.is_dir():
        pytest.fail(f"SciPy repo not found at: {SCIPY_REPO_PATH}\n"
                    "Please update SCIPY_REPO_PATH in this script.")
    if not PEAK_TEST_FILE.is_file():
        pytest.fail(f"Test file not found: {PEAK_TEST_FILE}")
    if not DETREND_TEST_FILE.is_file():
        pytest.fail(f"Test file not found: {DETREND_TEST_FILE}")


def run_tests_from_copy(source_test_file, test_class_name, tmp_path, test_id):
    """
    Copies a test file to a temporary directory with a unique name and runs pytest.
    The unique name (via test_id) is the key to avoiding Python's module caching issues.
    """
    # Create a unique name for the copied test to avoid module name clashes.
    dest_file = tmp_path / f"temp_{test_id}_{source_test_file.name}"
    shutil.copy(source_test_file, dest_file)

    test_target = f"{dest_file}::{test_class_name}"
    
    return pytest.main(['-v', test_target])


# --- Test Suite for find_peaks ---

def test_find_peaks_scipy_original(tmp_path):
    """Establishes a baseline by running tests against the installed SciPy."""
    print("\n\n--- Running tests for ORIGINAL scipy.signal.find_peaks ---\n")
    result = run_tests_from_copy(PEAK_TEST_FILE, "TestFindPeaks", tmp_path, "peaks_original")
    assert result == pytest.ExitCode.OK, "Original SciPy find_peaks failed its own tests!"

def test_find_peaks_standalone(tmp_path):
    """Runs the same tests, but with the SciPy function replaced by yours."""
    print("\n\n--- Running tests for STANDALONE find_peaks (monkey-patched) ---\n")
    with patch('scipy.signal.find_peaks', new=standalone.find_peaks):
        result = run_tests_from_copy(PEAK_TEST_FILE, "TestFindPeaks", tmp_path, "peaks_standalone")
    assert result == pytest.ExitCode.OK, "Standalone find_peaks failed SciPy's tests!"


# --- Test Suite for detrend ---

def test_detrend_scipy_original(tmp_path):
    """Establishes a baseline by running tests against the installed SciPy."""
    print("\n\n--- Running tests for ORIGINAL scipy.signal.detrend ---\n")
    result = run_tests_from_copy(DETREND_TEST_FILE, "TestDetrend", tmp_path, "detrend_original")
    assert result == pytest.ExitCode.OK, "Original SciPy detrend failed its own tests!"

def test_detrend_standalone(tmp_path):
    """Runs the same tests, but with the SciPy function replaced by yours."""
    print("\n\n--- Running tests for STANDALONE detrend (monkey-patched) ---\n")
    with patch('scipy.signal.detrend', new=standalone.detrend):
        result = run_tests_from_copy(DETREND_TEST_FILE, "TestDetrend", tmp_path, "detrend_standalone")
    assert result == pytest.ExitCode.OK, "Standalone detrend failed SciPy's tests!"