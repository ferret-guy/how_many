from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

from PyInstaller.utils.hooks import check_requirement
from PyInstaller.utils.hooks.qt import (
    ensure_single_qt_bindings_package,
    pyside6_library_info,
)

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from _qt_prune import filter_toc

ensure_single_qt_bindings_package("PySide6")

hiddenimports: list[str] = []
binaries: list[tuple[str, str, str]] = []
datas: list[tuple[str, str, str]] = []

if pyside6_library_info.version is not None:
    hiddenimports = ["shiboken6", "inspect"]
    if check_requirement("PySide6 >= 6.4.0"):
        hiddenimports.append("PySide6.support.deprecated")

    binaries = filter_toc(pyside6_library_info.collect_extra_binaries())

    datas = []

