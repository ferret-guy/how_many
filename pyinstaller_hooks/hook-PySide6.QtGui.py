from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies

HOOK_DIR = Path(__file__).resolve().parent
if str(HOOK_DIR) not in sys.path:
    sys.path.insert(0, str(HOOK_DIR))

from _qt_prune import filter_toc

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)

binaries = filter_toc(binaries)
datas = filter_toc(datas)
