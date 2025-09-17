from __future__ import annotations

import sys
from typing import Sequence, Tuple


_TocEntry = Tuple[str, str, str]

_DROP_TOKENS = {
    "Qt6Pdf",
    "Qt6Qml",
    "Qt6Quick",
    "Qt6Svg",
    "Qt6VirtualKeyboard",
    "Qt6Wayland",
    "Qt6WlShell",
    "Qt6Egl",
    "Qt6EglFs",
    "Qt6Network",
    "Qt6OpenGL",
    "opengl32sw.dll",
    "libqwayland",
    "libqeglfs",
    "libqvnc",
    "libqminimal",
    "libqminimalegl",
    "libqvkkhrdisplay",
    "libqxcb-egl-integration",
    "libqxcb-glx-integration",
    "libqlinuxfb",
    "libqgtk",
    "libqtuiotouchplugin",
    "libqevdev",
    "libqxdgdesktopportal",
}

_DROP_PREFIXES = (
    "PySide6/Qt/plugins/wayland-",
    "PySide6/Qt/plugins/egldeviceintegrations/",
    "PySide6/Qt/plugins/networkinformation/",
    "PySide6/Qt/plugins/platforminputcontexts/libqtvirtualkeyboardplugin",
    "PySide6/Qt/plugins/platformthemes/",
    "PySide6/Qt/plugins/iconengines/",
    "PySide6/Qt/plugins/imageformats/",
    "PySide6/Qt/plugins/generic/",
    "PySide6/Qt/translations",
)


def _should_keep(name: str) -> bool:
    lowered = name.lower()
    for prefix in _DROP_PREFIXES:
        if lowered.startswith(prefix.lower()):
            return False
    for token in _DROP_TOKENS:
        if token.lower() in lowered:
            return False
    return True


def filter_toc(items: Sequence[_TocEntry]) -> Sequence[_TocEntry]:
    kept = [entry for entry in items if _should_keep(entry[0])]
    # avoid noisy output; call sites can inspect sizes if desired
    try:
        return items.__class__(kept)
    except Exception:
        return kept
