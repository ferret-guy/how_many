"""Qt helper utilities."""

from __future__ import annotations

import numpy as np
from PySide6 import QtGui


def _rgba_to_bgr(rgba: np.ndarray) -> np.ndarray:
    """Return a contiguous BGR copy of an RGBA ``uint8`` image."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected an array shaped (H, W, 4) in RGBA order.")

    bgr = np.empty((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
    bgr[..., 0] = rgba[..., 2]
    bgr[..., 1] = rgba[..., 1]
    bgr[..., 2] = rgba[..., 0]
    return bgr


def qpixmap_to_bgr(pix: QtGui.QPixmap) -> np.ndarray:
    """Convert a :class:`~PySide6.QtGui.QPixmap` into a NumPy BGR array."""

    fmt = getattr(QtGui.QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QtGui.QImage.Format.Format_RGBA8888
    img: QtGui.QImage = pix.toImage().convertToFormat(fmt)
    width = img.width()
    height = img.height()
    bytes_per_line = img.bytesPerLine()
    buf = img.constBits()  # memoryview in PySide6
    if hasattr(buf, "setsize"):
        buf.setsize(height * bytes_per_line)
    arr = np.frombuffer(buf, np.uint8)
    arr = arr.reshape((height, bytes_per_line))  # include stride
    arr = arr[:, : width * 4]  # crop padding
    arr = arr.reshape((height, width, 4))
    return _rgba_to_bgr(arr)


__all__ = ["qpixmap_to_bgr"]
