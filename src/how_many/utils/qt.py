"""Qt helper utilities."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6 import QtGui


def qpixmap_to_bgr(pix: QtGui.QPixmap) -> np.ndarray:
    """Convert a :class:`~PySide6.QtGui.QPixmap` into an OpenCV BGR array."""

    fmt = getattr(QtGui.QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QtGui.QImage.Format.Format_RGBA8888
    img: QtGui.QImage = pix.toImage().convertToFormat(fmt)
    width = img.width()
    height = img.height()
    bytes_per_line = img.bytesPerLine()
    buf = img.constBits()  # memoryview in PySide6
    arr = np.frombuffer(buf, np.uint8)
    arr = arr.reshape((height, bytes_per_line))  # include stride
    arr = arr[:, : width * 4]  # crop padding
    arr = arr.reshape((height, width, 4))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bgr


__all__ = ["qpixmap_to_bgr"]
