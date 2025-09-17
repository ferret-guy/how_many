"""Qt helper utilities."""

from PySide6 import QtGui
import numpy as np


def qpixmap_to_bgr(pix: QtGui.QPixmap) -> np.ndarray:
    """Convert a :class:`~PySide6.QtGui.QPixmap` into a BGR NumPy array."""
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
    rgba = arr.astype(np.uint8, copy=False)
    rgb = rgba[..., :3]
    bgr = rgb[..., ::-1]
    return np.ascontiguousarray(bgr)


__all__ = ["qpixmap_to_bgr"]
