"""Geometry helpers used throughout the UI and analysis layers."""

import math
from typing import Tuple


def rotate_point(
    x: float, y: float, cx: float, cy: float, angle_deg: float
) -> Tuple[float, float]:
    """Rotate a point around ``(cx, cy)`` by ``angle_deg`` degrees."""
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x0 = x - cx
    y0 = y - cy
    xr = x0 * cos_t - y0 * sin_t + cx
    yr = x0 * sin_t + y0 * cos_t + cy
    return xr, yr


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the inclusive range ``[lo, hi]``."""
    return max(lo, min(hi, value))


__all__ = ["rotate_point", "clamp"]
