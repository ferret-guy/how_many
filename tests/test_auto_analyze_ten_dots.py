"""Regression tests for auto-analysis using the screenshot helpers."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from how_many.analysis import estimate_counts_from_screenshot


STRIPE_WIDTH_PX = 28
SPACING_PX = 20.0
DOT_RADIUS_PX = 5
MARGIN_PX = 48


def _synthetic_dot_screenshot(
    num_dots: int, angle_deg: float
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Generate a BGR screenshot with ``num_dots`` evenly spaced along ``angle_deg``."""

    assert num_dots >= 2

    angle_rad = math.radians(angle_deg)
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)

    offsets = (np.arange(num_dots) - (num_dots - 1) / 2.0) * SPACING_PX
    xs = offsets * cos_t
    ys = offsets * sin_t

    min_x = float(xs.min() - DOT_RADIUS_PX - MARGIN_PX)
    max_x = float(xs.max() + DOT_RADIUS_PX + MARGIN_PX)
    min_y = float(ys.min() - DOT_RADIUS_PX - MARGIN_PX)
    max_y = float(ys.max() + DOT_RADIUS_PX + MARGIN_PX)

    width = int(math.ceil(max_x - min_x)) + 1
    height = int(math.ceil(max_y - min_y)) + 1

    pad = int(2 * (DOT_RADIUS_PX + MARGIN_PX) + STRIPE_WIDTH_PX)
    width = max(width, pad)
    height = max(height, pad)

    canvas = np.full((height, width), 18.0, dtype=np.float64)

    centers = []
    sigma = DOT_RADIUS_PX / 1.5
    radius = int(math.ceil(3 * sigma))
    for x, y in zip(xs, ys):
        cx = float(x - min_x)
        cy = float(y - min_y)
        px = int(round(cx))
        py = int(round(cy))
        centers.append((float(px), float(py)))

        x0 = max(0, px - radius)
        x1 = min(width, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(height, py + radius + 1)

        patch_x = np.arange(x0, x1, dtype=np.float64)
        patch_y = np.arange(y0, y1, dtype=np.float64)
        gx, gy = np.meshgrid(patch_x, patch_y)
        dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
        canvas[y0:y1, x0:x1] += np.exp(-dist2 / (2.0 * sigma * sigma)) * 220.0

    canvas = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    screenshot = np.repeat(canvas[:, :, None], 3, axis=2)

    p1 = centers[0]
    p2 = centers[-1]
    return screenshot, p1, p2


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0, 135.0])
def test_estimate_counts_snap_angles(angle_deg: float) -> None:
    """The UI snap angles should yield confident counts for ten dots."""

    screenshot, p1, p2 = _synthetic_dot_screenshot(num_dots=10, angle_deg=angle_deg)
    profile, suggestions = estimate_counts_from_screenshot(
        screenshot,
        p1,
        p2,
        STRIPE_WIDTH_PX,
        max_candidates=5,
    )

    assert profile.size >= 16
    assert suggestions, "Auto-analysis returned no candidates for snap angle test."

    best = suggestions[0]
    assert best.count == 10
    assert best.confidence > 0.75


angles = st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False)
dot_counts = st.integers(min_value=2, max_value=100)


@given(angle_deg=angles, num_dots=dot_counts)
@settings(deadline=None, max_examples=45)
def test_estimate_counts_random_angle(angle_deg: float, num_dots: int) -> None:
    """Any angle between 0° and 180° should recover the dot count."""

    screenshot, p1, p2 = _synthetic_dot_screenshot(num_dots=num_dots, angle_deg=angle_deg)
    _, suggestions = estimate_counts_from_screenshot(
        screenshot,
        p1,
        p2,
        STRIPE_WIDTH_PX,
        max_candidates=6,
    )

    assert suggestions, "Expected at least one candidate for generated dot pattern."

    best = suggestions[0]
    assert best.count == num_dots
    assert best.confidence > 0.6
