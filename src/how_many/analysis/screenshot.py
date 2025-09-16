"""Helpers for deriving stripe profiles directly from screenshot imagery."""

from __future__ import annotations
from typing import List, Tuple

import math

import numpy as np

from ..models import Suggestion
from .core import estimate_counts_from_profile


Point = Tuple[float, float]


def _coerce_bgr(image: np.ndarray) -> np.ndarray:
    """Return a contiguous BGR image regardless of the input layout."""

    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            return np.ascontiguousarray(arr)
        if arr.shape[2] == 4:
            return np.ascontiguousarray(arr[..., :3])
    raise ValueError("Expected a grayscale or BGR/BGRA screenshot array.")


def _bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = x - x0
    wy = y - y0

    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]

    top = top_left * (1.0 - wx)[..., None] + top_right * wx[..., None]
    bottom = bottom_left * (1.0 - wx)[..., None] + bottom_right * wx[..., None]
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None]


def _extract_aligned_stripe_numpy(
    img: np.ndarray, p1: Point, p2: Point, stripe_w: int
) -> np.ndarray:
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 4.0:
        raise ValueError("Overlay line too short to extract a stripe.")

    stripe_len = max(4, int(math.ceil(length)))
    if stripe_len <= 1:
        step = 0.0
    else:
        step = length / float(stripe_len - 1)

    ux = dx / length
    uy = dy / length
    nx = -uy
    ny = ux

    positions = np.arange(stripe_len, dtype=np.float64) * step
    xs = x1 + ux * positions
    ys = y1 + uy * positions

    offsets = np.linspace(-(stripe_w - 1) / 2.0, (stripe_w - 1) / 2.0, stripe_w)
    grid_x = xs[None, :] + nx * offsets[:, None]
    grid_y = ys[None, :] + ny * offsets[:, None]

    stripe = _bilinear_sample(img.astype(np.float64, copy=False), grid_x, grid_y)
    return stripe


def _gaussian_blur_numpy(gray: np.ndarray, sigma: float) -> np.ndarray:
    base_radius = max(1, int(math.ceil(4.0 * sigma)))

    def convolve_axis(data: np.ndarray, axis: int) -> np.ndarray:
        length = data.shape[axis]
        radius = min(base_radius, max(0, length - 1))
        if radius == 0:
            return data

        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
        kernel /= float(np.sum(kernel))

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (radius, radius)
        padded = np.pad(data, pad_width, mode="reflect")
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis, padded
        )

    blurred = convolve_axis(gray, axis=0)
    blurred = convolve_axis(blurred, axis=1)
    return blurred


def extract_aligned_stripe(
    screenshot_bgr: np.ndarray,
    p1: Point,
    p2: Point,
    stripe_width_px: int,
) -> np.ndarray:
    """Sample a stripe aligned with ``p1``→``p2`` from a screenshot image.

    Parameters
    ----------
    screenshot_bgr:
        Screenshot pixels in BGR order.
    p1, p2:
        The two overlay handle locations expressed in screenshot coordinates.
    stripe_width_px:
        Desired stripe thickness in pixels.
    """

    img = _coerce_bgr(screenshot_bgr)
    original_dtype = img.dtype
    stripe_w = max(2, int(stripe_width_px))

    stripe = _extract_aligned_stripe_numpy(img, p1, p2, stripe_w)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        stripe = np.rint(stripe)
        stripe = np.clip(stripe, float(info.min), float(info.max))
        return stripe.astype(original_dtype)
    return stripe


def stripe_profile_from_screenshot(
    screenshot_bgr: np.ndarray,
    p1: Point,
    p2: Point,
    stripe_width_px: int,
    *,
    blur_sigma_px: float = 0.0,
) -> np.ndarray:
    """Return the averaged stripe profile underneath the overlay handles."""

    stripe = extract_aligned_stripe(screenshot_bgr, p1, p2, stripe_width_px)

    stripe_float = stripe.astype(np.float64, copy=False)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float64)
    gray = np.tensordot(stripe_float, weights, axes=([-1], [0]))

    if blur_sigma_px > 0.1:
        gray = _gaussian_blur_numpy(gray.astype(np.float64, copy=False), float(blur_sigma_px))

    profile = np.mean(gray, axis=0)
    return profile.astype(np.float64, copy=False)


def estimate_counts_from_screenshot(
    screenshot_bgr: np.ndarray,
    p1: Point,
    p2: Point,
    stripe_width_px: int,
    *,
    blur_sigma_px: float = 0.0,
    max_candidates: int = 5,
) -> Tuple[np.ndarray, List[Suggestion]]:
    """High-level helper mirroring the UI pipeline without Qt objects."""

    profile = stripe_profile_from_screenshot(
        screenshot_bgr,
        p1,
        p2,
        stripe_width_px,
        blur_sigma_px=blur_sigma_px,
    )
    suggestions = estimate_counts_from_profile(
        profile, max_candidates=max_candidates
    )
    return profile, suggestions


__all__ = [
    "extract_aligned_stripe",
    "stripe_profile_from_screenshot",
    "estimate_counts_from_screenshot",
]
