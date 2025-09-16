import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from how_many.analysis import screenshot


def _expected_stripe_len(p1: tuple[float, float], p2: tuple[float, float]) -> int:
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return max(4, int(math.ceil(length)))


def test_extract_aligned_stripe_from_grayscale_returns_bgr() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    p1 = (0.0, 2.0)
    p2 = (4.0, 2.0)

    stripe = screenshot.extract_aligned_stripe(image, p1, p2, 3)

    assert stripe.shape == (3, _expected_stripe_len(p1, p2), 3)
    assert stripe.dtype == np.uint8
    assert np.all(stripe[..., 0] == stripe[..., 1])
    assert np.all(stripe[..., 1] == stripe[..., 2])


def test_extract_aligned_stripe_trims_alpha_channel() -> None:
    base = np.zeros((6, 6, 4), dtype=np.uint8)
    base[..., 0] = 12
    base[..., 1] = 34
    base[..., 2] = 56
    base[..., 3] = 200

    p1 = (1.0, 1.0)
    p2 = (5.0, 1.0)
    stripe = screenshot.extract_aligned_stripe(base, p1, p2, 2)

    assert stripe.shape == (2, _expected_stripe_len(p1, p2), 3)
    assert np.all(stripe[..., 0] == 12)
    assert np.all(stripe[..., 1] == 34)
    assert np.all(stripe[..., 2] == 56)


def test_stripe_profile_matches_manual_pipeline() -> None:
    base = np.zeros((12, 12, 3), dtype=np.uint8)
    base[..., 0] = np.tile(np.arange(12, dtype=np.uint8), (12, 1))
    p1 = (2.0, 6.0)
    p2 = (9.5, 6.0)

    stripe = screenshot.extract_aligned_stripe(base, p1, p2, 4)
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float64)
    manual_gray = np.tensordot(stripe.astype(np.float64, copy=False), weights, axes=([-1], [0]))
    manual_profile = manual_gray.mean(axis=0)

    profile = screenshot.stripe_profile_from_screenshot(base, p1, p2, 4, blur_sigma_px=0.0)
    assert_allclose(profile, manual_profile, atol=1e-6)


def test_gaussian_blur_numpy_smooths_values() -> None:
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(8, 6))
    sigma = 1.35

    blurred = screenshot._gaussian_blur_numpy(data, sigma)  # type: ignore[attr-defined]
    assert blurred.shape == data.shape
    assert np.var(blurred) < np.var(data)
    assert abs(float(blurred.mean() - data.mean())) < 0.1
    assert float(blurred.min()) >= float(data.min())
    assert float(blurred.max()) <= float(data.max())


def test_extract_aligned_stripe_rejects_short_line() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        screenshot.extract_aligned_stripe(img, (1.0, 1.0), (1.5, 1.5), 3)
