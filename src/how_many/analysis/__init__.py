"""Core analysis routines used by the how_many overlay."""

from .core import estimate_counts_from_profile
from .peaks_detrend import find_peaks, detrend
from .screenshot import (
    estimate_counts_from_screenshot,
    extract_aligned_stripe,
    stripe_profile_from_screenshot,
)
from ..models import Suggestion

__all__ = [
    "estimate_counts_from_profile",
    "estimate_counts_from_screenshot",
    "extract_aligned_stripe",
    "stripe_profile_from_screenshot",
    "find_peaks",
    "detrend",
    "Suggestion",
]
