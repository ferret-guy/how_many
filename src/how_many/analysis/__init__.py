"""Core analysis routines used by the how_many overlay."""

from ..models import Suggestion
from .core import estimate_counts_from_profile
from .peaks_detrend import detrend, find_peaks
from .screenshot import (
    estimate_counts_from_screenshot,
    extract_aligned_stripe,
    stripe_profile_from_screenshot,
)

__all__ = [
    "estimate_counts_from_profile",
    "estimate_counts_from_screenshot",
    "extract_aligned_stripe",
    "stripe_profile_from_screenshot",
    "find_peaks",
    "detrend",
    "Suggestion",
]
