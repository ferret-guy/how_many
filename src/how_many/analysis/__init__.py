"""Core analysis routines used by the how_many overlay."""

from .core import estimate_counts_from_profile
from .peaks_detrend import find_peaks, detrend
from ..models import Suggestion

__all__ = ["estimate_counts_from_profile", "find_peaks", "detrend", "Suggestion"]
