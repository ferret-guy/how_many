"""Spectral and autocorrelation based counting helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..models import Suggestion
from ..utils import clamp
from .peaks_detrend import find_peaks, detrend


def _rank_and_cap(
    suggestions: Dict[int, Suggestion], profile_len: int, max_candidates: int
) -> List[Suggestion]:
    ranked = sorted(
        suggestions.values(),
        key=lambda s: (s.confidence, s.source.startswith("fft")),
        reverse=True,
    )
    filtered: List[Suggestion] = [
        s for s in ranked if 1 <= s.count <= max(1, profile_len)
    ]  # items can be up to length
    return filtered[: max(1, int(max_candidates))]


def estimate_counts_from_profile(
    profile: np.ndarray, max_candidates: int = 5
) -> List[Suggestion]:
    """Estimate the number of items (endpoints included) along the stripe profile."""

    N = int(profile.size)
    if N < 16:
        return []

    # Detrend and normalize
    p = detrend(profile.astype(np.float64))
    m = float(np.max(np.abs(p)))
    if m > 1e-9:
        p /= m

    # Windowing to reduce spectral leakage
    win = np.hanning(N)
    p_win = p * win

    suggestions: Dict[int, Suggestion] = {}

    def add_items(cycles: int, conf: float, source: str) -> None:
        # Convert cycles (intervals) -> items (endpoints included)
        items = int(max(1, cycles + 1))
        prev = suggestions.get(items)
        if (prev is None) or (conf > prev.confidence):
            suggestions[items] = Suggestion(
                count=items, confidence=float(clamp(conf, 0.0, 1.0)), source=source
            )

    # ---------------- FFT ----------------
    fft_mag = np.abs(np.fft.rfft(p_win))
    fft_mag[0] = 0.0  # ignore DC

    spectral = fft_mag.copy()
    spectral[0] = 0.0  # ignore DC but keep the first harmonic for short stripes

    spec_max = float(np.max(spectral)) if np.max(spectral) > 0 else 0.0
    if spec_max > 0:
        prominence = float(spec_max * 0.15)
        peaks, _ = find_peaks(spectral, prominence=prominence)

        if peaks.size > 0:
            for k in peaks:
                cycles = int(k)  # DFT bin index ~ cycles across the sampled window
                conf = float(spectral[k] / spec_max)
                add_items(cycles, conf, "fft-peak")

            # Consider harmonics
            top_idxs = sorted(
                peaks.tolist(), key=lambda i: float(spectral[i]), reverse=True
            )[:3]
            for i in top_idxs:
                base = int(i)
                base_conf = float(spectral[i] / spec_max)
                add_items(base, base_conf, "fft-top")
                if 2 * base < len(spectral):
                    add_items(2 * base, base_conf * 0.6, "harmonic-x2")
                if base % 2 == 0 and base > 0:
                    add_items(base // 2, base_conf * 0.6, "harmonic-x0.5")

    # ---------------- ACF ----------------
    p0 = p - float(np.mean(p))
    acf_full = np.correlate(p0, p0, mode="full")
    acf = acf_full[N - 1 :]  # lags 0..N-1
    if acf[0] != 0:
        acf = acf / float(acf[0])
    acf[:2] = 0.0  # ignore lag 0/1

    if N >= 32 and np.max(acf) > 0:
        acf_peaks, _ = find_peaks(acf, prominence=float(np.max(acf) * 0.1))
        if acf_peaks.size > 0:
            acf_max = float(np.max(acf))
            for lag in acf_peaks[:5]:
                if lag > 0:
                    cycles = int(
                        round((N - 1) / float(lag))
                    )  # use N-1 (interval length) for cycles
                    conf = float(acf[lag] / acf_max)
                    add_items(cycles, conf, "acf-peak")

    return _rank_and_cap(suggestions, N, max_candidates)


__all__ = ["estimate_counts_from_profile"]
