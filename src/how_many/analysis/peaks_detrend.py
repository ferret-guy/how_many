"""Standalone SciPy signal helpers used by how_many.

This module vendors the pure-Python translations of
``scipy.signal.find_peaks`` and ``scipy.signal.detrend`` from SciPy
version 1.15.3.  The original implementations are
``Copyright (c) 2001-2024 SciPy Developers`` and made available under the
SciPy BSD 3-Clause licence.  These copies remain covered by that licence
and are regression-tested against the matching SciPy release to confirm
behavioural parity within the how_many project.
"""

import math
from typing import Literal
import warnings

import numpy as np

SCIPY_REFERENCE_VERSION = "1.15.3"


class PeakPropertyWarning(RuntimeWarning):
    """Warn about inconsistent peak property values."""

    pass


def _local_maxima_1d(x):
    """Find local maxima in a 1D array.

    This is a pure Python implementation of the logic from the Cython original.
    """
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0
    i = 1
    i_max = x.shape[0] - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                i = i_ahead
        i += 1
    return midpoints[:m], left_edges[:m], right_edges[:m]


def _select_by_peak_distance(peaks, priority, distance):
    """Select peaks that respect the minimum distance constraint.

    This is a pure Python implementation of the logic from the Cython original.
    """
    peaks_size = peaks.shape[0]
    distance_ = np.ceil(distance)
    keep = np.ones(peaks_size, dtype=bool)
    priority_to_position = np.argsort(priority)

    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if not keep[j]:
            continue

        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance_:
            keep[k] = False
            k -= 1

        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = False
            k += 1
    return keep


def _peak_prominences(x, peaks, wlen):
    """Compute peak prominences for a sampled signal.

    This is a pure Python implementation of the logic from the Cython original.
    """
    show_warning = False
    prominences = np.empty(peaks.shape[0], dtype=np.float64)
    left_bases = np.empty(peaks.shape[0], dtype=np.intp)
    right_bases = np.empty(peaks.shape[0], dtype=np.intp)

    for peak_nr, peak in enumerate(peaks):
        i_min, i_max = 0, x.shape[0] - 1
        if not i_min <= peak <= i_max:
            raise ValueError(f"peak {peak} is not a valid index for `x`")

        if wlen >= 2:
            i_min = max(peak - wlen // 2, i_min)
            i_max = min(peak + wlen // 2, i_max)

        # Find left base
        i = peak
        left_bases[peak_nr] = peak
        left_min = x[peak]
        while i >= i_min and x[i] <= x[peak]:
            if x[i] < left_min:
                left_min = x[i]
                left_bases[peak_nr] = i
            i -= 1

        # Find right base
        i = peak
        right_bases[peak_nr] = peak
        right_min = x[peak]
        while i <= i_max and x[i] <= x[peak]:
            if x[i] < right_min:
                right_min = x[i]
                right_bases[peak_nr] = i
            i += 1

        prominences[peak_nr] = x[peak] - max(left_min, right_min)
        if prominences[peak_nr] == 0:
            show_warning = True

    if show_warning:
        warnings.warn("some peaks have a prominence of 0", PeakPropertyWarning)

    return prominences, left_bases, right_bases


def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    """Measure the widths of peaks at the requested relative height.

    This is a pure Python implementation of the logic from the Cython original.
    """
    if rel_height < 0:
        raise ValueError("`rel_height` must be greater or equal to 0.0")

    widths = np.empty(peaks.shape[0], dtype=np.float64)
    width_heights = np.empty(peaks.shape[0], dtype=np.float64)
    left_ips = np.empty(peaks.shape[0], dtype=np.float64)
    right_ips = np.empty(peaks.shape[0], dtype=np.float64)
    show_warning = False

    for p, peak in enumerate(peaks):
        i_min, i_max = left_bases[p], right_bases[p]
        if not 0 <= i_min <= peak <= i_max < x.shape[0]:
            raise ValueError(f"prominence data is invalid for peak {peak}")

        height = x[peak] - prominences[p] * rel_height
        width_heights[p] = height

        # Find intersection point on left side
        i = peak
        while i_min < i and height < x[i]:
            i -= 1
        left_ip = float(i)
        if x[i] < height:
            left_ip += (height - x[i]) / (x[i + 1] - x[i])

        # Find intersection point on right side
        i = peak
        while i < i_max and height < x[i]:
            i += 1
        right_ip = float(i)
        if x[i] < height:
            right_ip -= (height - x[i]) / (x[i - 1] - x[i])

        widths[p] = right_ip - left_ip
        if widths[p] == 0:
            show_warning = True
        left_ips[p] = left_ip
        right_ips[p] = right_ip

    if show_warning:
        warnings.warn("some peaks have a width of 0", PeakPropertyWarning)

    return widths, width_heights, left_ips, right_ips


def _arg_x_as_expected(value):
    """Validate the input signal as a 1-D float array."""
    value = np.asarray(value, order="C", dtype=np.float64)
    if value.ndim != 1:
        raise ValueError("`x` must be a 1-D array")
    return value


def _arg_wlen_as_expected(value):
    """Convert ``wlen`` to the SciPy-compatible sentinel form."""
    if value is None:
        value = -1
    elif 1 < value:
        value = (
            np.intp(math.ceil(value)) if isinstance(value, float) else np.intp(value)
        )
    else:
        raise ValueError(f"`wlen` must be larger than 1, was {value}")
    return value


def _unpack_condition_args(interval, x, peaks):
    """Normalise condition arguments to lower and upper bounds."""
    try:
        imin, imax = interval
    except (TypeError, ValueError):
        imin, imax = (interval, None)

    if isinstance(imin, np.ndarray):
        if imin.size != x.size:
            raise ValueError("lower border must match x")
        imin = imin[peaks]
    if isinstance(imax, np.ndarray):
        if imax.size != x.size:
            raise ValueError("upper border must match x")
        imax = imax[peaks]
    return imin, imax


def _select_by_property(peak_properties, pmin, pmax):
    """Filter peaks based on lower and upper bound constraints."""
    keep = np.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= pmin <= peak_properties
    if pmax is not None:
        keep &= peak_properties <= pmax
    return keep


def _select_by_peak_threshold(x, peaks, tmin, tmax):
    """Evaluate peak threshold conditions for left and right neighbours."""
    if peaks.size == 0:
        return np.array([], dtype=bool), np.array([]), np.array([])
    if np.min(peaks) == 0 or np.max(peaks) >= len(x) - 1:
        raise ValueError(
            "Threshold condition does not support peaks at signal boundary."
        )

    thresholds = np.vstack([x[peaks] - x[peaks - 1], x[peaks] - x[peaks + 1]])
    keep = np.ones(peaks.size, dtype=bool)
    if tmin is not None:
        keep &= tmin <= np.min(thresholds, axis=0)
    if tmax is not None:
        keep &= np.max(thresholds, axis=0) <= tmax

    return keep, thresholds[0], thresholds[1]


def find_peaks(
    x,
    height=None,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
):
    """Find peaks inside a signal based on peak properties.

    This is a standalone version of ``scipy.signal.find_peaks``.
    """
    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError("`distance` must be greater or equal to 1")

    peaks, left_edges, right_edges = _local_maxima_1d(x)
    properties = {}

    # The following checks are applied in order to reduce the number of peaks
    # to process in the more expensive calculations that follow.

    if plateau_size is not None:
        plateau_sizes = right_edges - left_edges + 1
        pmin, pmax = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        (
            peaks,
            properties["plateau_sizes"],
            properties["left_edges"],
            properties["right_edges"],
        ) = (
            peaks[keep],
            plateau_sizes[keep],
            left_edges[keep],
            right_edges[keep],
        )

    if height is not None:
        peak_heights = x[peaks]
        hmin, hmax = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks, properties["peak_heights"] = peaks[keep], peak_heights[keep]

    if threshold is not None:
        tmin, tmax = _unpack_condition_args(threshold, x, peaks)
        keep, left_thresholds, right_thresholds = _select_by_peak_threshold(
            x, peaks, tmin, tmax
        )
        peaks = peaks[keep]
        properties["left_thresholds"] = left_thresholds[keep]
        properties["right_thresholds"] = right_thresholds[keep]

    if distance is not None:
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]

    # For the remaining peaks, calculate the more expensive properties
    if prominence is not None or width is not None:
        wlen = _arg_wlen_as_expected(wlen)
        prominences, left_bases, right_bases = _peak_prominences(x, peaks, wlen=wlen)
        properties.update(
            {
                "prominences": prominences,
                "left_bases": left_bases,
                "right_bases": right_bases,
            }
        )

    if prominence is not None:
        pmin, pmax = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties["prominences"], pmin, pmax)
        peaks = peaks[keep]
        for key in ("prominences", "left_bases", "right_bases"):
            properties[key] = properties[key][keep]

    if width is not None:
        widths, width_heights, left_ips, right_ips = _peak_widths(
            x,
            peaks,
            rel_height,
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        )
        properties.update(
            {
                "widths": widths,
                "width_heights": width_heights,
                "left_ips": left_ips,
                "right_ips": right_ips,
            }
        )
        wmin, wmax = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties["widths"], wmin, wmax)
        peaks = peaks[keep]
        for key in (
            "prominences",
            "left_bases",
            "right_bases",
            "widths",
            "width_heights",
            "left_ips",
            "right_ips",
        ):
            if key in properties:
                properties[key] = properties[key][keep]

    # Prune properties dictionary from entries not kept
    for key in list(properties.keys()):
        if properties[key].shape[0] != peaks.shape[0]:
            properties[key] = properties[key][keep]

    return peaks, properties


# #############################################################################
# Standalone `detrend` function
# #############################################################################


def detrend(
    data,
    axis: int = -1,
    type: Literal["linear", "constant"] = "linear",
    bp=0,
    overwrite_data: bool = False,
) -> np.ndarray:
    """Remove linear or constant trends from the provided data.

    This is a standalone version of ``scipy.signal.detrend`` that uses NumPy.
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")

    data = np.asarray(data)
    dtype = data.dtype
    if dtype.kind not in "fc":  # 'f' for float, 'c' for complex
        dtype = np.float64

    if type in ["constant", "c"]:
        return data - np.mean(data, axis, keepdims=True)

    # --- Linear detrending ---
    dshape = data.shape
    N = dshape[axis]

    # Correctly form breakpoints
    bp = np.sort(np.unique(np.concatenate(([0], np.atleast_1d(bp), [N]))))
    if np.any(bp > N):
        raise ValueError(
            "Breakpoints must be less than length of data along given axis."
        )

    # Reshape data to be 2D
    if axis < 0:
        axis += data.ndim
    newdata = np.moveaxis(data, axis, 0)
    newdata_shape = newdata.shape
    newdata = newdata.reshape(N, -1)

    if not overwrite_data:
        newdata = newdata.copy()
    newdata = newdata.astype(dtype, copy=False)

    # Perform least-squares fit for each segment
    for m in range(len(bp) - 1):
        Npts = bp[m + 1] - bp[m]
        if Npts == 0:
            continue

        sl = slice(bp[m], bp[m + 1])
        # Create design matrix for linear fit
        A = np.vstack(
            [
                np.arange(1, Npts + 1, dtype=dtype) / Npts,
                np.ones(Npts, dtype=dtype),
            ]
        ).T

        # Fit and subtract
        coef = np.linalg.lstsq(A, newdata[sl], rcond=None)[0]
        newdata[sl] -= np.dot(A, coef)

    # Reshape back to original
    newdata = newdata.reshape(newdata_shape)
    return np.moveaxis(newdata, 0, axis)
