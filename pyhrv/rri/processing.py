"""
This module contains algorithms that process RR-interval time series and
produce a processed RR-interval time series.
"""
import math

import numpy as np
import scipy.signal as sps

from pyhrv.conf import get_val as v


def filtrr(t, rr,
           enable_range=v('filtrr.range.enable'),
           enable_moving_average=v('filtrr.moving_average.enable'),
           **kw):
    """
    Detects and removes outliers in RR interval data.
    Performs three types of different outlier detection: Range based
    detection, moving-average filter-based detection and quotient filter based
    detection.
    :param t: Time axis of RR intervals.
    :param rr: RR interval signal.
    :param enable_range: Whether to apply range filter.
    :param enable_moving_average: Whether to apply moving average filter.
    :param kw: Arguments for the different filters. See their respective
    functions: :meth:`filtrr_range`, :meth:`filtrr_moving_average`, and
    :meth:`filtrr_quotient`.
    :return: tuple of time axis and RR intervals after filtering.
    """
    idx = np.ones_like(rr).astype(np.bool)

    if enable_range:
        range_idx = _filtrr_range(rr, **kw)
        idx &= range_idx

    if enable_moving_average:
        ma_idx, rr_ma = _filtrr_ma(rr, **kw)
        idx &= ma_idx

    t_f, rr_f = t[idx], rr[idx]
    return t_f, rr_f


def _filtrr_range(rr,
                  rr_min=v('filtrr.range.rr_min'),
                  rr_max=v('filtrr.range.rr_max'), ):
    idx = ((rr >= rr_min) & (rr <= rr_max))
    return idx


def _filtrr_ma(rr,
               win_len=v('filtrr.moving_average.win_samples'),
               win_thresh=v('filtrr.moving_average.thresh_percent')):
    b_fir = np.r_[np.ones(win_len), 0., np.ones(win_len)].astype(np.float32)
    b_fir *= 1 / (2 * win_len)

    rr_ma = sps.filtfilt(b_fir, 1., rr)

    win_thresh /= 100
    idx = np.abs(rr - rr_ma) <= (win_thresh * rr_ma)

    return idx, rr_ma


def splitrr(t, rr, win_sec,
            rr_min=v('filtrr.range.rr_min'),
            rr_max=v('filtrr.range.rr_max'), ):
    """
    Split an RR-interval signal into windows of approximately equal duration.
    The segments will be zero-padded so that they all have the same length.
    :param t: Time axis of intervals.
    :param rr: Intervals.
    :param win_sec: Desired segment (window) duration.
    :param rr_min: minimal physiological RR-interval.
    :param rr_max: maximal physiological RR-interval.
    :return: A tensor of shape (N, L) where N is the number of segments and L
    is the maximal possible length of a segment, in intervals.
    """

    pad_len = math.ceil(win_sec / rr_min)
    sig_duration = t[-1]

    window_starts = (win_sec * i for i in range(2 ** 63 - 1))
    window_tensors = []

    for win_start in window_starts:
        win_end = win_start + win_sec
        if win_end > sig_duration:
            break

        rr_win = rr[(t >= win_start) & (t < win_end)]

        if len(rr_win) < (win_sec / rr_max):
            continue

        rr_win = np.pad(rr_win, (0, pad_len - len(rr_win)), 'constant',
                        constant_values=0)

        window_tensors.append(rr_win)

    return np.stack(window_tensors, axis=0)
