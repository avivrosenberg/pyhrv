import math
import numpy as np
import logging
import scipy.signal as sps
from typing import Callable

from pyhrv import utils

logger = logging.getLogger(__name__)


def pxx_lomb(
    rri: np.ndarray,
    f_axis: np.ndarray,
    trr: np.ndarray = None,
    t_win: float = None,
    win_func: Callable = sps.windows.hamming,
):
    """
    Lomb-Scargle periodogram of RR intervals.
    Can optionally split the data into equal-length windows and average
    the periodogram from each window.

    :param rri: RR intervals.
    :param f_axis: Frequencies, in Hz, to evaluate at.
    :param trr: RR intervals times.
    :param t_win: Duration in seconds of each window. If none, no windowing.
    :param win_func: Window function to apply to the data (e.g. Hamming).
    None to disable (use rectangular window).
    :return: Periodogram.
    """
    # Standardize input vectors
    rri, trr = utils.standardize_rri_trr(rri, trr)

    if not t_win:
        t_win = math.floor(trr[-1] - trr[0])

    if not win_func:
        win_func = sps.windows.boxcar

    # Convert frequency to normalized (angular freq)
    w_axis = f_axis * 2 * math.pi

    sig_duration = trr[-1]
    window_starts = (t_win * i for i in range(2 ** 63 - 1))

    pxx = np.zeros_like(f_axis)
    i = 0
    for i, win_start in enumerate(window_starts):
        win_end = win_start + t_win
        if win_end > sig_duration:
            break

        win_idx = (trr >= win_start) & (trr < win_end)
        rri_win = rri[win_idx]

        min_samples_nyq = math.ceil(2 * f_axis[-1] * t_win)
        if len(rri_win) < min_samples_nyq:
            logger.warning(
                f"Nyquist criterion not met for lomb periodogram "
                f"in window {i} "
                f"({len(rri_win)}/{min_samples_nyq} samples). "
            )

        # Apply window
        win_coeffs = win_func(len(rri_win))
        rri_win *= win_coeffs

        # Calculate periodogram
        pxx_win = sps.lombscargle(
            trr,
            rri,
            w_axis,
            precenter=True,
            normalize=False,
        )
        # Window gain correction
        pxx_win *= 1 / np.mean(win_coeffs)
        pxx += pxx_win

    pxx /= i
    return pxx


def build_uniform_freq_axis(
    t_win: float,
    f_min: float,
    f_max: float,
    resample_factor: float = 2,
    oversample_factor: float = 1,
):
    """
    Builds a frequency axis for an RR interval signal.
    Since RR intervals are by nature non-uniformly sampled, we assume that
    they will be resampled at least at twice the maximal frequency we wish
    to resolve prior to spectral analysis.

    Also, we handle the case where the signal will be split into windows of
    length t_win seconds prior to spectral analysis.

    :param t_win: Duration in seconds of the signal.
    :param f_min: Minimal frequency we wish to resolve.
    :param f_max: Maximal frequency we wish to resolve.
    :param resample_factor: Factor of f_max that the signal should be
    resampled at (at least 2 to maintain Nyquist criterion).
    :param oversample_factor: Factor to increase the frequency resolution of
    the created frequency axis.
    :return: (f_axis, fs_uni) a tuple of the frequency axis and the uniform
    resampling frequency.
    """

    t_win_min = 1 / f_min  # minimal window to resolve f_min
    t_win = max(t_win, t_win_min)

    # Uniform sampling freq: Take at least 2x more than f_max
    if resample_factor < 2:
        raise ValueError("resample_factor must be at least 2")
    fs_uni = resample_factor * f_max  # Hz

    # Num samples per window
    n_win_uni = math.floor(t_win / (1 / fs_uni))

    # Frequency axis
    ts = t_win / (n_win_uni - 1)  # sampling interval
    f_res = 1 / (n_win_uni * ts)  # freq resolution, aka f_min or delta_f
    f_res /= oversample_factor  # Oversampling factor -> interpolation in freq
    f_axis = np.r_[f_res:f_max:f_res]

    return f_axis, fs_uni
