import logging
import math
from typing import Tuple, Union, Callable

import numpy as np
import scipy.interpolate
import scipy.signal
import pyhrv.rri.frequency as frequency

import pyhrv.conf
from pyhrv import utils

logger = logging.getLogger(__name__)


def v(k):
    return pyhrv.conf.get_val(f'hrv_freq.{k}')


def hrv_freq(rri: np.ndarray, trr: np.ndarray = None,
             methods: Tuple[str, ...] = v('methods'),
             norm_method: str = v('norm_method'),
             vlf_band: Tuple[float] = v('vlf_band'),
             lf_band: Tuple[float] = v('lf_band'),
             hf_band: Tuple[float] = v('hf_band'),
             extra_bands: Tuple[float] = v('extra_bands'),
             window_minutes: float = v('window_minutes'),
             win_func: Union[str, Callable] = v('win_func'),
             oversample_factor: float = v('osf'),
             resample_factor: float = v('resample_factor'),
             welch_overlap: float = v('welch_overlap'),
             ar_order: int = v('ar_order')):
    """
    NN interval spectrum and frequency-domain HRV metrics.
    This function estimates
    the PSD (power spectral density) of a given nn-interval sequence, and
    calculates the power in various frequency bands.

    :param rri: RR/NN intervals, in seconds.
    :param trr: specify the time interval vector. If it is not
    specified then it will be computed from the nni time series.

    :param methods: A cell array of strings containing names of methods to use
    to estimate the spectrum. Supported methods are:
       - ``lomb``: Lomb-scargle periodogram.
       - ``ar``: Yule-Walker autoregressive model. Data will be resampled.
          No windowing will be performed for this method.
       - ``welch``: Welch's method (overlapping windows).

     In all cases, a window will be used on the samples according to the
     ``win_func`` parameter.  Data will be resampled for all methods except
     ``lomb``.

    :param norm_method: A string, either ``total`` or ``lf_hf``. If ``total``,
    then the power in each band will be normalized by the total
    power in the entire frequency spectrum. If ``lf_hf``, then only
    for the LF and HF bands, the normalization will be performed
    by the (LF+HF) power. This is the standard method used in
    many papers to normalize these bands. In any case, VLF and
    user-defined custom bands are not affected by this parameter.

    :param vlf_band: 2-element vector of frequencies in Hz defining the VLF
    band.

    :param lf_band: 2-element vector of frequencies in Hz defining the LF band.

    :param hf_band: 2-element vector of frequencies in Hz defining the HF band.

    :param extra_bands: A cell array of frequency pairs, for example
     ``{[f_start,f_end], ...}``. Each pair defines a custom band for
     which the power and normalized power will be calculated.

    :param window_minutes: Split intervals into windows of this length,
    calcualte the spectrum in each window, and average them. A window
    funciton will also be applied to each window after breaking the intervals
    into windows. Set to None if you want to disable windowing.

    :param win_func: The window function to apply to each segment. Should be a
    function that accepts one parameter (length in samples) and returns a
    window of that length.

    :param oversample_factor: Frequency oversampling factor. Increases the resolution in
    the frequency domain by oversampling.

    :param resample_factor: Time-domain resampling factor. Must be greater
    than 2. The maximal frequency in we want to resolve (top of hf_band)
    will be multiplied by this factor to determine the frequency at which to
    resample the rr intervals before applying frequency analysis (except for
    the lomb method).

    :param ar_order: Order of the autoregressive model to use if ``ar`` method
    is specific.

    :param welch_overlap: Percentage of overlap between windows when using
    Welch's method.

    :returns:
    """

    # Validate methods
    supported_methods = {'lomb', 'ar', 'welch'}
    methods = {m.lower() for m in methods}
    if not methods or not all(m in supported_methods for m in methods):
        raise ValueError(f"Entries in methods must were {methods}, but they "
                         f"must each be one of {supported_methods}")

    # Validate norm method
    supported_norm_methods = {'total', 'lf_af'}
    norm_method = norm_method.lower()
    if norm_method not in supported_norm_methods:
        raise ValueError(f"Unsupported norm_method ({norm_method}, must be "
                         f"one of {supported_norm_methods}.)")

    # Convert window func if needed
    if isinstance(win_func, str):
        win_func = utils.import_function_by_name(win_func)

    # Validate input vectors
    rri, trr = utils.standardize_rri_trr(rri, trr)

    # Validate bands
    if not len(vlf_band) == len(lf_band) == len(hf_band) == 2:
        raise ValueError("All frequency band vectors must have exactly two "
                         "elements.")

    # Use full signal if window_minutes is not defined
    if not window_minutes or window_minutes < 1:
        window_minutes = max(1, math.floor((trr[-1] - trr[0]) / 60))

    # Windowing
    t_max = trr[-1]
    f_min = vlf_band[0]
    f_max = hf_band[1]
    t_win_min = 1 / f_min  # minimal window to resolve f_min
    t_win = max(60 * window_minutes, t_win_min)

    # In case there's not enough data for one window, use entire signal length
    num_windows = math.floor(t_max / t_win)
    if num_windows < 1:
        num_windows = 1
        t_win = math.floor(trr[-1] - trr[0])

    # Uniform frequency axis
    f_axis, fs_uni = frequency.build_uniform_freq_axis(
        t_win, f_min, f_max, resample_factor, oversample_factor
    )

    # Uniform time axis
    trr_uni = np.r_[trr[0]:trr[-1]: 1 / fs_uni]
    n_win_uni = math.floor(t_win / (1 / fs_uni))  # num samples per window
    num_windows_uni = math.floor(len(trr_uni) / n_win_uni)

    # Check Nyquist criterion
    if n_win_uni < 2 * f_max * t_win:
        logger.warning('Nyquist criterion not met for given window length and '
                       'frequency bands')

    # Calculate spectrums
    pxx = {}
    if 'lomb' in methods:
        pxx['lomb'] = frequency.pxx_lomb(rri, f_axis, trr, t_win, win_func)

    # Resample on a uniform time axis to obtain spectral estimate
    rri_interpolator = scipy.interpolate.interp1d(
        trr, rri, kind='cubic', assume_sorted=True, fill_value='extrapolate'
    )
    rri_uni = rri_interpolator(trr_uni)

    if 'welch' in methods:
        welch_window = win_func(n_win_uni)
        welch_overlap = math.floor(n_win_uni * welch_overlap / 100)
        f_welch, pxx_welch = scipy.signal.welch(
            rri_uni, fs=fs_uni, nperseg=n_win_uni,
            noverlap=welch_overlap, window=welch_window, detrend='constant',
            scaling='density', return_onesided=True)

        pxx['welch'] = pxx_welch[f_welch <= f_max]

    return pxx, f_axis
