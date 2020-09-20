import numpy as np
import importlib
from typing import NamedTuple


def import_function_by_name(func_name):
    """
    Imports and returns a function obbject given it's full name (packages,
    modules and function name, e.g. foo.bar.baz).
    :param func_name: Full name.
    :return: Function object (callable).
    """
    mod_name, func_name = func_name.rsplit(".", 1)

    if not mod_name:
        raise ValueError("Must provide fully-qualified function name.")

    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def sec_to_time(sec: float):
    """
    Converts a time duration in seconds to days, hours, minutes, seconds and
    milliseconds.
    :param sec: Time in seconds.
    :return: An object with d, h, m, s and ms fields representing the above,
    respectively.
    """
    if sec < 0:
        raise ValueError("Invalid argument value")
    d = int(sec // (3600 * 24))
    h = int((sec // 3600) % 24)
    m = int((sec // 60) % 60)
    s = int(sec % 60)
    ms = int((sec % 1) * 1000)
    return __T(d, h, m, s, ms)


class __T(NamedTuple):
    d: int
    h: int
    m: int
    s: int
    ms: int

    def __repr__(self):
        return (
            f'{"" if self.d == 0 else f"{self.d}+"}'
            f"{self.h:02d}:{self.m:02d}:{self.s:02d}.{self.ms:03d}"
        )


def np_squeeze_check(a: np.ndarray) -> np.ndarray:
    """
    Converts a row/column (2d) to a 1d array. Does nothing if it's already 1d.
    Raises an error if it's not a row/col.
    :param a: An ndarray of shape (N,) or (N,1) or (N,1).
    :return: The same data, flattend to (N,).
    """
    a = a.squeeze()
    if a.ndim != 1:
        raise ValueError("The given array is not 1d")

    return a


def standardize_rri_trr(rri, trr=None):
    """
    Converts rri intervals and their times to 1d arrays. Creates zero-based
    times if not provided.
    :param rri: RR intervals.
    :param trr: RR intervals times.
    :return: rri, trr tuple after standardization.
    """
    rri = np_squeeze_check(rri)

    if trr is None:
        trr = np.r_[0.0, np.cumsum(rri)[:-1]]
    else:
        trr = np_squeeze_check(trr)
        if len(trr) != len(rri):
            raise ValueError("Shape mismatch between rri and trr")

    return rri, trr
