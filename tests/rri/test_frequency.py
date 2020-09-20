import pytest

import math
import scipy.signal as sps
import matplotlib.pyplot as plt

import pyhrv.wfdb.rri
import pyhrv.rri.frequency as frequency
import pyhrv.rri.processing

from ..wfdb import TEST_RESOURCES_PATH

WFDB_TEST_RESOURCES_PATH = TEST_RESOURCES_PATH.joinpath("wfdb")


class TestBuildUniformFrequencyAxis(object):
    def test_1(self):
        f_min, f_max = 0.003, 0.5
        t_win = 200  # too low for f_min
        resample_factor = 2.25
        osf = 4

        f_axis, fs = frequency.build_uniform_freq_axis(
            t_win, f_min, f_max, resample_factor=resample_factor, oversample_factor=osf
        )

        assert fs == f_max * resample_factor

        t_win = 1 / f_min
        n_win = math.floor(t_win / (1 / fs))
        f_res = (n_win - 1) / (n_win * t_win)  # freq resolution
        f_res /= osf  # Oversampling factor -> interpolation in freq

        assert f_axis[0] == pytest.approx(f_res, rel=1e-9)
        assert f_axis[-1] == pytest.approx(
            f_res + math.floor((f_max - f_res) / f_res) * f_res, rel=1e-9
        )


class TestPxxLomb(object):
    @classmethod
    def setup_class(cls):
        rec_path = WFDB_TEST_RESOURCES_PATH.joinpath("100")
        trr, rri = pyhrv.wfdb.rri.ecgrr(rec_path, ann_ext="atr")

        cls.trr, cls.rri = pyhrv.rri.processing.filtrr(trr, rri)

    def test_1(self):
        t_win = 330
        f_min, f_max = 1 / t_win, 0.4
        osf = 4

        f_axis, _ = frequency.build_uniform_freq_axis(
            t_win, f_min, f_max, oversample_factor=osf
        )

        i = 1
        win_idx = (self.trr >= t_win * i) & (self.trr < t_win * (i + 1))
        rri_win = self.rri[win_idx]

        pxx_lomb = frequency.pxx_lomb(rri_win, f_axis=f_axis)
        # plt.plot(f_axis, pxx_lomb)
        # plt.show()

        # Check location of highest peak
        peaks_idx, _ = sps.find_peaks(pxx_lomb, height=0.1)
        assert f_axis[peaks_idx[0]] == pytest.approx(0.168299, rel=1e-5)
