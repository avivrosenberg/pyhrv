import pytest
import numpy as np
import pyhrv.wfdb.rri as rri

from . import TEST_RESOURCES_PATH


class TestECGRR(object):
    def setup_method(self):
        self.resources = TEST_RESOURCES_PATH.joinpath('wfdb')

    def test_with_ann(self):
        t, rr = rri.ecgrr(self.resources/'100', ann_ext='atr')
        self.sanity_checks(t, rr)

    def test_without_ann(self):
        t, rr = rri.ecgrr(self.resources/'100')
        self.sanity_checks(t, rr)

    def test_with_ann_from_time(self):
        t, rr = rri.ecgrr(self.resources/'100', ann_ext='atr',
                          from_time='01:30')
        assert np.all(t >= 1.5*60)
        self.sanity_checks(t, rr)

    def test_without_ann_from_time(self):
        t, rr = rri.ecgrr(self.resources/'100', from_time='05:45')
        assert np.all(t >= 5.75*60)
        self.sanity_checks(t, rr)

    def test_with_ann_to_time(self):
        t, rr = rri.ecgrr(self.resources/'100', ann_ext='atr',
                          to_time='10:30')
        assert np.all(t <= 10.5*60)
        self.sanity_checks(t, rr)

    def test_without_ann_to_time(self):
        t, rr = rri.ecgrr(self.resources/'100', to_time='13:15')
        assert np.all(t <= 13.25*60)
        self.sanity_checks(t, rr)

    def test_with_ann_from_to_time(self):
        t, rr = rri.ecgrr(self.resources/'100', ann_ext='atr',
                          from_time='02:15', to_time='10:30')
        assert np.all((2.25 * 60 <= t) * (t <= 10.5 * 60))
        self.sanity_checks(t, rr)

    def test_without_ann_from_to_time(self):
        t, rr = rri.ecgrr(self.resources/'100',
                          from_time='00:02:15', to_time='00:10:30')
        assert np.all((2.25 * 60 <= t) * (t <= 10.5 * 60))
        self.sanity_checks(t, rr)

    @staticmethod
    def sanity_checks(t, rr):
        assert len(t) == len(rr)
        assert np.all(t >= 0)
        assert np.all(np.diff(t) > 0)  # t should be monotonically increasing
        assert np.all(rr > 0)
        assert np.mean(rr) == pytest.approx(0.8, rel=0.05)
