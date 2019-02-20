import pytest
import numpy as np
import pyhrv.wfdb.rri as rri

from tests.wfdb import TEST_RESOURCES_PATH


class TestECGRR(object):
    def setup_method(self):
        self.resources = TEST_RESOURCES_PATH.joinpath('wfdb')

    def test_with_ann(self):
        t, rr = rri.ecgrr(self.resources/'100', ann_ext='atr')
        self.sanity_checks(t, rr)

    def test_without_ann(self):
        t, rr = rri.ecgrr(self.resources/'100')
        self.sanity_checks(t, rr)

    @staticmethod
    def sanity_checks(t, rr):
        assert len(t) == len(rr)
        assert np.all(t >= 0)
        assert np.all(np.diff(t) > 0)  # t should be monotonically increasing
        assert np.all(rr > 0)
        assert np.mean(rr) == pytest.approx(0.8, rel=0.05)
