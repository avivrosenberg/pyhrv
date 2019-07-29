import glob
import os
from pathlib import Path
import wfdb

import pytest
from pyhrv.wfdb.qrs import ecgpuwave_wrapper

from . import TEST_RESOURCES_PATH

RESOURCES_PATH = f'{TEST_RESOURCES_PATH}/ecgpuwave'
TEST_ANN_EXT = 'test'


class TestECGPuWaveWrapper(object):
    def setup_method(self):
        self.test_rec = f'{RESOURCES_PATH}/100s'
        self.test_rec_ann_ext = 'atr'

    def teardown_method(self):
        if glob.glob(f'{RESOURCES_PATH}/fort.*'):
            self.fail("Found un-deleted temp files")

    @classmethod
    def teardown_class(cls):
        # Delete previously-created annotations
        for ann_file_path in Path(RESOURCES_PATH).glob(f"*.{TEST_ANN_EXT}"):
            os.remove(ann_file_path)

    def test_sig0_full_noatr_num_annotations(self):
        for signal_idx in [0, None]:
            res = ecgpuwave_wrapper(self.test_rec, TEST_ANN_EXT,
                                    channel=signal_idx)
            assert res
            self._helper_check_num_annotations(684)

    def test_sig1_full_noatr_num_annotations(self):
        res = ecgpuwave_wrapper(self.test_rec, TEST_ANN_EXT, channel=1)
        assert res
        self._helper_check_num_annotations(655)

    def test_sig0_first30s_noatr_num_annotations(self):
        for to_time in ['0:30', '0:0:30', '00:00:30', 's10800']:
            res = ecgpuwave_wrapper(self.test_rec, TEST_ANN_EXT,
                                    to_time=to_time)
            assert res
            self._helper_check_num_annotations(349)

    def test_sig1_last20s_noatr_num_annotations(self):
        for from_time in ['0:40', '0:0:40', '00:00:40', 's14400']:
            res = ecgpuwave_wrapper(self.test_rec, TEST_ANN_EXT, channel=1,
                                    from_time=from_time)
            assert res
            self._helper_check_num_annotations(209)

    def test_sig0_full_withatr_num_annotations(self):
        res = ecgpuwave_wrapper(self.test_rec, TEST_ANN_EXT,
                                in_ann_ext=self.test_rec_ann_ext)
        assert res
        self._helper_check_num_annotations(670)

    def test_invalid_record_should_fail(self):
        with pytest.raises(ValueError):
            res = ecgpuwave_wrapper(f'{RESOURCES_PATH}/foo', 'bar')

    def _helper_check_num_annotations(self, expected):
        ann = wfdb.rdann(self.test_rec, TEST_ANN_EXT)
        actual = len(ann.sample)
        assert expected == actual, "Incorrect number of annotations"
