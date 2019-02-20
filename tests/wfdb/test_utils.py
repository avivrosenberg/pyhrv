import re
from pathlib import Path

from tests.wfdb import TEST_RESOURCES_PATH
from pyhrv.wfdb import ECG_CHANNEL_PATTERN

import pyhrv.wfdb.utils as utils


class TestECGChannelRegex(object):
    def setup_method(self):
        self.pattern = re.compile(ECG_CHANNEL_PATTERN, re.IGNORECASE)

        self.positive_test_cases = [
            'ecg', 'foo ECG', 'ECG_bar', 'ECG1', 'ecg2', 'foo ecg3', 'foo_ecg',
            'MLI', 'MLII', 'MLIII', 'foo MLI', 'MLIIII bar',
            'V5', 'foo v1', 'v4 bar',
            'lead ii', 'foo lead iii', 'lead I bar', 'ECG Lead II',
            'iiii', 'foo II', 'II bar',
        ]

        self.negative_test_cases = [
            'foobar', 'foo bar baz', 'foo_bar', 'foo1', 'foo 1', '2 bar',
            '2_bar', 'foo foo bar', 'bar_foo_baz', 'IIfoo', 'fooIII',
            'wrist_ppg', 'PPG', 'resp', 'ART', 'PAP', 'CVP', 'resp imp.',
            'CO2',
        ]

    def test_positive_cases(self):
        for teststr in self.positive_test_cases:
            for teststr_case in [teststr, teststr.upper(), teststr.lower()]:
                match = self.pattern.search(teststr_case)
                assert match is not None, teststr_case

    def test_negative_cases(self):
        for teststr in self.negative_test_cases:
            for teststr_case in [teststr, teststr.upper(), teststr.lower()]:
                match = self.pattern.search(teststr_case)
                assert match is None, teststr_case


class TestFindECGChannel(object):
    def setup_method(self):
        self.resource_path = Path(f'{TEST_RESOURCES_PATH}/find_ecg_channel')

    def test_first_channel(self):
        ch = utils.find_ecg_channel(self.resource_path/'ch0')
        assert ch == 0

    def test_second_channel(self):
        ch = utils.find_ecg_channel(self.resource_path/'ch1')
        assert ch == 1

    def test_no_channel(self):
        ch = utils.find_ecg_channel(self.resource_path/'nochan')
        assert ch is None


class TestIsRecord(object):
    def setup_method(self):
        self.resource_path = Path(f'{TEST_RESOURCES_PATH}/wfdb')

    def test_non_existent_record(self):
        assert not utils.is_record(self.resource_path/'no_such_record_123')

    def test_header_only_record(self):
        assert not utils.is_record(self.resource_path/'foo')
        assert utils.is_record(self.resource_path/'foo', dat_ext=None)

    def test_header_data_record(self):
        assert utils.is_record(self.resource_path/'100')
        assert utils.is_record(self.resource_path/'101')

    def test_header_data_ann_record(self):
        assert utils.is_record(self.resource_path/'100', ann_exts=('atr',))
        assert utils.is_record(self.resource_path/'101', ann_exts='atr')
