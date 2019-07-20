import re
import random
from pathlib import Path

import pytest

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
        ch = utils.find_ecg_channel(self.resource_path / 'ch0')
        assert ch == 0

    def test_second_channel(self):
        ch = utils.find_ecg_channel(self.resource_path / 'ch1')
        assert ch == 1

    def test_no_channel(self):
        ch = utils.find_ecg_channel(self.resource_path / 'nochan')
        assert ch is None


class TestIsRecord(object):
    def setup_method(self):
        self.resource_path = Path(f'{TEST_RESOURCES_PATH}/wfdb')

    def test_non_existent_record(self):
        assert not utils.is_record(self.resource_path / 'no_such_record_123')
        assert not utils.is_record(self.resource_path / 'no_such_record_123',
                                   dat_ext=None)
        assert not utils.is_record(self.resource_path / 'no_such_record_123',
                                   dat_ext=None, ann_ext='foo')

    def test_ann_only_record(self):
        assert utils.is_record(self.resource_path / 'foo', ann_ext='bar')
        assert utils.is_record(self.resource_path / 'foo', ann_ext='bar',
                               dat_ext=None)

    def test_header_data_record(self):
        assert utils.is_record(self.resource_path / '100')
        assert utils.is_record(self.resource_path / '101')

    def test_header_data_ann_record(self):
        assert utils.is_record(self.resource_path / '101', ann_ext='atr')


class TestWFDBTimeToSamples(object):
    fs = 128

    def test_samples_string(self):
        assert utils.wfdb_time_to_samples('s234098234', 342) == 234098234
        assert utils.wfdb_time_to_samples('s123', 100) == 123
        assert utils.wfdb_time_to_samples('s0', 123) == 0

    def test_end_symbol(self):
        assert utils.wfdb_time_to_samples('e', 123) == -1
        assert utils.wfdb_time_to_samples('e', 321) == -1

    def test_hhmmss(self):
        assert utils.wfdb_time_to_samples('1:2:3', self.fs) == 3723 * self.fs
        assert utils.wfdb_time_to_samples('01:2:3', self.fs) == 3723 * self.fs
        assert utils.wfdb_time_to_samples('1:02:3', self.fs) == 3723 * self.fs
        assert utils.wfdb_time_to_samples('1:2:03', self.fs) == 3723 * self.fs
        assert utils.wfdb_time_to_samples('10:20:10',
                                          self.fs) == 37210 * self.fs

    def test_mmssyyy(self):
        assert utils.wfdb_time_to_samples('2:3',
                                          self.fs) == int(123 * self.fs)
        assert utils.wfdb_time_to_samples('02:3',
                                          self.fs) == int(123 * self.fs)
        assert utils.wfdb_time_to_samples('2:03',
                                          self.fs) == int(123 * self.fs)
        assert utils.wfdb_time_to_samples('02:03',
                                          self.fs) == int(123 * self.fs)
        assert utils.wfdb_time_to_samples('2:3.4',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('2:3.40',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('2:3.400',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('2:03.400',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('02:3.400',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('02:03.400',
                                          self.fs) == int(123.4 * self.fs)
        assert utils.wfdb_time_to_samples('59:59.999',
                                          self.fs) == int(3599.999 * self.fs)

    def test_invalid_formats(self):
        bad_formats = [
            '', 's', 'ss', 'ee', 'E', 'S123', 'EE',
            '1:61:2', '01:-1:01', '05:002:59', '02:999:49',
            '1:02:61', '01:01:-1', '05:59:002', '02:49:999', ':02:03',
            '03::03', '04::', '::05', ':09:',
            '2:3.', '01:02.1000', '3:4.1234', '03:4.5432',
            '61:02.123', '6:61:341',
        ]

        for bad_format in bad_formats:
            rand_fs = random.randrange(100, 1000)
            with pytest.raises(ValueError):
                utils.wfdb_time_to_samples(bad_format, rand_fs)


class TestSecToTime(object):
    def test_invalid(self):
        for val in [-1.3, -100, -123.234]:
            with pytest.raises(ValueError):
                utils.sec_to_time(val)

    def test_zero(self):
        assert str(utils.sec_to_time(0)) == "00:00:00.000"
        assert str(utils.sec_to_time(0.)) == "00:00:00.000"

    def test_milisec(self):
        assert str(utils.sec_to_time(0.2345)) == "00:00:00.234"

    def test_sec(self):
        assert str(utils.sec_to_time(1.2345)) == "00:00:01.234"
        assert str(utils.sec_to_time(59.345678)) == "00:00:59.345"

    def test_min(self):
        assert str(utils.sec_to_time(60.345678)) == "00:01:00.345"
        assert str(utils.sec_to_time(70.45678)) == "00:01:10.456"

    def test_hrs(self):
        assert str(utils.sec_to_time(3599.345678)) == "00:59:59.345"
        assert str(utils.sec_to_time(3600)) == "01:00:00.000"
        assert str(utils.sec_to_time(3600 + 123.456)) == "01:02:03.456"

    def test_days(self):
        assert str(utils.sec_to_time(
            23 * 3600 + 15 * 60 + 59.456)) == "23:15:59.456"
        assert str(utils.sec_to_time(
            24 * 3600 + 15 * 60 + 59.456)) == "1+00:15:59.456"
        assert str(utils.sec_to_time(
            97 * 3600 + 59 * 60 + 59.456)) == "4+01:59:59.456"
        assert str(utils.sec_to_time(
            97 * 3600 + 59 * 60 + 59.456 + 0.544)) == "4+02:00:00.000"
