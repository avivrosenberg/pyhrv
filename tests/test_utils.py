import numpy as np
import pytest

import pyhrv.utils as utils


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


class TestImportFunctionByName(object):
    def test_(self):
        f = utils.import_function_by_name('pyhrv.utils.sec_to_time')
        assert str(f(97 * 3600 + 59 * 60 + 59.456)) == "4+01:59:59.456"


class TestNpSqueezeCheck(object):
    @staticmethod
    def _check_data(expected, actual):
        assert actual.ndim == 1
        assert np.all(actual == expected.reshape(-1))

    def test_1d(self):
        a = np.array([1, 2, 3])
        b = utils.np_squeeze_check(a)
        self._check_data(a, b)

    def test_col(self):
        a = np.c_[1:10]
        b = utils.np_squeeze_check(a)
        self._check_data(a, b)

    def test_row(self):
        a = np.c_[1:10].T
        b = utils.np_squeeze_check(a)
        self._check_data(a, b)

    def test_2d(self):
        a = np.c_[1:10, 2:11]
        with pytest.raises(ValueError) as ex_info:
            utils.np_squeeze_check(a)

    def test_nd(self):
        a = np.random.randn(3, 4, 5, 6, 7, 9)
        with pytest.raises(ValueError) as ex_info:
            utils.np_squeeze_check(a)

    def test_nd_sqeezable(self):
        a = np.random.randn(1, 1, 1, 1, 100, 1, 1, 1, 1)
        b = utils.np_squeeze_check(a)
        self._check_data(a, b)


class TestStandardize(object):
    @staticmethod
    def _check_data(rri, trr, rri_new, trr_new):
        assert rri_new.ndim == 1
        assert trr_new.ndim == 1
        assert np.all(rri_new == rri.reshape(-1))
        assert np.all(trr_new == trr.reshape(-1))
        assert rri_new.shape == trr_new.shape

    def test_squeeze_dims(self):
        rri = np.random.randn(1, 100)
        trr = np.linspace(1.2, 111.1, num=100)

        rri_new, trr_new = utils.standardize_rri_trr(rri, trr)
        self._check_data(rri, trr, rri_new, trr_new)

    def test_create_trr(self):
        rri = np.random.randn(1, 100)

        rri_new, trr_new = utils.standardize_rri_trr(rri)

        trr_expected = np.r_[0, np.cumsum(rri)[:-1]]
        self._check_data(rri, trr_expected, rri_new, trr_new)
