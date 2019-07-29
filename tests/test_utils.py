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
