import importlib
from typing import NamedTuple


def import_function_by_name(func_name):
    mod_name, func_name = func_name.rsplit('.', 1)

    if not mod_name:
        raise ValueError("Must provide fully-qualified function name.")

    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def sec_to_time(sec: float):
    if sec < 0:
        raise ValueError('Invalid argument value')
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
        return f'{"" if self.d == 0 else f"{self.d}+"}' \
            f'{self.h:02d}:{self.m:02d}:{self.s:02d}.{self.ms:03d}'
