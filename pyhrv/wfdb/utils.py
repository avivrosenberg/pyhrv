import re
import sys

import os.path
import numpy as np
import wfdb

from typing import NamedTuple

from pyhrv.wfdb.consts import *


def rdann_by_type(rec_path: str, ann_ext: str,
                  from_time: str = None, to_time: str = None,
                  types: str = WFDB_ANN_ALL_PEAK_TYPES):
    """
    Reads WFDB annotation file and returns annotations of specific types.
    :param rec_path: Record path (without extension).
    :param ann_ext: extension of annotation file to load.
    :param from_time: Start time. A string in PhysioNet time format [1]_.
    :param to_time: End time.  A string in PhysioNet time format [1]_.
    :param types: A string of chars of the annotation types to find (see [2]_).
    :return: A dictionary, mapping from the annotation type (a
    char) to a numpy array of indices in the signal.

    .. [1] https://www.physionet.org/physiotools/wag/intro.htm#time
    .. [2] https://www.physionet.org/physiobank/annotations.shtml
    """
    if not is_record(rec_path, ann_ext=ann_ext):
        raise ValueError(f"Can't find record {rec_path}")

    ann_to_idx = {ann_type: [] for ann_type in types}

    # In case it's a Path object; wfdb can't handle that
    rec_path = str(rec_path)

    # Handle from/to by converting to samples
    sampfrom, sampto = 0, sys.maxsize
    if from_time is not None or to_time is not None:
        header = wfdb.rdheader(rec_path)
        if from_time is not None:
            sampfrom = wfdb_time_to_samples(from_time, header.fs)
        if to_time is not None:
            sampto = wfdb_time_to_samples(to_time, header.fs)

    # Read annotations
    ann = wfdb.rdann(rec_path, ann_ext, sampfrom, sampto)

    # Find annotations of requested type
    annotations_pattern = re.compile(fr'[{types}]')
    joined_ann = str.join('', ann.symbol)
    matches = list(annotations_pattern.finditer(joined_ann))

    for i, m in enumerate(matches):
        ann_type = m.group()
        ann_idx = m.start()

        # Save annotation sample
        ann_to_idx[ann_type].append(ann.sample[ann_idx])

    return {
        ann_type: np.array(idxs)
        for (ann_type, idxs) in ann_to_idx.items()
    }


def find_ecg_channel(rec_path):
    """
    Heuristically finds the index of the first ECG channel in a record.
    :param rec_path: Path to record without extension.
    :return: The index of the first ECG channel.
    """
    pattern = re.compile(ECG_CHANNEL_PATTERN, re.IGNORECASE)
    header = wfdb.rdheader(rec_path)
    for i, name in enumerate(header.sig_name):
        if pattern.match(name):
            return i
    return None


def is_record(rec_path: str, dat_ext: str = 'dat', ann_ext: str = None):
    """
    Checks whether the given recrod path is a PhysioNet record.
    A record must have at least a header file (.hea), and either a data file
    (.dat/.mat other) or an annotation file.
    See also [1]_.

    :param rec_path: Path of record, without any file extension, for example
    'db/mitdb/100'.
    :param dat_ext: File extension of data file (usually dat or mat). Can
    also be None to indicate a header-only record.
    :param ann_ext: Extension annotation file to check for.
    :return: True if the given rec_path corresponds to a PhysioNet record
    with matching header file, data file and annotation files. If the data
    is not present but atleast one annotation exists, it's still a record.

    .. [1] https://www.physionet.org/physiotools/wag/intro.htm
    """
    if not os.path.isfile(f'{rec_path}.hea'):
        return False

    if dat_ext and os.path.isfile(f'{rec_path}.{dat_ext}'):
        return True

    if ann_ext and os.path.isfile(f'{rec_path}.{ann_ext}'):
        return True

    return False


def wfdb_time_to_samples(time: str, fs):
    """
    Converts a WFDB time string [1]_ to samples.

    :param time: A string in the WFDB time format.
    :param fs: The sampling frequency of the record.
    :return: An integer equal to the number of samples the given time string
    represents.

    .. [1] https://www.physionet.org/physiotools/wag/intro.htm#time
    """

    # 'e' means last sample
    if time == 'e':
        return -1

    # 's1234' means sample 1234
    m = re.match(r'^s(\d+)$', time)
    if m:
        return int(m.group(1))

    # 'sss..s'
    m = re.match(r'^\d+$', time)
    if m:
        return int(int(time) * fs)

    # 'hh:mm:ss'
    m = re.match(r'^(\d{1,2}):(\d{1,2}):(\d{1,2})$', time)
    if m:
        h, m, s = m.groups()
        h, m, s = int(h), int(m), int(s)
        if m > 60 or s > 60:
            raise ValueError("Invalid number of minutes or seconds")
        seconds = h * 3600 + m * 60 + s
        return int(seconds * fs)

    # 'mm:ss.yyy'
    m = re.match(r'^(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?$', time)
    if m:
        m, s, yyy = m.groups()
        m, s = int(m), int(s)
        ms = int(yyy + (3 - len(yyy)) * '0') if yyy else 0
        if m > 60 or s > 60:
            raise ValueError("Invalid number of minutes or seconds")
        seconds = m * 60 + s + ms / 1000.
        return int(seconds * fs)

    raise ValueError(f"The given time string ({time}) in not in a "
                     f"recognised format.")


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
