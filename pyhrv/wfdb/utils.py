import re

import os.path
import numpy as np
import wfdb

from pyhrv.wfdb.consts import *


def rdann_by_type(rec_path: str, ann_ext: str,
                  types: str = WFDB_ANN_ALL_PEAK_TYPES):
    """
    Reads WFDB annotation file and returns annotations of specific types.
    :param rec_path: Record path (without extension).
    :param ann_ext: extension of annotation file to load.
    :param types: A string of chars of the annotation types to find (see [1]_).
    :return: A dictionary, mapping from the annotation type (a
    char) to a numpy array of indices in the signal.

    .. [1] https://www.physionet.org/physiobank/annotations.shtml
    """
    ann_to_idx = {ann_type: [] for ann_type in types}

    # In case it's a Path object; wfdb can't handle that
    rec_path = str(rec_path)

    # Read annotations
    ann = wfdb.rdann(rec_path, ann_ext)

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


def is_record(rec_path: str, dat_ext: str = 'dat', ann_exts=()):
    """
    Checks whether the given recrod path is a PhysioNet record.
    A record must have at least a header file (.hea).
    See also [1]_.

    :param rec_path: Path of record, without any file extension, for example
    'db/mitdb/100'.
    :param dat_ext: File extension of data file (usually dat or mat). Can
    also be None to indicate a header-only record.
    :param ann_exts: Iterable of annotation file extensions to check for.
    :return: True if the given rec_path corresponds to a PhysioNet record
    with matching header file, data file and annotation files.

    .. [1] https://www.physionet.org/physiotools/wag/intro.htm
    """
    if type(ann_exts) == str:
        ann_exts = (ann_exts,)

    exts_to_check = ('hea', dat_ext, *ann_exts)

    for ext in exts_to_check:
        if ext is None:
            continue
        if not os.path.isfile(f'{rec_path}.{ext}'):
            return False

    return True


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
    m = re.match(r'^(\d{1,2}):(\d{1,2})\.(\d{1,3})$', time)
    if m:
        m, s, yyy = m.groups()
        m, s = int(m), int(s)
        ms = int(yyy + (3-len(yyy)) * '0')
        if m > 60 or s > 60:
            raise ValueError("Invalid number of minutes or seconds")
        seconds = m * 60 + s + ms/1000.
        return int(seconds * fs)

    raise ValueError(f"The given time string ({time}) in not in a "
                     f"recognised format.")

