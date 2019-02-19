import re

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
    char) to an array of indices in the signal.

    .. [1] https://www.physionet.org/physiobank/annotations.shtml
    """
    ann_to_idx = {ann_type: [] for ann_type in types}

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
