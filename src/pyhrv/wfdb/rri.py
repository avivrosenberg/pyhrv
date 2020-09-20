import numpy as np
import wfdb
import pyhrv.wfdb.qrs as qrs
import pyhrv.wfdb.utils as utils


def ecgrr(rec_path, ann_ext=None, channel=None, from_time=None, to_time=None,
          detector=qrs.ecgpuwave_detect_rec, dtype=np.float32):
    """
    Returns an RR-interval time-series given a PhysioNet record.
    :param rec_path: The path to the record (without any file extension).
    :param ann_ext: Extension of annotation file to use. If provided,
    R-peaks will be read from this annotation file instead of performing
    peak-detection.
    :param channel: Number of ECG channel in the record. Will be
    heuristically estimated if missing.
    :param from_time: Start time. A string in the PhysioNet time format.
    :param to_time: End time. A string in the PhysioNet time format.
    :param detector: A function to use for peak-detection. Will only be used if
    the ann_ext parameter was not provided.
    :param dtype: Desired dtype of output tensors.
    :return: Tuple of time axis and interval durations.
    """
    if not utils.is_record(rec_path, ann_ext=ann_ext):
        raise ValueError(f"Can't find record {rec_path}")

    if ann_ext is not None:
        # Load r-peaks from annotation
        ann_type = 'N'
        ann = utils.rdann_by_type(rec_path, ann_ext,
                                  from_time, to_time, types=ann_type)
        sample_idxs = ann[ann_type]
    else:
        # Calculate r-peaks using a peak-detector
        sample_idxs = detector(rec_path, channel=channel,
                               from_time=from_time, to_time=to_time)

    header = wfdb.rdheader(rec_path)
    fs = float(header.fs)

    start_time = sample_idxs[0] / fs
    rri = np.diff(sample_idxs) / fs

    trr = np.empty_like(rri)
    np.cumsum(rri[0:-1], out=trr[1:])
    trr[0] = 0.
    trr += start_time

    return trr.astype(dtype), rri.astype(dtype)
