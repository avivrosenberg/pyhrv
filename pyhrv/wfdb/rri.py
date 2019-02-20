import numpy as np
import wfdb
import pyhrv.wfdb.qrs as qrs
import pyhrv.wfdb.utils as utils


def ecgrr(rec_path, ann_ext=None, channel=None, from_time=None, to_time=None,
          detector=qrs.ecgpuwave_detect_rec):

    if not utils.is_record(rec_path, ann_exts=(ann_ext,)):
        raise ValueError(f"Can't find record {rec_path}")

    header = wfdb.rdheader(rec_path)
    fs = float(header.fs)

    if ann_ext is not None:
        # Load r-peaks from annotation
        ann_type = 'N'
        ann = utils.rdann_by_type(rec_path, ann_ext, types=ann_type)
        sample_idxs = ann[ann_type]

    else:
        # Calculate r-peaks using a peak-detector
        sample_idxs = detector(rec_path, channel=channel,
                               from_time=from_time, to_time=to_time)

    start_time = sample_idxs[0] / fs
    rri = np.diff(sample_idxs) / fs

    trr = np.empty_like(rri)
    np.cumsum(rri[0:-1], out=trr[1:])
    trr[0] = 0.
    trr += start_time

    return trr, rri
