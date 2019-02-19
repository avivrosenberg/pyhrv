import os
import re
import subprocess
import warnings
import glob
import tempfile
import random

import wfdb

import pyhrv.wfdb.utils as utils
from pyhrv.wfdb.consts import ECGPUWAVE_BIN


def ecgpuwave_detect(sig=None, fs=None, rec_path=None, channel=None,
                     sampfrom=0, sampto=-1, peaks=True, **kw):
    """
    Detect QRS using the ecgpuwave detector.
    Can work with either a PhysioNet record name or data from a single channel
    :param sig: A 1-d numpy array containing one ECG channel. Can't be
    combined with rec_path.
    :param fs: The sampling frequency of that channel. Can't be
    combined with rec_path.
    :param rec_path: Path to PhysioNet record. If specified, sig and fs must
    be None.
    :param channel: Index of channel to read when using rec_path. If None,
    it will be heuristically estimated.
    :param sampfrom: Start sample index.
    :param sampto: End sample index. -1 for end of record.
    :param peaks: Whether to detect R-peaks (True) or QRS onset (False).
    :return: A numpy array of sample indices corresponding to detected
    features (either R-peaks or QRS onset).
    """

    if not ((sig is not None and fs is not None) or rec_path is not None):
        raise ValueError("Must set either sig and fs or rec_path, not both "
                         "or none")

    from_time = f's{sampfrom:d}'
    to_time = 'e' if sampto == -1 else f's{sampto:d}'

    # Write the signal data to a WFDB record
    delete_rec = False
    if not rec_path:
        assert sig.ndim == 1
        temp_dir = tempfile.gettempdir()
        temp_recname = f'tmp_{random.randint(0, 10000):04d}'
        sig = sig[sampfrom:sampto].reshape(-1, 1)
        wfdb.wrsamp(record_name=temp_recname, fs=fs, units=['mV'],
                    sig_name=['ECG'], p_signal=sig,
                    write_dir=temp_dir, fmt=['212'])
        rec_path = os.path.join(temp_dir, temp_recname)
        delete_rec = True
    else:
        if channel is None:
            channel = utils.find_ecg_channel(rec_path)

    # Apply peak detector
    try:
        detector = ECGPuWave()
        ann_ext = 'ecgatr'
        detector(rec_path, ann_ext, channel=channel,
                 from_time=from_time, to_time=to_time)
        ann_type = 'N' if peaks else '('
        ann_to_idx = utils.rdann_by_type(rec_path, ann_ext, types=ann_type)
        return ann_to_idx[ann_type]
    finally:
        if delete_rec:
            for f in glob.glob(os.path.join(rec_path, '.*')):
                os.remove(f)


class ECGPuWave(object):
    """
    A wrapper for PhysioNet's ecgpuwave [1]_ tool, which segments ECG beats.

    .. [1] https://www.physionet.org/physiotools/ecgpuwave/
    """

    def __init__(self, ecgpuwave_bin=ECGPUWAVE_BIN):
        self.ecgpuwave_bin = ecgpuwave_bin

    def __call__(self, record: str, out_ann_ext: str,
                 in_ann_ext: str = None, channel: int = None,
                 from_time: str = None, to_time: str = None):
        """
        Runs the ecgpuwave tool on a given record, producing an annotation
        file with a specified extension.

        :param record: Path to PhysioNet record, e.g. foo/bar/123 (no file
            extension allowed).
        :param out_ann_ext: The extension of the annotation file to create.
        :param in_ann_ext: Read an annotation file with the given extension
            as input to specify beat types in the record.
        :param channel: The index of the channel in the record to
            analyze.
        :param from_time: Start at the given time. Should be a string in one of
            the PhysioNet time formats (see [2]_).
        :param to_time: Stop at the given time. Should be a string in one of
            the PhysioNet time formats (see [2]_).
        :return: True if ran without error.

        .. [2] https://www.physionet.org/physiotools/wag/intro.htm#time
        """

        rec_dir = os.path.dirname(record)
        rec_name = os.path.basename(record)
        ecgpuwave_rel_path = os.path.abspath(self.ecgpuwave_bin)

        ecgpuwave_command = [
            ecgpuwave_rel_path,
            '-r', rec_name,
            '-a', out_ann_ext,
        ]

        if in_ann_ext:
            ecgpuwave_command += ['-i', in_ann_ext]

        if channel:
            ecgpuwave_command += ['-s', str(channel)]

        if from_time:
            ecgpuwave_command += ['-f', from_time]

        if to_time:
            ecgpuwave_command += ['-t', to_time]

        try:
            ecgpuwave_result = subprocess.run(
                ecgpuwave_command,
                check=True, shell=False, universal_newlines=True, timeout=10,
                cwd=rec_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # ecgpuwave can sometimes fail but still return 0, so need to
            # also check the stderr output.
            if ecgpuwave_result.stderr:
                # Annoying case: sometimes ecgpuwave writes to stderr but it's
                # not an error...
                if not re.match(r'Rearranging annotations[\w\s.]+done!',
                                ecgpuwave_result.stderr):
                    raise subprocess.CalledProcessError(0, ecgpuwave_command)

        except subprocess.CalledProcessError as process_err:
            warnings.warn(f'Failed to run ecgpuwave on record '
                          f'{record}:\n'
                          f'stderr: {ecgpuwave_result.stderr}\n'
                          f'stdout: {ecgpuwave_result.stdout}\n')
            return False

        except subprocess.TimeoutExpired as timeout_err:
            warnings.warn(f'Timed-out runnning ecgpuwave on record '
                          f'{record}: '
                          f'{ecgpuwave_result.stdout}')
            return False
        finally:
            # Remove tmp files created by ecgpuwave
            for tmpfile in glob.glob(f'{rec_dir}/fort.*'):
                os.remove(tmpfile)

        return True
