import os
import re
import subprocess
import warnings
import glob
import tempfile

import wfdb

import pyhrv.wfdb.utils as utils
from pyhrv.wfdb.consts import ECGPUWAVE_BIN


def ecgpuwave_detect_sig(sig, fs, from_samp=0, to_samp=-1, **kw):
    """
    Runs the ecgpuwave QRS detector on a single channel ECG signal, and returns
    the indices of detections.

    :param sig: A 1-d numpy array containing one ECG channel.
    :param fs: The sampling frequency of that channel.
    :param from_samp: Start sample index.
    :param to_samp: End sample index. -1 for end of record.
    :return: A numpy array of sample indices corresponding to detected
    features (either R-peaks or QRS onset).
    """
    assert sig.ndim == 1
    sig = sig[from_samp:to_samp].reshape(-1, 1)

    # We'll write the given data to a temporary record
    temp_dir = tempfile.gettempdir()
    temp_recname = f'tmprec_{os.getpid()}'
    rec_path = os.path.join(temp_dir, temp_recname)

    try:
        wfdb.wrsamp(record_name=temp_recname, fs=fs, units=['mV'],
                    sig_name=['ECG'], p_signal=sig,
                    write_dir=temp_dir, fmt=['212'])

        return ecgpuwave_detect_rec(rec_path, channel=0, **kw)
    finally:
        # Remove the temporary record
        for f in glob.glob(os.path.join(rec_path, '.*')):
            os.remove(f)


def ecgpuwave_detect_rec(rec_path, channel=None, from_time=None, to_time=None,
                         **kw):
    """
    Runs the ecgpuwave QRS detector on a single channel ECG signal from a
    given PhysioNet record, and returns the indices of detections.

    :param rec_path: Path to PhysioNet record without extension.
    :param channel: Index of channel to read when using rec_path. If None,
    it will be heuristically estimated.
    :param from_time: Start time. A string in PhysioNet time format [1]_.
    :param to_time: End time.  A string in PhysioNet time format [1]_.
    :return: A numpy array of sample indices corresponding to detected
    features (either R-peaks or QRS onset).

    .. [1] https://www.physionet.org/physiotools/wag/intro.htm#time
    """

    if channel is None:
        channel = utils.find_ecg_channel(rec_path)

    ann_ext = f'ecgatr{os.getpid()}'
    try:
        # Run ecgpuwave
        ecgpuwave_wrapper(rec_path, ann_ext, channel=channel,
                          from_time=from_time, to_time=to_time)

        # Read the annotations from the file it created.
        ann_type = 'N'
        ann_to_idx = utils.rdann_by_type(rec_path, ann_ext, types=ann_type)
        return ann_to_idx[ann_type]
    finally:
        # Remove the annotation file
        ann_file = f'{rec_path}.{ann_ext}'
        if os.path.exists(ann_file):
            os.remove(ann_file)


def ecgpuwave_wrapper(record: str, out_ann_ext: str,
                      in_ann_ext: str = None, channel: int = None,
                      from_time: str = None, to_time: str = None):
    """
    A wrapper for PhysioNet's ecgpuwave [1]_ tool, which segments ECG beats.

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

    .. [1] https://www.physionet.org/physiotools/ecgpuwave/
       [2] https://www.physionet.org/physiotools/wag/intro.htm#time
    """
    if not utils.is_record(record):
        raise ValueError(f"Can't find record {record}")

    rec_dir = os.path.dirname(record)
    rec_name = os.path.basename(record)
    ecgpuwave_rel_path = os.path.abspath(ECGPUWAVE_BIN)

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
            try:
                os.remove(tmpfile)
            except FileNotFoundError:
                # When running multiple ecgpuwave processes in parallel, it's
                # possible file was already deleted by another process
                pass

    return True

