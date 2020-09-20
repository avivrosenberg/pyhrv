import os.path as path

PHYSIONET_TOOLS_BIN_DIR = f"{path.abspath(path.dirname(__file__))}/bin"

ECGPUWAVE_BIN = f"{PHYSIONET_TOOLS_BIN_DIR}/ecgpuwave-1.3.3/ecgpuwave"

WFDB_ANN_ALL_PEAK_TYPES = "NVSFQLRBAaJrFejnE/f"

"""
Regex pattern that heuristically detects ECG channels in WFDB records.
"""
ECG_CHANNEL_PATTERN = r"ECG|lead\si+|MLI+|v\d|\bI+\b"
