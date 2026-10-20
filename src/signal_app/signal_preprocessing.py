import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 50_000  # measurements per second
SMOOTH_MS = 1.0       # smoothing window in milliseconds
HIGHPASS_HZ = 0.3     # drift slower than this is removed
EDGE_SECONDS = 2.0    # filter is unreliable this long at the start and end


def smooth(y, fs=SAMPLE_RATE, ms=SMOOTH_MS):
    n = int(ms / 1000 * fs)
    if n <= 1:
        return y
    return uniform_filter1d(y, size=n, mode="nearest")


def remove_baseline(y, fs=SAMPLE_RATE, cutoff_hz=HIGHPASS_HZ):
    y = y - np.median(y)
    if cutoff_hz <= 0 or len(y) < 1000:
        return y
    sos = butter(2, cutoff_hz, btype="highpass", fs=fs, output="sos")
    pad = min(len(y) - 1, int(2 * fs / cutoff_hz))
    return sosfiltfilt(sos, y, padlen=pad)


def preprocess(y, fs=SAMPLE_RATE, smooth_ms=SMOOTH_MS, cutoff_hz=HIGHPASS_HZ):
    y = np.asarray(y, dtype=np.float64)
    result = np.full(y.shape, np.nan)

    is_data = ~np.isnan(y)
    if not is_data.any():
        return result

    # Clean each stretch of real data separately (NaN stays NaN).
    change = np.diff(np.concatenate(([0], is_data.astype(int), [0])))
    starts = np.flatnonzero(change == 1)
    stops = np.flatnonzero(change == -1)
    for start, stop in zip(starts, stops):
        part = smooth(y[start:stop], fs, smooth_ms)
        result[start:stop] = remove_baseline(part, fs, cutoff_hz)
    return result


def good_samples(y, fs=SAMPLE_RATE, edge_seconds=EDGE_SECONDS):
    y = np.asarray(y)
    ok = ~np.isnan(y)
    edge = int(edge_seconds * fs)
    if edge == 0:
        return ok
    padded = np.concatenate(([True], np.isnan(y), [True]))
    near_gap = uniform_filter1d(padded.astype(float), size=2 * edge + 1,
                                mode="constant") > 0
    return ok & ~near_gap[1:-1]
