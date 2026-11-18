import numpy as np
from signal_app.signal_preprocessing import (
    remove_baseline_linear, baseline_drift, preprocess)

FS = 1000


def make_signal():
    t = np.arange(10 * FS) / FS
    y = 100 + 20 * t                  # slow drift, 100 -> 300
    y[5000] += 80                     # one spike
    return y


def test_linear_removes_drift_keeps_spike():
    out = remove_baseline_linear(make_signal(), fs=FS, block_seconds=1.0)
    assert abs(np.median(out)) < 2
    assert out[5000] > 70
    assert abs(out[2000]) < 5


def test_drift_before_and_after():
    y = make_signal()
    out = remove_baseline_linear(y, fs=FS, block_seconds=1.0)
    assert baseline_drift(y, fs=FS) > 150
    assert baseline_drift(out, fs=FS) < 5


def test_preprocess_linear_keeps_nan_and_length():
    y = make_signal()
    y[3000:3100] = np.nan
    out = preprocess(y, fs=FS, smooth_ms=0, method="linear")
    assert len(out) == len(y)
    assert np.isnan(out[3050]) and not np.isnan(out[2000])


def test_short_signal_does_not_crash():
    out = remove_baseline_linear(np.arange(10.0), fs=FS)
    assert len(out) == 10
