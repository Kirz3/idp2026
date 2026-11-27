import numpy as np

from signal_app.JuliasPeakDetector import (
    JuliasPeakDetector,
    estimate_noise,
    MIN_PROMINENCE,
)

FS = 50_000


def make_signal(noise_sigma, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, noise_sigma, FS * seconds)


def add_peak(signal, second, height, width_ms=10):
    t = np.arange(len(signal)) / FS
    sigma = width_ms / 1000 / 2.355
    peak = height * np.exp(-0.5 * ((t - second) / sigma) ** 2)
    return signal + peak


def test_noise_estimate():
    signal = make_signal(5.0)
    noise = estimate_noise(signal)
    assert abs(noise - 5.0) < 0.3


def test_noise_with_peaks():
    signal = make_signal(5.0)

    for second in (1.5, 3.5, 5.5, 7.5):
        signal = add_peak(signal, second, 300)

    noise = estimate_noise(signal)
    assert abs(noise - 5.0) < 0.3


def test_clear_peaks():
    signal = make_signal(5.0)

    for second in (2.5, 5.5, 8.5):
        signal = add_peak(signal, second, 300)

    detector = JuliasPeakDetector()
    peaks = detector.detect_peaks(signal)
    assert len(peaks) == 3


def test_different_noise_levels():
    quiet = add_peak(make_signal(1.0), 5.0, 50)
    noisy = add_peak(make_signal(30.0), 5.0, 50)

    detector = JuliasPeakDetector()

    assert len(detector.detect_peaks(quiet)) == 1
    assert len(detector.detect_peaks(noisy)) == 0


def test_min_prominence():
    signal = make_signal(0.5)
    signal = add_peak(signal, 5.0, MIN_PROMINENCE / 2)

    detector = JuliasPeakDetector()
    assert len(detector.detect_peaks(signal)) == 0


def test_flat_signal():
    signal = np.zeros(FS)
    detector = JuliasPeakDetector()

    assert len(detector.detect_peaks(signal)) == 0
