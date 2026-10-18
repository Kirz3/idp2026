import numpy as np

from signal_app.signal_preprocessing import (
    SAMPLE_RATE,
    good_samples,
    preprocess,
    remove_baseline,
    smooth,
)


def fake_signal(seconds=20, spikes=(), seed=0):
    """Noise + slow drift + a big offset, like a sensor reading.

    `spikes` is a list of (time in seconds, height) pairs.
    """
    rng = np.random.default_rng(seed)
    n = seconds * SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    y = rng.normal(0, 3, n) + 700 + 100 * np.sin(2 * np.pi * 0.05 * t)
    for time_s, height in spikes:
        i = int(time_s * SAMPLE_RATE)
        y[i : i + 10] += height
    return y


def test_output_has_same_length():
    y = fake_signal()
    assert len(preprocess(y)) == len(y)


def test_offset_and_drift_are_removed():
    y = fake_signal()
    clean = preprocess(y)
    assert abs(np.median(clean)) < 1          # the offset of 700 is gone
    assert clean.std() < 0.1 * y.std()        # the drift is gone


def test_noise_gets_smaller():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 3, 100_000)
    assert smooth(noise).std() < 0.5 * noise.std()


def test_spike_is_kept_and_stays_in_place():
    y = fake_signal(spikes=[(10, 200)])
    clean = preprocess(y)
    peak = np.argmax(clean)
    assert abs(peak - 10 * SAMPLE_RATE) < 0.005 * SAMPLE_RATE  # within 5 ms
    assert clean[peak] > 30  # still clearly visible (smoothing lowers it)


def test_bigger_spike_gives_bigger_value():
    y = fake_signal(spikes=[(5, 50), (10, 100), (15, 200)])
    clean = preprocess(y)
    heights = [clean[int(t * SAMPLE_RATE) : int(t * SAMPLE_RATE) + 10].max()
               for t in (5, 10, 15)]
    assert heights[0] < heights[1] < heights[2]


def test_missing_values_stay_missing():
    y = fake_signal()
    y[1000:2000] = np.nan
    clean = preprocess(y)
    assert np.isnan(clean[1000:2000]).all()
    assert not np.isnan(clean[3000:]).any()


def test_empty_end_of_file_does_not_change_the_data_part():
    # The last window of a file is filled with NaN after the data ends.
    # The cleaned data must be exactly the same as without the empty part.
    data = fake_signal(seconds=20)
    padded = np.concatenate([data, np.full(10 * SAMPLE_RATE, np.nan)])
    assert np.allclose(preprocess(padded)[: len(data)], preprocess(data))


def test_short_and_empty_input_do_not_crash():
    assert len(preprocess(np.array([]))) == 0
    assert len(preprocess(np.array([5.0]))) == 1
    assert len(preprocess(np.ones(500))) == 500
    assert np.isnan(preprocess(np.full(100, np.nan))).all()


def test_remove_baseline_on_a_flat_signal_gives_zero():
    assert np.allclose(remove_baseline(np.full(5000, 665.0)), 0)


def test_good_samples_skips_edges_and_gaps():
    y = fake_signal(seconds=20)
    y[10 * SAMPLE_RATE : 11 * SAMPLE_RATE] = np.nan
    ok = good_samples(preprocess(y))
    assert not ok[0] and not ok[-1]                        # array ends
    assert not ok[10 * SAMPLE_RATE - 1]                    # just before the gap
    assert not ok[int(10.5 * SAMPLE_RATE)]                 # inside the gap
    assert ok[5 * SAMPLE_RATE]                             # the safe middle
