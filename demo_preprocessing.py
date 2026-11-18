"""Shows the preprocessing on a real signal: raw, smoothed, baseline removed.

Run from the project folder:
    uv run python demo_preprocessing.py "C:\\path\\to\\signal.csv"
Optional: --col 1 (which column to use), --seconds 20 (how much to load)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from signal_app.signal_preprocessing import (
    SAMPLE_RATE, baseline_drift, preprocess, smooth)


def load(path, col, seconds):
    n = int(seconds * SAMPLE_RATE)
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
        data = np.asarray(data[:n * 2]).reshape(-1, 2) if data.ndim == 1 else np.asarray(data[:n])
    else:
        sep = "," if "," in path.read_text(errors="ignore")[:2000] else None
        data = np.genfromtxt(path, delimiter=sep, max_rows=n)
    data = data.astype(np.float64)
    if data.ndim == 1:
        return data
    if col is None:
        col = 1 if data.shape[1] > 1 else 0
    print(f"Using column {col} of {data.shape[1]}")
    return data[:, col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--col", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=20)
    a = ap.parse_args()

    y = load(Path(a.path), a.col, a.seconds)
    t = np.arange(len(y)) / SAMPLE_RATE

    # smooth() cannot take NaN, so fill gaps for the picture only
    filled = np.where(np.isnan(y), np.nanmedian(y), y)
    smoothed = smooth(filled)
    smoothed[np.isnan(y)] = np.nan
    high = preprocess(y, smooth_ms=1.0)
    lin = preprocess(y, smooth_ms=1.0, method="linear")

    print(f"Samples: {len(y)}  ({len(y) / SAMPLE_RATE:.1f} s)")
    print(f"Baseline drift before:           {baseline_drift(y):8.2f}")
    print(f"Baseline drift after high-pass:  {baseline_drift(high):8.2f}")
    print(f"Baseline drift after linear:     {baseline_drift(lin):8.2f}")

    step = 50  # plot every 50th point so the picture is fast
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for axis, data, name in zip(
            ax, [y, smoothed, high, lin],
            ["Raw signal", "Smoothed (1 ms)",
             "Smoothed + baseline removed (high-pass)",
             "Smoothed + baseline removed (linear)"]):
        axis.plot(t[::step], data[::step], linewidth=0.7)
        axis.set_title(name, fontsize=10)
    ax[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out = Path("preprocessing_demo.png")
    fig.savefig(out, dpi=130)
    print(f"Saved picture: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
