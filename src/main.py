"""Starter tremor analysis pipeline for AutoMe."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 1.0, high_hz: float = 12.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * fs
    low = low_hz / nyquist
    high = high_hz / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    fft_vals = np.abs(rfft(signal)) / n
    freqs = rfftfreq(n, d=1.0 / fs)
    return freqs, fft_vals


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoMe starter tremor analysis")
    parser.add_argument("--input", required=True, type=Path, help="Path to input CSV file")
    parser.add_argument("--column", default="accel_x", help="Column name to analyze")
    parser.add_argument("--sampling-rate", type=float, required=True, help="Sampling rate in Hz")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if args.column not in data.columns:
        raise ValueError(f"Column '{args.column}' not found. Available columns: {list(data.columns)}")

    signal = data[args.column].to_numpy(dtype=float)
    filtered = bandpass_filter(signal, fs=args.sampling_rate, low_hz=1.0, high_hz=12.0)

    freqs, spectrum = compute_fft(filtered, fs=args.sampling_rate)
    tremor_band = (freqs >= 3.0) & (freqs <= 7.0)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, spectrum, label="Filtered Signal FFT")
    plt.fill_between(freqs, spectrum, where=tremor_band, alpha=0.25, label="Tremor Band (3-7 Hz)")
    plt.title("AutoMe Tremor Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
