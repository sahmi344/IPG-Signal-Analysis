"""
Signal Processing: Cycle P2P, Coefficient of Variation, Band Power
======================================================================
Loads a CSV file (header row: Sample, Amplitude) and computes:

  1. Average Peak-to-Peak Voltage, cycle-by-cycle:
       - A "cycle" is defined as the segment between two consecutive
         valleys (local minima).
       - For each cycle, P2P = (peak value within the cycle) - (the
         FIRST valley value bounding that cycle).
       - Average P2P = mean of all per-cycle P2P values.

  2. Coefficient of Variation (CV):
       - Computed on the cycle-by-cycle P2P values (not raw amplitude),
         since raw-amplitude CV is unstable for signals oscillating near
         zero mean.
       - CV (%) = (std of P2P values / mean of P2P values) * 100

  3. Band Power (absolute energy) in a customizable band, default
     0.3-5 Hz, at a fixed sampling frequency of 500 Hz:
       - Computed via Welch's PSD method, then integrating the PSD
         curve over the band using the trapezoidal rule.
       - Units: V^2 (signal power/energy in that frequency band).

Install: pip install numpy scipy matplotlib
Run    : python signal_processing.py
"""

import os
import csv
import numpy as np
from scipy.signal import find_peaks, welch

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

FS = 500.0  # Sampling frequency in Hz (fixed)
DEFAULT_BAND_LOW = 0.3
DEFAULT_BAND_HIGH = 5.0


# ──────────────────────────────────────────────────────────────────────
# File loading
# ──────────────────────────────────────────────────────────────────────


def load_csv(filepath):
    """Load CSV with header row: Sample, Amplitude (V). Returns (samples, amps, header)."""
    samples = []
    amps = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0].strip() != "" and row[1].strip() != "":
                samples.append(float(row[0]))
                amps.append(float(row[1]))
    if not samples:
        raise ValueError(f"No data found in {filepath}")
    return (np.array(samples, dtype=np.float64),
            np.array(amps, dtype=np.float64), header)


def select_input_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select CSV file to analyze",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No file selected. Exiting.")
    return path


# ──────────────────────────────────────────────────────────────────────
# Signal Processing
# ──────────────────────────────────────────────────────────────────────

def detect_valleys(signal, fs=FS):
    """
    Detect valleys (local minima) in the signal.
    Uses adaptive minimum distance (based on estimated dominant frequency
    via zero-crossing rate) and a prominence threshold (10% of signal
    range) to avoid picking up noise as false valleys.
    """
    if len(signal) < 3:
        return np.array([], dtype=int)

    zero_crossings = np.sum(np.diff(np.signbit(signal - np.mean(signal))))
    duration = len(signal) / fs
    est_freq = max((zero_crossings / 2.0) / duration, 0.5) if duration > 0 else 1.0
    min_dist = int(max(fs / (est_freq * 4.0), 1))

    sig_range = np.max(signal) - np.min(signal)
    prominence = 0.1 * sig_range if sig_range > 0 else None

    valley_idx, _ = find_peaks(-signal, distance=min_dist, prominence=prominence)
    return valley_idx


def compute_cycle_p2p(signal, fs=FS):
    """
    Cycle-by-cycle peak-to-peak analysis.
    A cycle = the segment between two CONSECUTIVE valleys.
    For each cycle:
        peak_value   = max(signal) within that valley-to-valley window
        cycle_p2p    = peak_value - first_valley_value
    Returns:
        avg_p2p       : mean of all cycle P2P values
        cv_percent    : coefficient of variation of cycle P2P values (%)
        cycle_p2p_list: list of individual cycle P2P values
        valley_idx    : indices of detected valleys
        peak_idx_list : indices of the peak found within each cycle
    """
    valley_idx = detect_valleys(signal, fs)

    if len(valley_idx) < 2:
        return 0.0, 0.0, [], valley_idx, []

    cycle_p2p_list = []
    peak_idx_list = []

    for i in range(len(valley_idx) - 1):
        v_start = valley_idx[i]
        v_end = valley_idx[i + 1]

        segment = signal[v_start:v_end + 1]
        local_peak_offset = int(np.argmax(segment))
        peak_idx = v_start + local_peak_offset
        peak_value = signal[peak_idx]
        first_valley_value = signal[v_start]

        p2p = peak_value - first_valley_value
        cycle_p2p_list.append(p2p)
        peak_idx_list.append(peak_idx)

    cycle_p2p_arr = np.array(cycle_p2p_list)
    avg_p2p = float(np.mean(cycle_p2p_arr))

    if avg_p2p != 0:
        cv_percent = float(np.std(cycle_p2p_arr) / avg_p2p * 100.0)
    else:
        cv_percent = 0.0

    return avg_p2p, cv_percent, cycle_p2p_list, valley_idx, peak_idx_list


def compute_band_power(signal, fs, band_low, band_high):
    """
    Absolute band power (energy) in [band_low, band_high] Hz via Welch PSD,
    integrated using the trapezoidal rule. Units: V^2.
    """
    sig = signal - np.mean(signal)
    n = len(sig)
    if n < 8:
        return 0.0

    nperseg = min(n, int(fs * 4))
    nperseg = max(nperseg, 8)

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    band_mask = (freqs >= band_low) & (freqs <= band_high)

    if not np.any(band_mask):
        return 0.0

    band_power = trapz_fn(psd[band_mask], freqs[band_mask])
    return float(band_power)


# ──────────────────────────────────────────────────────────────────────
# Settings GUI
# ──────────────────────────────────────────────────────────────────────

def get_band_settings():
    """Small GUI to customize the band power frequency range (Hz)."""
    result = {"low": DEFAULT_BAND_LOW, "high": DEFAULT_BAND_HIGH}

    root = tk.Tk()
    root.title("Band Power Settings")
    root.geometry("340x170")

    tk.Label(root, text="Band Power Frequency Range (Hz)",
             font=("Arial", 10, "bold")).pack(pady=(12, 6))

    frame = tk.Frame(root)
    frame.pack(pady=4)
    tk.Label(frame, text="Low (Hz):").grid(row=0, column=0, padx=4, pady=4)
    low_entry = tk.Entry(frame, width=8)
    low_entry.insert(0, str(DEFAULT_BAND_LOW))
    low_entry.grid(row=0, column=1, padx=4, pady=4)

    tk.Label(frame, text="High (Hz):").grid(row=1, column=0, padx=4, pady=4)
    high_entry = tk.Entry(frame, width=8)
    high_entry.insert(0, str(DEFAULT_BAND_HIGH))
    high_entry.grid(row=1, column=1, padx=4, pady=4)

    tk.Label(root, text=f"Band Power = \u222b PSD over [low, high] Hz  (V\u00b2)\n"
                         f"Sampling Frequency fixed at {FS:.0f} Hz",
             font=("Arial", 8), fg="gray30").pack(pady=(2, 8))

    def submit():
        try:
            lo = float(low_entry.get())
            hi = float(high_entry.get())
            if lo < 0 or hi <= lo:
                raise ValueError
            result["low"], result["high"] = lo, hi
            root.destroy()
        except ValueError:
            messagebox.showerror("Invalid input", "Enter valid numbers with High > Low >= 0.")

    tk.Button(root, text="Run Analysis", command=submit, bg="#bbf7d0").pack(pady=6)
    root.mainloop()
    return result["low"], result["high"]


# ──────────────────────────────────────────────────────────────────────
# Results Display
# ──────────────────────────────────────────────────────────────────────

def show_plot_and_results(samples, amps, valley_idx, peak_idx_list,
                           avg_p2p, cv_percent, n_cycles,
                           band_power, band_low, band_high, csv_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.canvas.manager.set_window_title(f"Signal Analysis - {os.path.basename(csv_path)}")

    # ── Signal plot with valleys/peaks marked ──
    ax1.plot(samples, amps, color="#2563eb", linewidth=0.8, label="Signal", zorder=1)
    if len(valley_idx) > 0:
        ax1.scatter(samples[valley_idx], amps[valley_idx],
                    color="#dc2626", marker="v", s=40, zorder=3,
                    label=f"Valleys (n={len(valley_idx)})")
    if len(peak_idx_list) > 0:
        peak_idx_arr = np.array(peak_idx_list)
        ax1.scatter(samples[peak_idx_arr], amps[peak_idx_arr],
                    color="#16a34a", marker="^", s=40, zorder=3,
                    label=f"Cycle Peaks (n={len(peak_idx_list)})")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude (V)")
    ax1.set_title(f"{os.path.basename(csv_path)} — Valley-to-Valley Cycle Detection")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Results table ──
    ax2.axis("off")
    rows = [
        ["Number of Cycles Detected", f"{n_cycles}"],
        ["Average Peak-to-Peak Voltage", f"{avg_p2p:.5f} V"],
        ["Coefficient of Variation (CV)", f"{cv_percent:.3f} %"],
        [f"Band Power [{band_low}-{band_high} Hz]", f"{band_power:.6f} V\u00b2"],
        ["Sampling Frequency", f"{FS:.0f} Hz"],
    ]
    table = ax2.table(cellText=rows, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    csv_path = select_input_file()
    samples, amps, header = load_csv(csv_path)

    band_low, band_high = get_band_settings()

    avg_p2p, cv_percent, cycle_p2p_list, valley_idx, peak_idx_list = compute_cycle_p2p(amps, FS)
    band_power = compute_band_power(amps, FS, band_low, band_high)

    n_cycles = len(cycle_p2p_list)

    print("\n========== RESULTS ==========")
    print(f"File: {csv_path}")
    print(f"Sampling Frequency: {FS} Hz")
    print(f"Number of valleys detected: {len(valley_idx)}")
    print(f"Number of cycles (valley-to-valley): {n_cycles}")
    print(f"Average Peak-to-Peak Voltage: {avg_p2p:.5f} V")
    print(f"Coefficient of Variation (of cycle P2P values): {cv_percent:.3f} %")
    print(f"Band Power [{band_low}-{band_high} Hz]: {band_power:.6f} V^2")
    print("==============================\n")

    show_plot_and_results(samples, amps, valley_idx, peak_idx_list,
                          avg_p2p, cv_percent, n_cycles,
                          band_power, band_low, band_high, csv_path)


if __name__ == "__main__":
    main()
