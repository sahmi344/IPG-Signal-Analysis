"""
Plot CSV and Save Selected Region
===================================
- Loads a CSV file with header row: Sample, Amplitude (V)
- Plots Sample (x-axis) vs Amplitude in Volts (y-axis)
- Click-drag on the plot to select a region
- Click "Save Selected Region" to export just that region to a new CSV,
  keeping the original Sample numbers and the same 2-column format.

Install: pip install matplotlib numpy
Run    : python plot_and_save_selection.py
"""

import os
import csv
import numpy as np
from scipy.signal import butter, sosfiltfilt

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button

FS = 500.0  # Sampling frequency in Hz (samples are at 500 Hz)
DEFAULT_LOW = 0.3
DEFAULT_HIGH = 5.0


def load_csv(filepath):
    """Load CSV with header row: Sample, Amplitude (V). Returns (samples, amps)."""
    samples = []
    amps = []
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row
        for row in reader:
            if len(row) >= 2 and row[0].strip() != "" and row[1].strip() != "":
                samples.append(float(row[0]))
                amps.append(float(row[1]))
    if not samples:
        raise ValueError(f"No data found in {filepath}")
    return np.array(samples, dtype=np.float64), np.array(amps, dtype=np.float64), header


def bandpass_filter(signal, low_hz, high_hz, fs=FS, order=2):
    """
    Apply a Butterworth bandpass filter (zero-phase via sosfiltfilt) to the signal.
    low_hz / high_hz are the cutoff frequencies in Hz.

    Uses second-order-sections (SOS) form rather than transfer-function (b,a)
    form, since b,a coefficients become numerically unstable/inaccurate for
    narrow, low-frequency bands like 0.3-5 Hz at typical sampling rates —
    SOS form avoids this and gives a correct, stable filter.
    """
    nyquist = fs / 2.0
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist

    low_norm = max(low_norm, 1e-6)
    high_norm = min(high_norm, 0.999999)

    if low_norm >= high_norm:
        raise ValueError(
            f"Invalid band: low ({low_hz} Hz) must be less than high ({high_hz} Hz), "
            f"and both must be below Nyquist ({nyquist} Hz)."
        )

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")

    # sosfiltfilt needs the signal to be longer than ~3x the number of
    # second-order sections; pick padlen automatically to avoid errors on
    # very short signals.
    padlen = min(3 * 2 * len(sos), len(signal) - 1)
    if padlen < 0:
        padlen = 0

    filtered = sosfiltfilt(sos, signal, padlen=padlen)
    return filtered


def get_filter_settings():
    """Small GUI to customize the bandpass filter range (Hz) before plotting."""
    result = {"low": DEFAULT_LOW, "high": DEFAULT_HIGH, "enabled": True}

    root = tk.Tk()
    root.title("Bandpass Filter Settings")
    root.geometry("340x200")

    enabled_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        root, text="Apply Bandpass Filter", variable=enabled_var,
        font=("Arial", 10, "bold")
    ).pack(pady=(12, 4))

    frame = tk.Frame(root)
    frame.pack(pady=4)
    tk.Label(frame, text="Low cutoff (Hz):").grid(row=0, column=0, padx=4, pady=4)
    low_entry = tk.Entry(frame, width=8)
    low_entry.insert(0, str(DEFAULT_LOW))
    low_entry.grid(row=0, column=1, padx=4, pady=4)

    tk.Label(frame, text="High cutoff (Hz):").grid(row=1, column=0, padx=4, pady=4)
    high_entry = tk.Entry(frame, width=8)
    high_entry.insert(0, str(DEFAULT_HIGH))
    high_entry.grid(row=1, column=1, padx=4, pady=4)

    tk.Label(root, text=f"Butterworth bandpass filter (fs = {FS:.0f} Hz)\n"
                         f"Default: {DEFAULT_LOW}-{DEFAULT_HIGH} Hz",
             font=("Arial", 8), fg="gray30").pack(pady=(2, 8))

    def submit():
        try:
            lo = float(low_entry.get())
            hi = float(high_entry.get())
            if lo < 0 or hi <= lo:
                raise ValueError
            result["low"], result["high"] = lo, hi
            result["enabled"] = enabled_var.get()
            root.destroy()
        except ValueError:
            messagebox.showerror("Invalid input", "Enter valid numbers with High > Low >= 0.")

    tk.Button(root, text="Continue", command=submit, bg="#bbf7d0").pack(pady=6)
    root.mainloop()
    return result["low"], result["high"], result["enabled"]


def select_input_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select CSV file to plot",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not path:
        raise SystemExit("No file selected. Exiting.")
    return path


def save_selection_csv(samples, amps, mask, header, source_path, filter_enabled=True, low_hz=None, high_hz=None):
    """Save the masked (selected) rows to a new CSV, same 2-column format,
    keeping original sample numbers. Amplitude values are whatever was
    passed in (filtered, per the workflow)."""
    root = tk.Tk()
    root.withdraw()
    default_name = os.path.splitext(os.path.basename(source_path))[0] + "_selected.csv"
    out_path = filedialog.asksaveasfilename(
        title="Save selected region as",
        defaultextension=".csv",
        initialfile=default_name,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not out_path:
        return None

    sel_samples = samples[mask]
    sel_amps = amps[mask]

    col1_name = header[0] if header and len(header) >= 1 else "Sample"
    col2_name = header[1] if header and len(header) >= 2 else "Amplitude (V)"
    if filter_enabled and low_hz is not None and high_hz is not None:
        col2_name = f"{col2_name} [Filtered {low_hz}-{high_hz}Hz]"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col1_name, col2_name])
        for s, v in zip(sel_samples, sel_amps):
            # Keep sample as integer-looking if it was a whole number originally
            s_out = int(s) if float(s).is_integer() else s
            writer.writerow([s_out, v])

    return out_path, len(sel_samples)


def main():
    csv_path = select_input_file()
    samples, amps_raw, header = load_csv(csv_path)

    low_hz, high_hz, filter_enabled = get_filter_settings()

    if filter_enabled:
        amps = bandpass_filter(amps_raw, low_hz, high_hz, fs=FS)
        filter_label = f"Bandpass filtered ({low_hz}-{high_hz} Hz)"
    else:
        amps = amps_raw
        filter_label = "Unfiltered (raw)"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.canvas.manager.set_window_title(f"Plot & Select Region - {os.path.basename(csv_path)}")
    ax.plot(samples, amps, color="#2563eb", linewidth=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(
        f"{os.path.basename(csv_path)}  —  {filter_label}\n"
        f"Click-drag to select a region, then click 'Save Selected Region'"
    )
    ax.grid(alpha=0.3)

    selection = {"mask": None, "smin": None, "smax": None}

    info_text = ax.text(
        0.01, 0.98, "No region selected yet.", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, color="#b91c1c",
        bbox=dict(boxstyle="round", fc="white", ec="#b91c1c", alpha=0.85),
    )

    def on_select(xmin, xmax):
        smin, smax = sorted((xmin, xmax))
        mask = (samples >= smin) & (samples <= smax)
        n_selected = int(np.sum(mask))
        selection["mask"] = mask
        selection["smin"] = smin
        selection["smax"] = smax
        info_text.set_text(
            f"Selected samples {smin:.0f} to {smax:.0f}  "
            f"({n_selected} samples)"
        )
        fig.canvas.draw_idle()

    span = SpanSelector(
        ax, on_select, "horizontal", useblit=True,
        props=dict(alpha=0.25, facecolor="#22c55e"),
        interactive=True, drag_from_anywhere=True,
    )

    def on_save(event):
        if selection["mask"] is None or not np.any(selection["mask"]):
            messagebox.showwarning(
                "No Selection", "Please click-drag on the plot to select a region first."
            )
            return
        result = save_selection_csv(
            samples, amps, selection["mask"], header, csv_path,
            filter_enabled=filter_enabled, low_hz=low_hz, high_hz=high_hz
        )
        if result is None:
            return
        out_path, n_rows = result
        messagebox.showinfo(
            "Saved",
            f"Saved {n_rows} samples to:\n{out_path}"
        )

    btn_ax = fig.add_axes([0.78, 0.01, 0.20, 0.06])
    btn = Button(btn_ax, "Save Selected Region", color="#bbf7d0", hovercolor="#86efac")
    btn.on_clicked(on_save)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
