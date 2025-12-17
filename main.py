
from src.metrics import cross_spectrum_irregular, auto_spectrum_irregular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Main workflow (translation of your R script)
# -----------------------------------------------------------

def main():
    # -----------------------------------------------------------
    # Load and plot the first light curve (irregularly sampled)
    # -----------------------------------------------------------

    # Choose the files you want
    file1 = "OGLE-BLG-LPV-000018.dat"
    file2 = "OGLE-BLG-LPV-000009.dat"

    Irrlyr1 = read_ogle_dat(file1)
    Irrlyr2 = read_ogle_dat(file2)

    # Column mapping
    # time = col 0, magnitude = col 1, error = col 2 (optional)
    t1 = Irrlyr1.iloc[:, 0].to_numpy()
    x = Irrlyr1.iloc[:, 1].to_numpy()

    t2 = Irrlyr2.iloc[:, 0].to_numpy()
    y = Irrlyr2.iloc[:, 1].to_numpy()

    # Plot series 1
    plt.figure()
    plt.plot(t1, x, marker="o", linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title("Light curve 1")
    plt.gca().invert_yaxis()  # common for magnitudes; remove if undesired
    plt.show()

    # Plot series 2
    plt.figure()
    plt.plot(t2, y, marker="o", linestyle="-")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title("Light curve 2")
    plt.gca().invert_yaxis()
    plt.show()

    # -----------------------------------------------------------
    # Global spectral parameters: mixture weight and frequency grid
    # -----------------------------------------------------------

    alpha = 0.5
    beta = alpha * (1 - alpha)

    # Approximate average sampling interval Δt (using series 1)
    st = t1
    delta_t = (np.max(st) - np.min(st)) / len(st)

    delta_f = 0.001
    f_min = delta_f
    f_max = 1.0 / (2.0 * delta_t)

    f = np.arange(f_min, f_max + delta_f, delta_f)
    omega = 2.0 * np.pi * f

    # Diagnostics for each series
    delta_t1 = (np.max(t1) - np.min(t1)) / len(t1)
    delta_t2 = (np.max(t2) - np.min(t2)) / len(t2)

    print(f"delta_t  (global, from series 1): {delta_t}")
    print(f"delta_t1 (series 1): {delta_t1}")
    print(f"delta_t2 (series 2): {delta_t2}")
    print(f"f_min={f_min}, f_max≈{f_max}, #freq={len(f)}")

    # -----------------------------------------------------------
    # Auto-spectra (power or cross-spectrum with itself)
    # -----------------------------------------------------------

    # Vector-valued spectra across omega grid
    sx1 = auto_spectrum_irregular(omega, t1, x)
    sx2 = auto_spectrum_irregular(omega, t2, y)

    # Integrals over frequency band
    Ix = integrate_cross_spectrum_real(f_min, f_max, t1, x, t1, x)
    Iy = integrate_cross_spectrum_real(f_min, f_max, t2, y, t2, y)

    # -----------------------------------------------------------
    # Cross term: cross-spectrum between series 1 and 2
    # -----------------------------------------------------------

    f_max_xy = min(f_max, f_max)  # kept for structural similarity to your R
    Ixy = integrate_cross_spectrum_real(f_min, f_max_xy, t1, x, t2, y)

    # -----------------------------------------------------------
    # Jensen-type spectral divergence between the two series
    # -----------------------------------------------------------

    J = beta * (Ix + Iy - 2.0 * Ixy) / (2.0 * np.pi)

    print("\nResults:")
    print(f"Ix  = {Ix}")
    print(f"Iy  = {Iy}")
    print(f"Ixy = {Ixy}")
    print(f"J   = {J}")

    # Optional: plot real parts of auto-spectra for inspection
    plt.figure()
    plt.plot(f, np.real(sx1), label="Re S_xx (series 1)")
    plt.plot(f, np.real(sx2), label="Re S_yy (series 2)")
    plt.xlabel("Frequency")
    plt.ylabel("Real spectrum (arb. units)")
    plt.title("Auto-spectra (real parts)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
