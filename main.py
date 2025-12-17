"""
Spectral and cross-spectral analysis for irregularly sampled light curves
Python translation of the provided R workflow.

Assumptions:
- The original R function cross.spectrum likely expects the integration variable
  (frequency or angular frequency) as its first argument when used with integrate().
- We implement a practical irregular-sampling cross-spectrum estimator.

Files:
- OGLE-style .dat with columns: time, magnitude, magnitude error (optional)

Dependencies:
- numpy, pandas, scipy, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad


# -----------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------

def read_ogle_dat(path: str) -> pd.DataFrame:
    """
    Read an OGLE .dat file robustly.
    Expected columns:
        col1: time
        col2: magnitude
        col3: magnitude error (optional)
    """
    # Most OGLE light-curve files are whitespace-separated.
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    # Ensure at least two columns exist
    if df.shape[1] < 2:
        raise ValueError(f"File {path} does not appear to have at least 2 columns.")
    return df


# -----------------------------------------------------------
# Core spectral estimators (irregular sampling)
# -----------------------------------------------------------

def cross_spectrum_irregular(omega, t1, x1, t2, x2, demean=True, normalize=True):
    """
    Estimate cross-spectrum S_xy(omega) for irregularly sampled data.

    This is a cross-periodogram-like estimator:
        S_xy(ω) ≈ (1/(N1 N2)) Σ_i Σ_j x1_i x2_j * exp(-i ω (t1_i - t2_j))

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency grid (rad/time).
    t1, x1 : array-like
        Times and values for series 1.
    t2, x2 : array-like
        Times and values for series 2.
    demean : bool
        Remove mean from x1 and x2.
    normalize : bool
        Scale by (N1*N2). This keeps values in a stable numeric range.

    Returns
    -------
    np.ndarray or complex
        Complex cross-spectrum estimates at omega.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if demean:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)

    N1 = len(x1)
    N2 = len(x2)
    if N1 == 0 or N2 == 0:
        raise ValueError("Empty time series provided to cross_spectrum_irregular.")

    # Pairwise time differences matrix: shape (N1, N2)
    dt = t1[:, None] - t2[None, :]

    # Outer product of amplitudes
    amp = x1[:, None] * x2[None, :]

    # Vectorize over omega
    omega_arr = np.atleast_1d(omega).astype(float)
    out = np.empty_like(omega_arr, dtype=np.complex128)

    # Compute exp(-i ω Δt) for each ω
    # For moderate N, this is fine. For very large N, consider more efficient methods.
    for k, w in enumerate(omega_arr):
        out[k] = np.sum(amp * np.exp(-1j * w * dt))

    if normalize:
        out = out / (N1 * N2)

    # Return scalar if scalar input
    if np.ndim(omega) == 0:
        return out[0]
    return out


def auto_spectrum_irregular(omega, t, x, **kwargs):
    """
    Auto-spectrum S_xx(omega) computed as cross-spectrum of a series with itself.
    """
    return cross_spectrum_irregular(omega, t, x, t, x, **kwargs)


# -----------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------

def integrate_cross_spectrum_real(
    f_lower, f_upper, t1, x1, t2, x2, demean=True, normalize=True
):
    """
    Numerically integrate the *real part* of S_xy over frequency f
    with omega = 2π f.

    The R code uses integrate(cross.spectrum, lower=f.min, upper=f.max, ...).
    Since SciPy quad expects a real-valued integrand, we integrate Re{S_xy}.
    """
    def integrand(f):
        w = 2.0 * np.pi * f
        val = cross_spectrum_irregular(
            w, t1, x1, t2, x2, demean=demean, normalize=normalize
        )
        return float(np.real(val))

    value, err = quad(integrand, f_lower, f_upper, limit=200)
    return value


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
