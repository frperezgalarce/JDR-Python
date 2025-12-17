"""
metrics.py

Spectral and cross-spectral analysis for irregularly sampled light curves.

This module is a *numerically aligned* Python translation of the provided R workflow:
- tau() from tau.r (Fortran-consistent branch: atan(sin.sum/cos.sum) / (2*pi))
- periodogram.component() from periodogram.component.r
- cross.spectrum() (co-spectrum) as in cross.spectrum.r:
      co(omega) = 0.5 * (a1*a2 + b1*b2)
  where a,b are the normalized cosine/sine periodogram components.

Notes
-----
1) The original attached module implemented a different cross-spectrum estimator
   based on a double-sum over pairwise lags. This replacement follows the R code
   you provided (tau + normalized projections).

2) Time units:
   - omega must be angular frequency in rad / (time-unit of t).
   - If you integrate over cyclic frequency f, use omega = 2*pi*f.

3) Standardization:
   - The R code uses sample SD: sd(x) => numpy.std(ddof=1).

Files
-----
- OGLE-style .dat files: whitespace-separated with columns:
    time  magnitude  [magnitude_error]

Dependencies
------------
- numpy, pandas, scipy

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import quad


ArrayLike = Union[np.ndarray, list, tuple]


# -----------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------

def read_ogle_dat(path: str) -> pd.DataFrame:
    """
    Read an OGLE-style .dat light curve file robustly.

    Supports:
      - 2 columns: time, mag
      - 3 columns: time, mag, mag_err

    Returns
    -------
    pd.DataFrame with columns: time, mag [, mag_err]
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    ncol = df.shape[1]
    if ncol == 2:
        df.columns = ["time", "mag"]
    elif ncol == 3:
        df.columns = ["time", "mag", "mag_err"]
    else:
        raise ValueError(f"{path}: expected 2 or 3 columns, found {ncol}.")

    # Enforce numeric and drop invalid rows
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df


# -----------------------------------------------------------
# Core estimators aligned with the R workflow
# -----------------------------------------------------------

def tau(t: ArrayLike, omega: Union[float, ArrayLike]) -> Union[float, np.ndarray]:
    """
    Compute the time-shift parameter tau, aligned with the provided R function:

        two.omega <- 2 * omega
        sin.sum <- rowSums(sin(two.omega %o% t))
        cos.sum <- rowSums(cos(two.omega %o% t))
        two.omega.tau <- atan(sin.sum / cos.sum)          # NOT atan2
        tau <- two.omega.tau / (2*pi)

    Parameters
    ----------
    t : array-like, shape (N,)
        Sampling times.
    omega : float or array-like, shape (M,)
        Angular frequency (rad / time-unit).

    Returns
    -------
    float or np.ndarray
        tau value(s). Scalar in -> scalar out.
    """
    t = np.asarray(t, dtype=float)
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))

    two_omega = 2.0 * omega_arr                       # (M,)
    two_omega_t = two_omega[:, None] * t[None, :]     # (M, N)

    sin_sum = np.sum(np.sin(two_omega_t), axis=1)     # (M,)
    cos_sum = np.sum(np.cos(two_omega_t), axis=1)     # (M,)

    # Match R/Fortran branch exactly: atan(sin_sum / cos_sum)
    ratio = sin_sum / cos_sum                         # may be inf if cos_sum==0, like R
    two_omega_tau = np.arctan(ratio)

    tau_vals = two_omega_tau / (2.0 * np.pi)

    if np.ndim(omega) == 0:
        return float(tau_vals[0])
    return tau_vals


def periodogram_component(
    t: ArrayLike,
    x: ArrayLike,
    omega: Union[float, ArrayLike],
    tau_val: Union[float, ArrayLike],
    sincos: Literal["sin", "cos"],
) -> Union[float, np.ndarray]:
    """
    Compute one component of the periodogram, aligned with the provided R function:

        t.minus.tau <- outer(tau, t, FUN = `-`)
        sincos.vec <- sincos.fn(t.minus.tau * omega)
        (sincos.vec %*% x) / sqrt(rowSums(sincos.vec^2))

    Parameters
    ----------
    t : array-like, shape (N,)
    x : array-like, shape (N,)
        Series values (expected to be demeaned/standardized by the caller if desired).
    omega : float or array-like, shape (M,)
    tau_val : float or array-like, shape (M,)
    sincos : "sin" or "cos"

    Returns
    -------
    float or np.ndarray
        Component value(s). Scalar in -> scalar out.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    tau_arr = np.atleast_1d(np.asarray(tau_val, dtype=float))

    if sincos not in ("sin", "cos"):
        raise ValueError("sincos must be 'sin' or 'cos'.")

    # R: outer(tau, t, FUN='-') => tau[m] - t[n]
    t_minus_tau = tau_arr[:, None] - t[None, :]               # (M, N)
    arg = t_minus_tau * omega_arr[:, None]                    # (M, N)

    trig = np.sin(arg) if sincos == "sin" else np.cos(arg)    # (M, N)

    num = trig @ x                                            # (M,)
    denom = np.sqrt(np.sum(trig**2, axis=1))                  # (M,)

    comp = num / denom

    if np.ndim(omega) == 0:
        return float(comp[0])
    return comp


def cospectrum(
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
    omega: Union[float, ArrayLike],
    demeaned: bool = False,
    standardized: bool = False,
) -> Union[float, np.ndarray]:
    """
    Co-spectrum (real cross-spectrum) aligned with the provided R cross.spectrum():

        if (!demeaned)     x <- x - mean(x)
        if (!standardized) x <- x / sd(x)    # sample sd
        tau1 <- tau(t1, omega)
        tau2 <- tau(t2, omega)
        a <- periodogram.component(..., "cos")
        b <- periodogram.component(..., "sin")
        co <- 0.5 * (a1*a2 + b1*b2)

    Returns
    -------
    float or np.ndarray
        co-spectrum at omega.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if not demeaned:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)

    if not standardized:
        x1 = x1 / np.std(x1, ddof=1)
        x2 = x2 / np.std(x2, ddof=1)


    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))

    tau1 = tau(t1, omega_arr)
    tau2 = tau(t2, omega_arr)

    a1 = periodogram_component(t1, x1, omega_arr, tau1, "cos")
    a2 = periodogram_component(t2, x2, omega_arr, tau2, "cos")
    b1 = periodogram_component(t1, x1, omega_arr, tau1, "sin")
    b2 = periodogram_component(t2, x2, omega_arr, tau2, "sin")

    co = 0.5 * (a1 * a2 + b1 * b2)

    if np.ndim(omega) == 0:
        return float(np.atleast_1d(co)[0])
    return co


def quadrature_spectrum(
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
    omega: Union[float, ArrayLike],
    demeaned: bool = False,
    standardized: bool = False,
) -> Union[float, np.ndarray]:
    """
    Quadrature spectrum (imaginary counterpart) consistent with the same components:

        qu <- 0.5 * (a1*b2 - b1*a2)

    This is not present in your pasted R snippet but is the standard companion.
    """
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if not demeaned:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)

    if not standardized:
        x1 = x1 / np.std(x1, ddof=1)
        x2 = x2 / np.std(x2, ddof=1)

    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))

    tau1 = tau(t1, omega_arr)
    tau2 = tau(t2, omega_arr)

    a1 = periodogram_component(t1, x1, omega_arr, tau1, "cos")
    a2 = periodogram_component(t2, x2, omega_arr, tau2, "cos")
    b1 = periodogram_component(t1, x1, omega_arr, tau1, "sin")
    b2 = periodogram_component(t2, x2, omega_arr, tau2, "sin")

    qu = 0.5 * (a1 * b2 - b1 * a2)

    if np.ndim(omega) == 0:
        return float(np.atleast_1d(qu)[0])
    return qu


def cross_spectrum_irregular(
    omega: Union[float, ArrayLike],
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
    demean: bool = True,
    normalize: bool = True,
) -> Union[complex, np.ndarray]:
    """
    Backward-compatible entry point matching the previous module's signature.

    IMPORTANT:
    - To align with the R workflow, this function now returns the *complex*
      cross-spectrum S_xy(omega) = co(omega) + i*qu(omega), computed from the
      R-aligned co-spectrum and quadrature spectrum definitions.
    - The previous implementation used a different estimator (pairwise-lag double sum).

    Parameters
    ----------
    omega : float or array-like
        Angular frequency (rad/time-unit).
    demean : bool
        If True, demean inside this function (R does this when demeaned=FALSE).
    normalize : bool
        If True, standardize (divide by sample SD) inside this function (R does this when standardized=FALSE).

    Returns
    -------
    complex or np.ndarray of complex
    """
    # Map flags to R semantics
    demeaned = not demean
    standardized = not normalize

    co = cospectrum(t1, x1, t2, x2, omega, demeaned=demeaned, standardized=standardized)
    qu = quadrature_spectrum(t1, x1, t2, x2, omega, demeaned=demeaned, standardized=standardized)

    return co + 1j * qu


def auto_spectrum_irregular(
    omega: Union[float, ArrayLike],
    t: ArrayLike,
    x: ArrayLike,
    demean: bool = True,
    normalize: bool = True,
) -> Union[complex, np.ndarray]:
    """
    Auto-spectrum computed as cross-spectrum of a series with itself (complex form).
    The real part corresponds to the co-spectrum; imaginary part should be ~0.
    """
    return cross_spectrum_irregular(omega, t, x, t, x, demean=demean, normalize=normalize)


# -----------------------------------------------------------
# Integration helpers
# -----------------------------------------------------------

def integrate_cross_spectrum_real(
    f_lower: float,
    f_upper: float,
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
    demean: bool = True,
    normalize: bool = True,
    jacobian: Literal["df", "domega"] = "df",
    quad_limit: int = 200,
) -> float:
    """
    Integrate the real part of the cross-spectrum over a frequency band.

    Uses omega = 2*pi*f and integrates Re{S_xy(omega)}.

    Parameters
    ----------
    f_lower, f_upper : float
        Frequency bounds in cycles per time-unit.
    jacobian : {"df","domega"}
        - "df": computes ∫ Re(S_xy(2πf)) df
        - "domega": computes ∫ Re(S_xy(ω)) dω, but using f as the variable, i.e.
                   ∫ Re(S_xy(2πf)) * (2π) df
    """
    factor = 1.0 if jacobian == "df" else 2.0 * np.pi

    def integrand(f):
        w = 2.0 * np.pi * f
        val = cross_spectrum_irregular(w, t1, x1, t2, x2, demean=demean, normalize=normalize)
        return float(np.real(val)) * factor

    value, _err = quad(integrand, f_lower, f_upper, limit=quad_limit, epsabs=1e-10, epsrel=1e-10)
    return float(value)
