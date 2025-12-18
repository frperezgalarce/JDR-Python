"""
metrics.py (compatible, robust R integration)

Changes vs previous:
- Rscript output no longer prints locale ("[1] \"C\"") because we wrap Sys.setlocale in invisible().
- R integrate: uses stop.on.error=FALSE to avoid hard failure on roundoff.
- Python wrapper retries with relaxed tolerances / higher subdivisions when R reports an error.
- If R still fails or returns non-finite, falls back to SciPy quad (so the rest of your pipeline can proceed).

You still get:
- integrate_cross_spectrum_real(...)      -> SciPy quad
- integrate_cross_spectrum_real_R(...)    -> R integrate via Rscript, with retries + fallback

Environment variables:
- METRICS_RSCRIPT_EXE : path to Rscript (default "Rscript")
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike as NpArrayLike
from scipy.integrate import quad

ArrayLike = Union[np.ndarray, list, tuple, NpArrayLike]


# -----------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------

def read_ogle_dat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    ncol = df.shape[1]
    if ncol == 2:
        df.columns = ["time", "mag"]
    elif ncol == 3:
        df.columns = ["time", "mag", "mag_err"]
    else:
        raise ValueError(f"{path}: expected 2 or 3 columns, found {ncol}.")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


# -----------------------------------------------------------
# Core estimators aligned with the R workflow
# -----------------------------------------------------------

def tau(t: ArrayLike, omega: Union[float, ArrayLike]) -> Union[float, np.ndarray]:
    t = np.asarray(t, dtype=float)
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))

    two_omega = 2.0 * omega_arr
    two_omega_t = two_omega[:, None] * t[None, :]

    sin_sum = np.sum(np.sin(two_omega_t), axis=1)
    cos_sum = np.sum(np.cos(two_omega_t), axis=1)

    ratio = sin_sum / cos_sum
    tau_vals = np.arctan(ratio) / (2.0 * np.pi)

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
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    tau_arr = np.atleast_1d(np.asarray(tau_val, dtype=float))

    if sincos not in ("sin", "cos"):
        raise ValueError("sincos must be 'sin' or 'cos'.")

    t_minus_tau = tau_arr[:, None] - t[None, :]
    arg = t_minus_tau * omega_arr[:, None]

    trig = np.sin(arg) if sincos == "sin" else np.cos(arg)

    num = trig @ x
    denom = np.sqrt(np.sum(trig**2, axis=1))
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
    return cross_spectrum_irregular(omega, t, x, t, x, demean=demean, normalize=normalize)


# -----------------------------------------------------------
# Integration backends
# -----------------------------------------------------------

_RSCRIPT_DEFAULT = os.environ.get("METRICS_RSCRIPT_EXE", "Rscript")


def _rscript_integrate_cospec(
    f_lower: float,
    f_upper: float,
    t1: np.ndarray,
    x1: np.ndarray,
    t2: np.ndarray,
    x2: np.ndarray,
    *,
    demean: bool,
    normalize: bool,
    factor: float,
    rel_tol: float,
    abs_tol: float,
    subdivisions: int,
    rscript_exe: str,
) -> tuple[float, str, str]:
    """
    Returns (value, status, message)
      status: "ok" | "error"
    """
    t1 = np.asarray(t1, float)
    x1 = np.asarray(x1, float)
    t2 = np.asarray(t2, float)
    x2 = np.asarray(x2, float)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        s1 = td / "s1.csv"
        s2 = td / "s2.csv"
        np.savetxt(s1, np.c_[t1, x1], delimiter=",")
        np.savetxt(s2, np.c_[t2, x2], delimiter=",")

        r_code = f"""
        options(warn=1)
        suppressWarnings({{
          try(invisible(Sys.setlocale("LC_ALL","C")), silent=TRUE)
        }})

        lower <- {f_lower}
        upper <- {f_upper}
        demean <- {str(demean).upper()}
        normalize <- {str(normalize).upper()}
        factor <- {factor}
        rel_tol <- {rel_tol}
        abs_tol <- {abs_tol}
        subdivisions <- {int(subdivisions)}

        s1 <- read.csv("{s1.as_posix()}", header=FALSE)
        s2 <- read.csv("{s2.as_posix()}", header=FALSE)
        t1 <- as.numeric(s1[[1]]); x1 <- as.numeric(s1[[2]])
        t2 <- as.numeric(s2[[1]]); x2 <- as.numeric(s2[[2]])

        if (demean) {{
          x1p <- x1 - mean(x1)
          x2p <- x2 - mean(x2)
        }} else {{
          x1p <- x1
          x2p <- x2
        }}
        if (normalize) {{
          x1p <- x1p / sd(x1p)
          x2p <- x2p / sd(x2p)
        }}

        tau_scalar <- function(t, omega) {{
          atan(sum(sin(2*omega*t)) / sum(cos(2*omega*t))) / (2*pi)
        }}

        pc_scalar <- function(t, x, omega, tau, sc) {{
          z <- (tau - t) * omega
          tr <- if (sc=="sin") sin(z) else cos(z)
          sum(tr*x)/sqrt(sum(tr^2))
        }}

        cospec_scalar <- function(omega) {{
          tau1 <- tau_scalar(t1, omega)
          tau2 <- tau_scalar(t2, omega)
          a1 <- pc_scalar(t1, x1p, omega, tau1, "cos")
          a2 <- pc_scalar(t2, x2p, omega, tau2, "cos")
          b1 <- pc_scalar(t1, x1p, omega, tau1, "sin")
          b2 <- pc_scalar(t2, x2p, omega, tau2, "sin")
          0.5*(a1*a2 + b1*b2)
        }}

        cospec_vec <- function(omega_vec) {{
          sapply(omega_vec, function(w) cospec_scalar(w))
        }}

        integrand <- function(f_vec) {{
          omega_vec <- 2*pi*f_vec
          factor * cospec_vec(omega_vec)
        }}

        res <- try(
          integrate(integrand, lower=lower, upper=upper,
                    rel.tol=rel_tol, abs.tol=abs_tol,
                    subdivisions=subdivisions,
                    stop.on.error=FALSE),
          silent=TRUE
        )

        if (inherits(res, "try-error")) {{
          cat("ERROR\\tNA\\t", as.character(res), "\\n", sep="")
        }} else {{
          msg <- if (!is.null(res$message)) res$message else ""
          cat("OK\\t", format(res$value, digits=17), "\\t", msg, "\\n", sep="")
        }}
        """

        proc = subprocess.run([rscript_exe, "-e", r_code], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Rscript failed.\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )

        last = proc.stdout.strip().splitlines()[-1]
        parts = last.split("\t")
        if len(parts) < 2:
            raise RuntimeError(f"Unexpected R output:\n{proc.stdout}")

        status = parts[0].strip()
        if status == "OK":
            value = float(parts[1])
            msg = parts[2] if len(parts) >= 3 else ""
            return value, "ok", msg
        else:
            # "ERROR\tNA\t<message>"
            msg = parts[2] if len(parts) >= 3 else last
            return float("nan"), "error", msg


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
    quad_limit: int = 500,
) -> float:
    factor = 1.0 if jacobian == "df" else 2.0 * np.pi

    def integrand(f: float) -> float:
        w = 2.0 * np.pi * f
        val = cross_spectrum_irregular(w, t1, x1, t2, x2, demean=demean, normalize=normalize)
        return float(np.real(val)) * factor

    value, _err = quad(integrand, f_lower, f_upper, limit=quad_limit, epsabs=1e-10, epsrel=1e-10)
    return float(value)


def integrate_cross_spectrum_real_R(
    f_lower: float,
    f_upper: float,
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
    demean: bool = True,
    normalize: bool = True,
    jacobian: Literal["df", "domega"] = "df",
    quad_limit: int = 5000,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-10,
    rscript_exe: str = _RSCRIPT_DEFAULT,
    fallback_to_scipy: bool = False,
) -> float:
    """
    R backend (Rscript). Retries on numerical issues and optionally falls back to SciPy.

    Practical behavior:
    - First attempt: your requested tolerances
    - Retry 1: relax tolerances (1e-8), increase subdivisions
    - Retry 2: relax tolerances (1e-6), increase subdivisions further
    - If still non-finite and fallback_to_scipy=True: return SciPy quad() result
    """
    factor = 1.0 if jacobian == "df" else 2.0 * np.pi

    t1a = np.asarray(t1, float)
    x1a = np.asarray(x1, float)
    t2a = np.asarray(t2, float)
    x2a = np.asarray(x2, float)

    attempts = [
        (rel_tol, abs_tol, int(quad_limit)),
        (max(rel_tol, 1e-8), max(abs_tol, 1e-8), max(int(quad_limit), 5000)),
        (max(rel_tol, 1e-6), max(abs_tol, 1e-6), max(int(quad_limit), 20000)),
    ]

    last_msg = ""
    for rt, at, sub in attempts:
        val, status, msg = _rscript_integrate_cospec(
            f_lower, f_upper, t1a, x1a, t2a, x2a,
            demean=demean, normalize=normalize,
            factor=factor,
            rel_tol=rt, abs_tol=at,
            subdivisions=sub,
            rscript_exe=rscript_exe,
        )
        last_msg = msg
        if status == "ok" and np.isfinite(val):
            return float(val)

    if fallback_to_scipy:
        print('Fallback to Scipy')
        return integrate_cross_spectrum_real(
            f_lower, f_upper, t1a, x1a, t2a, x2a,
            demean=demean, normalize=normalize,
            jacobian=jacobian,
            quad_limit=5000,
        )

    raise RuntimeError(f"R integrate failed after retries. Last message: {last_msg}")
