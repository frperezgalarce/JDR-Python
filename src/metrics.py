"""Spectral coefficients with explicit standard and historical conventions.

Standard mode fits intercept + cosine + sine in a common time coordinate.
Legacy mode reproduces the supplied R/Fortran tau and normalized projections.
Neither mode silently interpolates, changes precision, or accepts failed quadrature.
"""

from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from src.read_data import read_ogle_dat


def validate_series(t, x):
    t, x = np.asarray(t, dtype=np.float64), np.asarray(x, dtype=np.float64)
    if t.ndim != 1 or x.shape != t.shape or len(t) < 3:
        raise ValueError(
            "time and values must be matching 1D arrays with at least three observations."
        )
    if not np.isfinite(t).all() or not np.isfinite(x).all():
        raise ValueError("Observations must be finite.")
    if np.unique(t).size != len(t):
        raise ValueError("Observation times must be distinct.")
    return t, x


def _frequencies(omega):
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Angular frequencies must be finite and positive.")
    return w


def tau(t, omega, convention="standard"):
    t = np.asarray(t, dtype=float)
    w = _frequencies(omega)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all():
        raise ValueError("Invalid observation times.")
    angles = 2 * w[:, None] * t
    s, c = np.sin(angles).sum(axis=1), np.cos(angles).sum(axis=1)
    if convention == "standard":
        values = np.arctan2(s, c) / (2 * w)
    elif convention == "legacy":
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.arctan(s / c) / (2 * np.pi)
        if not np.isfinite(values).all():
            raise ValueError("Degenerate legacy tau.")
    else:
        raise ValueError("convention must be standard or legacy.")
    return float(values[0]) if np.ndim(omega) == 0 else values


def periodogram_component(t, x, omega, tau_val, sincos):
    t, x = validate_series(t, x)
    w = _frequencies(omega)
    shift = np.broadcast_to(np.atleast_1d(tau_val), w.shape)
    if sincos not in ("sin", "cos"):
        raise ValueError("sincos must be sin or cos.")
    trig = (np.sin if sincos == "sin" else np.cos)((shift[:, None] - t) * w[:, None])
    norm = np.linalg.norm(trig, axis=1)
    if np.any(norm <= np.finfo(float).eps):
        raise ValueError("Degenerate trigonometric basis.")
    out = trig @ x / norm
    return float(out[0]) if np.ndim(omega) == 0 else out


def spectral_coefficients(
    t, x, omega, *, convention="standard", demean=True, normalize=True, errors=None
):
    """Return (cos, sin) arrays in a common phase basis (standard mode).

    Weighted least squares includes a floating intercept when demean=True.
    Normalization uses weighted population variance (standard) or sample SD (legacy).
    Coefficients are amplitudes, so observation count does not directly rescale them.
    """
    t, x = validate_series(t, x)
    w = _frequencies(omega)
    if convention not in ("standard", "legacy"):
        raise ValueError("convention must be standard or legacy.")
    weights = np.ones(len(t))
    if errors is not None:
        errors = np.asarray(errors, float)
        if (
            errors.shape != t.shape
            or not np.isfinite(errors).all()
            or np.any(errors <= 0)
        ):
            raise ValueError("errors must be finite, positive and match observations.")
        if convention == "legacy":
            raise ValueError(
                "Legacy convention does not support measurement weighting."
            )
        weights = (np.min(errors) / errors) ** 2
    mean = np.average(x, weights=weights)
    y = x - mean if demean else x.copy()
    if normalize:
        sd = (
            np.std(y, ddof=1)
            if convention == "legacy"
            else np.sqrt(np.average((x - mean) ** 2, weights=weights))
        )
        if not np.isfinite(sd) or sd <= np.finfo(float).eps * max(
            1.0, np.max(np.abs(x))
        ):
            raise ValueError("Cannot normalize a constant signal.")
        y = y / sd
    if convention == "legacy":
        shifts = tau(t, w, convention="legacy")
        return (
            periodogram_component(t, y, w, shifts, "cos"),
            periodogram_component(t, y, w, shifts, "sin"),
        )
    # Batched weighted normal equations; rank/conditioning is checked before solving.
    out = np.empty((len(w), 2))
    for start in range(0, len(w), 512):
        phase = w[start : start + 512, None] * t
        cols = [np.cos(phase), np.sin(phase)]
        if demean:
            cols.insert(0, np.ones_like(phase))
        design = np.stack(cols, axis=-1)
        gram = np.einsum("fni,fnj,n->fij", design, design, weights)
        rhs = np.einsum("fni,n,n->fi", design, y, weights)
        if np.any(np.linalg.cond(gram) > 1e12):
            raise ValueError(
                "Ill-conditioned spectral fit; revise frequency band or sampling."
            )
        coef = np.linalg.solve(gram, rhs[..., None])[..., 0]
        out[start : start + len(coef)] = coef[:, -2:]
    return out[:, 0], out[:, 1]


def cross_spectrum_irregular(
    omega, t1, x1, t2, x2, demean=True, normalize=True, convention="standard"
):
    a, b = spectral_coefficients(
        t1, x1, omega, convention=convention, demean=demean, normalize=normalize
    )
    c, d = spectral_coefficients(
        t2, x2, omega, convention=convention, demean=demean, normalize=normalize
    )
    values = 0.5 * ((a * c + b * d) + 1j * (a * d - b * c))
    return complex(values[0]) if np.ndim(omega) == 0 else values


def auto_spectrum_irregular(omega, t, x, **kwargs):
    return cross_spectrum_irregular(omega, t, x, t, x, **kwargs)


def cospectrum(
    t1, x1, t2, x2, omega, demeaned=False, standardized=False, convention="standard"
):
    return np.real(
        cross_spectrum_irregular(
            omega, t1, x1, t2, x2, not demeaned, not standardized, convention
        )
    )


def quadrature_spectrum(
    t1, x1, t2, x2, omega, demeaned=False, standardized=False, convention="standard"
):
    return np.imag(
        cross_spectrum_irregular(
            omega, t1, x1, t2, x2, not demeaned, not standardized, convention
        )
    )


def _integration_options(lo, hi, jacobian, limit):
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError("Require 0 < f_lower < f_upper, both finite.")
    if jacobian not in ("df", "domega"):
        raise ValueError("jacobian must be df or domega.")
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("quad_limit must be a positive integer.")
    return 1.0 if jacobian == "df" else 2 * np.pi


def integrate_cross_spectrum_real(
    f_lower,
    f_upper,
    t1,
    x1,
    t2,
    x2,
    demean=True,
    normalize=True,
    jacobian="df",
    quad_limit=500,
    convention="standard",
    rel_tol=1e-8,
    abs_tol=1e-10,
):
    factor = _integration_options(f_lower, f_upper, jacobian, quad_limit)
    if not np.isfinite([rel_tol, abs_tol]).all() or min(rel_tol, abs_tol) <= 0:
        raise ValueError("Tolerances must be finite and positive.")

    def integrand(f):
        return factor * float(
            np.real(
                cross_spectrum_irregular(
                    2 * np.pi * f, t1, x1, t2, x2, demean, normalize, convention
                )
            )
        )

    result = quad(
        integrand,
        f_lower,
        f_upper,
        limit=quad_limit,
        epsabs=abs_tol,
        epsrel=rel_tol,
        full_output=1,
    )
    value, error = result[:2]
    if (
        len(result) > 3
        or not np.isfinite([value, error]).all()
        or error > max(abs_tol, rel_tol * abs(value))
    ):
        raise RuntimeError(
            f"Quadrature did not converge: {result[3] if len(result)>3 else error}"
        )
    return float(value)


def _rscript_integrate_cospec(
    f_lower,
    f_upper,
    t1,
    x1,
    t2,
    x2,
    *,
    demean,
    normalize,
    factor,
    rel_tol,
    abs_tol,
    subdivisions,
    rscript_exe,
    convention="standard",
):
    """Independent R reference; standard fits use QR least squares, not normal equations."""
    if convention not in ("standard", "legacy"):
        raise ValueError("convention must be standard or legacy.")
    t1, x1 = validate_series(t1, x1)
    t2, x2 = validate_series(t2, x2)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        np.savetxt(root / "a.csv", np.c_[t1, x1], delimiter=",")
        np.savetxt(root / "b.csv", np.c_[t2, x2], delimiter=",")
        code = f"""
        a <- as.matrix(read.csv("{root.as_posix()}/a.csv", header=FALSE))
        b <- as.matrix(read.csv("{root.as_posix()}/b.csv", header=FALSE))
        coeff <- function(v,w) {{
          t <- v[,1]; x <- v[,2]
          if ({str(demean).upper()}) x <- x-mean(x)
          if ({str(normalize).upper()}) {{
            scale <- if ("{convention}"=="legacy") sd(x) else sqrt(mean((x-mean(x))^2))
            if (!is.finite(scale) || scale<=.Machine$double.eps) stop("constant signal")
            x <- x/scale
          }}
          if ("{convention}"=="legacy") {{
            shift <- atan(sum(sin(2*w*t))/sum(cos(2*w*t)))/(2*pi)
            c <- cos(w*(shift-t)); s <- sin(w*(shift-t))
            return(c(sum(c*x)/sqrt(sum(c*c)),sum(s*x)/sqrt(sum(s*s))))
          }}
          X <- cbind(cos(w*t),sin(w*t))
          if ({str(demean).upper()}) X <- cbind(1,X)
          fit <- qr.solve(X,x)
          tail(fit,2)
        }}
        fun <- function(f) sapply(f,function(v) {{
          x <- coeff(a,2*pi*v); y <- coeff(b,2*pi*v)
          {factor}*.5*sum(x*y)
        }})
        res <- tryCatch(integrate(fun,{f_lower},{f_upper},rel.tol={rel_tol},abs.tol={abs_tol},
                         subdivisions={subdivisions},stop.on.error=FALSE),error=function(e)e)
        if (inherits(res,"error")) cat("error\\tNaN\\t",conditionMessage(res),sep="") else {{
          status <- if (res$message=="OK" && is.finite(res$value) &&
                        res$abs.error<=max({abs_tol},{rel_tol}*abs(res$value))) "ok" else "error"
          cat(status,"\\t",format(res$value,digits=17),"\\t",res$message,sep="")
        }}
        """
        (root / "reference.R").write_text(code)
        proc = subprocess.run(
            [rscript_exe, str(root / "reference.R")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode:
            raise RuntimeError(f"Rscript failed: {proc.stderr}")
        fields = proc.stdout.strip().split("\t", 2)
        if len(fields) != 3:
            raise RuntimeError(f"Unexpected R result: {proc.stdout}")
        return float(fields[1]), fields[0], fields[2]


def integrate_cross_spectrum_real_R(
    f_lower,
    f_upper,
    t1,
    x1,
    t2,
    x2,
    demean=True,
    normalize=True,
    jacobian="df",
    quad_limit=5000,
    rel_tol=1e-8,
    abs_tol=1e-10,
    rscript_exe=None,
    fallback_to_scipy=False,
    convention="standard",
):
    factor = _integration_options(f_lower, f_upper, jacobian, quad_limit)
    if not np.isfinite([rel_tol, abs_tol]).all() or min(rel_tol, abs_tol) <= 0:
        raise ValueError("Tolerances must be finite and positive.")
    msg = ""
    for multiplier in (1, 2, 4):
        try:
            value, status, msg = _rscript_integrate_cospec(
                f_lower,
                f_upper,
                t1,
                x1,
                t2,
                x2,
                demean=demean,
                normalize=normalize,
                factor=factor,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                subdivisions=quad_limit * multiplier,
                rscript_exe=rscript_exe
                or os.environ.get("METRICS_RSCRIPT_EXE", "Rscript"),
                convention=convention,
            )
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            msg = str(exc)
            break
        if status == "ok" and msg.strip() == "OK" and np.isfinite(value):
            return value
    if fallback_to_scipy:
        return integrate_cross_spectrum_real(
            f_lower,
            f_upper,
            t1,
            x1,
            t2,
            x2,
            demean,
            normalize,
            jacobian,
            quad_limit,
            convention,
            rel_tol,
            abs_tol,
        )
    raise RuntimeError(f"R integration failed: {msg}")
