"""Deterministic, uncertainty-weighted period estimation and periodic shape fitting."""

from dataclasses import dataclass
import numpy as np
from astropy.timeseries import LombScargle
from src.read_data import read_ogle_dat


@dataclass(frozen=True)
class FoldConfig:
    min_period: float = 0.2
    max_period: float = 2.0
    samples_per_peak: int = 5
    harmonics: int = 4
    phase_points: int = 256
    max_frequencies: int = 500000

    def __post_init__(self):
        if (
            not np.isfinite([self.min_period, self.max_period]).all()
            or not 0 < self.min_period < self.max_period
        ):
            raise ValueError("Require 0 < min_period < max_period.")
        for key in ("samples_per_peak", "harmonics", "phase_points", "max_frequencies"):
            val = getattr(self, key)
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, np.integer))
                or val < 1
            ):
                raise ValueError(f"{key} must be a positive integer.")
        if self.phase_points < 8 * self.harmonics:
            raise ValueError("phase_points must resolve the harmonic model.")


def harmonic_design(phase, harmonics):
    phase = np.asarray(phase)
    return np.column_stack(
        [np.ones(len(phase))]
        + [
            f(2 * np.pi * k * phase)
            for k in range(1, harmonics + 1)
            for f in (np.cos, np.sin)
        ]
    )


def _harmonic_fit(t, x, err, period, order):
    phase = ((t - t.min()) / period) % 1.0
    A = harmonic_design(phase, order)
    scale = np.ones(len(t)) if err is None else np.min(err) / err
    coef, _, rank, _ = np.linalg.lstsq(A * scale[:, None], x * scale, rcond=None)
    if rank != A.shape[1]:
        raise ValueError("Rank-deficient periodic fit.")
    residual = x - A @ coef
    # Weighted residual; the caller applies a separate BIC complexity penalty.
    loss = float(np.sum((residual * scale) ** 2))
    return coef, loss


def fold_light_curve(path, config=FoldConfig()):
    df = read_ogle_dat(path)
    t = df.time.to_numpy()
    x = df.mag.to_numpy()
    err = df.mag_err.to_numpy() if "mag_err" in df else None
    if len(t) <= 2 * config.harmonics + 2:
        raise ValueError("Too few observations for the requested harmonic order.")
    t = (
        t - t.min()
    )  # per-object epoch is intentional; fitted shape is canonically aligned below.
    lo, hi = 1 / config.max_period, 1 / config.min_period
    count = max(32, int(np.ceil((hi - lo) * np.ptp(t) * config.samples_per_peak)) + 1)
    if count > config.max_frequencies:
        raise ValueError(
            "Period grid exceeds max_frequencies; configure a narrower physical range."
        )
    freq = np.linspace(lo, hi, count)
    power = LombScargle(t, x, dy=err, fit_mean=True).power(freq)
    if not np.isfinite(power).all():
        raise ValueError("Non-finite periodogram.")
    # Refine several distinct local peaks; include half/double-period alternatives.
    from scipy.signal import find_peaks

    peaks = np.unique(np.r_[find_peaks(power)[0], np.argmax(power)])
    peaks = peaks[np.argsort(power[peaks])[-5:]]
    from scipy.optimize import minimize_scalar

    candidates = []
    for index in peaks:
        a, b = freq[max(0, index - 1)], freq[min(len(freq) - 1, index + 1)]
        fit = minimize_scalar(
            lambda f: -float(LombScargle(t, x, dy=err).power(f)),
            bounds=(a, b),
            method="bounded",
            options={"xatol": max(1e-12, (b - a) * 1e-5)},
        )
        for p in (1 / fit.x, 0.5 / fit.x, 2 / fit.x):
            if config.min_period <= p <= config.max_period:
                # Select harmonic complexity with BIC. A doubled period must not
                # win merely by spending extra harmonics on the same waveform.
                center = 1 / p
                half = freq[1] - freq[0]
                bounds = (max(lo, center - half), min(hi, center + half))
                for order in range(1, config.harmonics + 1):
                    optimum = minimize_scalar(
                        lambda f: _harmonic_fit(t, x, err, 1 / f, order)[1],
                        bounds=bounds,
                        method="bounded",
                        options={"xatol": 1e-10},
                    )
                    period = 1 / optimum.x
                    coef, loss = _harmonic_fit(t, x, err, period, order)
                    bic = len(t) * np.log(max(loss / len(t), np.finfo(float).tiny)) + (
                        2 * order + 2
                    ) * np.log(len(t))
                    candidates.append((float(bic), float(period), coef, loss, order))
    candidates.sort(key=lambda v: v[0])
    bic, period, coef, loss, order = candidates[0]
    # Use the smooth fitted brightest phase, not the most extreme observed datum.
    fine = np.arange(4096) / 4096
    shift = float(fine[np.argmin(harmonic_design(fine, order) @ coef)])
    phase = np.arange(config.phase_points) / config.phase_points
    shape = harmonic_design((phase + shift) % 1, order) @ coef
    diagnostics = {
        "period": float(period),
        "weighted_residual": loss,
        "phase_shift": shift,
        "observations": len(t),
        "period_grid_size": len(freq),
        "candidate_periods": [float(c[1]) for c in candidates[:5]],
        "amplitude": float(np.ptp(shape)),
        "harmonics": order,
        "bic": bic,
    }
    return phase, shape, diagnostics
