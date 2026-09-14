"""Paper Eq. (1)/(7), with explicitly clarified complex-amplitude cross-spectrum.

No floating-intercept least-squares amplitude substitution. Frequencies are angular;
the default integration interval is [0, 2*pi]. Inputs are centered before projection.
The nonnegative J is squared Euclidean; sqrt(J), not J, is a metric.
"""

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.spatial.distance import pdist, squareform
from src.metrics import validate_series


@dataclass(frozen=True)
class PaperJDRConfig:
    n_frequencies: int = 4097
    alpha: float = 0.5
    normalize: bool = True

    def __post_init__(self):
        if (
            isinstance(self.n_frequencies, bool)
            or not isinstance(self.n_frequencies, (int, np.integer))
            or self.n_frequencies < 3
        ):
            raise ValueError("Need at least three frequency points.")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("Require 0 < alpha < 1.")

    @property
    def omega(self):
        return np.linspace(0, 2 * np.pi, self.n_frequencies)


def paper_amplitude(t, x, omega, normalize=True):
    """A=(C+iS) exp(i*omega*tau)/sqrt(2), so |A|^2 equals paper Eq. (1).

    C,S are projections onto unit-norm shifted cosine/sine vectors, NOT fitted
    sinusoid amplitudes. Rotation restores a shared phase basis across samplings.
    atan2 selects a branch robustly; the rotated amplitude is branch invariant.
    omega=0 uses the continuous right-hand limit for centered input.
    """
    t, x = validate_series(t, x)
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Angular frequencies must be finite and nonnegative.")
    y = x - x.mean()
    sd = np.sqrt(np.mean(y * y))
    if sd <= np.finfo(float).eps:
        raise ValueError("Constant or numerically degenerate input.")
    if normalize:
        y = y / sd
    out = np.empty(len(w), dtype=complex)
    zero = w == 0
    centered = t - t.mean()
    out[zero] = 1j * np.dot(y, centered) / np.linalg.norm(centered) / np.sqrt(2)
    for start in range(0, len(w), 1024):
        indices = np.arange(start, min(start + 1024, len(w)))
        indices = indices[~zero[indices]]
        if not len(indices):
            continue
        angles = w[indices, None] * t
        theta = 0.5 * np.arctan2(
            np.sin(2 * angles).sum(axis=1), np.cos(2 * angles).sum(axis=1)
        )
        c, s = np.cos(angles - theta[:, None]), np.sin(angles - theta[:, None])
        nc, ns = np.linalg.norm(c, axis=1), np.linalg.norm(s, axis=1)
        if (nc == 0).any() or (ns == 0).any():
            raise ValueError("Degenerate trigonometric projection.")
        C, S = (c @ y) / nc, (s @ y) / ns
        out[indices] = (C + 1j * S) * np.exp(1j * theta) / np.sqrt(2)
    return out


def paper_features(t, x, config=PaperJDRConfig()):
    amp = paper_amplitude(t, x, config.omega, config.normalize)
    q = np.full(config.n_frequencies, 2 * np.pi / (config.n_frequencies - 1))
    q[[0, -1]] *= 0.5
    scale = np.sqrt(config.alpha * (1 - config.alpha) * q / (2 * np.pi))
    return np.r_[amp.real * scale, amp.imag * scale]


def paper_matrix(curves, config=PaperJDRConfig()):
    z = np.array([paper_features(t, x, config) for t, x in curves])
    return squareform(pdist(z, "sqeuclidean")), z


def scalar_amplitude_reference(t, x, w, normalize=True):
    """Scalar reference using the printed arctan ratio, independently of vector code."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    y = x - x.mean()
    if normalize:
        y = y / np.sqrt(np.mean(y * y))
    if w == 0:
        centered = t - t.mean()
        return 1j * np.sum(y * centered) / np.sqrt(np.sum(centered**2)) / np.sqrt(2)
    numerator = np.sum(np.sin(2 * w * t))
    denominator = np.sum(np.cos(2 * w * t))
    theta = (
        0.5 * np.arctan(numerator / denominator)
        if denominator != 0
        else np.sign(numerator) * np.pi / 4
    )
    c = np.cos(w * t - theta)
    s = np.sin(w * t - theta)
    C = np.sum(y * c) / np.sqrt(np.sum(c * c))
    S = np.sum(y * s) / np.sqrt(np.sum(s * s))
    return complex(C, S) * np.exp(1j * theta) / np.sqrt(2)


def adaptive_three_integrals(first, second, alpha=0.5):
    """Scalar-reference quadrature in d-omega, with explicit convergence failure."""

    def integrand(w, kind):
        a = scalar_amplitude_reference(*first, w)
        b = scalar_amplitude_reference(*second, w)
        return float(
            abs(a) ** 2
            if kind == "xx"
            else abs(b) ** 2 if kind == "yy" else (a * np.conj(b)).real
        )

    values = []
    errors = []
    for kind in ["xx", "yy", "xy"]:
        result = quad(
            integrand,
            0,
            2 * np.pi,
            args=(kind,),
            epsabs=1e-8,
            epsrel=1e-9,
            limit=500,
            full_output=1,
        )
        value, error = result[:2]
        if (
            len(result) > 3
            or not np.isfinite([value, error]).all()
            or error > max(1e-8, 1e-9 * abs(value))
        ):
            raise RuntimeError(f"Independent quadrature did not converge: {result}")
        values.append(float(value))
        errors.append(float(error))
    j = alpha * (1 - alpha) / (2 * np.pi) * (values[0] + values[1] - 2 * values[2])
    return {
        "integral_xx": values[0],
        "integral_yy": values[1],
        "integral_xy_real": values[2],
        "j_adaptive": float(j),
        "quad_error_bound_j": float(
            alpha * (1 - alpha) / (2 * np.pi) * (errors[0] + errors[1] + 2 * errors[2])
        ),
    }
