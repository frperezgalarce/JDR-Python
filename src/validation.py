"""Shared matrix contract for complete, symmetric dissimilarities."""

import numpy as np


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def validate_distance_matrix(D):
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or not len(D):
        raise ValueError("Distance matrix must be nonempty and square.")
    if not np.isfinite(D).all() or np.any(D < 0):
        raise ValueError("Distances must be finite and nonnegative.")
    if not np.allclose(D, D.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Distances must be symmetric.")
    if not np.allclose(np.diag(D), 0, rtol=0, atol=1e-12):
        raise ValueError("Distance diagonal must be zero.")
    # Normalize only already-validated floating point roundoff.
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    return D
