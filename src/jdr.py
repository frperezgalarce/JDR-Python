"""Shared-grid JDR. The returned J is a squared Euclidean dissimilarity."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.metrics import spectral_coefficients
from src.read_data import read_ogle_dat


@dataclass(frozen=True)
class JDRConfig:
    f_min: float = 0.001
    f_max: float = 1.0
    n_frequencies: int = 4097
    alpha: float = 0.5
    convention: str = "standard"
    time_origin: float = 0.0
    use_errors: bool = False

    def __post_init__(self):
        if not np.isfinite(
            [self.f_min, self.f_max, self.alpha, self.time_origin]
        ).all():
            raise ValueError("Configuration numbers must be finite.")
        if not 0 < self.f_min < self.f_max or not 0 < self.alpha < 1:
            raise ValueError("Require 0 < f_min < f_max and 0 < alpha < 1.")
        if (
            isinstance(self.n_frequencies, bool)
            or not isinstance(self.n_frequencies, (int, np.integer))
            or self.n_frequencies < 3
        ):
            raise ValueError("n_frequencies must be an integer >= 3.")
        if self.convention not in ("standard", "legacy"):
            raise ValueError("convention must be standard or legacy.")
        if self.use_errors and self.convention == "legacy":
            raise ValueError("Legacy mode does not support errors.")

    @property
    def frequencies(self):
        return np.linspace(self.f_min, self.f_max, self.n_frequencies)


def feature_vector(t, x, config=JDRConfig(), errors=None):
    f = config.frequencies
    a, b = spectral_coefficients(
        np.asarray(t, dtype=np.float64) - config.time_origin,
        x,
        2 * np.pi * f,
        convention=config.convention,
        errors=errors if config.use_errors else None,
    )
    weights = np.full(len(f), (config.f_max - config.f_min) / (len(f) - 1))
    weights[[0, -1]] *= 0.5
    scale = np.sqrt(config.alpha * (1 - config.alpha) / (4 * np.pi) * weights)
    return np.concatenate([scale * a, scale * b])


def _file_feature(args):
    path, config = args
    df = read_ogle_dat(path)
    if config.use_errors and "mag_err" not in df:
        raise ValueError(f"{path}: use_errors=True requires mag_err.")
    return feature_vector(df.time, df.mag, config, df.get("mag_err"))


def build_distance_matrix(files, config=JDRConfig(), n_jobs=1):
    """Read/transform each curve once, with one shared configuration for all pairs."""
    files = list(files)
    if not files:
        raise ValueError("No light curves supplied.")
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs < 1
    ):
        raise ValueError("n_jobs must be a positive integer.")
    args = [(f, config) for f in files]
    if n_jobs == 1:
        z = np.array([_file_feature(a) for a in args])
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            z = np.array(list(pool.map(_file_feature, args)))
    D = squareform(pdist(z, metric="sqeuclidean"))
    return D


def jdr(file1, file2, alpha=0.5, delta_f=0.001, *, config=None):
    """Pair interface. Defaults use fixed [delta_f,1] cycles/time, never pair cadence.

    For a scientifically chosen band, pass JDRConfig explicitly. Grid resolution
    must be checked for the observation baseline; delta_f is the lower bound.
    """
    cfg = config if config is not None else JDRConfig(alpha=alpha, f_min=delta_f)
    return float(build_distance_matrix([file1, file2], cfg)[0, 1])


def jdr_parallel(file1, file2, alpha=0.5, delta_f=0.001, n_jobs=3, *, config=None):
    cfg = config if config is not None else JDRConfig(alpha=alpha, f_min=delta_f)
    return float(build_distance_matrix([file1, file2], cfg, n_jobs=n_jobs)[0, 1])


def save_distance_matrix(path, D, files, config, extra=None):
    from src.validation import validate_distance_matrix

    D = validate_distance_matrix(D)
    files = [Path(f) for f in files]
    if len(files) != len(D) or len({f.stem for f in files}) != len(files):
        raise ValueError("Matrix rows require unique, matching object IDs.")
    manifest = {
        "version": 2,
        "distance": "squared_jdr",
        "config": asdict(config),
        "files": [str(f.resolve()) for f in files],
        "object_ids": [f.stem for f in files],
        "sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f in files],
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, D=D, manifest=json.dumps(manifest))


def load_distance_matrix(path, files, config, expected_extra=None):
    from src.validation import validate_distance_matrix

    with np.load(path, allow_pickle=False) as archive:
        D = validate_distance_matrix(archive["D"])
        m = json.loads(str(archive["manifest"]))
    files = [Path(f) for f in files]
    if (
        m.get("version") != 2
        or m["config"] != asdict(config)
        or m.get("extra", {}) != (expected_extra or {})
        or m["object_ids"] != [f.stem for f in files]
        or len(D) != len(files)
        or m["sha256"] != [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    ):
        raise ValueError(
            "Cached distance provenance does not match inputs/configuration."
        )
    return D, m
