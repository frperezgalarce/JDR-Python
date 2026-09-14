from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from astropy.time import Time
from astropy import units as u
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from src.jdr import JDRConfig, feature_vector
from src.preprocessing import FoldConfig, fold_light_curve
from src.conversion import lightcurve_to_ogle
from src.JDRKMedoids import JDRKMedoids


def test_synthetic_class_recovery():
    rng = np.random.default_rng(22)
    z = []
    truth = []
    config = JDRConfig(f_min=0.5, f_max=4, n_frequencies=513)
    for k in range(3):
        for _ in range(8):
            t = np.sort(rng.uniform(0, 1, 150))
            x = (
                np.cos(2 * np.pi * t)
                + [0, 0.5, -0.5][k] * np.cos(4 * np.pi * t)
                + 0.03 * rng.normal(size=len(t))
            )
            z.append(feature_vector(t, x, config))
            truth.append(k)
    D = squareform(pdist(z, "sqeuclidean"))
    models = [JDRKMedoids(3, random_state=i).fit(D) for i in range(8)]
    labels = min(models, key=lambda m: m.inertia_).labels_
    assert adjusted_rand_score(truth, labels) > 0.95


def test_period_recovery_and_repeatability(tmp_path):
    rng = np.random.default_rng(11)
    t = np.sort(rng.uniform(0, 20, 180))
    period = 0.67
    x = (
        15
        + np.cos(2 * np.pi * t / period)
        + 0.4 * np.sin(4 * np.pi * t / period)
        + rng.normal(0, 0.02, len(t))
    )
    p = tmp_path / "synthetic.dat"
    np.savetxt(p, np.c_[t, x, np.full(len(t), 0.02)])
    cfg = FoldConfig(min_period=0.4, max_period=1.5, harmonics=4)
    a = fold_light_curve(p, cfg)
    b = fold_light_curve(p, cfg)
    assert abs(a[2]["period"] / period - 1) < 0.01
    assert np.array_equal(a[1], b[1])
    # Shifting observational epoch must not change folded shape.
    np.savetxt(p, np.c_[t + 2454833.0, x, np.full(len(t), 0.02)])
    c = fold_light_curve(p, cfg)
    assert np.allclose(a[1], c[1], atol=1e-5)


def test_conversion_filtering(tmp_path):
    lc = SimpleNamespace(
        time=Time(2454833 + np.arange(6) * 0.01, format="jd", scale="tdb"),
        flux=np.array([100, 110, -1, 90, 105, 100]) * u.electron / u.s,
        flux_err=np.array([1, 1, 1, 1, 1, np.nan]) * u.electron / u.s,
    )
    p = tmp_path / "converted.dat"
    result = lightcurve_to_ogle(lc, p)
    data = np.loadtxt(p)
    assert result["dropped"] == 2 and len(data) == 4
    assert np.allclose(data[:, 0], lc.time.tdb.jd[[0, 1, 3, 4]], rtol=0, atol=1e-9)
    assert np.isfinite(data).all()
