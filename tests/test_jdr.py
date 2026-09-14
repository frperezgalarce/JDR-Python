import importlib
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from src.metrics import (
    tau,
    spectral_coefficients,
    cospectrum,
    integrate_cross_spectrum_real,
    integrate_cross_spectrum_real_R,
)
from src.jdr import (
    JDRConfig,
    feature_vector,
    jdr,
    jdr_parallel,
    build_distance_matrix,
    save_distance_matrix,
    load_distance_matrix,
)
from src.read_data import read_ogle_dat
from src.JDRKMedoids import JDRKMedoids
from src.JDRDBSCAN import JDRDBSCAN
from src.evaluation import aligned_labels, evaluate_clustering


@pytest.fixture
def curves(tmp_path):
    rng = np.random.default_rng(10)
    paths = []
    for i in range(4):
        t = np.sort(rng.uniform(0, 3 + i, 45 + i))
        x = np.sin(2 * np.pi * 0.7 * t + i * 0.2) + 0.2 * np.cos(2 * np.pi * 1.2 * t)
        p = tmp_path / f"curve{i}.dat"
        np.savetxt(p, np.c_[t, x])
        paths.append(p)
    return paths


def test_raw_reader_and_validation(curves, tmp_path):
    assert len(read_ogle_dat(curves[0])) == 45
    assert len(read_ogle_dat(curves[0], fixed_length=70)) == 70
    for data in [
        [[0, 1], [1, 1], [2, 1]],
        [[0, 1], [0, 2], [1, 3]],
        [[0, 1], [1, np.inf], [2, 3]],
        [[0, 1, -1], [1, 2, 1], [2, 3, 1]],
    ]:
        p = tmp_path / "bad.dat"
        np.savetxt(p, data)
        with pytest.raises(ValueError):
            read_ogle_dat(p)


def test_order_parallel_identity(curves):
    cfg = JDRConfig(f_min=0.2, f_max=1.5, n_frequencies=513)
    D = build_distance_matrix(curves, cfg)
    assert np.array_equal(np.diag(D), np.zeros(4))
    assert np.all(D >= 0)
    order = [2, 0, 3, 1]
    assert np.allclose(
        build_distance_matrix([curves[i] for i in order], cfg), D[np.ix_(order, order)]
    )
    assert jdr(curves[0], curves[1], config=cfg) == jdr(
        curves[1], curves[0], config=cfg
    )
    assert jdr_parallel(curves[0], curves[1], config=cfg) == jdr(
        curves[0], curves[1], config=cfg
    )
    assert np.allclose(build_distance_matrix(curves, cfg, n_jobs=2), D)


def test_algebra_grid_quad(curves):
    a, b = [read_ogle_dat(p) for p in curves[:2]]
    cfg = JDRConfig(f_min=0.3, f_max=0.8, n_frequencies=2049)
    z1 = feature_vector(a.time, a.mag, cfg)
    z2 = feature_vector(b.time, b.mag, cfg)
    w = 2 * np.pi * cfg.frequencies
    integrand = (
        cospectrum(a.time, a.mag, a.time, a.mag, w)
        + cospectrum(b.time, b.mag, b.time, b.mag, w)
        - 2 * cospectrum(a.time, a.mag, b.time, b.mag, w)
    )
    value = 0.25 / (2 * np.pi) * np.trapezoid(integrand, cfg.frequencies)
    assert np.dot(z1 - z2, z1 - z2) == pytest.approx(value, rel=1e-12)
    terms = [
        integrate_cross_spectrum_real(0.3, 0.8, u.time, u.mag, v.time, v.mag)
        for u, v in [(a, a), (b, b), (a, b)]
    ]
    assert value == pytest.approx(
        0.25 / (2 * np.pi) * (terms[0] + terms[1] - 2 * terms[2]), rel=1e-5
    )


def test_standard_tau_and_known_coefficients():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 4, 60))
    w = 1.7
    shift = tau(t, w)
    assert abs(np.sum(np.sin(w * (t - shift)) * np.cos(w * (t - shift)))) < 1e-12
    x = 3 + 2 * np.cos(w * t) - 0.7 * np.sin(w * t)
    a, b = spectral_coefficients(t, x, w, normalize=False)
    assert a[0] == pytest.approx(2)
    assert b[0] == pytest.approx(-0.7)
    # Standard coefficient phase is common even if cadences differ.
    t2 = np.sort(rng.uniform(0, 4, 80))
    x2 = 3 + 2 * np.cos(w * t2) - 0.7 * np.sin(w * t2)
    c, d = spectral_coefficients(t2, x2, w, normalize=False)
    assert np.allclose([a, b], [c, d])


def test_affine_precision_and_alpha():
    t = 2454833.0 + np.arange(50) * 0.01
    x = np.sin(np.arange(50) * 0.2)
    cfg = JDRConfig(f_min=0.5, f_max=2, n_frequencies=129, time_origin=2454833.0)
    a = feature_vector(t, x, cfg)
    b = feature_vector(t, 3 * x + 20, cfg)
    assert np.allclose(a, b)
    assert len(np.unique(t)) == 50
    assert np.allclose(
        feature_vector(t, x, replace(cfg, alpha=0.2)), a * np.sqrt(0.16 / 0.25)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"alpha": 2},
        {"f_min": 2, "f_max": 1},
        {"n_frequencies": 2},
        {"n_frequencies": 4.5},
        {"convention": "wrong"},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        JDRConfig(**kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 2)),
        np.array([[0, -1], [-1, 0]]),
        np.array([[0, 1], [2, 0]]),
        np.array([[0, np.inf], [np.inf, 0]]),
        np.empty((0, 0)),
    ],
)
def test_matrix_contract(bad):
    for model in [JDRKMedoids(1), JDRDBSCAN(1)]:
        with pytest.raises(ValueError):
            model.fit(bad)


def test_medoids_ties_rng_predict():
    m = JDRKMedoids(2, init_medoids=[0, 1]).fit(np.zeros((3, 3)))
    assert all(m.clusters_)
    D = squareform(pdist(np.random.default_rng(1).normal(size=(15, 2))))
    m = JDRKMedoids(3, random_state=6)
    np.random.seed(1)
    a = m.fit_predict(D).copy()
    np.random.normal(size=100)
    assert np.array_equal(a, m.fit_predict(D))
    with pytest.raises(ValueError):
        m.predict([[np.inf, 1, 2]])
    with pytest.raises(ValueError):
        JDRKMedoids(2, max_iter=0).fit(D)


def test_dbscan_reference():
    rng = np.random.default_rng(8)
    for _ in range(30):
        D = squareform(pdist(rng.normal(size=(25, 3))))
        for eps in [0.3, 1, 2]:
            a = JDRDBSCAN(eps, 3).fit(D)
            b = DBSCAN(eps=eps, min_samples=3, metric="precomputed").fit(D)
            assert np.array_equal(a.labels_, b.labels_)
            assert a.core_samples_ == b.core_sample_indices_.tolist()


def test_metadata_and_noise():
    df = pd.DataFrame({"id": ["B", "A"], "type": ["b", "a"]})
    assert aligned_labels(["A.dat", "B.dat"], df).tolist() == ["a", "b"]
    with pytest.raises(ValueError):
        aligned_labels(["C.dat"], df)
    r = evaluate_clustering(["a"] * 100, [0] * 5 + [-1] * 95)
    assert r["coverage"] == 0.05 and r["purity_assigned"] == 1 and r["n_noise"] == 95
    assert evaluate_clustering(["a", "b"], [-1, -1])["purity_assigned"] is None


def test_cache(curves, tmp_path):
    cfg = JDRConfig(n_frequencies=33, f_min=0.3, f_max=1)
    D = build_distance_matrix(curves, cfg)
    p = tmp_path / "cache.npz"
    save_distance_matrix(p, D, curves, cfg)
    assert np.array_equal(load_distance_matrix(p, curves, cfg)[0], D)
    with pytest.raises(ValueError):
        load_distance_matrix(p, curves[::-1], cfg)
    with pytest.raises(ValueError):
        load_distance_matrix(p, curves, replace(cfg, alpha=0.2))
    curves[0].write_text(curves[0].read_text() + "\n")
    with pytest.raises(ValueError):
        load_distance_matrix(p, curves, cfg)


def test_r_failure_and_quadrature_failure():
    t = np.array([0.0, 0.2, 0.9, 1.2])
    x = np.array([1.0, 2.0, 1.0, 3.0])
    with patch(
        "src.metrics._rscript_integrate_cospec",
        return_value=(1.0, "ok", "subdivisions reached"),
    ) as call:
        with pytest.raises(RuntimeError):
            integrate_cross_spectrum_real_R(0.2, 0.5, t, x, t, x)
        assert call.call_count == 3
    with patch("src.metrics.quad", return_value=(1.0, 0.2, {}, "failed")):
        with pytest.raises(RuntimeError):
            integrate_cross_spectrum_real(0.2, 0.5, t, x, t, x)
    with pytest.raises(ValueError):
        integrate_cross_spectrum_real(0.2, 0.5, t, x, t, x, jacobian="bad")


def test_generic_builder_refuses_asymmetry():
    def asymmetric(a, b):
        return float(a + 2 * b)

    with pytest.raises(ValueError):
        JDRKMedoids.build_distance_matrix([1, 2], asymmetric, verbose=False)
    D = JDRKMedoids.build_distance_matrix_parallel(
        [1, 2], lambda a, b: abs(a - b), verbose=False, n_jobs=2
    )
    assert np.array_equal(D, [[0, 1], [1, 0]])


@pytest.mark.parametrize("convention", ["standard", "legacy"])
def test_independent_r_reference(convention):
    import shutil

    if not shutil.which("Rscript"):
        pytest.skip("Rscript not installed")
    t = np.array([0.0, 0.1, 0.3, 0.7, 1.1, 1.4])
    x = np.sin(2 * np.pi * 0.7 * t)
    y = np.cos(2 * np.pi * 0.4 * t)
    python = integrate_cross_spectrum_real(0.3, 0.35, t, x, t, y, convention=convention)
    r = integrate_cross_spectrum_real_R(0.3, 0.35, t, x, t, y, convention=convention)
    assert r == pytest.approx(python, rel=1e-7, abs=1e-10)
