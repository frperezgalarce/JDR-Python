import numpy as np
from astropy.timeseries import LombScargle
from src.paper_jdr import (
    PaperJDRConfig,
    paper_amplitude,
    paper_features,
    paper_matrix,
    scalar_amplitude_reference,
    adaptive_three_integrals,
)


def sample():
    t = np.sort(np.random.default_rng(51).uniform(0, 1, 71))
    x = np.sin(2 * np.pi * t) + 0.3 * np.cos(4 * np.pi * t + 0.2)
    return t, x


def test_eq1_matches_independent_astropy_classical_lsp():
    t, x = sample()
    w = np.linspace(0.02, 2 * np.pi, 95)
    y = (x - x.mean()) / x.std()
    reference = LombScargle(
        t, y, fit_mean=False, center_data=False, normalization="psd"
    ).power(w / (2 * np.pi), method="slow")
    np.testing.assert_allclose(
        abs(paper_amplitude(t, x, w)) ** 2, reference, rtol=1e-9, atol=1e-10
    )


def test_scalar_arctan_and_vector_atan2_branches_agree():
    t, x = sample()
    w = np.linspace(0.001, 2 * np.pi, 115)
    reference = np.array([scalar_amplitude_reference(t, x, f) for f in w])
    np.testing.assert_allclose(
        paper_amplitude(t, x, w), reference, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        paper_amplitude(t, x, [0]), paper_amplitude(t, x, [1e-8]), atol=1e-6
    )


def test_j_matches_three_angular_integrals_and_squared_euclidean_identity():
    t, x = sample()
    y = np.sin(2 * np.pi * t + 0.35) + 0.2 * np.cos(4 * np.pi * t)
    J, z = paper_matrix([(t, x), (t, y)])
    independent = adaptive_three_integrals((t, x), (t, y))
    np.testing.assert_allclose(J[0, 1], independent["j_adaptive"], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(J[0, 1], np.sum((z[0] - z[1]) ** 2), rtol=1e-12)
    assert np.array_equal(np.diag(J), [0, 0]) and J[0, 1] > 0 and J[0, 1] == J[1, 0]


def test_literal_power_product_is_not_a_distance_and_j_need_not_satisfy_triangle():
    t, x = sample()
    w = np.linspace(0, 2 * np.pi, 4097)
    power = abs(paper_amplitude(t, x, w)) ** 2
    literal = 0.25 / (2 * np.pi) * np.trapezoid(2 * power - 2 * power**2, w)
    assert literal < 0
    J, _ = paper_matrix(
        [(t, x), (t, 2 * x), (t, 3 * x)], PaperJDRConfig(normalize=False)
    )
    assert J[0, 2] > J[0, 1] + J[1, 2]
    assert np.sqrt(J[0, 2]) <= np.sqrt(J[0, 1]) + np.sqrt(J[1, 2]) + 1e-12


def test_standardization_translation_and_measurement_order():
    t, x = sample()
    w = np.linspace(0.01, 2 * np.pi, 87)
    np.testing.assert_allclose(
        paper_amplitude(t, x, w),
        paper_amplitude(t, 3 * x + 18, w),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        paper_amplitude(t[::-1], x[::-1], w), paper_amplitude(t, x, w), atol=1e-10
    )
    # Common time translation rotates both spectra equally and preserves their real cross product.
    y = np.cos(2 * np.pi * t)
    first = paper_amplitude(t, x, w) * paper_amplitude(t, y, w).conj()
    second = paper_amplitude(t + 0.37, x, w) * paper_amplitude(t + 0.37, y, w).conj()
    np.testing.assert_allclose(first, second, atol=1e-9)
