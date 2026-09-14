from itertools import combinations


def run_self_tests():
    rng = np.random.default_rng(72)
    phase = np.sort(rng.uniform(0, 1, 101))
    signal = (
        15
        + 0.7 * np.cos(2 * np.pi * phase + 0.3)
        + 0.21 * np.cos(4 * np.pi * phase + 1.2)
        + 0.07 * np.cos(6 * np.pi * phase - 0.4)
        + 0.035 * np.cos(8 * np.pi * phase + 0.2)
    )
    f, _ = classical_features(phase, signal, 0.57)
    np.testing.assert_allclose(
        [f["R21"], f["R31"], f["R41"]], [0.3, 0.1, 0.05], atol=1e-12
    )
    np.testing.assert_allclose(
        [f["cos_phi21"], f["sin_phi21"]], [np.cos(0.6), np.sin(0.6)], atol=1e-12
    )
    shifted, _ = classical_features(phase, 3 * signal + 9, 0.57)
    np.testing.assert_allclose(
        [f[k] for k in SHAPE_FEATURES], [shifted[k] for k in SHAPE_FEATURES], atol=1e-10
    )
    reordered = np.argsort((phase + 0.27) % 1)
    shiftphase = (phase + 0.27)[reordered] % 1
    rephased, _ = classical_features(shiftphase, signal[reordered], 0.57)
    np.testing.assert_allclose(
        [f[k] for k in SHAPE_FEATURES],
        [rephased[k] for k in SHAPE_FEATURES],
        atol=1e-10,
    )
    omega = np.linspace(0.01, 2 * np.pi, 47)
    y = (signal - signal.mean()) / signal.std()
    expected = LombScargle(
        phase, y, fit_mean=False, center_data=False, normalization="psd"
    ).power(omega / (2 * np.pi), method="slow")
    np.testing.assert_allclose(
        abs(paper_amplitude(phase, signal, omega)) ** 2, expected, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        paper_amplitude(phase, signal, omega),
        [scalar_amplitude_reference(phase, signal, w) for w in omega],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        paper_amplitude(phase, signal, [0]),
        paper_amplitude(phase, signal, [1e-8]),
        atol=1e-6,
    )
    raw = np.array([[1, 2], [3, np.nan], [5, 6]], float)
    transformed, scaler = robust_fit_transform(raw)
    np.testing.assert_allclose(transformed, [[-1, -1], [0, 0], [1, 1]])
    assert scaler["imputed_counts"] == [0, 1]
    subset, _ = robust_fit_transform(np.array([[0.0], [1.0], [2.0]]))
    full, _ = robust_fit_transform(np.array([[0.0], [1.0], [2.0], [100.0]]))
    assert not np.allclose(subset[:, 0], full[:3, 0])
    points = np.array([[0], [0.1], [4], [4.1], [9], [9.1], [16], [16.1]], float)
    D = squared_matrix(points)
    labels, info = fit_models(D, points, PROTOCOL["seed"])
    optimal = min(np.min(D[:, ids], axis=1).sum() for ids in combinations(range(8), 4))
    np.testing.assert_allclose(info["medoid_objective"], optimal, atol=1e-12)
    assert adjusted_rand_score(np.repeat(range(4), 2), labels["K-medoids"]) == 1
    different = signal + 0.3 * np.sin(2 * np.pi * phase)
    J, z = paper_matrix([(phase, signal), (phase, different)])
    audit = adaptive_three_integrals((phase, signal), (phase, different))
    np.testing.assert_allclose(J[0, 1], audit["j_adaptive"], rtol=1e-5, atol=1e-8)
    assert J[0, 0] == 0 and J[1, 1] == 0 and J[0, 1] >= 0
    return [
        "Fourier ratios and phases",
        "magnitude/phase invariances of shape features",
        "independent Astropy power",
        "scalar cross-spectrum and zero-frequency limit",
        "median imputation and IQR scaling",
        "subset scaling isolation",
        "enumerated K-medoids optimum",
        "independent three-integral identity",
    ]


with threadpool_limits(limits=1):
    SELF_TEST_RESULTS = run_self_tests()
print("Self-tests passed:", len(SELF_TEST_RESULTS))
