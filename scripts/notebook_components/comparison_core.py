# This source is embedded verbatim in the notebook: no project imports are required.
SHAPE_FEATURES = [
    "skewness",
    "excess_kurtosis",
    "bowley_skewness",
    "tail_ratio",
    "phase_eta",
    "R21",
    "R31",
    "R41",
    "cos_phi21",
    "sin_phi21",
    "cos_phi31",
    "sin_phi31",
    "residual_fraction",
]
FULL_FEATURES = SHAPE_FEATURES + [
    "log10_period",
    "log10_std_mag",
    "log10_q95_q05_amplitude",
]
REPRESENTATIONS = ["JDR", "Classical shape", "Classical full"]
MODEL_NAMES = ["K-medoids", "K-means", "Average linkage", "DBSCAN", "HDBSCAN"]
PROTOCOL = dict(
    seed=20260912,
    n_objects=1000,
    n_clusters=3,
    n_init=20,
    subsamples=30,
    subset_fraction=0.8,
    grid_sizes=[1025, 2049, 4097],
    alpha=0.5,
    dbscan_min_samples=5,
    dbscan_quantile=0.70,
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=5,
    harmonics=4,
    dissimilarity="Squared Euclidean for every representation; J for JDR",
    scaling="JDR quadrature weights unchanged; classical median/IQR with median imputation, refitted within each subset",
)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_inputs(data_dir):
    data_dir = Path(data_dir)
    table = pd.read_csv(data_dir / "cohort.csv")
    if (
        len(table) != 1000
        or not table.object_id.is_unique
        or not table.sha256.is_unique
    ):
        raise ValueError(
            "The frozen cohort must contain 1000 distinct IDs and contents."
        )
    if table.object_id.tolist() != sorted(table.object_id):
        raise ValueError("Cohort order changed.")
    acquisition = json.loads((data_dir / "acquisition.json").read_text())
    # The portable bundle includes catalogs and selected files, not the original full archive.
    catalog_hashes = {}
    for record in acquisition["source_files"]:
        name = Path(record["file"]).name
        if name == "phot.tar.gz":
            continue
        path = data_dir / "source" / name
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Changed source: {name}")
        catalog_hashes[name] = digest
    ident = {
        line[:19].strip(): line[21:25].strip()
        for line in (data_dir / "source/ident.dat").read_text().splitlines()
    }
    periods = {}
    for kind in ["RRab", "RRc", "RRd"]:
        for line in (data_dir / "source" / f"{kind}.dat").read_text().splitlines():
            periods[line[:19].strip()] = (float(line[36:46]), float(line[59:69]))
    curves = []
    for row in table.itertuples():
        path = data_dir / "photometry" / f"{row.object_id}.dat"
        if sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Changed photometry: {row.object_id}")
        if ident[row.object_id] != row.catalog_type or not np.allclose(
            periods[row.object_id],
            [row.catalog_period, row.catalog_epoch],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Catalog period, epoch or subtype mismatch.")
        raw = np.loadtxt(path, dtype=np.float64)
        if (
            raw.ndim != 2
            or raw.shape != (row.n_observations, 3)
            or len(raw) < 30
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Invalid raw measurements.")
        t, x, e = raw.T
        if np.any(e <= 0) or len(np.unique(t)) != len(t) or np.std(x) <= 0:
            raise ValueError("Invalid errors, times or constant signal.")
        phase = ((t - row.catalog_epoch) / row.catalog_period) % 1
        order = np.argsort(phase, kind="stable")
        if len(np.unique(phase)) != len(phase):
            raise ValueError("Duplicate phases need an explicit handling policy.")
        curves.append((phase[order], x[order], e[order]))
    return table, curves, catalog_hashes


def classical_features(phase, mag, period):
    # Unweighted Fourier fit matches JDR's use of the original observations without error weighting.
    y = (mag - np.mean(mag)) / np.std(mag)
    q05, q25, q50, q75, q95 = np.quantile(mag, [0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = q75 - q25
    out = dict(
        skewness=float(np.mean(y**3)),
        excess_kurtosis=float(np.mean(y**4) - 3),
        bowley_skewness=float((q75 + q25 - 2 * q50) / iqr) if iqr > 1e-12 else np.nan,
        tail_ratio=float((q95 - q05) / iqr) if iqr > 1e-12 else np.nan,
        phase_eta=float(np.mean((np.roll(y, -1) - y) ** 2)),
        log10_period=float(np.log10(period)),
        log10_std_mag=float(np.log10(np.std(mag))),
        log10_q95_q05_amplitude=float(np.log10(q95 - q05)) if q95 > q05 else np.nan,
    )
    design = np.column_stack(
        [np.ones(len(phase))]
        + [fn(2 * np.pi * k * phase) for k in range(1, 5) for fn in (np.cos, np.sin)]
    )
    coef, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    condition = float(singular[0] / singular[-1])
    bad = rank != 9 or condition > 1e10
    amp = np.hypot(coef[1::2], coef[2::2])
    angle = np.arctan2(-coef[2::2], coef[1::2])
    for k in [2, 3, 4]:
        out[f"R{k}1"] = (
            float(amp[k - 1] / amp[0]) if not bad and amp[0] > 1e-8 else np.nan
        )
    for k in [2, 3]:
        delta = angle[k - 1] - k * angle[0]
        valid = not bad and min(amp[0], amp[k - 1]) > 1e-8
        out[f"cos_phi{k}1"] = float(np.cos(delta)) if valid else np.nan
        out[f"sin_phi{k}1"] = float(np.sin(delta)) if valid else np.nan
    out["residual_fraction"] = (
        float(np.mean((y - design @ coef) ** 2)) if not bad else np.nan
    )
    diagnostic = dict(
        fourier_rank=int(rank),
        fourier_condition=condition,
        undefined_features=int(sum(not np.isfinite(v) for v in out.values())),
    )
    return out, diagnostic


def robust_fit_transform(raw):
    raw = np.asarray(raw, float)
    if np.isinf(raw).any() or np.isnan(raw).all(axis=0).any():
        raise ValueError("Infinite or entirely missing feature.")
    med = np.nanmedian(raw, axis=0)
    filled = np.where(np.isnan(raw), med, raw)
    quartiles = np.percentile(filled, [25, 75], axis=0)
    scale = quartiles[1] - quartiles[0]
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (filled - med) / scale, dict(
        median=med.tolist(),
        iqr_scale=scale.tolist(),
        imputed_counts=np.isnan(raw).sum(axis=0).tolist(),
    )


def squared_matrix(z):
    D = squareform(pdist(z, "sqeuclidean"))
    if not np.isfinite(D).all() or (D < 0).any():
        raise ValueError("Invalid dissimilarity.")
    return D


def fit_models(D, z, seed):
    # Same initial index sets, objective convention and stopping criterion for all representations.
    best = None
    for start in range(PROTOCOL["n_init"]):
        medoids = np.random.default_rng(seed + start).choice(len(D), 3, replace=False)
        for iteration in range(300):
            labels = np.argmin(D[:, medoids], axis=1)
            labels[medoids] = np.arange(3)
            updated = []
            for k in range(3):
                ids = np.flatnonzero(labels == k)
                updated.append(ids[np.argmin(D[np.ix_(ids, ids)].sum(axis=0))])
            updated = np.asarray(updated)
            if np.array_equal(medoids, updated):
                break
            medoids = updated
        else:
            raise RuntimeError(
                "K-medoids failed to converge; do not silently return a capped fit."
            )
        labels = np.argmin(D[:, medoids], axis=1)
        labels[medoids] = np.arange(3)
        objective = float(D[np.arange(len(D)), medoids[labels]].sum())
        if best is None or objective < best[0]:
            best = (objective, labels.copy(), medoids.copy(), iteration + 1)
    eps = max(float(np.quantile(np.sort(D, axis=1)[:, 4], 0.70)), np.finfo(float).eps)
    labels = {
        "K-medoids": best[1],
        "K-means": KMeans(n_clusters=3, n_init=20, random_state=seed).fit_predict(z),
        "Average linkage": AgglomerativeClustering(
            n_clusters=3, metric="precomputed", linkage="average"
        ).fit_predict(D),
        "DBSCAN": DBSCAN(eps=eps, min_samples=5, metric="precomputed").fit_predict(D),
        "HDBSCAN": HDBSCAN(
            min_cluster_size=5, min_samples=5, metric="precomputed", copy=True
        ).fit_predict(D),
    }
    return labels, dict(
        medoid_objective=best[0],
        medoid_indices=best[2].tolist(),
        medoid_iterations=best[3],
        dbscan_eps=eps,
    )


def metrics_for(truth, labels, D):
    mask = labels != -1
    groups = np.unique(labels[mask])
    count = len(groups)
    purity = (
        float(
            sum(
                np.unique(truth[(labels == k)], return_counts=True)[1].max()
                for k in groups
            )
            / mask.sum()
        )
        if mask.any()
        else None
    )
    return dict(
        ari_all=float(adjusted_rand_score(truth, labels)),
        ami_all=float(adjusted_mutual_info_score(truth, labels)),
        coverage=float(mask.mean()),
        n_noise=int((~mask).sum()),
        n_clusters=count,
        purity_assigned=purity,
        ari_assigned=(
            float(adjusted_rand_score(truth[mask], labels[mask]))
            if mask.sum() > 1
            else None
        ),
        silhouette_assigned=(
            float(
                silhouette_score(
                    D[np.ix_(mask, mask)], labels[mask], metric="precomputed"
                )
            )
            if 1 < count < mask.sum()
            else None
        ),
    )


def run_pipeline(data_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    with threadpool_limits(limits=1):
        table, curves, catalog_hashes = read_inputs(data_dir)
        raw_features = []
        quality = []
        for row, (phase, mag, err) in zip(table.itertuples(), curves):
            features, diagnostic = classical_features(phase, mag, row.catalog_period)
            raw_features.append(features)
            quality.append({"object_id": row.object_id, **diagnostic})
        features = pd.DataFrame(raw_features)[FULL_FEATURES]
        # Recompute JDR from observations, never read the earlier experiment.npz.
        previous = None
        convergence = []
        for size in PROTOCOL["grid_sizes"]:
            config = PaperJDRConfig(n_frequencies=size)
            J, z = paper_matrix([(t, x) for t, x, e in curves], config)
            if previous is not None:
                nn = lambda m: np.argmin(m + np.eye(len(m)) * 1e99, axis=1)
                check = dict(
                    coarse_grid=size // 2 + 1,
                    fine_grid=size,
                    all_pairs_close=bool(
                        np.allclose(previous, J, rtol=1e-3, atol=1e-8)
                    ),
                    neighbor_agreement=float(np.mean(nn(previous) == nn(J))),
                    relative_max_error=float(abs(previous - J).max() / J.max()),
                )
                if not check["all_pairs_close"] or check["neighbor_agreement"] != 1:
                    raise AssertionError(f"Grid not converged: {check}")
                convergence.append(check)
            previous = J
        matrices = {"JDR": J}
        embeddings = {"JDR": z}
        scalers = {}
        for name, columns in [
            ("Classical shape", SHAPE_FEATURES),
            ("Classical full", FULL_FEATURES),
        ]:
            embeddings[name], scalers[name] = robust_fit_transform(
                features[columns].to_numpy()
            )
            matrices[name] = squared_matrix(embeddings[name])
        permutation = np.random.default_rng(PROTOCOL["seed"]).permutation(1000)
        permuted = np.array(
            [paper_features(curves[i][0], curves[i][1], config) for i in permutation]
        )
        assert np.allclose(
            squared_matrix(permuted),
            J[np.ix_(permutation, permutation)],
            rtol=1e-12,
            atol=1e-14,
        )
        audit = []
        upper = np.triu_indices(1000, 1)
        ordered = np.argsort(J[upper], kind="stable")
        for quantile in [0.1, 0.5, 0.9]:
            pos = ordered[int(quantile * (len(ordered) - 1))]
            i, j = upper[0][pos], upper[1][pos]
            result = adaptive_three_integrals(curves[i][:2], curves[j][:2])
            grid = float(J[i, j])
            assert np.isclose(result["j_adaptive"], grid, rtol=1e-3, atol=1e-8)
            audit.append(
                dict(
                    quantile=quantile,
                    object_i=table.object_id.iloc[i],
                    object_j=table.object_id.iloc[j],
                    **result,
                    j_grid=grid,
                    relative_difference=abs(result["j_adaptive"] - grid)
                    / max(abs(result["j_adaptive"]), 1e-30),
                )
            )
        truth = table.catalog_type.to_numpy()
        all_labels = {}
        info = {}
        scores = []
        retention = []
        composition = []
        for representation in REPRESENTATIONS:
            labels, info[representation] = fit_models(
                matrices[representation], embeddings[representation], PROTOCOL["seed"]
            )
            all_labels[representation] = labels
            for model, predicted in labels.items():
                scores.append(
                    dict(
                        representation=representation,
                        model=model,
                        **metrics_for(truth, predicted, matrices[representation]),
                    )
                )
                for kind in ["RRab", "RRc", "RRd"]:
                    retention.append(
                        dict(
                            representation=representation,
                            model=model,
                            catalog_type=kind,
                            n=int(sum(truth == kind)),
                            retention=float(np.mean(predicted[truth == kind] != -1)),
                        )
                    )
                    for group in sorted(np.unique(predicted)):
                        composition.append(
                            dict(
                                representation=representation,
                                model=model,
                                cluster=int(group),
                                catalog_type=kind,
                                count=int(sum((predicted == group) & (truth == kind))),
                            )
                        )
        # Identical subsets across representations; imputation/scaling are fitted anew on each subset.
        rng = np.random.default_rng(PROTOCOL["seed"] + 10)
        memberships = []
        stability = []
        subset_scalers = []
        for replicate in range(PROTOCOL["subsamples"]):
            indices = np.sort(rng.choice(1000, size=800, replace=False))
            memberships.append(indices.tolist())
            for representation in REPRESENTATIONS:
                if representation == "JDR":
                    sub_z = z[indices]
                    sub_D = J[np.ix_(indices, indices)]
                else:
                    columns = (
                        SHAPE_FEATURES
                        if representation == "Classical shape"
                        else FULL_FEATURES
                    )
                    sub_z, scaler = robust_fit_transform(
                        features[columns].to_numpy()[indices]
                    )
                    sub_D = squared_matrix(sub_z)
                    subset_scalers.append(
                        dict(
                            replicate=replicate, representation=representation, **scaler
                        )
                    )
                labels, _ = fit_models(sub_D, sub_z, PROTOCOL["seed"] + 100 + replicate)
                for model, predicted in labels.items():
                    reference = all_labels[representation][model][indices]
                    common = (predicted != -1) & (reference != -1)
                    valid = (
                        common.sum() > 1
                        and len(np.unique(predicted[common])) > 1
                        and len(np.unique(reference[common])) > 1
                    )
                    stability.append(
                        dict(
                            representation=representation,
                            model=model,
                            replicate=replicate,
                            ari_stability_all=float(
                                adjusted_rand_score(reference, predicted)
                            ),
                            ari_stability_common_assigned=(
                                float(
                                    adjusted_rand_score(
                                        reference[common], predicted[common]
                                    )
                                )
                                if valid
                                else None
                            ),
                            common_assigned_fraction=float(common.mean()),
                            coverage=float(np.mean(predicted != -1)),
                            catalog_ari_subset=float(
                                adjusted_rand_score(truth[indices], predicted)
                            ),
                        )
                    )
            if (replicate + 1) % 10 == 0:
                print(
                    f"Completed {replicate+1}/30 shared subsamples for all representations.",
                    flush=True,
                )
        # Predeclared dissimilarity-convention sensitivity, not a model/representation selector.
        convention = []
        for representation in REPRESENTATIONS:
            D = np.sqrt(matrices[representation])
            labels, _ = fit_models(D, embeddings[representation], PROTOCOL["seed"])
            for model, predicted in labels.items():
                convention.append(
                    dict(
                        representation=representation,
                        model=model,
                        convention="Euclidean sqrt sensitivity",
                        **metrics_for(truth, predicted, D),
                    )
                )
        scores = pd.DataFrame(scores)
        stability = pd.DataFrame(stability)
        paired = scores.pivot(
            index="model", columns="representation", values="ari_all"
        ).reindex(MODEL_NAMES)
        paired["full_minus_JDR"] = paired["Classical full"] - paired["JDR"]
        paired["shape_minus_JDR"] = paired["Classical shape"] - paired["JDR"]
        # PCA never enters clustering. Independent per-representation coordinates are not directly comparable axes.
        arrays = dict(
            object_ids=table.object_id.to_numpy(dtype=str),
            raw_features=features.to_numpy(),
            raw_phase=np.concatenate([t for t, x, e in curves]),
            raw_mag=np.concatenate([x for t, x, e in curves]),
            raw_error=np.concatenate([e for t, x, e in curves]),
            offsets=np.r_[0, np.cumsum([len(t) for t, x, e in curves])],
        )
        explained = {}
        nuisance = []
        for index, representation in enumerate(REPRESENTATIONS):
            pca = PCA(n_components=2, svd_solver="full")
            xy = pca.fit_transform(embeddings[representation])
            arrays.update(
                {
                    f"embedding_{index}": embeddings[representation],
                    f"squared_distance_{index}": matrices[representation],
                    f"pca_{index}": xy,
                }
            )
            explained[representation] = pca.explained_variance_ratio_.tolist()
            for col in ["n_observations", "baseline_days"]:
                rho, p = spearmanr(xy[:, 0], table[col])
                nuisance.append(
                    dict(
                        representation=representation,
                        quantity=col,
                        pc1_spearman=float(rho),
                        interpretation="Descriptive association, not causal evidence; PCA sign is arbitrary.",
                    )
                )
        table.to_csv(output / "cohort.csv", index=False)
        features.assign(object_id=table.object_id).to_csv(
            output / "classical_features.csv", index=False
        )
        pd.DataFrame(quality).to_csv(output / "feature_quality.csv", index=False)
        scores.to_csv(output / "model_comparison.csv", index=False)
        paired.to_csv(output / "paired_ari_comparison.csv")
        stability.to_csv(output / "subsampling_stability.csv", index=False)
        pd.DataFrame(retention).to_csv(output / "class_retention.csv", index=False)
        pd.DataFrame(composition).to_csv(
            output / "cluster_composition.csv", index=False
        )
        pd.DataFrame(audit).to_csv(output / "spectral_integral_audit.csv", index=False)
        pd.DataFrame(convention).to_csv(
            output / "distance_convention_sensitivity.csv", index=False
        )
        pd.DataFrame(nuisance).to_csv(output / "sampling_diagnostics.csv", index=False)
        pd.DataFrame(
            {
                "object_id": table.object_id,
                **{
                    f"{r} | {m}": v
                    for r, labels in all_labels.items()
                    for m, v in labels.items()
                },
            }
        ).to_csv(output / "cluster_labels.csv", index=False)
        np.savez_compressed(output / "comparison_arrays.npz", **arrays)
        json_write(output / "scalers.json", scalers)
        json_write(output / "subset_scalers.json", subset_scalers)
        json_write(output / "subsample_membership.json", memberships)
        summary = dict(
            protocol=PROTOCOL,
            class_counts=table.catalog_type.value_counts().to_dict(),
            representations=REPRESENTATIONS,
            shape_features=SHAPE_FEATURES,
            full_features=FULL_FEATURES,
            grid_convergence=convergence,
            fit_info=info,
            pca_explained=explained,
            interpretation="Exploratory in-sample representation comparison. Classical full explicitly retains period/amplitude; shape control removes those scalars. No catalog-based feature/hyperparameter selection. Eight RRd stars are insufficient for rare-class generalization.",
        )
        json_write(output / "summary.json", summary)
        provenance = dict(
            protocol=PROTOCOL,
            input_sha256=table.sha256.tolist(),
            object_ids=table.object_id.tolist(),
            catalog_sha256=catalog_hashes,
            cohort_manifest_sha256=sha256(
                (Path(data_dir) / "cohort.csv").read_bytes()
            ).hexdigest(),
            acquisition_sha256=sha256(
                (Path(data_dir) / "acquisition.json").read_bytes()
            ).hexdigest(),
            python=sys.version,
            platform=platform.platform(),
            numerical_threads=1,
            versions={
                p: version(p)
                for p in [
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "matplotlib",
                    "astropy",
                    "threadpoolctl",
                ]
            },
        )
        json_write(output / "provenance.json", provenance)
    json_write(
        output / "timing.json",
        dict(
            elapsed_seconds=time.perf_counter() - clock,
            scope="All representations, grids, 15 primary fits, 30 paired subset refits, convention sensitivity and integral audits; local, excludes figures.",
        ),
    )
    return output


def compare_runs(first, second):
    first, second = Path(first), Path(second)
    for p in sorted(first.glob("*.csv")):
        assert p.read_bytes() == (second / p.name).read_bytes(), p.name
    for name in [
        "summary.json",
        "provenance.json",
        "scalers.json",
        "subset_scalers.json",
        "subsample_membership.json",
    ]:
        assert json.loads((first / name).read_text()) == json.loads(
            (second / name).read_text()
        ), name
    with (
        np.load(first / "comparison_arrays.npz", allow_pickle=False) as a,
        np.load(second / "comparison_arrays.npz", allow_pickle=False) as b,
    ):
        assert a.files == b.files
        for key in a.files:
            assert (
                np.array_equal(a[key], b[key], equal_nan=True)
                if a[key].dtype.kind not in "US"
                else np.array_equal(a[key], b[key])
            ), key
    result = dict(
        independent_full_recomputations=2,
        arrays_identical=True,
        tables_identical=True,
        provenance_identical=True,
        scope="Exact in the recorded environment; wall time and compressed-file timestamps excluded.",
    )
    json_write(first / "reproducibility.json", result)
    return result
