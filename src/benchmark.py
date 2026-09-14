"""Reproducible census benchmark; catalog labels never select model parameters."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import platform
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from src.evaluation import aligned_labels, evaluate_clustering, load_metadata
from src.jdr import JDRConfig, feature_vector
from src.JDRDBSCAN import JDRDBSCAN
from src.JDRKMedoids import JDRKMedoids
from src.preprocessing import FoldConfig, fold_light_curve
from src.read_data import read_ogle_dat
from src.validation import validate_distance_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["K-medoids", "K-means", "Average linkage", "DBSCAN", "HDBSCAN"]
PROTOCOL = {
    "version": 1,
    "n_objects": 100,
    "seed": 20260912,
    "n_clusters": 3,
    "n_init": 20,
    "dbscan_min_samples": 5,
    "dbscan_eps_quantile": 0.70,
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples": 5,
    "subsamples": 30,
    "subsample_fraction": 0.80,
    "grid_sizes": [1025, 2049, 4097],
    "grid_rtol": 1e-3,
    "grid_atol": 1e-8,
    "distance": "sqrt(J); common Euclidean geometry for all methods",
}


def cohort():
    files = sorted((ROOT / "data").glob("*RRLYR*.dat"))
    if len(files) != 100:
        raise ValueError(
            f"This frozen census requires exactly 100 files, found {len(files)}."
        )
    catalogs = sorted((ROOT / "notebooks").glob("ident*.dat"))
    metadata = load_metadata(catalogs)
    truth = aligned_labels(files, metadata)
    rows = []
    for file, label in zip(files, truth):
        frame = read_ogle_dat(file)
        rows.append(
            {
                "object_id": file.stem,
                "catalog_type": label,
                "region": file.stem.split("-")[1],
                "file": str(file.relative_to(ROOT)),
                "sha256": sha256(file.read_bytes()).hexdigest(),
                "n_observations": len(frame),
                "baseline_days": float(np.ptp(frame.time)),
                "has_errors": "mag_err" in frame,
            }
        )
    table = pd.DataFrame(rows)
    if table.object_id.duplicated().any() or table.sha256.duplicated().any():
        raise ValueError(
            "Duplicate IDs or byte-identical curves; resolve before benchmarking."
        )
    return files, table, catalogs


def fit_models(distance, features, protocol=PROTOCOL, seed=None):
    """All matrix models receive sqrt(J); k-means receives its exact feature embedding."""
    distance = validate_distance_matrix(distance)
    seed = protocol["seed"] if seed is None else seed
    k = protocol["n_clusters"]
    candidates = [
        JDRKMedoids(k, random_state=seed + i).fit(distance)
        for i in range(protocol["n_init"])
    ]
    medoid = min(candidates, key=lambda m: m.inertia_)
    q = protocol["dbscan_eps_quantile"]
    m = protocol["dbscan_min_samples"]
    eps = float(np.quantile(np.sort(distance, axis=1)[:, m - 1], q))
    eps = max(eps, np.finfo(float).eps)
    labels = {
        "K-medoids": medoid.labels_,
        "K-means": KMeans(
            n_clusters=k, n_init=protocol["n_init"], random_state=seed
        ).fit_predict(features),
        "Average linkage": AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(distance),
        "DBSCAN": JDRDBSCAN(eps=eps, min_samples=m).fit_predict(distance),
        "HDBSCAN": HDBSCAN(
            min_cluster_size=protocol["hdbscan_min_cluster_size"],
            min_samples=protocol["hdbscan_min_samples"],
            metric="precomputed",
            copy=True,
        ).fit_predict(distance),
    }
    reference = DBSCAN(eps=eps, min_samples=m, metric="precomputed").fit_predict(
        distance
    )
    if not np.array_equal(labels["DBSCAN"], reference):
        raise AssertionError("Custom DBSCAN does not match reference on the census.")
    return labels, {
        "dbscan_eps": eps,
        "medoid_indices": list(map(int, medoid.medoids_)),
        "medoid_objective": float(medoid.inertia_),
        "dbscan_reference_agreement": True,
    }


def scored_models(truth, labels, distance, regions):
    rows = []
    details = {}
    for name in MODEL_NAMES:
        predicted = labels[name]
        metrics = evaluate_clustering(truth, predicted)
        mask = predicted != -1
        count = len(np.unique(predicted[mask]))
        silhouette = (
            float(
                silhouette_score(
                    distance[np.ix_(mask, mask)], predicted[mask], metric="precomputed"
                )
            )
            if 1 < count < mask.sum()
            else None
        )
        metrics["silhouette_assigned"] = silhouette
        metrics["ari_region_all"] = (
            float(adjusted_rand_score(regions, predicted))
            if len(np.unique(regions)) > 1
            else None
        )
        rows.append(
            {
                "model": name,
                **{k: v for k, v in metrics.items() if not isinstance(v, dict)},
            }
        )
        details[name] = metrics
    return pd.DataFrame(rows), details


def subsampling_stability(distance, features, reference_labels, protocol=PROTOCOL):
    """Refit on shared 80% subsets. Quantiles are empirical stability, not confidence intervals."""
    rng = np.random.default_rng(protocol["seed"] + 10)
    n = len(distance)
    size = int(n * protocol["subsample_fraction"])
    rows, memberships = [], []
    for replicate in range(protocol["subsamples"]):
        indices = np.sort(rng.choice(n, size=size, replace=False))
        memberships.append(indices.tolist())
        predictions, _ = fit_models(
            distance[np.ix_(indices, indices)],
            features[indices],
            protocol,
            seed=protocol["seed"] + 100 + replicate,
        )
        for name, new in predictions.items():
            full = reference_labels[name][indices]
            assigned = (new != -1) & (full != -1)
            valid = (
                assigned.sum() > 1
                and len(np.unique(new[assigned])) > 1
                and len(np.unique(full[assigned])) > 1
            )
            rows.append(
                {
                    "replicate": replicate,
                    "model": name,
                    "ari_stability_all": float(adjusted_rand_score(full, new)),
                    "ari_stability_common_assigned": (
                        float(adjusted_rand_score(full[assigned], new[assigned]))
                        if valid
                        else None
                    ),
                    "common_assigned_fraction": float(assigned.mean()),
                    "coverage": float(np.mean(new != -1)),
                    "n_clusters": len(set(new) - {-1}),
                }
            )
    return pd.DataFrame(rows), memberships


def sensitivity(distance, truth):
    """Report the whole predeclared grid, never promote its best catalog-label score."""
    rows = []
    for min_samples in (3, 5, 8):
        kdist = np.sort(distance, axis=1)[:, min_samples - 1]
        for q in np.linspace(0.10, 0.95, 18):
            eps = max(float(np.quantile(kdist, q)), np.finfo(float).eps)
            labels = JDRDBSCAN(eps, min_samples).fit_predict(distance)
            metrics = evaluate_clustering(truth, labels)
            rows.append(
                {
                    "min_samples": min_samples,
                    "eps_quantile": float(q),
                    "eps": eps,
                    **{
                        k: metrics[k]
                        for k in ("coverage", "ari_all", "n_clusters", "n_noise")
                    },
                }
            )
    return pd.DataFrame(rows)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_benchmark(output_dir=None, protocol=None, cohort_loader=None):
    """Recompute every period fit and matrix; no implicit result-cache reuse."""
    protocol = dict(PROTOCOL if protocol is None else protocol)
    output = Path(output_dir) if output_dir else ROOT / "results" / "benchmark_100"
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        files, table, catalogs = (cohort if cohort_loader is None else cohort_loader)()
        if len(files) != protocol["n_objects"]:
            raise ValueError("Cohort size does not match the frozen protocol.")
        fold_config = FoldConfig(**protocol.get("fold_config", {}))
        curves = []
        for i, path in enumerate(files):
            curves.append(fold_light_curve(path, fold_config))
            if (i + 1) % 20 == 0:
                print(f"Folded {i+1}/{len(files)} light curves", flush=True)
        diagnostics = pd.DataFrame(
            [{"object_id": p.stem, **c[2]} for p, c in zip(files, curves)]
        )
        # Candidate lists are saved separately to keep the tabular artifact tidy.
        candidates = diagnostics.pop("candidate_periods").tolist()
        table = table.merge(
            diagnostics, on="object_id", validate="one_to_one", sort=False
        )
        phase = curves[0][0]
        shapes = np.array([c[1] for c in curves])
        matrices, embeddings = [], []
        configs = []
        for size in protocol["grid_sizes"]:
            config = JDRConfig(f_min=0.5, f_max=6.0, n_frequencies=size)
            configs.append(config)
            z = np.array([feature_vector(t, x, config) for t, x, _ in curves])
            J = validate_distance_matrix(squareform(pdist(z, "sqeuclidean")))
            matrices.append(J)
            embeddings.append(z)
        convergence = []
        for index in range(1, len(matrices)):
            old, new = matrices[index - 1 : index + 1]
            delta = np.abs(old - new)
            old_nn = np.argmin(old + np.eye(len(old)) * 1e99, axis=1)
            new_nn = np.argmin(new + np.eye(len(new)) * 1e99, axis=1)
            check = {
                "coarse_grid": protocol["grid_sizes"][index - 1],
                "fine_grid": protocol["grid_sizes"][index],
                "relative_max_error": float(delta.max() / new.max()),
                "all_pairs_close": bool(
                    np.allclose(
                        old, new, rtol=protocol["grid_rtol"], atol=protocol["grid_atol"]
                    )
                ),
                "neighbor_agreement": float(np.mean(old_nn == new_nn)),
            }
            if not check["all_pairs_close"] or check["neighbor_agreement"] != 1.0:
                raise RuntimeError(f"Grid convergence failed: {check}")
            convergence.append(check)
        J, z = matrices[-1], embeddings[-1]
        distance = validate_distance_matrix(np.sqrt(J))
        # Recompute a permuted feature matrix, not just an already-computed matrix slice.
        permutation = np.random.default_rng(protocol["seed"]).permutation(len(files))
        shuffled = np.array(
            [
                feature_vector(curves[i][0], curves[i][1], configs[-1])
                for i in permutation
            ]
        )
        assert np.allclose(
            squareform(pdist(shuffled, "sqeuclidean")),
            J[np.ix_(permutation, permutation)],
            rtol=1e-12,
            atol=1e-14,
        )
        labels, fit_info = fit_models(distance, z, protocol)
        scores, details = scored_models(
            table.catalog_type.to_numpy(), labels, distance, table.region.to_numpy()
        )
        stability, memberships = subsampling_stability(distance, z, labels, protocol)
        sweep = sensitivity(distance, table.catalog_type.to_numpy())
        # PCA is visualization only. Model fitting above uses full features/distances.
        from sklearn.decomposition import PCA

        projection_model = PCA(n_components=2, svd_solver="full")
        coordinates = projection_model.fit_transform(z)
        majority = max(table.catalog_type.value_counts()) / len(table)
        summary = {
            "scope": protocol.get(
                "scope",
                "Full local 100-object census; exploratory in-sample comparison, not held-out accuracy.",
            ),
            "protocol": protocol,
            "fold_config": asdict(fold_config),
            "jdr_config": asdict(configs[-1]),
            "n_objects": len(files),
            "class_counts": table.catalog_type.value_counts().to_dict(),
            "region_counts": table.region.value_counts().to_dict(),
            "majority_class_fraction": float(majority),
            "grid_convergence": convergence,
            "permutation_invariance": True,
            "fit_info": fit_info,
            "models": details,
            "pca_explained_variance_ratio": projection_model.explained_variance_ratio_.tolist(),
            "limitations": [
                f"Only {int(sum(table.catalog_type == 'RRd'))} RRd; no reliable RRd class-level inference.",
                "Catalog subtype and sky region are imbalanced; region agreement is descriptive, not a causal confounding test.",
                "All-star ARI groups rejected points under the noise label; assigned-only metrics and coverage are also reported.",
                "Subsampling refits clustering on fixed fitted curves; period/photometry uncertainty is not propagated.",
                "No performance confidence intervals or best-model significance claims from this census.",
                "The standard spectral representation is version 2, not a verified reproduction of an unidentified JDR paper.",
            ],
        }
        code_paths = [
            ROOT / "src" / n
            for n in [
                "benchmark.py",
                "metrics.py",
                "jdr.py",
                "preprocessing.py",
                "read_data.py",
                "evaluation.py",
                "validation.py",
                "JDRDBSCAN.py",
                "JDRKMedoids.py",
                "publication_figures.py",
            ]
        ]
        provenance = {
            "protocol": protocol,
            "fold_config": asdict(fold_config),
            "jdr_config": asdict(configs[-1]),
            "object_ids": table.object_id.tolist(),
            "input_sha256": table.sha256.tolist(),
            "catalog_sha256": {
                str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                for p in catalogs
            },
            "code_sha256": {
                str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
                for p in code_paths
            },
            "python": sys.version,
            "platform": platform.platform(),
            "numerical_threads": 1,
            "versions": {
                name: version(name)
                for name in [
                    "numpy",
                    "scipy",
                    "pandas",
                    "scikit-learn",
                    "astropy",
                    "matplotlib",
                    "threadpoolctl",
                ]
            },
        }
        table.to_csv(output / "cohort.csv", index=False)
        scores.to_csv(output / "model_comparison.csv", index=False)
        stability.to_csv(output / "subsampling_stability.csv", index=False)
        sweep.to_csv(output / "dbscan_sensitivity.csv", index=False)
        pd.DataFrame({"object_id": table.object_id, **labels}).to_csv(
            output / "cluster_labels.csv", index=False
        )
        np.savez_compressed(
            output / "experiment.npz",
            J=J,
            distance=distance,
            features=z,
            phase=phase,
            shapes=shapes,
            pca=coordinates,
            object_ids=table.object_id.to_numpy(dtype=str),
        )
        _json(output / "summary.json", summary)
        _json(output / "provenance.json", provenance)
        _json(output / "subsample_membership.json", memberships)
        _json(output / "period_candidates.json", candidates)
    return output


def compare_runs(first, second):
    """Compare semantic numerical results; compressed ZIP/PDF timestamps are irrelevant."""
    first, second = Path(first), Path(second)
    for name in [
        "summary.json",
        "provenance.json",
        "subsample_membership.json",
        "period_candidates.json",
    ]:
        if json.loads((first / name).read_text()) != json.loads(
            (second / name).read_text()
        ):
            raise AssertionError(f"Repeated run differs: {name}")
    for name in [
        "cohort.csv",
        "model_comparison.csv",
        "subsampling_stability.csv",
        "dbscan_sensitivity.csv",
        "cluster_labels.csv",
    ]:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise AssertionError(f"Repeated table differs: {name}")
    with (
        np.load(first / "experiment.npz", allow_pickle=False) as a,
        np.load(second / "experiment.npz", allow_pickle=False) as b,
    ):
        for name in a.files:
            if not np.array_equal(a[name], b[name]):
                raise AssertionError(f"Repeated arrays differ: {name}")
    result = {
        "independent_full_recomputations": 2,
        "tables_identical": True,
        "arrays_identical": True,
        "provenance_identical": True,
        "comparison": "Exact within the recorded environment, not promised across BLAS/platform versions.",
    }
    _json(first / "reproducibility.json", result)
    return result
