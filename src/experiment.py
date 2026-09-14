"""Small reproducible experiment shared by CLI and notebooks. No import-time work."""

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.jdr import JDRConfig, feature_vector, save_distance_matrix
from src.preprocessing import FoldConfig, fold_light_curve
from src.evaluation import load_metadata, aligned_labels, evaluate_clustering
from src.JDRKMedoids import JDRKMedoids
from src.JDRDBSCAN import JDRDBSCAN

ROOT = Path(__file__).resolve().parents[1]


def run_small_experiment(n_files=18, seed=42, output_dir=None):
    """Exploratory in-sample comparison, never advertised as held-out accuracy.

    Sample available RRab/RRc/RRd classes deterministically. Selection uses labels
    only to represent subtypes; parameters and model choices are fixed beforehand.
    """
    from src.validation import positive_integer

    positive_integer(n_files, "n_files")
    output = Path(output_dir) if output_dir else ROOT / "results" / "validation_v2"
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(sorted((ROOT / "notebooks").glob("ident*.dat")))
    available = {p.stem: p for p in sorted((ROOT / "data").glob("*RRLYR*.dat"))}
    pool = metadata[metadata.id.isin(available)]
    if n_files < 6 or n_files > len(pool):
        raise ValueError("Choose between 6 and the number of available labeled stars.")
    rng = np.random.default_rng(seed)
    ids = []
    groups = [
        rng.permutation(group.id.to_numpy()).tolist()
        for _, group in pool.groupby("type")
    ]
    while len(ids) < n_files:
        for group in groups:
            if group and len(ids) < n_files:
                ids.append(group.pop())
    files = [available[i] for i in ids]
    truth = aligned_labels(files, metadata)
    fold_config = FoldConfig()
    curves = [fold_light_curve(p, fold_config) for p in files]
    config = JDRConfig(f_min=0.5, f_max=6.0, n_frequencies=1025)
    z = np.array([feature_vector(t, x, config) for t, x, _ in curves])
    D = squareform(pdist(z, "sqeuclidean"))
    fine_config = JDRConfig(f_min=0.5, f_max=6.0, n_frequencies=2049)
    fine = squareform(
        pdist(
            np.array([feature_vector(t, x, fine_config) for t, x, _ in curves]),
            "sqeuclidean",
        )
    )
    error = float(np.max(np.abs(D - fine)) / max(float(np.max(fine)), 1e-12))
    nn = np.argmin(D + np.eye(len(D)) * 1e99, axis=1)
    nn_fine = np.argmin(fine + np.eye(len(D)) * 1e99, axis=1)
    if not np.allclose(D, fine, rtol=1e-3, atol=1e-8) or not np.array_equal(
        nn, nn_fine
    ):
        raise RuntimeError(f"Grid convergence gate failed: relative max error={error}")
    finest_config = JDRConfig(f_min=0.5, f_max=6.0, n_frequencies=4097)
    finest = squareform(
        pdist(
            np.array([feature_vector(t, x, finest_config) for t, x, _ in curves]),
            "sqeuclidean",
        )
    )
    finest_error = float(
        np.max(np.abs(fine - finest)) / max(float(np.max(finest)), 1e-12)
    )
    if not np.allclose(fine, finest, rtol=1e-3, atol=1e-8) or not np.array_equal(
        nn, np.argmin(finest + np.eye(len(D)) * 1e99, axis=1)
    ):
        raise RuntimeError("Second grid refinement failed convergence/neighbor gate.")
    # Unsupervised multi-start objective selection, not selection by truth labels.
    fits = [JDRKMedoids(n_clusters=3, random_state=seed + i).fit(D) for i in range(10)]
    medoid = min(fits, key=lambda m: m.inertia_)
    # Fix a transparent k-distance quantile heuristic before inspecting class scores.
    eps = float(np.quantile(np.sort(D, axis=1)[:, 3], 0.7))
    db = JDRDBSCAN(eps=eps, min_samples=4).fit(D)
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering

    models = {
        "kmedoids": medoid.labels_,
        "dbscan": db.labels_,
        "hdbscan": HDBSCAN(
            min_cluster_size=4, min_samples=3, metric="precomputed", copy=True
        ).fit_predict(np.sqrt(D)),
        "average_linkage": AgglomerativeClustering(
            n_clusters=3, metric="precomputed", linkage="average"
        ).fit_predict(np.sqrt(D)),
    }
    report = {
        "scope": "Small in-sample smoke validation; not a generalization or improvement claim.",
        "seed": seed,
        "n_files": n_files,
        "object_ids": ids,
        "true_labels": truth.tolist(),
        "config": asdict(config),
        "fold_config": asdict(fold_config),
        "fold_diagnostics": {p.stem: c[2] for p, c in zip(files, curves)},
        "grid_relative_max_error": error,
        "second_refinement_relative_max_error": finest_error,
        "nearest_neighbors_unchanged": bool(np.array_equal(nn, nn_fine)),
        "dbscan_eps": eps,
        "kmedoids_inertia": medoid.inertia_,
        "models": {k: evaluate_clustering(truth, v) for k, v in models.items()},
        "labels": {k: v.tolist() for k, v in models.items()},
    }
    save_distance_matrix(
        output / "distances.npz",
        D,
        files,
        config,
        {
            "fold_config": asdict(fold_config),
            "fold_diagnostics": report["fold_diagnostics"],
        },
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D)
    fig.colorbar(im, ax=ax, label="Squared JDR")
    ax.set_xticks(range(n_files), truth, rotation=90)
    ax.set_yticks(range(n_files), truth)
    ax.set_title(f"Validated JDR: {n_files} folded light curves")
    fig.tight_layout()
    fig.savefig(output / "distances.png")
    plt.close(fig)
    return report
