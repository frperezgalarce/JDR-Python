"""Exercise the notebook's actual inline code, not a separate implementation."""

import ast
from pathlib import Path
import nbformat
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from src.JDRKMedoids import JDRKMedoids

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/jdr_vs_classical_features_1000.ipynb"
)


def load_definitions():
    # Import scientific dependencies without running input discovery, plotting or full study cells.
    namespace = {}
    imports = """from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
from dataclasses import dataclass
import os,sys,json,tempfile,time,platform
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.integrate import quad
from scipy.stats import spearmanr
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score,silhouette_score
from threadpoolctl import threadpool_limits
from astropy.timeseries import LombScargle
from itertools import combinations
"""
    exec(imports, namespace)
    for cell in nbformat.read(NOTEBOOK, 4).cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("src")
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith("src") for n in node.names)
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            or (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id
                    in {
                        "SHAPE_FEATURES",
                        "FULL_FEATURES",
                        "REPRESENTATIONS",
                        "MODEL_NAMES",
                        "PROTOCOL",
                    }
                    for t in n.targets
                )
            )
        ]
        if nodes:
            exec(
                compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"),
                namespace,
            )
    return namespace


def test_inline_self_checks():
    ns = load_definitions()
    with ns["threadpool_limits"](limits=1):
        assert len(ns["run_self_tests"]()) == 8


def test_inline_medoid_agrees_with_existing_algorithm():
    ns = load_definitions()
    rng = np.random.default_rng(55)
    z = np.vstack([rng.normal(c, 0.3, size=(10, 4)) for c in [0, 3, 6]])
    D = squareform(pdist(z, "sqeuclidean"))
    seed = ns["PROTOCOL"]["seed"]
    with ns["threadpool_limits"](limits=1):
        predicted, details = ns["fit_models"](D, z, seed)
        best = min(
            [JDRKMedoids(3, random_state=seed + i).fit(D) for i in range(20)],
            key=lambda m: m.inertia_,
        )
    assert adjusted_rand_score(predicted["K-medoids"], best.labels_) == 1
    np.testing.assert_allclose(details["medoid_objective"], best.inertia_, rtol=1e-12)


def test_missing_features_do_not_drop_stars():
    ns = load_definitions()
    raw = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    z, scaler = ns["robust_fit_transform"](raw)
    assert (
        z.shape == raw.shape
        and np.isfinite(z).all()
        and scaler["imputed_counts"] == [0, 1]
    )
    import pytest

    with pytest.raises(ValueError):
        ns["robust_fit_transform"](np.array([[1, np.nan], [2, np.nan]]))
