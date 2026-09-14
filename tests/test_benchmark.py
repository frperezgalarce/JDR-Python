import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from src.benchmark import (
    PROTOCOL,
    cohort,
    fit_models,
    scored_models,
    subsampling_stability,
    MODEL_NAMES,
)


def test_census_is_complete_and_unique():
    files, table, catalogs = cohort()
    assert len(files) == 100 and table.sha256.is_unique
    assert table.catalog_type.value_counts().to_dict() == {
        "RRab": 68,
        "RRc": 31,
        "RRd": 1,
    }


def test_model_inputs_stability_and_determinism():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [rng.normal(loc=center, scale=0.2, size=(10, 3)) for center in (0, 3, 6)]
    )
    D = squareform(pdist(X))
    protocol = dict(PROTOCOL, n_init=3, subsamples=2)
    labels, details = fit_models(D, X, protocol)
    repeated, _ = fit_models(D, X, protocol)
    for name in MODEL_NAMES:
        assert np.array_equal(labels[name], repeated[name])
    truth = np.repeat(["a", "b", "c"], 10)
    scores, _ = scored_models(truth, labels, D, np.repeat(["x", "y", "z"], 10))
    assert set(scores.model) == set(MODEL_NAMES)
    assert (scores.coverage >= 0).all() and (scores.coverage <= 1).all()
    stability, memberships = subsampling_stability(D, X, labels, protocol)
    assert len(stability) == 10 and all(len(x) == 24 for x in memberships)
    assert details["dbscan_reference_agreement"]


def test_benchmark_notebook_code_parses():
    import ast
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "experiment_100_journal.ipynb"
    )
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
