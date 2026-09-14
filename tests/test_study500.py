"""Checks for the separate data cohort and independent validation paths."""

import ast
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from src.jdr import JDRConfig, feature_vector
from src.study500 import cohort500, integral_audit, period_audit, ROOT


def test_frozen500_cohort_is_verified_and_separate():
    files, table, catalogs = cohort500()
    assert len(files) == 500
    assert all(p.parent.name == "photometry" for p in files)
    assert table.region.unique().tolist() == ["GD"]
    assert table.catalog_type.value_counts().to_dict() == {
        "RRab": 394,
        "RRc": 101,
        "RRd": 5,
    }


def test_integral_audit_against_analytic_harmonic_shapes(tmp_path):
    phase = np.arange(256) / 256
    shapes = np.array(
        [
            np.sin(2 * np.pi * phase + shift) + 0.2 * np.cos(4 * np.pi * phase)
            for shift in [0, 0.3, 0.7, 1.4]
        ]
    )
    cfg = JDRConfig(f_min=0.5, f_max=6, n_frequencies=4097)
    z = np.array([feature_vector(phase, shape, cfg) for shape in shapes])
    J = squareform(pdist(z, "sqeuclidean"))
    np.savez(
        tmp_path / "experiment.npz",
        phase=phase,
        shapes=shapes,
        J=J,
        object_ids=np.array(["a", "b", "c", "d"]),
    )
    rows = integral_audit(tmp_path)
    assert len(rows) == 3 and all(row["passed"] for row in rows)
    assert max(row["relative_difference"] for row in rows) < 1e-3


def test_catalog_period_check_recognizes_mode_alias_and_failure(tmp_path):
    pd.DataFrame(
        {
            "object_id": ["a", "b", "c", "d"],
            "catalog_type": ["RRab", "RRab", "RRd", "RRc"],
            "period": [0.5001, 1, 0.7, 0.25],
            "catalog_period": [0.5, 0.5, 0.5, 0.33],
            "catalog_secondary_period": [np.nan, np.nan, 0.7, np.nan],
        }
    ).to_csv(tmp_path / "cohort.csv", index=False)
    audit = period_audit(tmp_path)
    assert audit["within_1_percent"] == 2
    assert audit["half_double_alias_only"] == 1


def test_notebook500_code_parses():
    notebook = json.loads(
        (ROOT / "notebooks" / "experiment_500_journal.ipynb").read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
