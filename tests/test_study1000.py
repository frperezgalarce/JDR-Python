import ast
import json
import numpy as np
import pandas as pd
from src.study1000 import ROOT, cohort1000, prepare_catalog_curves


def test_cohort1000_nested_and_period_metadata_valid():
    files, table, acquisition = cohort1000()
    assert len(files) == 1000 and table.catalog_type.value_counts().to_dict() == {
        "RRab": 778,
        "RRc": 214,
        "RRd": 8,
    }
    assert set(pd.read_csv(ROOT / "data/ogle_gd_500/cohort.csv").object_id).issubset(
        set(table.object_id)
    )
    assert acquisition["excluded_before_1000"] == 11


def test_known_period_keeps_raw_irregular_samples(tmp_path, monkeypatch):
    import src.preprocessing

    def forbidden(*args, **kwargs):
        raise AssertionError("Period search must not run.")

    monkeypatch.setattr(src.preprocessing, "fold_light_curve", forbidden)
    t = np.sort(np.random.default_rng(9).uniform(100, 120, 65))
    p = 0.57312
    epoch = 99.82
    x = 15 + np.sin(2 * np.pi * (t - epoch) / p)
    path = tmp_path / "test.dat"
    np.savetxt(path, np.c_[t, x, np.full(len(t), 0.05)], fmt="%.15g")
    table = pd.DataFrame(
        [
            dict(
                object_id="test",
                catalog_period=p,
                catalog_epoch=epoch,
                n_observations=len(t),
            )
        ]
    )
    curves, phase, shape, diagnostics, errors = prepare_catalog_curves([path], table)
    assert len(curves[0][0]) == len(t) and len(phase) == 256
    np.testing.assert_allclose(curves[0][0], np.sort(((t - epoch) / p) % 1), atol=2e-12)
    assert diagnostics.period.iloc[0] == p and diagnostics.epoch.iloc[0] == epoch
    np.testing.assert_allclose(np.sort(curves[0][1]), np.sort(x), atol=1e-12)


def test_notebook1000_code_parses():
    n = json.loads((ROOT / "notebooks/experiment_1000_paper.ipynb").read_text())
    for c in n["cells"]:
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
