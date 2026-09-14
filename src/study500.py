"""Frozen 500-star OGLE experiment, acquisition validation and integral audit."""

from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from src.benchmark import ROOT, PROTOCOL, run_benchmark, compare_runs, _json
from src.metrics import integrate_cross_spectrum_real
from src.read_data import read_ogle_dat

DATA = ROOT / "data" / "ogle_gd_500"
PROTOCOL500 = {
    **PROTOCOL,
    "version": 2,
    "n_objects": 500,
    "scope": "Seeded eligible OGLE Galactic Disk I-band sample; exploratory in-sample comparison, not held-out accuracy.",
    "fold_config": {"min_period": 0.15},
    "acquisition_seed": 20260912,
}


def cohort500():
    table = pd.read_csv(DATA / "cohort.csv")
    if (
        len(table) != 500
        or table.object_id.duplicated().any()
        or table.sha256.duplicated().any()
    ):
        raise ValueError("Expected 500 distinct IDs and photometry contents.")
    if table.object_id.tolist() != sorted(table.object_id):
        raise ValueError("Cohort order changed.")
    files = [ROOT / p for p in table.file]
    for path, row in zip(files, table.itertuples()):
        if not path.is_relative_to(DATA / "photometry") or path.stem != row.object_id:
            raise ValueError("Manifest path/ID mismatch.")
        if sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Photometry hash mismatch: {path.name}")
        frame = read_ogle_dat(path)
        if len(frame) != row.n_observations or len(frame) < 30:
            raise ValueError(f"Observation count mismatch: {path.name}")
    acquisition = json.loads((DATA / "acquisition.json").read_text())
    for record in acquisition["source_files"]:
        source_path = ROOT / record["file"]
        if not source_path.is_relative_to(DATA / "source"):
            raise ValueError("Invalid archived source path.")
        if sha256(source_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Archived source hash mismatch: {source_path.name}")
    labels = {
        line[:19].strip(): line[21:25].strip()
        for line in (DATA / "source" / "ident.dat").read_text().splitlines()
    }
    for row in table.itertuples():
        if labels.get(row.object_id) != row.catalog_type:
            raise ValueError("Selected subtype disagrees with original catalog.")
    if table.catalog_type.value_counts().to_dict() != acquisition["class_counts"]:
        raise ValueError("Selected class counts disagree with acquisition record.")
    # Manifest, selection record and original fixed-width catalogs are provenance inputs.
    catalogs = [
        DATA / "cohort.csv",
        DATA / "acquisition.json",
        DATA / "eligibility_exclusions.csv",
    ]
    catalogs += sorted((DATA / "source").glob("*.dat")) + [DATA / "source" / "README"]
    return files, table, catalogs


def integral_audit(output):
    """Three deterministic distance-quantile pairs, independent adaptive integration."""
    output = Path(output)
    with np.load(output / "experiment.npz", allow_pickle=False) as a:
        phase, shapes, J, ids = a["phase"], a["shapes"], a["J"], a["object_ids"]
    upper = np.triu_indices(len(J), 1)
    values = J[upper]
    order = np.argsort(values, kind="stable")
    rows = []
    for quantile in (0.1, 0.5, 0.9):
        index = order[int(quantile * (len(order) - 1))]
        i, j = int(upper[0][index]), int(upper[1][index])

        def integral(x, y):
            return integrate_cross_spectrum_real(0.5, 6.0, phase, x, phase, y)

        ix, iy, ixy = (
            integral(shapes[i], shapes[i]),
            integral(shapes[j], shapes[j]),
            integral(shapes[i], shapes[j]),
        )
        adaptive = 0.25 / (2 * np.pi) * (ix + iy - 2 * ixy)
        grid = float(J[i, j])
        passed = bool(np.isclose(adaptive, grid, rtol=1e-3, atol=1e-8))
        rows.append(
            dict(
                distance_quantile=quantile,
                object_i=ids[i],
                object_j=ids[j],
                integral_xx=ix,
                integral_yy=iy,
                integral_xy=ixy,
                j_adaptive=adaptive,
                j_grid=grid,
                absolute_difference=abs(adaptive - grid),
                relative_difference=abs(adaptive - grid) / max(abs(adaptive), 1e-30),
                passed=passed,
            )
        )
        if not passed:
            raise AssertionError(f"Spectral integral check failed: {rows[-1]}")
    pd.DataFrame(rows).to_csv(output / "spectral_integral_audit.csv", index=False)
    return rows


def period_audit(output):
    """Post-fit diagnostic only; catalog periods never enter period/model selection."""
    output = Path(output)
    table = pd.read_csv(output / "cohort.csv")
    rows = []
    for row in table.itertuples():
        references = [row.catalog_period]
        if np.isfinite(row.catalog_secondary_period):
            references.append(row.catalog_secondary_period)
        errors = [abs(row.period / p - 1) for p in references]
        err = min(errors)
        alias_error = min(
            abs(row.period / (p * alias) - 1)
            for p in references
            for alias in (0.5, 2.0)
        )
        rows.append(
            dict(
                object_id=row.object_id,
                catalog_type=row.catalog_type,
                fitted_period=row.period,
                catalog_period=row.catalog_period,
                catalog_secondary_period=row.catalog_secondary_period,
                relative_error_nearest_mode=err,
                relative_error_half_double=alias_error,
                agrees_within_1_percent=bool(err < 0.01),
                half_double_alias_only=bool(err >= 0.01 and alias_error < 0.01),
            )
        )
    pd.DataFrame(rows).to_csv(output / "period_audit.csv", index=False)
    return {
        "n_objects": len(rows),
        "within_1_percent": sum(r["agrees_within_1_percent"] for r in rows),
        "half_double_alias_only": sum(r["half_double_alias_only"] for r in rows),
        "interpretation": "Descriptive post-fit check, including either catalog mode for RRd; not used to change fits, exclude objects or select clustering.",
    }


def run_study500(output_dir=None):
    output = Path(output_dir) if output_dir else ROOT / "results" / "benchmark_500"
    start = time.perf_counter()
    run_benchmark(output, PROTOCOL500, cohort_loader=cohort500)
    integrals = integral_audit(output)
    diagnostic = period_audit(output)
    _json(
        output / "additional_checks.json",
        {
            "spectral_integrals_passed": all(r["passed"] for r in integrals),
            "period_diagnostic": diagnostic,
            "acquisition": json.loads((DATA / "acquisition.json").read_text()),
        },
    )
    provenance = json.loads((output / "provenance.json").read_text())
    for relative in ["src/study500.py", "scripts/prepare_ogle500.py"]:
        provenance["code_sha256"][relative] = sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()
    _json(output / "provenance.json", provenance)
    _json(
        output / "timing.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "scope": "Full 500-object fits, three spectral grids, permutation recomputation, clustering, stability, sensitivity, PCA and independent integral audit; excludes acquisition, rendering and second run.",
            "execution": "Local Python process; one numerical thread; no remote compute service.",
        },
    )
    return output


def compare_study500(first, second):
    result = compare_runs(first, second)
    for name in [
        "spectral_integral_audit.csv",
        "period_audit.csv",
        "additional_checks.json",
    ]:
        if (Path(first) / name).read_bytes() != (Path(second) / name).read_bytes():
            raise AssertionError(f"Independent diagnostic differs: {name}")
    result["additional_checks_identical"] = True
    _json(Path(first) / "reproducibility.json", result)
    return result
