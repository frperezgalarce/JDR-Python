"""1000 original irregular light curves, catalog periods and paper-aligned JDR."""

from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
import json
import platform
import sys
import time
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from src.benchmark import (
    ROOT,
    PROTOCOL,
    fit_models,
    scored_models,
    subsampling_stability,
    sensitivity,
    _json,
)
from src.paper_jdr import (
    PaperJDRConfig,
    paper_matrix,
    paper_features,
    paper_amplitude,
    scalar_amplitude_reference,
    adaptive_three_integrals,
)
from src.preprocessing import harmonic_design
from src.read_data import read_ogle_dat
from src.validation import validate_distance_matrix

DATA = ROOT / "data" / "ogle_gd_1000"
PROTOCOL1000 = {
    **PROTOCOL,
    "version": 3,
    "n_objects": 1000,
    "distance": "J from paper Eq. (7), clarified amplitude cross-spectrum; matrix methods use J, K-means full embedding with squared objective",
    "period_policy": "Catalog period and epoch, primary mode (first overtone for RRd); no period search",
    "input": "Original irregular phase-folded measurements, centered and population-SD standardized; no smoothing or interpolation in JDR",
    "spectral_convention": "Paper Eq. (1) normalized projections, shared-basis complex amplitudes; Re cross-spectrum",
    "omega_band": [0.0, float(2 * np.pi)],
    "frequency_measure": "domega",
    "alpha": 0.5,
}


def cohort1000():
    table = pd.read_csv(DATA / "cohort.csv")
    acquisition = json.loads((DATA / "acquisition.json").read_text())
    assert len(table) == 1000 and table.object_id.is_unique and table.sha256.is_unique
    assert table.object_id.tolist() == sorted(table.object_id)
    for r in acquisition["source_files"]:
        if sha256((ROOT / r["file"]).read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError("Frozen source changed.")
    for kind in ["RRab", "RRc", "RRd"]:
        catalog = {
            line[:19].strip(): (float(line[36:46]), float(line[59:69]))
            for line in (DATA / "source" / f"{kind}.dat").read_text().splitlines()
        }
        for row in table[table.catalog_type == kind].itertuples():
            if row.object_id not in catalog or not np.allclose(
                [row.catalog_period, row.catalog_epoch],
                catalog[row.object_id],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Catalog period/epoch/type mismatch.")
    files = [ROOT / path for path in table.file]
    for path, row in zip(files, table.itertuples()):
        if path.stem != row.object_id or path.parent != DATA / "photometry":
            raise ValueError("Path or ID mismatch.")
        if sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError("Raw observation changed.")
    return files, table, acquisition


def prepare_catalog_curves(files, table):
    """Periods/epochs are exact catalog inputs; Fourier fits are for display only."""
    curves, shapes, diagnostics, errors = [], [], [], []
    grid = np.arange(256) / 256
    for path, row in zip(files, table.itertuples()):
        frame = read_ogle_dat(path)
        if len(frame) != row.n_observations:
            raise ValueError("Count mismatch.")
        phase = ((frame.time.to_numpy() - row.catalog_epoch) / row.catalog_period) % 1
        order = np.argsort(phase, kind="stable")
        phase, mag, err = (
            phase[order],
            frame.mag.to_numpy()[order],
            frame.mag_err.to_numpy()[order],
        )
        if np.unique(phase).size != len(phase):
            raise ValueError(
                "Exact duplicate folded phases; require an explicit handling policy."
            )
        curves.append((phase, mag))
        errors.append(err)
        fits = []
        for h in range(1, 5):
            design = harmonic_design(phase, h)
            w = np.min(err) / err
            coef, _, rank, _ = np.linalg.lstsq(design * w[:, None], mag * w, rcond=None)
            if rank != design.shape[1]:
                continue
            loss = float(np.sum(((mag - design @ coef) * w) ** 2))
            bic = len(mag) * np.log(max(loss / len(mag), np.finfo(float).tiny)) + (
                2 * h + 1
            ) * np.log(len(mag))
            fits.append((bic, h, coef, loss))
        if not fits:
            raise ValueError("No full-rank display fit.")
        bic, h, coef, loss = min(fits, key=lambda x: x[0])
        shapes.append(harmonic_design(grid, h) @ coef)
        diagnostics.append(
            {
                "object_id": row.object_id,
                "period": row.catalog_period,
                "epoch": row.catalog_epoch,
                "period_source": "OGLE catalog",
                "display_harmonics": h,
                "display_weighted_residual": loss,
                "display_bic": bic,
                "phase_observations": len(phase),
            }
        )
    return curves, grid, np.array(shapes), pd.DataFrame(diagnostics), errors


def audit_integrals(curves, J, ids, output):
    upper = np.triu_indices(len(J), 1)
    order = np.argsort(J[upper], kind="stable")
    rows = []
    for quantile in [0.1, 0.5, 0.9]:
        k = order[int(quantile * (len(order) - 1))]
        i, j = int(upper[0][k]), int(upper[1][k])
        result = adaptive_three_integrals(curves[i], curves[j])
        grid = float(J[i, j])
        passed = bool(np.isclose(result["j_adaptive"], grid, rtol=1e-3, atol=1e-8))
        rows.append(
            {
                "distance_quantile": quantile,
                "object_i": ids[i],
                "object_j": ids[j],
                **result,
                "j_grid": grid,
                "relative_difference": abs(result["j_adaptive"] - grid)
                / max(abs(result["j_adaptive"]), 1e-30),
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError("Independent Eq. (7) integral mismatch.")
    pd.DataFrame(rows).to_csv(output / "spectral_integral_audit.csv", index=False)
    # Literal Eq. (6) uses powers instead of amplitudes: diagnostic, never clustered.
    w = np.linspace(0, 2 * np.pi, 4097)
    a = paper_amplitude(*curves[0], w)
    power = abs(a) ** 2
    literal = 0.25 / (2 * np.pi) * np.trapezoid(2 * power - 2 * power**2, w)
    _json(
        output / "equation_audit.json",
        {
            "eq1_scalar_reference_max_abs_error": float(
                max(
                    np.max(
                        np.abs(
                            abs(
                                paper_amplitude(
                                    *curves[i], np.linspace(0.013, 2 * np.pi, 31)
                                )
                            )
                            ** 2
                            - np.array(
                                [
                                    abs(scalar_amplitude_reference(*curves[i], w)) ** 2
                                    for w in np.linspace(0.013, 2 * np.pi, 31)
                                ]
                            )
                        )
                    )
                    for i in [0, len(curves) // 2, len(curves) - 1]
                )
            ),
            "literal_power_product_self_j": float(literal),
            "literal_self_object": ids[0],
            "clarified_self_j": float(J[0, 0]),
            "clarification": "User approved complex-amplitude cross-spectrum. Integrate its real part. J is squared Euclidean; sqrt(J) has triangle inequality.",
            "paper_pdf_sha256": sha256(
                (ROOT / "review/jdr_paper_1000/reference.pdf").read_bytes()
            ).hexdigest(),
        },
    )


def run_study1000(output_dir=None):
    output = Path(output_dir) if output_dir else ROOT / "results" / "benchmark_1000"
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        files, table, acquisition = cohort1000()
        curves, phase, shapes, diagnostics, errors = prepare_catalog_curves(
            files, table
        )
        print(
            "Prepared 1000 curves using catalog periods; no period search.", flush=True
        )
        table = table.merge(
            diagnostics, on="object_id", validate="one_to_one", sort=False
        )
        matrices = []
        configs = []
        convergence = []
        for size in PROTOCOL1000["grid_sizes"]:
            config = PaperJDRConfig(n_frequencies=size)
            configs.append(config)
            J, z = paper_matrix(curves, config)
            J = validate_distance_matrix(J)
            if matrices:
                old = matrices[-1]
                delta = abs(old - J)
                nn = lambda m: np.argmin(m + np.eye(len(m)) * 1e99, axis=1)
                check = {
                    "coarse_grid": configs[-2].n_frequencies,
                    "fine_grid": size,
                    "relative_max_error": float(delta.max() / J.max()),
                    "all_pairs_close": bool(np.allclose(old, J, rtol=1e-3, atol=1e-8)),
                    "neighbor_agreement": float(np.mean(nn(old) == nn(J))),
                }
                if not check["all_pairs_close"] or check["neighbor_agreement"] != 1:
                    raise RuntimeError(f"Grid convergence failed: {check}")
                convergence.append(check)
            matrices.append(J)
            print(f"Paper JDR grid {size} complete.", flush=True)
        permutation = np.random.default_rng(PROTOCOL1000["seed"]).permutation(
            len(curves)
        )
        shuffled = np.array([paper_features(*curves[i], config) for i in permutation])
        assert np.allclose(
            squareform(pdist(shuffled, "sqeuclidean")),
            J[np.ix_(permutation, permutation)],
            rtol=1e-12,
            atol=1e-14,
        )
        # Paper Eq. (10) uses J itself. K-means sums squared Euclidean feature distances.
        labels, fit_info = fit_models(J, z, PROTOCOL1000)
        scores, details = scored_models(
            table.catalog_type.to_numpy(), labels, J, table.region.to_numpy()
        )
        print(
            "Five primary clustering models fitted; running stability refits.",
            flush=True,
        )
        stability, memberships = subsampling_stability(J, z, labels, PROTOCOL1000)
        sweep = sensitivity(J, table.catalog_type.to_numpy())
        pca = PCA(n_components=2, svd_solver="full")
        coordinates = pca.fit_transform(z)
        audit_integrals(curves, J, table.object_id.tolist(), output)
        summary = {
            "scope": "1000-object nested OGLE Galactic Disk sample; paper Eq. (1)/(7) with approved Eq. (6) clarification; catalog periods. Exploratory in-sample comparison.",
            "protocol": PROTOCOL1000,
            "jdr_config": asdict(config),
            "n_objects": 1000,
            "class_counts": table.catalog_type.value_counts().to_dict(),
            "region_counts": table.region.value_counts().to_dict(),
            "majority_class_fraction": float(
                table.catalog_type.value_counts().max() / 1000
            ),
            "grid_convergence": convergence,
            "permutation_invariance": True,
            "fit_info": fit_info,
            "models": details,
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "limitations": [
                "Eq. (6) requires the documented, user-approved amplitude interpretation; this is not a literal reproduction of the inconsistent power-product formula.",
                "The printed triangle-inequality claim does not hold for J in general; sqrt(J) is the Euclidean metric.",
                "Catalog phase folding, standardization and choice of first-overtone mode for RRd are explicit application choices, not prescribed by the paper.",
                "Original irregular observations feed Eq. (1); its projection power depends on observation count and sampling, which are not normalized away.",
                "Single-mode folding is incomplete for RRd and modulated stars; periods and epochs have uncertainty.",
                "Only a small RRd subgroup; no reliable rare-class generalization.",
                "One I band and the LSP estimator of Eq. (7); multitaper Eq. (8)/(9) and multiband combination are not implemented in this experiment.",
                "K=3 is a declared subtype-informed assumption; catalog labels are not used to optimize model settings.",
                "Subsampling holds folded observations fixed; no photometric or period uncertainty propagation or model-superiority significance claim.",
                "No direct score comparison with the old 500-study: preprocessing, spectral convention, integration band and matrix dissimilarity have changed.",
            ],
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
            distance=J,
            euclidean_distance=np.sqrt(J),
            features=z,
            phase=phase,
            shapes=shapes,
            pca=coordinates,
            object_ids=table.object_id.to_numpy(dtype=str),
            raw_phase=np.concatenate([t for t, x in curves]),
            raw_magnitude=np.concatenate([x for t, x in curves]),
            raw_error=np.concatenate(errors),
            raw_offsets=np.r_[0, np.cumsum([len(t) for t, x in curves])],
        )
        code_names = [
            "src/study1000.py",
            "src/paper_jdr.py",
            "src/benchmark.py",
            "src/JDRKMedoids.py",
            "src/JDRDBSCAN.py",
            "src/read_data.py",
            "src/metrics.py",
            "src/preprocessing.py",
            "src/evaluation.py",
            "src/validation.py",
            "src/publication_figures1000.py",
            "scripts/prepare_ogle1000.py",
        ]
        source_paths = [
            DATA / "cohort.csv",
            DATA / "acquisition.json",
            DATA / "eligibility_exclusions.csv",
            ROOT / "review/jdr_paper_1000/reference.pdf",
        ] + [ROOT / r["file"] for r in acquisition["source_files"]]
        provenance = {
            "protocol": PROTOCOL1000,
            "object_ids": table.object_id.tolist(),
            "input_sha256": table.sha256.tolist(),
            "code_sha256": {
                name: sha256((ROOT / name).read_bytes()).hexdigest()
                for name in code_names
            },
            "catalog_sha256": {
                str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                for path in source_paths
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
        _json(output / "summary.json", summary)
        _json(output / "provenance.json", provenance)
        _json(output / "subsample_membership.json", memberships)
    _json(
        output / "timing.json",
        {
            "elapsed_seconds": time.perf_counter() - start,
            "scope": "Full local 1000-curve analysis, all grids, primary models, 30 stability refits, sensitivity and scalar-reference integrals; excludes rendering. No period search.",
            "execution": "Local Python, one numerical thread.",
        },
    )
    return output


def compare_study1000(first, second):
    first, second = Path(first), Path(second)
    for name in [
        "summary.json",
        "provenance.json",
        "subsample_membership.json",
        "equation_audit.json",
    ]:
        assert json.loads((first / name).read_text()) == json.loads(
            (second / name).read_text()
        ), name
    for name in [
        "cohort.csv",
        "model_comparison.csv",
        "cluster_labels.csv",
        "subsampling_stability.csv",
        "dbscan_sensitivity.csv",
        "spectral_integral_audit.csv",
    ]:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    with (
        np.load(first / "experiment.npz", allow_pickle=False) as a,
        np.load(second / "experiment.npz", allow_pickle=False) as b,
    ):
        assert a.files == b.files
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key
    result = {
        "independent_full_recomputations": 2,
        "arrays_identical": True,
        "tables_identical": True,
        "provenance_identical": True,
        "additional_checks_identical": True,
        "comparison": "Exact within the recorded environment; not promised across numerical library or platform versions.",
    }
    _json(first / "reproducibility.json", result)
    return result
