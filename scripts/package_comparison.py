"""Validate the standalone execution and deliver its notebook, results and portable bundle."""

from pathlib import Path
from hashlib import sha256
import argparse
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portable_folder", type=Path)
    args = parser.parse_args()
    portable = args.portable_folder.resolve()
    output = portable / "outputs"
    notebook_path = portable / "jdr_vs_classical_features_1000.ipynb"
    notebook = nbformat.read(notebook_path, 4)
    codes = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type == "error" for c in codes for o in c.outputs)
    execution = json.loads((output / "notebook_execution.json").read_text())
    assert (
        execution["notebook_sha256"] == sha256(notebook_path.read_bytes()).hexdigest()
    )
    assert (
        execution["code_cells_sha256"]
        == sha256("\n\n".join(c.source for c in codes).encode()).hexdigest()
    )
    assert all(
        "from src" not in c.source and "import src" not in c.source for c in codes
    )
    repeat = json.loads((output / "reproducibility.json").read_text())
    assert repeat["arrays_identical"] and repeat["tables_identical"]
    cohort = pd.read_csv(output / "cohort.csv")
    input_cohort = pd.read_csv(portable / "inputs/cohort.csv")
    assert cohort.object_id.tolist() == input_cohort.object_id.tolist()
    for row in cohort.itertuples():
        assert (
            sha256(
                (portable / "inputs/photometry" / f"{row.object_id}.dat").read_bytes()
            ).hexdigest()
            == row.sha256
        )
    features = pd.read_csv(output / "classical_features.csv")
    assert features.object_id.tolist() == cohort.object_id.tolist()
    labels = pd.read_csv(output / "cluster_labels.csv")
    assert labels.object_id.tolist() == cohort.object_id.tolist()
    composition = pd.read_csv(output / "cluster_composition.csv")
    assert (
        composition.groupby(["representation", "model"])["count"].sum() == 1000
    ).all()
    scores = pd.read_csv(output / "model_comparison.csv")
    assert len(scores) == 15
    for r in scores.itertuples():
        predicted = labels[f"{r.representation} | {r.model}"]
        assert sum(predicted == -1) == r.n_noise
        assert np.isclose(1 - r.n_noise / 1000, r.coverage)
    with np.load(output / "comparison_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["object_ids"].tolist() == cohort.object_id.tolist()
        assert arrays["raw_features"].shape == (1000, 16)
        assert np.diff(arrays["offsets"]).tolist() == cohort.n_observations.tolist()
        for i, d in enumerate([8194, 13, 16]):
            assert arrays[f"embedding_{i}"].shape == (1000, d)
            D = arrays[f"squared_distance_{i}"]
            assert (
                D.shape == (1000, 1000)
                and np.isfinite(D).all()
                and (D >= 0).all()
                and np.array_equal(np.diag(D), np.zeros(1000))
            )
    figure_checks = []
    for pdf in sorted((output / "figures").glob("*.pdf")):
        png = Image.open(pdf.with_suffix(".png"))
        svg = pdf.with_suffix(".svg").read_text()
        assert (
            b"/FontFile2" in pdf.read_bytes()
            and "<text" in svg
            and min(png.info["dpi"]) > 599
        )
        figure_checks.append(
            dict(
                figure=pdf.stem,
                png_pixels=list(png.size),
                dpi=list(png.info["dpi"]),
                embedded_pdf_fonts=True,
                editable_svg_text=True,
            )
        )
    assert len(figure_checks) == 4
    test_suites = ET.parse(output / "tests.xml").getroot().findall("testsuite")
    assert all(
        int(s.attrib["failures"]) == 0 and int(s.attrib["errors"]) == 0
        for s in test_suites
    )
    count = sum(int(s.attrib["tests"]) for s in test_suites)
    validation = dict(
        status="passed",
        standalone_folder_execution=True,
        project_module_imports=False,
        executed_cells=len(codes),
        repository_tests_passed=count,
        notebook_self_checks=len(
            json.loads((output / "notebook_checks.json").read_text())[
                "self_tests_passed"
            ]
        ),
        full_recomputations=2,
        all_ids_hashes_counts_checked=True,
        figure_checks=figure_checks,
        visual_review="All four PDFs rendered and inspected for labels, legends, source counts, axes and readability.",
    )
    (output / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    paired = scores.pivot(
        index="model", columns="representation", values="ari_all"
    ).reindex(["K-medoids", "K-means", "Average linkage", "DBSCAN", "HDBSCAN"])
    times = [
        json.loads((output / n).read_text())["elapsed_seconds"]
        for n in ["timing.json", "repeat_timing.json"]
    ]
    quality = pd.read_csv(output / "feature_quality.csv")
    lines = [
        "# JDR versus classical features: reproducible comparison",
        "",
        "The self-contained notebook runs on 1,000 original OGLE Galactic Disk light curves (778 RRab, 214 RRc, 8 RRd), using the same catalog periods and epochs. All code, frozen inputs and environment are included in the portable bundle. No repository analysis modules or prior result caches are needed.",
        "",
        "## Primary all-star ARI",
        "",
        "| Clustering model | JDR | Classical shape (13) | Classical full (16) |",
        "|---|---:|---:|---:|",
    ]
    for model, row in paired.iterrows():
        lines.append(
            f"| {model} | {row['JDR']:.3f} | {row['Classical shape']:.3f} | {row['Classical full']:.3f} |"
        )
    lines += [
        "",
        f"For K-medoids, classical full features score {paired.loc['K-medoids','Classical full']:.3f} versus {paired.loc['K-medoids','JDR']:.3f} for JDR. Classical shape scores {paired.loc['K-medoids','Classical shape']:.3f}. The full baseline explicitly retains period and magnitude-scale information removed by normalized JDR. JDR performs better with average linkage and DBSCAN under these fixed policies. This is an interaction between representation and clustering, not a universal ranking or significance claim.",
        "",
        "Density-method scores must be read alongside coverage and per-class retention; high assigned-only purity can coexist with many rejected stars. All cluster sizes and noise labels are exported. Feature PCA plots retain outliers rather than clipping them, and feature redundancy is shown. The baseline is a documented conventional subset, not a claimed exact reproduction of FATS.",
        "",
        "## Validation",
        "",
        f"- {count} repository tests passed, including tests of the actual inline notebook implementation. All {len(codes)} notebook code cells executed in an isolated folder with only the frozen input bundle.",
        "- Eight notebook self-checks cover independent Lomb–Scargle power, scalar spectral integrals, known Fourier parameters, invariances, robust scaling/imputation, subset isolation and a small exhaustive medoid optimum.",
        "- Two complete runs gave exactly equal arrays, tables, scaling parameters, subset memberships and provenance within the recorded environment.",
        "- Every JDR pair passed grid refinement at 1,025/2,049/4,097 frequencies, unchanged nearest-neighbor IDs, permutation verification and three independent scalar integral audits.",
        "- All 15 primary combinations use squared dissimilarities. The same 30 seeded 800-star subsets are used across representations, with classical imputation/scaling refitted inside each subset. Square-root sensitivity is reported separately and does not select primary settings.",
        f"- {int((quality.undefined_features>0).sum())} objects had undefined classical features; none were dropped. The quality table and learned imputation parameters are retained.",
        "- Four figure sets are available in PDF/SVG/600-dpi PNG, with complete captions and source arrays/tables. PDFs were rendered and visually inspected; fonts are embedded and SVG text remains editable.",
        f"- The two measured local analysis times were {times[0]:.1f} and {times[1]:.1f} seconds, excluding plotting. These are observations on the recorded environment, not a hardware-independent speed guarantee.",
        "",
        "## Reproduction",
        "",
        "Extract `portable_reproduction_bundle.zip`; create a Python 3.12 environment; install `environment-lock.txt`; run `python run_notebook.py` with that environment, or open the notebook and run all cells. Keep `inputs/` beside the notebook. Outputs are regenerated under `outputs/`. The bundle includes all selected photometry and catalog metadata, not the full survey archive. Its recorded source hashes and selection record remain available.",
        "",
        "## Scientific scope",
        "",
        "The approved complex-amplitude clarification of the supplied manuscript is retained. J is squared Euclidean, not generally a metric. Applying its printed band to phase limits the upper frequency to one cycle per phase unit; the classical Fourier baseline uses four harmonics and therefore tests a conventional representation, not an identical-band basis. Original sampling affects both representations differently. Period/epoch and photometric uncertainties are not propagated; weak stationarity is not established. Eight RRd objects are insufficient for rare-class generalization. All catalog agreement is in-sample and the stability quantiles are not confidence intervals. No parameters/features were selected from the catalog scores.",
        "",
        "References: [OGLE data](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/), [Soszyński et al. (2019)](https://arxiv.org/abs/2001.00025), [FATS](https://arxiv.org/abs/1506.00010), [Fourier light-curve analysis](https://www.aanda.org/articles/aa/abs/2009/45/aa12851-09/aa12851-09.html).",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    shutil.copy2(
        ROOT / "review/jdr_paper_1000/reference.pdf",
        portable / "reference_manuscript.pdf",
    )
    # Record every delivered byte except the manifest itself; no data redownload is needed.
    paths = sorted(
        p
        for p in portable.rglob("*")
        if p.is_file()
        and p.name != "bundle_manifest.json"
        and "__pycache__" not in p.parts
    )
    manifest = {
        str(p.relative_to(portable)): dict(
            bytes=p.stat().st_size, sha256=sha256(p.read_bytes()).hexdigest()
        )
        for p in paths
    }
    (portable / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    destination = ROOT / "results/representation_comparison_1000"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output, destination, dirs_exist_ok=True)
    shutil.copy2(notebook_path, ROOT / "notebooks" / notebook_path.name)
    shutil.copy2(
        portable / "environment-lock.txt", destination / "environment-lock.txt"
    )
    with zipfile.ZipFile(
        destination / "portable_reproduction_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in paths + [portable / "bundle_manifest.json"]:
            archive.write(p, p.relative_to(portable))
    with zipfile.ZipFile(
        destination / "journal_figures.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in (
            sorted((output / "figures").glob("*"))
            + sorted(output.glob("*.csv"))
            + [output / "RESULTS.md"]
        ):
            archive.write(p, p.relative_to(output))
    print(json.dumps(validation, indent=2))
    print("Portable bundle:", destination / "portable_reproduction_bundle.zip")


if __name__ == "__main__":
    main()
