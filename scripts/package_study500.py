"""Validate and package the executed 500-star study; does not rerun the analysis."""

from pathlib import Path
from hashlib import sha256
import json
import sys
import zipfile
import xml.etree.ElementTree as ET

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "benchmark_500"


def main():
    notebook_path = ROOT / "notebooks" / "experiment_500_journal.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    provenance = json.loads((OUTPUT / "provenance.json").read_text())
    for path, digest in {
        **provenance["code_sha256"],
        **provenance["catalog_sha256"],
    }.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    cohort = pd.read_csv(OUTPUT / "cohort.csv")
    labels = pd.read_csv(OUTPUT / "cluster_labels.csv")
    assert labels.object_id.tolist() == cohort.object_id.tolist()
    assert cohort.sha256.tolist() == provenance["input_sha256"]
    for row in cohort.itertuples():
        assert sha256((ROOT / row.file).read_bytes()).hexdigest() == row.sha256
    comparison = pd.read_csv(OUTPUT / "model_comparison.csv")
    composition = pd.read_csv(OUTPUT / "cluster_composition.csv")
    assert (composition.groupby("model")["count"].sum() == 500).all()
    for row in comparison.itertuples():
        assert int(sum(labels[row.model] == -1)) == row.n_noise
        assert np.isclose(1 - row.n_noise / 500, row.coverage)
    with np.load(OUTPUT / "experiment.npz", allow_pickle=False) as a:
        assert a["object_ids"].tolist() == cohort.object_id.tolist()
        assert a["J"].shape == (500, 500) and a["features"].shape == (500, 8194)
        np.savez_compressed(
            OUTPUT / "figure_source_data.npz",
            **{
                key: a[key]
                for key in ["J", "distance", "phase", "shapes", "pca", "object_ids"]
            },
        )
    repeat = json.loads((OUTPUT / "reproducibility.json").read_text())
    assert (
        repeat["arrays_identical"]
        and repeat["tables_identical"]
        and repeat["additional_checks_identical"]
    )
    figure_checks = []
    for pdf in sorted((OUTPUT / "figures").glob("*.pdf")):
        png = Image.open(pdf.with_suffix(".png"))
        svg = pdf.with_suffix(".svg").read_text()
        assert b"/FontFile2" in pdf.read_bytes() and "<text" in svg
        assert min(png.info["dpi"]) > 599
        figure_checks.append(
            {
                "figure": pdf.stem,
                "png_dimensions": list(png.size),
                "png_dpi": list(png.info["dpi"]),
                "pdf_embedded_fonts": True,
                "svg_editable_text": True,
            }
        )
    assert len(figure_checks) == 4
    test_suites = ET.parse(OUTPUT / "tests.xml").getroot().findall("testsuite")
    n_tests = sum(int(s.attrib["tests"]) for s in test_suites)
    assert all(
        int(s.attrib["failures"]) == 0 and int(s.attrib["errors"]) == 0
        for s in test_suites
    )
    validation = {
        "status": "passed",
        "executed_code_cells": len(code_cells),
        "notebook_sha256": sha256(notebook_path.read_bytes()).hexdigest(),
        "source_and_input_hashes_match": True,
        "all_cluster_counts_match": True,
        "n_tests_passed": n_tests,
        "figure_exports": figure_checks,
        "visual_review": "Four preview figures inspected for labels, legends, axes and 500-object counts. Dense heatmap rasterized at 600 dpi; other marks and text retain vector form.",
    }
    (OUTPUT / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    timings = [
        json.loads((OUTPUT / name).read_text())["elapsed_seconds"]
        for name in ["timing.json", "repeat_timing.json"]
    ]
    checks = json.loads((OUTPUT / "additional_checks.json").read_text())
    integral = pd.read_csv(OUTPUT / "spectral_integral_audit.csv")
    lines = [
        "# 500-light-curve study: results and validation",
        "",
        "Executed notebook: `notebooks/experiment_500_journal.ipynb`.",
        "",
        "This separate seeded sample contains 500 original OGLE Galactic Disk I-band light curves: 394 RRab, 101 RRc and 5 RRd. Six candidates failed the predeclared minimum observation count. All selected measurements and selection/source hashes are retained. It is not a controlled enlargement of the earlier mixed-region 100-object census.",
        "",
        "## Model comparison",
        "",
        "| Model | ARI, all stars | AMI, all stars | Assigned fraction | Clusters | Noise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples():
        lines.append(
            f"| {r.model} | {r.ari_all:.3f} | {r.ami_all:.3f} | {r.coverage:.1%} | {r.n_clusters} | {r.n_noise} |"
        )
    lines += [
        "",
        "K-means has the largest all-star ARI under this fixed protocol; this is descriptive, not evidence of statistically significant superiority. Density methods improve conditional purity while rejecting roughly one fifth of the cohort. No settings were chosen from the external-score sensitivity sweep.",
        "",
        "## Completed checks",
        "",
        f"- {n_tests} tests passed; all {len(code_cells)} notebook code cells executed without an error in a fresh kernel.",
        "- Two complete notebook computations produced exactly equal numerical arrays, tables and additional diagnostics within the recorded environment.",
        "- Every distance entry passed the 1,025 → 2,049 → 4,097 grid comparisons; nearest neighbors were unchanged. Recomputed permutations and the custom/reference DBSCAN check passed.",
        f"- Independent adaptive integration of three spectral-integral terms for three distance-quantile pairs agreed with the grid embedding; maximum relative discrepancy was {integral.relative_difference.max():.3g}.",
        "- All 500 input hashes and recorded code/catalog hashes match. Cluster/noise totals and figure source ordering were checked.",
        "- Four figure sets were visually inspected. PDF fonts are embedded, SVG text is editable and PNG exports are 600 dpi. Captions and figure source data accompany the exports.",
        f"- Measured local analysis runtimes were {timings[0]:.1f} and {timings[1]:.1f} seconds, excluding acquisition/rendering. Each includes 500 period fits, all distance grids, model comparisons, 30 stability refits and integral checks. These are observed wall times, not a portable hardware benchmark.",
        "",
        "## Scientific limitations",
        "",
        f"Fitted periods agree with a catalog mode within 1% for {checks['period_diagnostic']['within_1_percent']}/500 stars. Ten disagreements are retained and listed in `period_audit.csv`; they were not relabeled, removed or repaired using catalog periods. The finite peak search and single-period shape model therefore remain a source of scientific error despite passing numerical checks.",
        "",
        "Five RRd stars cannot support reliable rare-class conclusions. All-star agreement treats noise as one label; inspect assigned-only metrics and coverage together. Subsample quantiles measure clustering stability on fixed fitted curves, not photometric uncertainty or confidence intervals. The frequency band, harmonic model, alignment and K=3 assumption require separate scientific sensitivity studies. These are exploratory in-sample comparisons, not held-out generalization or a verified reproduction of an unspecified JDR paper.",
        "",
        "## Reproduction and data credit",
        "",
        "Use Python 3.12 with `environment-lock.txt` and run `python scripts/validate_study500_notebook.py` from the repository. See the notebook for source acquisition, sampling, data management and the full spectral-integral derivation. Run `python scripts/package_study500.py` after notebook execution to refresh this report and the figure bundle.",
        "",
        "Data: [OGLE OCVS Galactic Disk RR Lyrae](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/). Cite [Soszyński et al. (2019), Acta Astronomica 69, 321](https://arxiv.org/abs/2001.00025). The archived source hashes, not the potentially updated live catalog, define this experiment.",
        "",
    ]
    (OUTPUT / "STUDY_REPORT.md").write_text("\n".join(lines))
    with zipfile.ZipFile(
        OUTPUT / "journal_figures.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        paths = sorted((OUTPUT / "figures").glob("*")) + sorted(OUTPUT.glob("*.csv"))
        paths += [
            OUTPUT / name
            for name in [
                "figure_source_data.npz",
                "summary.json",
                "provenance.json",
                "reproducibility.json",
                "artifact_validation.json",
                "STUDY_REPORT.md",
                "environment-lock.txt",
            ]
        ]
        for path in paths:
            z.write(path, path.relative_to(OUTPUT))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
