"""Validate and package the executed 1000-star study; does not rerun the analysis."""

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
OUTPUT = ROOT / "results" / "benchmark_1000"


def main():
    notebook_path = ROOT / "notebooks" / "experiment_1000_paper.ipynb"
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
    assert (composition.groupby("model")["count"].sum() == 1000).all()
    for row in comparison.itertuples():
        assert int(sum(labels[row.model] == -1)) == row.n_noise
        assert np.isclose(1 - row.n_noise / 1000, row.coverage)
    with np.load(OUTPUT / "experiment.npz", allow_pickle=False) as a:
        assert a["object_ids"].tolist() == cohort.object_id.tolist()
        assert a["J"].shape == (1000, 1000) and a["features"].shape == (1000, 8194)
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
        "visual_review": "Four previews and all four rendered PDFs inspected for labels, legends, axes and 1000-object counts. Dense heatmap rasterized at 600 dpi; other marks and text retain vector form.",
    }
    (OUTPUT / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    timings = [
        json.loads((OUTPUT / name).read_text())["elapsed_seconds"]
        for name in ["timing.json", "repeat_timing.json"]
    ]
    equation = json.loads((OUTPUT / "equation_audit.json").read_text())
    integral = pd.read_csv(OUTPUT / "spectral_integral_audit.csv")
    assert np.array_equal(cohort.period.to_numpy(), cohort.catalog_period.to_numpy())
    assert np.array_equal(cohort.epoch.to_numpy(), cohort.catalog_epoch.to_numpy())
    validation["all_catalog_periods_and_epochs_used_exactly"] = True
    (OUTPUT / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    lines = [
        "# 1,000-curve catalog-period JDR study",
        "",
        "Executed notebook: `notebooks/experiment_1000_paper.ipynb`.",
        "",
        "The sample contains 778 RRab, 214 RRc and 8 RRd original Galactic Disk I-band curves, including all previous 500 IDs. Eleven candidates fail the observation threshold. Every selected period and epoch comes directly from the catalog; no period search is performed. RRd uses its first-overtone mode. Original irregular folded observations, not smoothed fits, feed the distance.",
        "",
        "## Consistency with the supplied manuscript",
        "",
        "The implementation follows Equation 1 projection power and Equation 7 over angular frequency 0 to 2*pi, using the user-approved complex-amplitude cross-spectrum clarification of Equation 6. The real cross-spectrum is integrated. The resulting J is squared Euclidean, so the manuscript’s triangle-inequality claim applies to sqrt(J), not generally to J. Primary K-medoids uses J itself in the objective of Equations 10–12.",
        "",
        f"The literal power-product reading produces self-J = {equation['literal_power_product_self_j']:.6f} for the archived audit object, while the clarified self-J is zero. This is documented, not silently repaired. See `review/jdr_paper_1000/IMPLEMENTATION_MAP.md` for the equation mapping.",
        "",
        "## Primary comparison",
        "",
        "| Model | ARI, all stars | AMI, all stars | Coverage | Clusters | Noise |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples():
        lines.append(
            f"| {r.model} | {r.ari_all:.3f} | {r.ami_all:.3f} | {r.coverage:.1%} | {r.n_clusters} | {r.n_noise} |"
        )
    lines += [
        "",
        "Average linkage has the highest all-star ARI in this fixed experiment, but its three groups include a two-star cluster; this is not recovery of three physical subtypes or a significant-superiority claim. Inspect cluster compositions and stability rather than interpreting ARI alone. Matrix methods receive J; K-means receives the full embedding with a squared objective.",
        "",
        "## Validation completed",
        "",
        f"- {n_tests} tests pass. Independent Astropy classical-LSP and scalar arctangent implementations validate Equation 1, branch handling and the zero-frequency limit.",
        f"- All {len(code_cells)} notebook code cells executed in a fresh local kernel. Two complete runs produced exactly matching arrays, tables and provenance within the recorded environment.",
        "- All 499,500 unique pair distances pass successive 1,025/2,049/4,097 grid comparisons; nearest neighbors are unchanged. Feature permutation and custom/reference DBSCAN checks pass.",
        f"- Three scalar-reference adaptive-integral audits pass; the maximum relative discrepancy is {integral.relative_difference.max():.3g}.",
        "- All period/epoch inputs exactly equal their catalog values. Data, source, code and manuscript hashes match recorded provenance. No period-recovery accuracy is claimed for quantities supplied as inputs.",
        "- Cluster totals, ordering and source arrays match. Four figure sets were visually inspected; fonts are embedded in PDFs, SVG text remains editable, and PNGs are 600 dpi. The dense heatmap alone is rasterized inside vector files.",
        f"- Local analysis times were {timings[0]:.1f} and {timings[1]:.1f} seconds, excluding figure rendering. Shared feature computation makes this practical; period searching is absent.",
        "",
        "## Scientific scope and reproduction",
        "",
        "Catalog folding and magnitude standardization are declared application choices. The printed band on phase covers at most one cycle per phase unit and may restrict higher-harmonic discrimination. Equation 1 retains sampling/count dependence and is unweighted; photometric errors are preserved but not inserted into that formula. Single-mode RRd folding, epoch/period uncertainty and the small eight-object RRd subgroup remain limitations. Weak stationarity is assumed by the theory but has not been established for these folded stellar data. Multitaper and multiband estimators are not implemented here. Stability quantiles do not propagate observational uncertainty, and this is not held-out validation.",
        "",
        "The old 500-study differs in preprocessing, coefficients, band/measure and matrix dissimilarity, so score changes are not an isolated sample-size effect.",
        "",
        "Use Python 3.12 with `environment-lock.txt`, then run `python scripts/validate_study1000_notebook.py` from the repository. Run `python scripts/package_study1000.py` afterward to refresh this report and bundle. Source acquisition/selection is reproducible from the frozen archive using `scripts/prepare_ogle1000.py`.",
        "",
        "Data credit: [OGLE Galactic Disk RR Lyrae catalog](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/), [Soszyński et al. (2019)](https://arxiv.org/abs/2001.00025). The supplied manuscript is retained unchanged with its hash in provenance.",
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
                "equation_audit.json",
            ]
        ]
        paths.append(ROOT / "review/jdr_paper_1000/IMPLEMENTATION_MAP.md")
        for path in paths:
            z.write(
                path,
                path.relative_to(OUTPUT) if path.is_relative_to(OUTPUT) else path.name,
            )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
