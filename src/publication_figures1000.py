"""Editable journal-scale figures and source tables for the 1000-object paper-aligned JDR study."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from src.benchmark import MODEL_NAMES

CLASS_COLORS = {"RRab": "#0072B2", "RRc": "#D55E00", "RRd": "#009E73"}
MODEL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
SHORT = ["K-medoids", "K-means", "Average linkage", "DBSCAN", "HDBSCAN"]
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "jdr-paper-1000-v1",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}
WIDTH = 183 / 25.4


def panel(ax, label, title):
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="bottom",
    )
    ax.set_title(title, loc="left", pad=7)
    ax.spines[["top", "right"]].set_visible(False)


def export(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        folder / f"{name}.pdf",
        dpi=600,
        metadata={
            "Creator": "JDR reproducible benchmark",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(folder / f"{name}.svg", dpi=600, metadata={"Date": None})
    fig.savefig(folder / f"{name}.png", dpi=600)
    fig.savefig(folder / f"{name}_preview.png", dpi=160)
    plt.close(fig)


def make_figures(output_dir):
    root = Path(output_dir)
    folder = root / "figures"
    cohort = pd.read_csv(root / "cohort.csv")
    scores = (
        pd.read_csv(root / "model_comparison.csv").set_index("model").loc[MODEL_NAMES]
    )
    stability = pd.read_csv(root / "subsampling_stability.csv")
    sweep = pd.read_csv(root / "dbscan_sensitivity.csv")
    predictions = pd.read_csv(root / "cluster_labels.csv")
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "experiment.npz", allow_pickle=False) as archive:
        D = archive["distance"]
        J = archive["J"]
        shapes = archive["shapes"]
        phase = archive["phase"]
        xy = archive["pca"]
        assert (
            archive["object_ids"].tolist()
            == cohort.object_id.tolist()
            == predictions.object_id.tolist()
        )
    n_objects = len(cohort)
    classes = ["RRab", "RRc", "RRd"]
    with plt.rc_context(STYLE):
        # Figure 1: census composition and representations.
        fig, axes = plt.subplots(
            2, 2, figsize=(WIDTH, 128 / 25.4), layout="constrained"
        )
        ax = axes[0, 0]
        regions = sorted(cohort.region.unique())
        bottom = np.zeros(len(regions))
        for kind in classes:
            counts = np.array(
                [
                    sum((cohort.region == r) & (cohort.catalog_type == kind))
                    for r in regions
                ]
            )
            ax.bar(
                regions,
                counts,
                bottom=bottom,
                color=CLASS_COLORS[kind],
                label=kind,
                width=0.58,
                edgecolor="white",
                linewidth=0.35,
            )
            bottom += counts
        for i, n in enumerate(bottom):
            ax.text(i, n + 0.8, str(int(n)), ha="center", fontsize=6)
        ax.set_ylim(0, max(bottom) * 1.20)
        ax.set_ylabel("Number of light curves")
        ax.legend(frameon=False, ncol=3, loc="upper right")
        panel(ax, "a", f"OGLE cohort (N = {n_objects})")
        ax = axes[0, 1]
        chosen = []
        for kind in classes:
            indices = np.flatnonzero(cohort.catalog_type.to_numpy() == kind)
            index = indices[np.argmin(D[np.ix_(indices, indices)].sum(axis=1))]
            normalized = (shapes[index] - np.mean(shapes[index])) / np.std(
                shapes[index]
            )
            chosen.append(
                {
                    "class": kind,
                    "object_id": cohort.object_id.iloc[index],
                    "selection": "within-class distance medoid, for display only",
                }
            )
            ax.plot(
                phase,
                normalized,
                color=CLASS_COLORS[kind],
                label=kind,
                linestyle={"RRab": "-", "RRc": "--", "RRd": ":"}[kind],
                linewidth=1.2,
            )
        ax.invert_yaxis()
        ax.set_xlabel("Aligned phase")
        ax.set_ylabel("Standardized fitted magnitude")
        ax.legend(frameon=False, ncol=3, loc="upper center")
        panel(ax, "b", "Representative fitted shapes")
        pd.DataFrame(chosen).to_csv(root / "figure_representatives.csv", index=False)
        ax = axes[1, 0]
        order = (
            cohort.assign(
                _class=pd.Categorical(
                    cohort.catalog_type, categories=classes, ordered=True
                )
            )
            .sort_values(["_class", "region", "object_id"])
            .index.to_numpy()
        )
        image = ax.pcolormesh(
            np.arange(n_objects + 1),
            np.arange(n_objects + 1),
            J[np.ix_(order, order)],
            cmap="viridis",
            shading="flat",
            rasterized=n_objects > 200,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        counts = [sum(cohort.catalog_type == c) for c in classes]
        edges = np.r_[0, np.cumsum(counts)]
        # The singleton RRd is marked separately to avoid an unreadably crowded tick.
        centers = (edges[:-1] + edges[1:]) / 2
        ax.set_xticks(centers[:2], classes[:2])
        ax.set_yticks(centers[:2], classes[:2])
        for edge in edges[1:-1]:
            ax.axhline(edge, color="white", lw=0.6)
            ax.axvline(edge, color="white", lw=0.6)
        ax.text(
            0.5,
            -0.14,
            f"RRd: final {counts[2]} rows/columns",
            transform=ax.transAxes,
            fontsize=6,
            ha="center",
        )
        fig.colorbar(
            image, ax=ax, fraction=0.047, pad=0.025, label="Paper-aligned JDR, J"
        )
        panel(ax, "c", "Distances ordered by catalog subtype")
        pd.DataFrame(
            {
                "display_index": range(n_objects),
                "object_id": cohort.object_id.iloc[order].to_numpy(),
            }
        ).to_csv(root / "figure_distance_order.csv", index=False)
        ax = axes[1, 1]
        for kind, marker in zip(classes, ["o", "^", "D"]):
            mask = cohort.catalog_type == kind
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=12 if kind != "RRd" else 28,
                marker=marker,
                c=CLASS_COLORS[kind],
                alpha=0.8,
                label=kind,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )
        variance = 100 * np.array(summary["pca_explained_variance_ratio"])
        ax.set_xlabel(f"PC1 ({variance[0]:.1f}% of feature variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}%)")
        ax.legend(frameon=False, ncol=3, loc="best")
        panel(ax, "d", "PCA visualization of full JDR features")
        export(fig, folder, "figure_1_cohort_geometry")
        # Figure 2: different diagnostics remain in separate axes (no dual-axis charts).
        fig, axes = plt.subplots(
            2, 2, figsize=(WIDTH, 119 / 25.4), layout="constrained"
        )
        y = np.arange(5)
        ax = axes[0, 0]
        for metric, shift, marker, color, label in [
            ("ari_all", -0.12, "o", "#0072B2", "ARI"),
            ("ami_all", 0.12, "s", "#D55E00", "AMI"),
        ]:
            ax.scatter(
                scores[metric],
                y + shift,
                s=19,
                marker=marker,
                c=color,
                label=label,
                zorder=3,
            )
        ax.axvline(0, color=".7", lw=0.5)
        ax.set_xlim(-0.1, 1.05)
        ax.set_yticks(y, SHORT)
        ax.invert_yaxis()
        ax.set_xlabel("Agreement with catalog labels (all stars)")
        ax.set_ylim(5.05, -0.5)
        ax.legend(frameon=False, ncol=2, loc="lower right")
        panel(ax, "a", "External agreement")
        ax = axes[0, 1]
        ax.scatter(
            scores.coverage, y - 0.12, s=20, marker="o", c="#0072B2", label="Coverage"
        )
        ax.scatter(
            scores.purity_assigned,
            y + 0.12,
            s=20,
            marker="s",
            facecolors="none",
            edgecolors="#D55E00",
            label="Assigned-only purity",
        )
        ax.axvline(summary["majority_class_fraction"], ls=":", c=".5", lw=0.7)
        ax.set_xlim(0, 1.06)
        ax.set_yticks(y, SHORT)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction")
        ax.legend(frameon=False, loc="lower left", ncol=1)
        panel(ax, "b", "Retention and conditional purity")
        ax = axes[1, 0]
        rng = np.random.default_rng(44)
        for i, name in enumerate(MODEL_NAMES):
            v = stability.loc[stability.model == name, "ari_stability_all"].to_numpy()
            lo, median, hi = np.quantile(v, [0.1, 0.5, 0.9])
            ax.plot([lo, hi], [i, i], color=MODEL_COLORS[i], lw=2.5, alpha=0.65)
            ax.scatter(
                v,
                i + rng.uniform(-0.14, 0.14, len(v)),
                c=MODEL_COLORS[i],
                s=5,
                alpha=0.45,
                linewidths=0,
            )
            ax.scatter([median], [i], c="black", s=12, marker="|", zorder=4)
        ax.set_xlim(-0.1, 1.05)
        ax.set_yticks(y, SHORT)
        ax.invert_yaxis()
        ax.set_xlabel("ARI with restricted full-data partition")
        panel(ax, "c", "Clustering stability: 30 × 80% subsamples")
        ax = axes[1, 1]
        retention = np.array(
            [
                [summary["models"][m]["class_retention"][c] for c in classes]
                for m in MODEL_NAMES
            ]
        )
        im = ax.imshow(retention, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(y, SHORT)
        ax.set_xticks(
            range(3), [f"{c}\n(n = {sum(cohort.catalog_type==c)})" for c in classes]
        )
        for i in range(5):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{retention[i,j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if retention[i, j] > 0.55 else "black",
                )
        panel(ax, "d", "Retention within each catalog subtype")
        export(fig, folder, "figure_2_model_comparison")
        # Figure 3: actual cluster sizes and noise, no post-hoc class mapping.
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(
                WIDTH,
                max(68, 24 + 4.5 * max(predictions[m].nunique() for m in MODEL_NAMES))
                / 25.4,
            ),
            layout="constrained",
        )
        contingency = []
        for i, (ax, name) in enumerate(zip(axes, MODEL_NAMES)):
            labels = predictions[name].to_numpy()
            groups = sorted(set(labels) - {-1}) + ([-1] if -1 in labels else [])
            left = np.zeros(len(groups))
            for kind in classes:
                values = np.array(
                    [
                        sum((labels == g) & (cohort.catalog_type.to_numpy() == kind))
                        for g in groups
                    ]
                )
                ax.barh(
                    np.arange(len(groups)),
                    values,
                    left=left,
                    color=CLASS_COLORS[kind],
                    edgecolor="white",
                    linewidth=0.3,
                    height=0.68,
                )
                for group, count in zip(groups, values):
                    contingency.append(
                        {
                            "model": name,
                            "cluster": int(group),
                            "catalog_type": kind,
                            "count": int(count),
                        }
                    )
                left += values
            for row, total in enumerate(left):
                ax.text(total + 1, row, str(int(total)), va="center", fontsize=6)
            ax.set_yticks(
                range(len(groups)), ["Noise" if g == -1 else f"C{g}" for g in groups]
            )
            ax.invert_yaxis()
            ax.set_xlim(0, n_objects * 1.05)
            ax.set_xticks([0, n_objects // 2, n_objects])
            ax.set_xlabel("Number of stars")
            panel(ax, chr(97 + i), name)
        fig.legend(
            handles=[
                Line2D([], [], color=CLASS_COLORS[c], lw=4, label=c) for c in classes
            ],
            loc="outside lower center",
            ncol=3,
            frameon=False,
        )
        pd.DataFrame(contingency).to_csv(root / "cluster_composition.csv", index=False)
        export(fig, folder, "figure_3_cluster_composition")
        # Figure 4: numerical convergence and the entire sensitivity grid.
        fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 65 / 25.4), layout="constrained")
        ax = axes[0]
        convergence = summary["grid_convergence"]
        errors = [r["relative_max_error"] for r in convergence]
        grids = [r["fine_grid"] for r in convergence]
        ax.semilogy(grids, errors, "o-", color="#0072B2", ms=3)
        ax.set_xticks(grids, [str(n) for n in grids])
        ax.set_xlabel("Refined frequency-grid size")
        ax.set_ylabel("Maximum discrepancy / max(J)")
        ax.text(
            0.04,
            0.08,
            "Nearest neighbors: 100% agreement\nat both refinements",
            transform=ax.transAxes,
            fontsize=6,
        )
        panel(ax, "a", "Frequency-grid convergence")
        for ax, metric, letter, title in [
            (axes[1], "ari_all", "b", "DBSCAN agreement sensitivity"),
            (axes[2], "coverage", "c", "DBSCAN coverage sensitivity"),
        ]:
            for m, color, marker in [
                (3, "#0072B2", "o"),
                (5, "#D55E00", "s"),
                (8, "#009E73", "^"),
            ]:
                subset = sweep[sweep.min_samples == m]
                ax.plot(
                    subset.eps_quantile,
                    subset[metric],
                    color=color,
                    marker=marker,
                    ms=2,
                    markevery=3,
                    lw=0.9,
                    label=f"m = {m}",
                )
            ax.axvline(0.7, c=".5", ls=":", lw=0.7)
            ax.set_ylim(-0.1 if metric == "ari_all" else 0, 1.05)
            ax.set_xlabel("k-distance quantile defining ε")
            ax.set_ylabel(
                "ARI (all stars)" if metric == "ari_all" else "Assigned fraction"
            )
            ax.legend(frameon=False, loc="best")
            panel(ax, letter, title)
        export(fig, folder, "figure_4_numerical_sensitivity")
    captions = {
        "figure_1_cohort_geometry": f"Figure 1 | Catalog-period OGLE cohort and paper-aligned JDR. a, The 1,000 original Galactic Disk I-band curves comprise 778 RRab, 214 RRc and 8 RRd. The sample contains the previous 500 IDs, with an unchanged selection seed. b, Uncertainty-weighted harmonic fits at the catalog period and epoch of within-subtype J-medoids, for display only. JDR uses the original irregular folded observations, not these fitted curves. c, J from Eq. (7), with the approved complex-amplitude cross-spectrum interpretation of Eq. (6), ordered by subtype then ID. The small RRd block is indicated. d, Two-component PCA of the complete quadrature embedding, for display only. Models use J or the full embedding, never the PCA projection. Source IDs and display order are exported.",
        "figure_2_model_comparison": "Figure 2 | Fixed-protocol comparison on 1,000 stars. a, All-star ARI and AMI against catalog subtypes; rejected objects retain the common noise label -1. b, Coverage and purity conditional on assignment; the dotted line is the 0.778 majority-class fraction. c, Thirty shared seeded 800-star subsets: clustering is refitted using the fixed model policies, and each partition is compared with the full partition restricted to the same subset. Points are replicate ARIs, thick segments empirical 10th-90th percentiles, ticks medians. These are stability summaries, not confidence intervals or measurement/period uncertainty propagation. d, Subtype retention with denominators. Eight RRd stars do not support rare-class generalization. K-medoids minimizes sums of J as in the paper; K-means minimizes squared Euclidean feature distances, and the other matrix models also receive J. Assigned silhouette, where reported in tables, is calculated in J dissimilarity, not sqrt(J).",
        "figure_3_cluster_composition": "Figure 3 | Composition of every learned cluster. Segment colors denote catalog subtypes and end labels give observed counts. Cluster IDs are arbitrary and need not correspond across methods. Noise is shown explicitly. Every panel uses a 0-1050 count axis; no class relabeling or exclusion of noise is applied. J is the primary dissimilarity for all matrix methods. Bar totals account for all 1,000 stars in each model.",
        "figure_4_numerical_sensitivity": "Figure 4 | Numerical integration and model sensitivity. a, J matrices from 1,025, 2,049 and 4,097 angular frequencies over [0, 2*pi] are compared. Every entry meets rtol=1e-3 and atol=1e-8, with unchanged nearest neighbors at both refinements. Independent scalar-reference adaptive quadrature also checks three pairs in the exported integral audit. b,c, All 54 DBSCAN configurations (minimum samples 3, 5 or 8 including self; 18 J-neighbor-distance quantiles from 0.10 to 0.95). The primary policy remains minimum samples 5 and quantile 0.70, marked by the dotted line. These post-fit catalog-score diagnostics do not select the primary model. The study is exploratory and in-sample.",
    }
    (folder / "captions.json").write_text(json.dumps(captions, indent=2) + "\n")
    (folder / "CAPTIONS.md").write_text(
        "# Figure captions\n\n" + "\n\n".join(captions.values()) + "\n"
    )
    (folder / "FIGURE_SPECIFICATIONS.md").write_text("""# Figure specifications

All figures are 183 mm wide with 7-point main type, 6-point ticks and 8-point panel labels. Colorblind-accessible subtype colors are reinforced with marker/line differences where relevant. PDF and SVG preserve vector linework and editable text; PDFs embed TrueType fonts. For cohorts larger than 200, the dense distance heatmap alone is rasterized at 600 dpi to keep vector files practical. PNG exports are 600 dpi. Separate 160-dpi previews are for notebook display only. Heatmaps use a sequential scale; no smoothing or selective recoloring is applied. Raster-image DPI is not a substitute for vector quality.

Figures were designed with the Nature research figure guide in mind, not certified for a particular journal. Confirm the destination journal's final dimensions, fonts and color requirements before submission:
https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/

No performance confidence intervals or significance claims are implied. Source tables and captions are included with the experiment.
""")
    return folder
