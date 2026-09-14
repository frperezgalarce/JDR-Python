# All figure code is included in the notebook; figures use only freshly computed outputs.
REP_COLORS = {
    "JDR": "#0072B2",
    "Classical shape": "#D55E00",
    "Classical full": "#009E73",
}
CLASS_COLORS = {"RRab": "#0072B2", "RRc": "#D55E00", "RRd": "#009E73"}
FIG_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "jdr-feature-comparison",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
}


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        directory / f"{name}.pdf",
        dpi=600,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(directory / f"{name}.svg", dpi=600, metadata={"Date": None})
    fig.savefig(directory / f"{name}.png", dpi=600)
    fig.savefig(directory / f"{name}_preview.png", dpi=160)
    plt.close(fig)


def draw_figures(output):
    output = Path(output)
    folder = output / "figures"
    scores = pd.read_csv(output / "model_comparison.csv")
    stability = pd.read_csv(output / "subsampling_stability.csv")
    cohort = pd.read_csv(output / "cohort.csv")
    features = pd.read_csv(output / "classical_features.csv")
    summary = json.loads((output / "summary.json").read_text())
    composition = pd.read_csv(output / "cluster_composition.csv")
    retention = pd.read_csv(output / "class_retention.csv")
    with np.load(output / "comparison_arrays.npz", allow_pickle=False) as a:
        xy = [a[f"pca_{i}"].copy() for i in range(3)]
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(
            2, 2, figsize=(183 / 25.4, 121 / 25.4), layout="constrained"
        )
        for ax, metric, letter, title in zip(
            axes.flat,
            ["ari_all", "ami_all", "coverage", "ari_stability_all"],
            "abcd",
            [
                "Catalog agreement: ARI",
                "Catalog agreement: AMI",
                "Assigned fraction",
                "Partition stability",
            ],
        ):
            for r, representation in enumerate(REPRESENTATIONS):
                y = np.arange(5) + (r - 1) * 0.20
                if metric != "ari_stability_all":
                    values = (
                        scores[scores.representation == representation]
                        .set_index("model")
                        .loc[MODEL_NAMES, metric]
                    )
                    ax.scatter(
                        values,
                        y,
                        s=15,
                        marker=["o", "s", "^"][r],
                        color=REP_COLORS[representation],
                        label=representation,
                        zorder=3,
                    )
                else:
                    for k, model in enumerate(MODEL_NAMES):
                        values = stability[
                            (stability.representation == representation)
                            & (stability.model == model)
                        ][metric]
                        lo, med, hi = np.quantile(values, [0.1, 0.5, 0.9])
                        ax.plot(
                            [lo, hi],
                            [y[k], y[k]],
                            lw=1.6,
                            color=REP_COLORS[representation],
                        )
                        ax.scatter(
                            [med],
                            [y[k]],
                            s=10,
                            color=REP_COLORS[representation],
                            marker=["o", "s", "^"][r],
                        )
                ax.set_yticks(range(5), MODEL_NAMES)
                ax.set_ylim(4.6, -0.6)
                ax.set_xlim(-0.1 if metric != "coverage" else 0, 1.04)
                ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
                ax.set_xlabel(
                    "ARI with restricted full partition"
                    if metric == "ari_stability_all"
                    else (
                        "Fraction"
                        if metric == "coverage"
                        else metric.split("_")[0].upper()
                    )
                )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, folder, "figure_1_representation_comparison")
        fig, axes = plt.subplots(
            1, 3, figsize=(183 / 25.4, 67 / 25.4), layout="constrained"
        )
        for i, (ax, representation) in enumerate(zip(axes, REPRESENTATIONS)):
            for kind, marker in zip(["RRab", "RRc", "RRd"], ["o", "^", "D"]):
                mask = cohort.catalog_type == kind
                ax.scatter(
                    xy[i][mask, 0],
                    xy[i][mask, 1],
                    s=7 if kind != "RRd" else 20,
                    marker=marker,
                    color=CLASS_COLORS[kind],
                    alpha=0.65,
                    label=kind,
                    linewidths=0.2,
                    edgecolors="white",
                    rasterized=True,
                )
            ev = summary["pca_explained"][representation]
            ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)")
            ax.set_title(
                f"{chr(97+i)}  {representation}", loc="left", fontweight="bold"
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, folder, "figure_2_representation_geometry")
        fig, axes = plt.subplots(
            2, 3, figsize=(183 / 25.4, 126 / 25.4), layout="constrained"
        )
        for i, representation in enumerate(REPRESENTATIONS):
            ax = axes[0, i]
            subset = composition[
                (composition.representation == representation)
                & (composition.model == "K-medoids")
            ]
            left = np.zeros(3)
            for kind in ["RRab", "RRc", "RRd"]:
                counts = (
                    subset[subset.catalog_type == kind]
                    .set_index("cluster")["count"]
                    .reindex(range(3), fill_value=0)
                    .to_numpy()
                )
                ax.barh(
                    range(3),
                    counts,
                    left=left,
                    color=CLASS_COLORS[kind],
                    label=kind,
                    height=0.65,
                )
                left += counts
            for k, count in enumerate(left):
                ax.text(count + 8, k, str(int(count)), va="center", fontsize=6)
            ax.set_yticks(range(3), ["C0", "C1", "C2"])
            ax.invert_yaxis()
            ax.set_xlim(0, 1050)
            ax.set_xticks([0, 500, 1000])
            ax.set_xlabel("Number of stars")
            ax.set_title(
                f"{chr(97+i)}  {representation}: K-medoids",
                loc="left",
                fontweight="bold",
            )
            ax = axes[1, i]
            sub = retention[retention.representation == representation]
            values = (
                sub.pivot(index="model", columns="catalog_type", values="retention")
                .reindex(MODEL_NAMES)[["RRab", "RRc", "RRd"]]
                .to_numpy()
            )
            ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_yticks(range(5), MODEL_NAMES)
            ax.set_xticks(range(3), ["RRab\n(n=778)", "RRc\n(n=214)", "RRd\n(n=8)"])
            for k in range(5):
                for j in range(3):
                    ax.text(
                        j,
                        k,
                        f"{values[k,j]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if values[k, j] > 0.55 else "black",
                    )
            ax.set_title(
                f"{chr(100+i)}  Retention: {representation}",
                loc="left",
                fontweight="bold",
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        save_figure(fig, folder, "figure_3_cluster_composition_retention")
        fig, axes = plt.subplots(
            2, 2, figsize=(183 / 25.4, 119 / 25.4), layout="constrained"
        )
        ax = axes[0, 0]
        for kind, marker in zip(["RRab", "RRc", "RRd"], ["o", "^", "D"]):
            mask = cohort.catalog_type == kind
            ax.scatter(
                features.loc[mask, "log10_period"],
                10 ** features.loc[mask, "log10_q95_q05_amplitude"],
                s=8 if kind != "RRd" else 24,
                marker=marker,
                color=CLASS_COLORS[kind],
                alpha=0.65,
                label=kind,
                edgecolors="white",
                linewidths=0.2,
                rasterized=True,
            )
        ax.set_xlabel("log10 catalog period [days]")
        ax.set_ylabel("Q95 − Q05 magnitude range [mag]")
        ax.set_title(
            "a  Information added by full features", loc="left", fontweight="bold"
        )
        ax.legend(frameon=False, ncol=3)
        ax = axes[0, 1]
        corr = features[SHAPE_FEATURES].corr(method="spearman").to_numpy()
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(13), range(1, 14), fontsize=5)
        ax.set_yticks(range(13), range(1, 14), fontsize=5)
        ax.set_xlabel("Shape feature index (notebook dictionary)")
        ax.set_ylabel("Shape feature index")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Spearman correlation")
        ax.set_title("b  Feature redundancy", loc="left", fontweight="bold")
        ax = axes[1, 0]
        nuisance = pd.read_csv(output / "sampling_diagnostics.csv")
        for i, quantity in enumerate(["n_observations", "baseline_days"]):
            v = (
                nuisance[nuisance.quantity == quantity]
                .set_index("representation")
                .loc[REPRESENTATIONS, "pc1_spearman"]
                .abs()
            )
            ax.scatter(
                v,
                np.arange(3) + (i - 0.5) * 0.18,
                marker=["o", "s"][i],
                s=18,
                label=["Observation count", "Time baseline"][i],
            )
        ax.set_yticks(range(3), REPRESENTATIONS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("|Spearman correlation with PC1|")
        ax.legend(frameon=False, loc="lower right", fontsize=6)
        ax.set_title(
            "c  Sampling associations (descriptive)", loc="left", fontweight="bold"
        )
        ax = axes[1, 1]
        alternative = pd.read_csv(output / "distance_convention_sensitivity.csv")
        for r, representation in enumerate(REPRESENTATIONS):
            primary = (
                scores[scores.representation == representation]
                .set_index("model")
                .loc[MODEL_NAMES, "ari_all"]
            )
            alt = (
                alternative[alternative.representation == representation]
                .set_index("model")
                .loc[MODEL_NAMES, "ari_all"]
            )
            ax.scatter(
                alt.to_numpy() - primary.to_numpy(),
                np.arange(5) + (r - 1) * 0.2,
                s=13,
                color=REP_COLORS[representation],
                marker=["o", "s", "^"][r],
                label=representation,
            )
        ax.axvline(0, c=".5", ls=":", lw=0.8)
        ax.set_yticks(range(5), MODEL_NAMES)
        ax.invert_yaxis()
        ax.set_xlabel("ARI(sqrt dissimilarity) − ARI(primary)")
        ax.set_title(
            "d  Distance-convention sensitivity", loc="left", fontweight="bold"
        )
        ax.legend(frameon=False, loc="lower left", fontsize=6)
        save_figure(fig, folder, "figure_4_feature_and_protocol_diagnostics")
    captions = {
        "figure_1_representation_comparison": "Figure 1 | Representation comparison with identical clustering policies on 1,000 original light curves. JDR is contrasted with 13 standardized classical shape features and those same features augmented by log period, log magnitude SD and log Q95−Q05 amplitude (16 total). ARI/AMI retain rejected points under label −1; coverage is shown separately. Stability intervals are empirical 10th–90th percentiles over 30 shared 800-star subsets, with median markers. Imputation and scaling are refitted inside every classical-feature subset. These are stability summaries, not confidence intervals, and no model is selected from the catalog scores. All matrix methods receive squared dissimilarities in the primary comparison.",
        "figure_2_representation_geometry": "Figure 2 | Two-dimensional PCA views of the three representations. All 1,000 objects are shown, colored by catalog subtype with distinct markers; eight RRd stars cannot support rare-class inference. Each PCA is fitted separately; axes and explained-variance fractions refer to different spaces and are not directly comparable. Models use full representations, never these two coordinates. Dense point layers are rasterized at 600 dpi in vector exports.",
        "figure_3_cluster_composition_retention": "Figure 3 | Primary K-medoids composition and all-model retention. Top: counts in all three medoid clusters for each representation, with observed sizes; IDs are arbitrary across panels. Bottom: assigned fraction within each catalog subtype for every model, including all rejected stars in the denominator. All cluster compositions, including density-method noise, are exported in cluster_composition.csv. Fully assigned partitions need not recover astrophysical classes.",
        "figure_4_feature_and_protocol_diagnostics": "Figure 4 | Interpreting the representation comparison. a, Catalog period and robust magnitude range are additional scalar information in Classical full; JDR and Classical shape remove them as scalars. b, Spearman feature correlations indicate redundancy; feature indices follow the notebook dictionary. c, Absolute PC1 associations with observation count and time baseline are descriptive, not evidence of causal confounding. d, Change in all-star ARI when matrix methods receive square-root dissimilarities instead of their primary squared dissimilarities. K-means retains its full-vector squared objective. This predeclared sensitivity is not used to choose primary settings.",
    }
    json_write(folder / "captions.json", captions)
    (folder / "CAPTIONS.md").write_text(
        "# Figure captions\n\n" + "\n\n".join(captions.values()) + "\n"
    )
    return folder
