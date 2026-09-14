# 1,000-curve catalog-period JDR study

Executed notebook: `notebooks/experiment_1000_paper.ipynb`.

The sample contains 778 RRab, 214 RRc and 8 RRd original Galactic Disk I-band curves, including all previous 500 IDs. Eleven candidates fail the observation threshold. Every selected period and epoch comes directly from the catalog; no period search is performed. RRd uses its first-overtone mode. Original irregular folded observations, not smoothed fits, feed the distance.

## Consistency with the supplied manuscript

The implementation follows Equation 1 projection power and Equation 7 over angular frequency 0 to 2*pi, using the user-approved complex-amplitude cross-spectrum clarification of Equation 6. The real cross-spectrum is integrated. The resulting J is squared Euclidean, so the manuscript’s triangle-inequality claim applies to sqrt(J), not generally to J. Primary K-medoids uses J itself in the objective of Equations 10–12.

The literal power-product reading produces self-J = -422.945226 for the archived audit object, while the clarified self-J is zero. This is documented, not silently repaired. See `review/jdr_paper_1000/IMPLEMENTATION_MAP.md` for the equation mapping.

## Primary comparison

| Model | ARI, all stars | AMI, all stars | Coverage | Clusters | Noise |
|---|---:|---:|---:|---:|---:|
| K-medoids | 0.292 | 0.365 | 100.0% | 3 | 0 |
| K-means | 0.307 | 0.374 | 100.0% | 3 | 0 |
| Average linkage | 0.673 | 0.504 | 100.0% | 3 | 0 |
| DBSCAN | 0.535 | 0.371 | 82.0% | 8 | 180 |
| HDBSCAN | 0.242 | 0.254 | 63.7% | 7 | 363 |

Average linkage has the highest all-star ARI in this fixed experiment, but its three groups include a two-star cluster; this is not recovery of three physical subtypes or a significant-superiority claim. Inspect cluster compositions and stability rather than interpreting ARI alone. Matrix methods receive J; K-means receives the full embedding with a squared objective.

## Validation completed

- 42 tests pass. Independent Astropy classical-LSP and scalar arctangent implementations validate Equation 1, branch handling and the zero-frequency limit.
- All 7 notebook code cells executed in a fresh local kernel. Two complete runs produced exactly matching arrays, tables and provenance within the recorded environment.
- All 499,500 unique pair distances pass successive 1,025/2,049/4,097 grid comparisons; nearest neighbors are unchanged. Feature permutation and custom/reference DBSCAN checks pass.
- Three scalar-reference adaptive-integral audits pass; the maximum relative discrepancy is 3.46e-08.
- All period/epoch inputs exactly equal their catalog values. Data, source, code and manuscript hashes match recorded provenance. No period-recovery accuracy is claimed for quantities supplied as inputs.
- Cluster totals, ordering and source arrays match. Four figure sets were visually inspected; fonts are embedded in PDFs, SVG text remains editable, and PNGs are 600 dpi. The dense heatmap alone is rasterized inside vector files.
- Local analysis times were 48.3 and 47.6 seconds, excluding figure rendering. Shared feature computation makes this practical; period searching is absent.

## Scientific scope and reproduction

Catalog folding and magnitude standardization are declared application choices. The printed band on phase covers at most one cycle per phase unit and may restrict higher-harmonic discrimination. Equation 1 retains sampling/count dependence and is unweighted; photometric errors are preserved but not inserted into that formula. Single-mode RRd folding, epoch/period uncertainty and the small eight-object RRd subgroup remain limitations. Weak stationarity is assumed by the theory but has not been established for these folded stellar data. Multitaper and multiband estimators are not implemented here. Stability quantiles do not propagate observational uncertainty, and this is not held-out validation.

The old 500-study differs in preprocessing, coefficients, band/measure and matrix dissimilarity, so score changes are not an isolated sample-size effect.

Use Python 3.12 with `environment-lock.txt`, then run `python scripts/validate_study1000_notebook.py` from the repository. Run `python scripts/package_study1000.py` afterward to refresh this report and bundle. Source acquisition/selection is reproducible from the frozen archive using `scripts/prepare_ogle1000.py`.

Data credit: [OGLE Galactic Disk RR Lyrae catalog](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/), [Soszyński et al. (2019)](https://arxiv.org/abs/2001.00025). The supplied manuscript is retained unchanged with its hash in provenance.
