# 500-light-curve study: results and validation

Executed notebook: `notebooks/experiment_500_journal.ipynb`.

This separate seeded sample contains 500 original OGLE Galactic Disk I-band light curves: 394 RRab, 101 RRc and 5 RRd. Six candidates failed the predeclared minimum observation count. All selected measurements and selection/source hashes are retained. It is not a controlled enlargement of the earlier mixed-region 100-object census.

## Model comparison

| Model | ARI, all stars | AMI, all stars | Assigned fraction | Clusters | Noise |
|---|---:|---:|---:|---:|---:|
| K-medoids | 0.542 | 0.493 | 100.0% | 3 | 0 |
| K-means | 0.551 | 0.498 | 100.0% | 3 | 0 |
| Average linkage | 0.536 | 0.491 | 100.0% | 3 | 0 |
| DBSCAN | 0.384 | 0.377 | 80.4% | 8 | 98 |
| HDBSCAN | 0.411 | 0.389 | 81.0% | 8 | 95 |

K-means has the largest all-star ARI under this fixed protocol; this is descriptive, not evidence of statistically significant superiority. Density methods improve conditional purity while rejecting roughly one fifth of the cohort. No settings were chosen from the external-score sensitivity sweep.

## Completed checks

- 34 tests passed; all 8 notebook code cells executed without an error in a fresh kernel.
- Two complete notebook computations produced exactly equal numerical arrays, tables and additional diagnostics within the recorded environment.
- Every distance entry passed the 1,025 → 2,049 → 4,097 grid comparisons; nearest neighbors were unchanged. Recomputed permutations and the custom/reference DBSCAN check passed.
- Independent adaptive integration of three spectral-integral terms for three distance-quantile pairs agreed with the grid embedding; maximum relative discrepancy was 1.34e-06.
- All 500 input hashes and recorded code/catalog hashes match. Cluster/noise totals and figure source ordering were checked.
- Four figure sets were visually inspected. PDF fonts are embedded, SVG text is editable and PNG exports are 600 dpi. Captions and figure source data accompany the exports.
- Measured local analysis runtimes were 86.8 and 85.4 seconds, excluding acquisition/rendering. Each includes 500 period fits, all distance grids, model comparisons, 30 stability refits and integral checks. These are observed wall times, not a portable hardware benchmark.

## Scientific limitations

Fitted periods agree with a catalog mode within 1% for 490/500 stars. Ten disagreements are retained and listed in `period_audit.csv`; they were not relabeled, removed or repaired using catalog periods. The finite peak search and single-period shape model therefore remain a source of scientific error despite passing numerical checks.

Five RRd stars cannot support reliable rare-class conclusions. All-star agreement treats noise as one label; inspect assigned-only metrics and coverage together. Subsample quantiles measure clustering stability on fixed fitted curves, not photometric uncertainty or confidence intervals. The frequency band, harmonic model, alignment and K=3 assumption require separate scientific sensitivity studies. These are exploratory in-sample comparisons, not held-out generalization or a verified reproduction of an unspecified JDR paper.

## Reproduction and data credit

Use Python 3.12 with `environment-lock.txt` and run `python scripts/validate_study500_notebook.py` from the repository. See the notebook for source acquisition, sampling, data management and the full spectral-integral derivation. Run `python scripts/package_study500.py` after notebook execution to refresh this report and the figure bundle.

Data: [OGLE OCVS Galactic Disk RR Lyrae](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/). Cite [Soszyński et al. (2019), Acta Astronomica 69, 321](https://arxiv.org/abs/2001.00025). The archived source hashes, not the potentially updated live catalog, define this experiment.
