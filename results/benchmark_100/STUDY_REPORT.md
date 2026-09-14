# JDR clustering comparison: 100 local light curves

## Design

This is a census of the available 100 RR Lyrae files: 68 RRab, 31 RRc and one RRd; 50 BLG, 45 GD and five LMC. No stars are silently skipped or substituted. Object order and SHA-256 input/catalog hashes are recorded. Catalog labels are used for evaluation and display, never to choose the primary hyperparameters. The complete computation is repeated independently in the notebook.

The period/model fitting and common-basis spectral coefficients follow the documented version 2 implementation. JDR uses the fixed 0.5–6 cycles/phase band. All matrix-based methods use sqrt(J); k-means uses its exact full feature embedding. PCA is visualization only. K=3 is a fixed reference; density methods may return other cluster counts. Primary settings were specified before inspecting this experiment's scores. No density-model score is reported without coverage/noise accounting.

## Results

| Method | ARI, all objects | AMI, all objects | Coverage | Clusters | Rejected |
|---|---:|---:|---:|---:|---:|
| K-medoids | 0.604 | 0.557 | 100% | 3 | 0 |
| K-means | 0.624 | 0.566 | 100% | 3 | 0 |
| Average linkage | 0.699 | 0.569 | 100% | 3 | 0 |
| DBSCAN | 0.536 | 0.526 | 78% | 2 | 22 |
| HDBSCAN | 0.760 | 0.606 | 98% | 2 | 2 |

HDBSCAN has the highest all-object ARI in this census, but the study does not demonstrate a statistically significant or generalizable winner. DBSCAN's high assigned-only purity comes with 22 rejected curves. Noise points share label −1 in all-object ARI/AMI; assigned-only ARI, silhouette, class retention and conditional purity are also exported. A one-cluster majority-label purity baseline is 0.68.

## Validation and reproducibility

- Shared-grid refinements use 1025, 2049 and 4097 frequencies. Every pair must meet rtol=10⁻³ and atol=10⁻⁸; every nearest-neighbor identity must be preserved. The finest matrix is used for all primary fits.
- Nonnegativity, zero diagonal, finite output, symmetry and feature-recomputation permutation invariance are asserted.
- Custom DBSCAN must match scikit-learn on the full census matrix.
- Thirty deterministic shared 80-object subsets refit every model. Percentiles summarize clustering stability; they are not confidence intervals. Period fits remain fixed in these subset runs.
- A separate sensitivity grid reports all 54 combinations of three neighborhood sizes and 18 distance quantiles. Its catalog-label scores are not used to select primary settings.
- `reproducibility.json` records exact equality of tables, numerical arrays and provenance after two full original-data recomputations. This is an environment-specific guarantee, not a bitwise cross-platform promise.
- `tests.xml` records the regression suite. `notebook_execution.json` records fresh-kernel execution. Package versions, a complete environment lock and source hashes are included.

## Figures

Four multi-panel figure sets are saved in `figures/`, each with PDF, SVG, 600-dpi PNG and a smaller display preview. Artwork is 183 mm wide with editable text, embedded PDF TrueType fonts, accessible colors and explicit panel labels. The full captions are in `figures/CAPTIONS.md`; `figures/FIGURE_SPECIFICATIONS.md` documents export settings and the journal-style reference.

1. Cohort, representative fitted shapes, ordered distance matrix and PCA visualization.
2. Model agreement, retention, conditional purity, clustering stability and subtype retention.
3. Catalog composition and size of every cluster, including noise.
4. Numerical convergence and the complete DBSCAN sensitivity curves.

All plotted values can be traced to CSV/NPZ artifacts, including representative IDs and matrix display order. High-resolution artwork is prepared for manuscript use; final destination-journal specifications still need to be applied.

## Scientific limits

One RRd does not permit reliable RRd performance estimation. Period estimates have not been checked against independent catalog periods, and full photometric/period uncertainty is not propagated. Region agreement is descriptive, not a causal confounding test. This local census is not a survey-separated or held-out benchmark. No confidence intervals, statistical winner claims, or exact correspondence to an unidentified original JDR paper are asserted. These are boundaries of the experiment, not hidden exceptions to its numerical checks.

## Reproduce

Install `requirements-dev.txt` (or the captured lock in an appropriate Python 3.12 environment), then execute `notebooks/experiment_100_journal.ipynb` from a fresh kernel. The notebook runs both independent full computations and creates the figures. The command-line equivalent is `python scripts/validate_benchmark_notebook.py` from the repository root.
