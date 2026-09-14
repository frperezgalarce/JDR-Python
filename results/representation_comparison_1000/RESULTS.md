# JDR versus classical features: reproducible comparison

The self-contained notebook runs on 1,000 original OGLE Galactic Disk light curves (778 RRab, 214 RRc, 8 RRd), using the same catalog periods and epochs. All code, frozen inputs and environment are included in the portable bundle. No repository analysis modules or prior result caches are needed.

## Primary all-star ARI

| Clustering model | JDR | Classical shape (13) | Classical full (16) |
|---|---:|---:|---:|
| K-medoids | 0.292 | 0.380 | 0.665 |
| K-means | 0.307 | 0.314 | 0.427 |
| Average linkage | 0.673 | 0.022 | 0.013 |
| DBSCAN | 0.535 | 0.074 | 0.073 |
| HDBSCAN | 0.242 | 0.152 | 0.259 |

For K-medoids, classical full features score 0.665 versus 0.292 for JDR. Classical shape scores 0.380. The full baseline explicitly retains period and magnitude-scale information removed by normalized JDR. JDR performs better with average linkage and DBSCAN under these fixed policies. This is an interaction between representation and clustering, not a universal ranking or significance claim.

Density-method scores must be read alongside coverage and per-class retention; high assigned-only purity can coexist with many rejected stars. All cluster sizes and noise labels are exported. Feature PCA plots retain outliers rather than clipping them, and feature redundancy is shown. The baseline is a documented conventional subset, not a claimed exact reproduction of FATS.

## Validation

- 45 repository tests passed, including tests of the actual inline notebook implementation. All 14 notebook code cells executed in an isolated folder with only the frozen input bundle.
- Eight notebook self-checks cover independent Lomb–Scargle power, scalar spectral integrals, known Fourier parameters, invariances, robust scaling/imputation, subset isolation and a small exhaustive medoid optimum.
- Two complete runs gave exactly equal arrays, tables, scaling parameters, subset memberships and provenance within the recorded environment.
- Every JDR pair passed grid refinement at 1,025/2,049/4,097 frequencies, unchanged nearest-neighbor IDs, permutation verification and three independent scalar integral audits.
- All 15 primary combinations use squared dissimilarities. The same 30 seeded 800-star subsets are used across representations, with classical imputation/scaling refitted inside each subset. Square-root sensitivity is reported separately and does not select primary settings.
- 0 objects had undefined classical features; none were dropped. The quality table and learned imputation parameters are retained.
- Four figure sets are available in PDF/SVG/600-dpi PNG, with complete captions and source arrays/tables. PDFs were rendered and visually inspected; fonts are embedded and SVG text remains editable.
- The two measured local analysis times were 53.4 and 53.6 seconds, excluding plotting. These are observations on the recorded environment, not a hardware-independent speed guarantee.

## Reproduction

Extract `portable_reproduction_bundle.zip`; create a Python 3.12 environment; install `environment-lock.txt`; run `python run_notebook.py` with that environment, or open the notebook and run all cells. Keep `inputs/` beside the notebook. Outputs are regenerated under `outputs/`. The bundle includes all selected photometry and catalog metadata, not the full survey archive. Its recorded source hashes and selection record remain available.

## Scientific scope

The approved complex-amplitude clarification of the supplied manuscript is retained. J is squared Euclidean, not generally a metric. Applying its printed band to phase limits the upper frequency to one cycle per phase unit; the classical Fourier baseline uses four harmonics and therefore tests a conventional representation, not an identical-band basis. Original sampling affects both representations differently. Period/epoch and photometric uncertainties are not propagated; weak stationarity is not established. Eight RRd objects are insufficient for rare-class generalization. All catalog agreement is in-sample and the stability quantiles are not confidence intervals. No parameters/features were selected from the catalog scores.

References: [OGLE data](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/), [Soszyński et al. (2019)](https://arxiv.org/abs/2001.00025), [FATS](https://arxiv.org/abs/1506.00010), [Fourier light-curve analysis](https://www.aanda.org/articles/aa/abs/2009/45/aa12851-09/aa12851-09.html).
