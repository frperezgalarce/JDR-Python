# Exact OGLE-III JDR versus classical features

## Scope and status

The complete original OGLE-III catalogs contain 44,217 RR Lyrae IDs: BLG 16,836, LMC 24,906 and SMC 2,475. All catalog entries are inventoried and all original I-band photometry is retained. The frozen analysis cohort contains 43,057 eligible stars and 26,514,552 observations. Exclusions: 1,151 objects with exact duplicate folded phases and 9 with fewer than 30 observations. The existing strict policy excludes whole objects rather than combining duplicate phases or clipping measurements. This selection must be acknowledged in scientific reporting; these objects are not intrinsically unsuitable for other analyses.

This delivery prepares the full exact experiment for a larger-memory machine. Only the explicitly labeled 512-star pilot is executed locally. Full-catalog clustering, numerical convergence and peak resource use remain unvalidated until that run completes. Pilot results cannot be presented as full-catalog findings.

## Run

Use Python 3.12 in a fresh environment and install `requirements-lock.txt`. Extract the portable archive and keep `inputs/` beside `jdr_vs_classical_features_ogle3_all.ipynb`. Open Jupyter, choose this environment's kernel, and Restart Kernel → Run All.

Default `RUN_MODE="pilot"` selects 512 stars with seed 20260912. Set `RUN_MODE="full"` in the first code cell to use every eligible star. Full mode requires at least 192 GiB **available** memory and 200 GiB free disk under a conservative guard. Plan for a machine with at least 256 GiB installed RAM. Actual peak memory and runtime have not been measured at full scale. No cloud execution, paid service or automatic upload is configured.

Outputs go into separate `outputs/pilot/` and `outputs/full/` directories (or the corresponding repository results directory). The independent repeat should use the same output filesystem. Full execution includes three quadrature resolutions, all pair distances, 15 primary models, 30 paired 80% subsamples, square-root distance sensitivity, and a second independent end-to-end computation. This is intentionally expensive, as requested. A failed numerical check stops the study; never bypass it to obtain figures.

## Preserved method and declared changes

JDR uses original irregular phase-folded observations with catalog periods/epochs, normalized unit spectral projections, alpha=1/2, and the 0–2π angular-frequency integral. RRd uses the catalog first-overtone mode. The approved clarification uses complex amplitudes in the cross-spectrum. The notebook explains the spectral integrals and independent checks in detail. It includes the complete implementation without project imports.

The three representations are JDR, 13 classical shape features, and 16 classical full features (adding period and amplitude scalars). Models remain K-medoids, K-means, average linkage, DBSCAN and HDBSCAN with the original policies and squared matrix dissimilarities. Exact distances are used; alternating K-medoids with 20 starts does not guarantee a global optimum. K=4 is explicitly substituted for K=3 because original OGLE-III includes RRab, RRc, RRd and historical RRe labels. No RRe labels are merged or dropped.

All model scores are descriptive and in-sample. Noise remains label −1 for all-object ARI/AMI; coverage and subtype retention are separate. Resampling intervals describe stability, not confidence intervals. Cadence, region, exclusions, bandwidth differences and sampling-count dependence prevent a simple interpretation as astrophysical superiority.

## Provenance and data reconstruction

`inputs/sources_complete.json` records original source URLs, sizes and SHA-256 values. `source/{blg,lmc,smc}/` holds original fixed-width catalogs, READMEs and photometry archives. `inventory.csv`, `cohort.csv` and `exclusions.csv` make selection auditable. The notebook checks source hashes, all eligible raw file hashes and region-specific catalog parsing on every run. It never reads prior clustering results as input.

The supplied acquisition and inventory scripts can reconstruct inputs in a repository layout; the notebook itself needs only the supplied frozen input directory. Preserve archives and manifests together. Original catalog photometry can include OGLE-II supplements; no undocumented time cut is imposed. Sources: https://ftp.astrouw.edu.pl/ogle/ogle3/OIII-CVS/ (BLG/LMC/SMC rrlyr catalogs); cite the source READMEs and Soszyński et al. 2009, 2010, 2011. The supplied JDR manuscript is included unchanged as `reference.pdf`.

Figures are exported as PDF, editable SVG, 600-dpi PNG, and notebook previews. Every figure and caption identifies the execution mode and actual sample size. Dense scatter layers are rasterized in vector outputs. Inspect journal-specific size and font requirements before submission.
