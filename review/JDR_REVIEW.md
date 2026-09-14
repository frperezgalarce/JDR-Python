# JDR implementation and clustering review

Reviewed 12 September 2026. **The current repository cannot be considered correctly validated end to end.** DBSCAN's core expansion logic passed reference checks, but the distance definition, preprocessing, evaluation, and execution paths have consequential defects.

This is an audit, not a replacement implementation. Application code and notebooks were left unchanged. `verify.py` contains repeatable probes; `verification_results.json` records their results and `verification_requirements.txt` captures the temporary Python 3.12 environment. Run from the repository root with `PYTHONPATH=. python review/verify.py`. This script reports known defects as evidence; successful completion does not mean the implementation is correct.

## Executed verification and limits

- 90 random Euclidean-matrix cases: custom DBSCAN labels and core indices matched scikit-learn exactly. This covers ordinary valid inputs, not every edge case.
- A real two-curve serial JDR check returned 0.007940107789427947 in both directions. Both curves have the same 932.17781-day span, so this case does not expose band asymmetry. All six underlying integrals emitted `IntegrationWarning` after warnings were explicitly re-enabled; the value is not certified converged.
- An isolated unequal-span band-selection test, using controlled constant auto-spectra and zero cross-spectrum, returned 0.7956951379879308 versus 0.39780778025819236. This is a control-flow unit probe, not a physical-data result.
- The active reader expanded the two real curves from 95 and 108 observations to 200 each.
- Root `D_folded.npy` is 5×5 with maximum distance 10.37476; `notebooks/D_folded.npy` is 100×100 with maximum 0.19400. Both are finite and symmetric, but neither includes the ordered IDs or configuration needed to establish provenance. Their different scales make transferring eps settings unjustified.
- Reproduced the metadata ordering bug, empty k-medoids cluster, acceptance of malformed matrices, tau orthogonality discrepancy and R-wrapper convergence-status defect.
- Confirmed `main.main()` fails with `NameError: read_ogle_dat`. All Python files parsed; notebook syntax checking found the unexpected indentation in experiment1 cell 6.
- Simple pointwise checks passed for nonnegative finite auto-spectrum and positive-affine invariance; sign inversion changes the cross-spectrum as expected for phase-sensitive coefficients.

No full notebook reruns, full-distance recomputation, actual R backend execution, serial/process-pool equivalence test, or astronomical subtype improvement benchmark was performed. Proposed method improvements remain hypotheses to evaluate. The tests establish specific defects and a limited working baseline, not that every other code path is correct.

## Priority findings

### 1. Pair order changes the frequency band (high)

`src/jdr.py:35–40` and `158–172` derive the upper frequency bound exclusively from the first curve. Consequently J(A,B) and J(B,A) need not agree. `min(f_max, f_max)` has no effect. Both matrix builders default to computing one direction and copying it into the other triangle; a symmetric saved matrix therefore does not demonstrate symmetry of JDR. Changing file order can change clustering even after restoring the original row order.

Use an explicitly defined dataset-wide band and grid. A symmetric pair-specific band would remove direction dependence but would still compare different pairs in different feature spaces, undermining a common geometric interpretation and efficient caching. Time-domain frequencies and folded-phase harmonics require separate configurations and units.

### 2. The active reader destroys the irregular sampling representation (high)

`src/jdr.py` imports `src.read_data.read_ogle_dat`, which always interpolates onto 200 uniformly spaced points (`src/read_data.py:45–51`). A different reader with the same name in `src/metrics.py` preserves observations. Thus nominally similar entry points can analyze different data. Interpolation across long gaps invents smooth behavior and can erase short-period variability. The upper frequency bound then depends on the artificial 200-point count, not the original cadence. The README's “no interpolation required” description does not match the active workflow.

Keep raw observations by default; make resampling explicit and optional. Handle duplicate times, non-finite values, zero variance, insufficient observations and zero time span deliberately. On folded data, ordinary interpolation also ignores the periodic boundary between phases 1 and 0. If a phase grid is desired, use a periodic fitted curve with uncertainty estimates.

### 3. Purity can be calculated against the wrong stars (high)

`get_true_labels_from_clusters` in `results/run.py:483–507` and experiments 3/4 cell 22 uses `df.iloc[light_curve]` when given cluster indices. These indices refer to the distance matrix's file list, not the metadata table. This fails for reordered, filtered or combined surveys. A two-object reordered metadata probe reproduces the wrong labels.

Construct `object_ids = [Path(f).stem for f in files]`, validate metadata IDs are unique, and explicitly reindex metadata by `object_ids`. Reject missing IDs instead of silently skipping them. Save this ordered ID list with every matrix and use `model.labels_` for evaluation.

### 4. Reported purity rewards almost complete rejection (high)

Experiments 3 and 4 save a “best” configuration with purity 0.8, **one cluster and 95 noise objects out of 100**: only five objects contribute to the score. Even this figure is subject to the label-mapping bug. In addition, `clusters_` excludes noise before scoring, so setting `ignore_noise=False` cannot restore rejected observations. Optimizing and reporting on the same labels adds selection bias.

Report coverage, cluster sizes, ARI/AMI and class-specific retention alongside conditional purity. Show a coverage-versus-quality frontier and impose a predeclared minimum coverage if selecting a single operating point. Evaluate selected settings on untouched stars or survey fields. ARI calculated with all noise as one label has its own interpretation caveat, so report both all-object and assigned-object variants with coverage.

### 5. Integration failures are silently accepted (high)

`src/jdr.py:5` suppresses warnings globally. `src/metrics.py:374` discards quadrature error estimates. In the R wrapper, any returned integration object is printed as `OK`, including a non-convergence message (`316–317`); the caller accepts any finite value without inspecting that message (`427–428`). A mocked non-converged R result is accepted on the first attempt. This tests Python control flow, not an actual R execution.

Remove global warning suppression; expose status and estimated error; distinguish convergence from a finite answer. Validate parameters, including frequency order, `alpha` and the Jacobian enum. Do not silently clip substantial negative distances or replace non-finite distances with zero. Integrate the squared feature difference directly on a shared grid to avoid cancellation between three independently approximated integrals, and verify grid convergence.

### 6. Legacy tau matches R code, but not the standard orthogonalizing shift (scientific decision required)

`src/metrics.py:67` uses `atan(sum(sin(2ωt))/sum(cos(2ωt)))/(2π)`. The standard unweighted Lomb–Scargle shift is `atan2(sum(sin(2ωt)), sum(cos(2ωt)))/(2ω)`. The supplied `R files/tau.r` explicitly distinguishes its Fortran-compatible version from the paper-compatible version; Python copied the former. A probe shows a nonzero sine–cosine cross term for the legacy formula and numerical zero for the standard formula.

This is not merely a Python translation error. Preserve an explicitly named legacy mode for historical reproduction and establish a separately validated scientific mode against the intended JDR paper. The repository's JDR reference is `xxxx`, so exact agreement with the intended published definition cannot be certified. Changing tau also requires a consistent phase convention for cross-series coefficients; independently orthogonalized coefficient bases should not automatically be interpreted as physical cross phases. Audit the `df` versus `dω` normalization against the source definition before changing the final `2π` factor.

### 7. Parallel JDR changes numerical precision (high for absolute dates)

`src/jdr.py:29–33` converts both times and magnitudes to float32; serial JDR does not. Float32 spacing around JD 2,454,833 is **0.25 days**, enough to collapse closely spaced observations. Converting back to float64 inside the estimator cannot recover lost information. Preserve float64; explicitly choose a time origin consistent with the phase convention. Add serial/parallel equivalence and timestamp-resolution tests.

### 8. Execution and validation gaps (medium)

- `main.py:20` raises `NameError` because `read_ogle_dat` is not imported; the integrator import is also missing, and file paths do not point into the data directory.
- `run.sh:73` launches nonexistent `notebooks/run.py`; the actual script is `results/run.py`. Its installation command includes standard-library modules such as `random`, `subprocess` and `tempfile`.
- `results/run.py` uses inconsistent working-directory assumptions and performs the full workflow at import time. A main guard is required for portable process spawning, particularly on macOS/Windows. Its final plotting loop uses `../data` despite initial reads using `data`.
- Python 3.9 advertised in the README is incompatible with the eagerly evaluated `int | None` annotation in `JDRKMedoids.py` without postponed annotations. Declare a supported Python version and package dependencies.
- K-medoids accepts negative, asymmetric and infinite matrices. Both estimators accept asymmetric matrices and nonzero diagonals; DBSCAN also accepts infinity. Define and enforce a matrix contract, distinguishing deliberately missing distances if supported.
- K-medoids' nearest-medoid ties can leave requested clusters empty: an all-zero 3×3 matrix with two medoids returns one nonempty cluster. Handle medoid self-assignment/ties explicitly. It is alternating medoid optimization, not full PAM swap search; use multiple starts and report achieved objective/stability.
- Seeding global random generators in the constructor makes results depend on unrelated intervening random calls. Use a per-estimator generator.
- Starting a fresh three-worker process pool per pair incurs repeated process startup. Pairwise parallel mode can also nest pools if passed `jdr_parallel`; avoid that combination.

## Notebook audit

Cell numbers below are one-based and include markdown cells. All ten notebooks were inspected as stored source and outputs; complete notebook pipelines were not rerun because several regenerate/delete folded data or download datasets, and the expensive full matrix runs are not needed to reproduce the review findings.

| Notebook | Assessment |
|---|---|
| `experiment1.ipynb` | Cell 6 has an unexpected leading indent and fails syntax parsing. Saved outputs therefore are not a clean execution of current source. First 50 sorted RR Lyrae files, one K=5 run and visual inspection are exploratory, not validated class recovery. |
| `experiment2.ipynb` and `experiment2 copy.ipynb` | Initialization/imports are markdown in cell 1; cell 2 references undefined `PROJECT_ROOT` in a fresh kernel. Stored content of the copies is identical in the audit extraction. Purity-free inspection is useful but no held-out evidence is provided. |
| `experiment3.ipynb` and `experiment4.ipynb` | Extracted source and text outputs are identical. Current `n_files=500` conflicts with saved 100-object outputs. Random 300-point subsampling occurs without a seed, and period estimation reads another random subset. Hardcoded Linux paths prevent portability. Metadata loads `ident.dat` twice and excludes loaded `df_4` from concatenation. Cache files contain no ID/configuration manifest. Best saved purity rejects 95% of stars. |
| `experiment_mnist.ipynb` | Uses PCA–Euclidean distances, so it tests generic medoid clustering, not JDR. Cell 9 reports `y[i]` instead of `y_sub[i]`; displayed medoid/cluster digits are wrong after random subsampling. Thirty images and ten clusters are too small for substantive accuracy claims. |
| `calculte_distances.ipynb` | Useful backend exploration, but stored quadrature warnings indicate unverified integrals. A comment gives R JDR 0.001591215 whereas displayed Python JDR is 0.003116824; these are not a controlled matched-input assertion. Saved sampling diagnostics also need regeneration after reader changes. |
| `test.ipynb` | The stored ~0.256% grid/quad difference concerns one auto-spectrum, while both ordinary and tighter quadrature report subdivision failures. Zero auto-spectrum imaginary part follows algebraically and is insufficient evidence of correct JDR. No assertions. |
| `convert_keplet.ipynb.ipynb` | Download/conversion utility, not a clustering validation. Earlier conversion does not guard nonpositive flux; later conversion replaces invalid flux with the smallest positive value, inventing observations. Prefer filtering with an aligned mask. Preserve Astropy time metadata instead of inferring formats from numerical magnitude. Verify downloaded products and dependencies before a fresh run. |
| `R files/light_curves.ipynb` | Stored `NameError` outputs show R commands executing in a Python kernel. Use an R kernel or an explicit R bridge. |

## Recommended JDR improvement strategy

### First: establish a common, testable spectral representation

For the current coefficient definitions, write the cosine and sine components as a_i(f), b_i(f). On a **common band** and with identical preprocessing in all terms, the implemented algebra gives

`J_ij = β/(4π) ∫ [(a_i-a_j)^2 + (b_i-b_j)^2] df`.

With common positive quadrature weights q_m, concatenate `sqrt(β q_m/(4π)) * [a_i(f_m), b_i(f_m)]` into a feature vector z_i. Then the grid approximation is exactly `J_ij = ||z_i-z_j||²`. This is a derivation from the current code, not a claim that its estimator is already scientifically correct.

Compute each star's representation once, then use vectorized distances or blocked Gram-matrix operations. This avoids repeated file reads, repeated normalization and three adaptive integrals per pair. Feature construction costs roughly O(NLM) for N stars, L observations and M frequencies; dense pairwise comparison remains O(N²M) and storage O(N²). For larger datasets, use blocks, memory mapping, landmark medoids or a sparse neighbor graph after validating approximation error.

`J` is a squared-distance dissimilarity under these assumptions, not generally a metric satisfying the triangle inequality. `sqrt(J)` is a Euclidean distance in the finite-grid feature representation, potentially only a pseudometric on original curves. DBSCAN neighborhoods are unchanged if thresholds are transformed consistently (`eps_sqrt = sqrt(eps_J)`), whereas k-medoids' sum objective can change. Do not claim a new clustering gain merely from rescaling alpha: for 0<alpha<1 it only multiplies every fixed-representation distance by the same β.

### Second: make the representation match the scientific question

For RR Lyrae subtype clustering, test a phase-shape representation while retaining period information separately:

1. Estimate periods from original observations with measurement errors and a floating mean; refine candidate peaks and check half/double-period aliases. Set grid resolution from baseline and desired peak sampling, rather than using 20,000 frequencies for every span. Irregular sampling does not justify a universal average-Nyquist cutoff. [Astropy frequency-grid documentation](https://docs.astropy.org/en/stable/api/astropy.timeseries.LombScargle.html), [VanderPlas's methodological paper](https://arxiv.org/abs/1703.09824).
2. Fit a robust periodic harmonic model to folded observations. Compare several harmonic orders using validation or a complexity penalty. Avoid alignment to a single brightest, potentially noisy observation.
3. Compare canonical alignment to minimization over a **single global circular phase shift**. Do not independently align each harmonic, which can erase shape information. Validate phase invariance and aliases explicitly; do not assume the existing tau-based representation supplies this invariance.
4. Compare full phase-sensitive coefficients against normalized power/amplitude features. Power-only features remove phase information and may lose subtype discrimination; this is an ablation, not an automatic improvement.
5. Test a composite dissimilarity combining standardized JDR shape, log-period difference, amplitude and harmonic ratios/phase differences. Choose scales on development data and assess each added feature independently. Existing standardization removes positive amplitude scaling, so min–max magnitude scaling before JDR is largely redundant; amplitude must be retained separately if useful.
6. Perturb photometry using its errors and resample observation subsets; quantify distance and cluster stability. Compare same-shape curves under different cadences to identify sampling-driven clustering. Use robust cleaning rather than treating interpolated magnitude errors as independent measurements.

### Third: compare clustering algorithms after fixing the distance

Keep the custom DBSCAN as a checked baseline; select eps from empirical k-neighbor distances rather than arbitrary ranges transferred between raw and folded units. Compare multi-start k-medoids, average-linkage clustering and HDBSCAN with precomputed distances. HDBSCAN can explore variable density without choosing one global eps, but cannot repair a flawed distance and is not guaranteed to improve subtype recovery. [HDBSCAN precomputed-distance documentation](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html).

Use a frozen, ID-aligned development/validation/test manifest spanning classes and survey fields. Report ARI, AMI, coverage, per-class retention, cluster sizes and resampling stability; retain purity only as a supplementary diagnostic. Account for the fact that density clustering on a held-out cohort is a transductive evaluation, whereas prediction against fixed medoids is inductive. [scikit-learn clustering evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation).

## Proposed acceptance gates

1. Clean-kernel execution for every supported notebook and script; deterministic file manifests, periods and seeds; no silent metadata gaps.
2. Tests for pair symmetry, file-order permutation invariance, self-distance, nonnegativity, finite output and serial/parallel agreement; malformed inputs rejected explicitly.
3. Exact pointwise agreement between squared-feature differences and auto/cross-term algebra; convergence across M, 2M and 4M grid sizes with stable nearest neighbors. Choose tolerances from downstream stability, not only integration defaults.
4. Matched Python/R fixtures with fixed input arrays, precision, tau convention, integration units and tolerances. Check convergence status, not only returned values. [SciPy quadrature diagnostics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).
5. Synthetic recovery experiments varying phase, amplitude, noise, cadence, baseline and period independently; compare legacy JDR, corrected/common-grid JDR and proposed shape variants.
6. Compare algorithms on the same corrected representation and untouched evaluation cohorts. Promote a change only when improvements persist across resampling and do not depend on discarding most stars.

The recommended next implementation is the common-grid JDR engine with explicit preprocessing and ID-aligned evaluation. Optimizing DBSCAN parameters before these repairs would produce more precise-looking results without resolving their validity.
