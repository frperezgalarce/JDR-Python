# Version 2 — fixes and small-set validation

Completed 12 September 2026. The implementation defects identified in the audit were repaired, and the revised pipeline was validated on synthetic signals and 18 local light curves. These are engineering and small-sample scientific checks, not a claim that all astronomical cases or the unidentified original JDR paper have been validated.

## Results

- **27 automated tests pass**, including 90 DBSCAN comparisons with scikit-learn and actual independent R/Python integration checks for both standard and legacy conventions.
- Synthetic morphology clustering passes an ARI > 0.95 recovery gate under irregular sampling and added noise.
- The synthetic 0.67-day periodic signal is recovered within 1%; repeated fits are identical, and adding an absolute Julian-date epoch preserves the fitted shape within the test tolerance. This test initially exposed a double-period alias; BIC-based harmonic-order selection fixed it.
- Pair reversal, file permutation, self-distance, nonnegativity, float64 timestamp handling, serial/thread equivalence, quadrature/grid algebra, metadata reordering, invalid matrices, zero-distance medoid ties, cache mismatch rejection and conversion masks are tested.
- The synthetic notebook and the 18-star notebook execute from fresh kernels. All ten notebook sources parse. The remaining eight notebooks have not been executed end to end in this revision.
- The CLI and fresh-kernel notebook produce identical ordered IDs, labels and distance matrices for the same seed and inputs (checked separately).

The real sample contains **9 RRab, 8 RRc and 1 RRd**. It was drawn deterministically from the locally available labeled files across subtype groups, using seed 42. It is too small, especially for RRd, to establish generalization.

| Method | ARI, all stars | Coverage | Purity, assigned only | Clusters | Rejected |
|---|---:|---:|---:|---:|---:|
| kmedoids | 0.514 | 100.0% | 0.889 | 3 | 0 |
| dbscan | 0.752 | 83.3% | 1.000 | 2 | 3 |
| hdbscan | 0.697 | 100.0% | 0.889 | 2 | 0 |
| average_linkage | 0.812 | 100.0% | 0.944 | 3 | 0 |

DBSCAN's perfect conditional purity does **not** mean perfect classification: it rejects three stars, including the only RRd. Average linkage has the highest all-star ARI in this particular sample; that does not establish it as the best method. K=3, multi-start objective selection, the DBSCAN k-distance quantile and HDBSCAN settings were fixed policies rather than choices made to maximize catalog-label scores.

The grid checks use 1025 → 2049 → 4097 frequency points. Maximum absolute matrix discrepancy divided by the refined matrix maximum is **6.4006659e-06**, then **1.6002421e-06**. Every entry also passes `rtol=1e-3, atol=1e-8`, and all nearest neighbors remain unchanged at both refinements.

## Repair map

| Audit issue | Resolution |
|---|---|
| First-object integration band | One explicit JDRConfig for the dataset; no cadence-derived pair bounds. |
| Hidden interpolation / duplicate readers | One raw-data reader, re-exported by metrics; resampling only when explicitly requested. |
| Float32 parallel path | Float64 throughout; shared feature engine with bounded threads. |
| Incorrect tau / incompatible coefficient bases | Correct standard tau utility; standard distance uses common-basis least-squares amplitudes. Historical tau/projections are explicit legacy mode. |
| Suppressed convergence failures | No global warning filter; error/status gates in SciPy and R; no silent tolerance relaxation. |
| Wrong metadata and MNIST labels | Explicit object-ID joins; subset label indexing; no catalog-row shortcut. |
| Purity without rejection accounting | Full label vector retained; coverage, class retention, ARI/AMI and conditional purity reported. |
| Nondeterministic subsampling and extrema alignment | All observations used; weighted period search, alias candidates, BIC harmonic selection and smooth fitted phase alignment. |
| Stale matrices | Versioned .npz archives containing ID order, source hashes, full distance configuration and preprocessing provenance. |
| Invalid matrices and empty medoid clusters | Shared validation contract; medoids own their tie assignments; local random generator; callback symmetry checks. |
| Repeated process pools / pair recomputation | One feature transformation per curve; vectorized pairwise distances; no nested process pools. |
| Broken entry points / notebook state | Working-directory-independent CLI, main guards, shared notebook modules and corrected batch launcher. |
| Kepler/TESS conversion | Astropy time metadata, joint valid-row mask and positive-flux checks; offline conversion test. |
| Missing environment declaration | Python >=3.11, runtime/dev requirements, package metadata and captured validation environment. |

## Important method distinction

Version 2 standard mode is a **revised JDR representation**: normalized floating-mean least-squares amplitudes in a common basis, integrated with the repository's documented df scaling. Legacy mode preserves the old coefficient convention for comparison, not the defective old full pipeline. Therefore old matrices and eps thresholds cannot be reused. Exact correspondence to a particular published JDR definition remains unverified because the repository supplied no identifiable reference; that cannot be solved by guessing a formula.

The phase-shape fit uses uncertainty-weighted original observations, but downstream shape distances do not propagate full period/model covariance. A larger survey-separated evaluation and uncertainty/stability study remain research work, not prerequisites to the bounded small-set verification completed here.

## Reproduce

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python main.py --n-files 18 --seed 42
python scripts/validate_notebooks.py
```

Results are in `results/validation_v2/`: `report.json`, `distances.npz`, `distances.png`, `tests.xml`, `notebook_execution.json` and `requirements-lock.txt`. The actual fresh-kernel 18-star notebook also writes `results/notebook_experiment4/`.

The automatic approval review rejected a broad all-notebook kernel run. The narrower synthetic-plus-18-star notebook execution was subsequently approved and completed. Other notebook checks were limited to source inspection/syntax, rather than working around that restriction.
