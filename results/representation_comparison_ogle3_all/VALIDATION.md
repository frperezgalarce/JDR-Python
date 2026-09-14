# Validation scope

The delivered analysis is a **512-star pilot of a prepared full-catalog pipeline**. Full-catalog clustering has not been executed. Catalog inventory covers 44,217 IDs, with 43,057 eligible objects.

- Fresh local notebook execution: all 15 code cells passed, including eight independent numerical self-checks.
- Two independent end-to-end computations: numeric arrays, tables, scalers, memberships and provenance identical in the recorded environment.
- Pilot subtype counts: 380 RRab, 101 RRc, 18 RRd, 13 RRe.
- All three representations and all five clustering models completed, including 30 shared 80% subsamples and square-root convention sensitivity.
- JDR grid refinement 1,025 → 2,049 → 4,097 passed all-pair tolerances, with 100% nearest-neighbor agreement at both refinements. Relative maximum errors were approximately 7.25e-8 and 1.81e-8.
- Three representative pairs passed independent adaptive spectral-integral comparisons.
- Five dedicated OGLE-III tests passed: numerical identities, medoid agreement with the existing implementation, missing-feature handling, full-mode resource guard, and region-specific catalog parsing.
- Four PDF figure sets were rendered and visually inspected; plots and captions identify PILOT and N=512. Vector PDF/SVG and 600-dpi PNG exports are included.

The repository-wide check returned 49 passed and one existing failure: `tests/test_comparison_notebook.py::test_inline_self_checks` cannot find `run_self_tests` in the current older `jdr_vs_classical_features_1000.ipynb`. That notebook was left unchanged. The new notebook explicitly includes and executes its self-test cell.

These checks do not prove full-scale peak memory, full-cohort grid convergence, runtime, astrophysical validity or performance on an independent survey. They also do not establish cross-platform bitwise identity. Retain the recorded environment, and treat a failed full-mode numerical check as unresolved rather than bypassing it.

Portable validation: the final notebook was executed in `/tmp/jdr-ogle3-portable-validation` without repository imports, with all 15 code cells passing and both independent recomputations identical. Numeric arrays and CSV tables also matched the repository pilot; all four rendered PDFs were byte-identical to the visually reviewed versions.
