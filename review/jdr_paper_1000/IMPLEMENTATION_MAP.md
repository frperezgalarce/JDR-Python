# Mapping the supplied manuscript to the 1,000-curve implementation

Source: `reference.pdf`, an unchanged copy of the user-supplied manuscript by Contreras-Reyes, Pérez-Galarce and Ojeda. The PDF hash is saved in each run's provenance. The user explicitly approved interpreting Equation 6 as a complex-amplitude product.

| Manuscript item | Implementation and qualification |
|---|---|
| Equation 1, page 3 | Unit-norm cosine/sine projections, with the correct frequency-dependent tau. Power equals half the sum of their squares. Independent Astropy classical-LSP and scalar-reference tests pass. This replaces the previous floating-intercept fitted-amplitude representation. |
| Tau below Equation 1 | Robust atan2 branch; complex amplitudes rotate back to a common phase coordinate. The scalar reference uses the printed arctan ratio and matches. |
| Equation 6, page 5 | Approved clarification: use complex amplitudes A with power equal to absolute-square A. Use the real part of A_i times conjugate A_j in the JDR integral. Multiplying the powers literally fails self-distance/nonnegativity; the numerical counterexample is retained. |
| Equations 5 and 7, page 5 | beta/(2*pi) times the three integrals over angular frequency 0 to 2*pi. beta=alpha*(1-alpha), alpha=0.5. The integration measure is d-omega, with the zero-frequency right limit evaluated explicitly. |
| Equation 7 numerical integration | Positive trapezoidal embedding at 1,025, 2,049 and 4,097 nodes. Every pair passes grid checks. Independent scalar adaptive quadrature audits three pairs without calling the vector implementation. |
| Claimed triangle inequality, page 5 | The clarified integral is squared Euclidean and need not satisfy the triangle inequality. Its square root does. A regression counterexample documents this rather than asserting the draft's claim. |
| Equations 10–12 and Algorithm 1, pages 6–7 | K-medoids receives J and alternates assignment and medoid updates. Twenty seeded starts are compared by sum of J to medoids. No catalog-score selection. |
| Multitaper Equations 2–4, 8–9 | Outside the present LSP-only experiment; not claimed as implemented or validated. |
| Multiband proposal | Outside this single I-band study. |

## Explicit application choices

- Use known OGLE periods and epochs; never search or refine them. RRd uses the catalog first-overtone mode; both catalog modes remain stored.
- Fold original observations into irregular phase, sort them, and center/standardize magnitudes. Do not replace observations with harmonic fits. Fits are for figure display only.
- Use the printed unweighted LSP. Measurement errors remain in raw artifacts and weight display fits, but are not silently inserted into Equation 1.
- Applying the printed angular band to phase means an upper frequency of one cycle per phase unit. This choice removes absolute period from the distance coordinate and may restrict higher-harmonic discrimination; it is not prescribed by the paper.
- Preserve the observation-count dependence of Equation 1. No undocumented division by sample count is added.
- Compare all matrix models on J, with K-means using the full embedding and its squared objective. The primary K-medoids objective therefore follows the manuscript. The previous 500-object matrix models used sqrt(J), so their scores are not directly comparable.
- Report catalog agreement as exploratory in-sample assessment, with noise retention and clustering stability. Eight RRd stars, uncertain epochs/periods, single-mode folding and unpropagated measurement uncertainty remain scientific limitations.

The theoretical weak-stationarity assumption is not established by centering or phase folding; it has not been tested for this cohort.
