"""Build an entirely self-contained analysis notebook from readable code sections."""

from pathlib import Path
import ast
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "scripts/notebook_components"


def definitions(path, names=None):
    source = path.read_text()
    tree = ast.parse(source)
    return "\n\n".join(
        (
            ast.get_source_segment(source, node)
            if not getattr(node, "decorator_list", None)
            else "\n".join(
                source.splitlines()[
                    min(d.lineno for d in node.decorator_list) - 1 : node.end_lineno
                ]
            )
        )
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (names is None or node.name in names)
    )


def main():
    cells = []

    def md(s):
        cells.append(nb.v4.new_markdown_cell(s))

    def code(s):
        cells.append(nb.v4.new_code_cell(s))

    md(
        r"""# JDR versus classical light-curve features: a self-contained 1,000-star experiment

**Question:** How does the paper-aligned JDR representation compare with classical descriptive/Fourier features when the data and clustering policies are held fixed?

This notebook contains **all analysis and plotting code**, rather than calling hidden project modules. It reads the original photometry, uses catalog periods, builds all representations and pairwise matrices, fits five clustering models, assesses stability, repeats the whole analysis and exports publication figures. It does not read earlier distance or clustering results.

Use **Restart Kernel → Run All**. In the portable bundle, keep `inputs/` beside this notebook; outputs are written to `outputs/`. Inside the original repository, the notebook finds `data/ogle_gd_1000/` and writes to `results/representation_comparison_1000/`. The exact dependency versions accompany the bundle. Python 3.12 and a few hundred MB of working memory are expected; computation is local. All code is visible and editable; changing any feature or protocol requires a new labeled experiment, not reuse of these conclusions.

The three representations are **JDR**, **Classical shape (13 features)** and **Classical full (16 features)**. The shape control removes period/amplitude as scalars; the full baseline restores them. This distinguishes a representation comparison from an information-content comparison. This is an exploratory in-sample experiment, not held-out validation or a significance claim."""
    )
    code("""from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
from dataclasses import dataclass
import os, sys, json, tempfile, time, platform
os.environ.setdefault('LOKY_MAX_CPU_COUNT','1')
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import quad
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from threadpoolctl import threadpool_limits
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

candidates=[Path.cwd(),*Path.cwd().parents]
portable=next((p for p in candidates if (p/'inputs/cohort.csv').exists()),None)
repository=next((p for p in candidates if (p/'data/ogle_gd_1000/cohort.csv').exists()),None)
if portable is not None:
    DATA=portable/'inputs';OUTPUT=portable/'outputs'
elif repository is not None:
    DATA=repository/'data/ogle_gd_1000';OUTPUT=repository/'results/representation_comparison_1000'
else:
    raise FileNotFoundError('Extract the portable bundle with inputs/ beside this notebook, or run in the supplied repository.')
print('Input directory:',DATA)
print('Output directory:',OUTPUT)""")
    md(
        r"""## 1. Frozen data, provenance and comparison protocol

The inputs are **1,000 original OGLE Galactic Disk I-band RR Lyrae curves**: 778 RRab, 214 RRc and 8 RRd. They are the nested sample used in the preceding JDR study. A seeded candidate permutation (20260912) selected the first 1,000 eligible curves; 11 of the first 1,011 candidates failed the minimum 30-observation threshold. This sample is not subtype balanced. IDs, labels and selection are frozen before this comparison.

Source: [OGLE OCVS](https://www.astrouw.edu.pl/ogle/ogle4/OCVS/gd/rrlyr/), [Soszyński et al. (2019)](https://arxiv.org/abs/2001.00025). Cite the source when publishing these data. The input bundle contains all selected original files, their manifest, the fixed-width source catalogs, README, acquisition record and exclusions. The full survey archive is not needed to reproduce this frozen selection. Its original URL/hash remains in acquisition metadata; regenerating selection from a later live archive may give different inputs.

The reader validates all selected-file SHA-256 values, source-catalog hashes, exact catalog IDs/subtypes and periods/epochs, positive errors, finite values and unique timestamps. Period bytes 37–46 and epoch bytes 60–69 are parsed from the source catalogs; every row is cross-checked. For RRd, use the first-overtone period/epoch, exactly as in the preceding experiment.

$$\phi_r=\operatorname{frac}[(t_r-T_0)/P].$$

No period search, clipping, interpolation, new observations or catalog-driven feature selection occurs. Errors are retained, but both JDR and the classical Fourier fit use **unweighted** original observations for this controlled comparison. This is a methodological choice, not a claim to optimal treatment of heteroscedastic noise.

The fixed five models are K-medoids, K-means, average linkage, DBSCAN and HDBSCAN. The first three use $K=3$, an explicit subtype-informed assumption. Twenty seeded starts are used for K-medoids/K-means. DBSCAN uses minimum samples 5 including self and the 70th percentile of each object's corresponding neighbor dissimilarity as radius. HDBSCAN uses minimum cluster size/minimum samples 5. We retain the preceding paper-aligned convention: **squared Euclidean dissimilarities for every matrix-based model**, so JDR receives $J$ and features receive $\|u_i-u_j\|^2$. K-means uses full vectors with its usual squared objective. A separate square-root sensitivity checks this convention without selecting settings."""
    )
    source = (COMP / "comparison_core.py").read_text()
    cut = source.index("def classical_features")
    code(source[:cut])
    code("""cohort,curves,catalog_hashes=read_inputs(DATA)
display(pd.Series(PROTOCOL,name='Frozen settings').to_frame())
display(pd.crosstab(cohort.region,cohort.catalog_type,margins=True))
display(cohort[['n_observations','baseline_days','catalog_period']].describe())
print('Unique pair distances per representation:',len(cohort)*(len(cohort)-1)//2)
print('Original observations:',int(cohort.n_observations.sum()))""")
    md(
        r"""## 2. Classical features: definitions and information controls

The baseline is an explicit, conventional descriptive/Fourier feature set, motivated by [FATS (Nun et al.)](https://arxiv.org/abs/1506.00010) and [Fourier light-curve analysis (Deb & Singh)](https://www.aanda.org/articles/aa/abs/2009/45/aa12851-09/aa12851-09.html). It is **not** claimed to reproduce either software package's complete feature collection or exact conventions.

Let $y=(m-\bar m)/s$ with population SD $s$. Quantiles $Q_p$ use NumPy's default linear interpolation. In phase order, fit the fixed four-harmonic model by unweighted least squares:

$$y(\phi)=c+\sum_{h=1}^{4}[a_h\cos(2\pi h\phi)+b_h\sin(2\pi h\phi)],\quad
A_h=\sqrt{a_h^2+b_h^2},\quad \psi_h=\operatorname{atan2}(-b_h,a_h).$$

Thus the cosine convention is $A_h\cos(2\pi h\phi+\psi_h)$. Relative phases $\psi_{h1}=\psi_h-h\psi_1$ are encoded with sine/cosine to avoid an artificial discontinuity at $\pm\pi$. No optimal harmonic order is selected using catalog scores.

| Index | Shape feature | Definition |
|---:|---|---|
| 1 | Skewness | $\operatorname{mean}(y^3)$ |
| 2 | Excess kurtosis | $\operatorname{mean}(y^4)-3$ |
| 3 | Bowley skewness | $(Q_{.75}+Q_{.25}-2Q_{.50})/(Q_{.75}-Q_{.25})$ |
| 4 | Tail ratio | $(Q_{.95}-Q_{.05})/(Q_{.75}-Q_{.25})$ |
| 5 | Circular phase eta | Mean squared successive differences of $y$, including last-to-first; cadence dependent |
| 6–8 | $R_{21},R_{31},R_{41}$ | $A_2/A_1,A_3/A_1,A_4/A_1$ |
| 9–10 | Relative phase 21 | $\cos\psi_{21},\sin\psi_{21}$ |
| 11–12 | Relative phase 31 | $\cos\psi_{31},\sin\psi_{31}$ |
| 13 | Residual fraction | Mean squared residual of the standardized four-harmonic fit |

**Classical full** adds $\log_{10}P$, $\log_{10}s$ and $\log_{10}(Q_{.95}-Q_{.05})$. Periods are in days and magnitude ranges in magnitudes. Apparent mean magnitude, coordinates, subtype, ID, observation count and measurement errors are not predictors. The full baseline has access to the same original data as JDR, but explicitly retains information that normalized phase-JDR discards. Shape-only is a closer information control, not a claim of identical invariances: sampling and epoch sensitivities differ.

Undefined ratios/phases, ill-conditioned fits and degenerate quantile features become missing values with an exported quality flag; stars are never silently removed. A feature missing for every star stops the analysis. Median imputation and median/IQR scaling are learned on the full cohort for the primary transductive experiment, then **refitted within each subsample** for stability. Zero IQR uses scale 1. There is no clipping or supervised weighting. Correlated features and the two-coordinate phase encoding imply an explicit Euclidean weighting choice, investigated descriptively in the figures."""
    )
    code(source[cut : source.index("def squared_matrix")])
    md(
        r"""## 3. JDR: paper formula and approved clarification

The supplied manuscript *Comparing irregular sampled time series using the Jensen-distance rate* is the reference for this implementation. As approved in the preceding study, Equation 6 is interpreted as a product of **complex spectral amplitudes**, not a product of the powers in Equation 1. A literal power product fails self-distance and can be negative.

For standardized original observations at irregular phase coordinates, define $\theta_i=\omega\tau_i$ with $\tau_i$ from the paper, and unit-norm projections

$$C_i=\frac{\sum_r y_{ir}\cos(\omega\phi_{ir}-\theta_i)}{\sqrt{\sum_r\cos^2(\omega\phi_{ir}-\theta_i)}},\quad
S_i=\frac{\sum_r y_{ir}\sin(\omega\phi_{ir}-\theta_i)}{\sqrt{\sum_r\sin^2(\omega\phi_{ir}-\theta_i)}}.$$

The common-basis complex amplitude is $A_i=(C_i+\mathrm{i}S_i)e^{\mathrm{i}\theta_i}/\sqrt2$, so the Equation 1 power is $|A_i|^2=(C_i^2+S_i^2)/2$. Its rotation removes arctangent-branch artifacts and aligns the sampling-dependent phase bases. The code uses robust `atan2`; the independent scalar audit uses the printed arctangent ratio.

$$J_{ij}=\frac{\beta}{2\pi}\left[\int_0^{2\pi}|A_i|^2d\omega+\int_0^{2\pi}|A_j|^2d\omega-2\int_0^{2\pi}\operatorname{Re}(A_i\overline A_j)d\omega\right],\quad\beta=\alpha(1-\alpha),\;\alpha=\tfrac12.$$

This is integration in **angular frequency**, not cyclic frequency. At zero frequency, use the right-hand limit for centered data; the sine projection tends to the projection on centered time. Positive trapezoidal weights give

$$J_{ij}\simeq\|z_i-z_j\|^2,\qquad z_i=[\sqrt{\beta q_k/(2\pi)}\operatorname{Re}A_i,\sqrt{\beta q_k/(2\pi)}\operatorname{Im}A_i]_k.$$

The embedding is calculated once per curve per grid. It is not standardized again across spectral coordinates, since that would change the JDR integral. The classical features require scaling because they have heterogeneous units; their scaling is part of the baseline definition.

$J$ is squared Euclidean; **$\sqrt J$**, not generally $J$, satisfies the triangle inequality. The printed band on phase spans at most one cycle per phase unit and may limit higher-harmonic discrimination. The classical four-harmonic fit intentionally offers a conventional alternative, not a matched spectral-band experiment. JDR retains sampling/count dependence. Weak stationarity is not established by phase folding; no such assumption is claimed to be verified. Multitaper and multiband extensions are not included."""
    )
    validate = definitions(ROOT / "src/metrics.py", {"validate_series"})
    paper = definitions(ROOT / "src/paper_jdr.py")
    code(validate + "\n\n" + paper)
    md(
        """## 4. Identical clustering and evaluation code

The following code is shared by all representations. K-medoids alternates assignment and within-cluster medoid updates and fails explicitly if it does not converge. Starting medoid index sets and random seeds are shared. Density-method radii are recalculated from each representation's own geometry using the same quantile policy, rather than transferring a numerical radius between incompatible units.

All-star ARI/AMI keep noise label −1. Assigned-only scores, coverage, cluster sizes and class retention remain visible. Silhouette is computed on the primary squared dissimilarities and should not be treated as a universal cross-representation ranking. Thirty shared 80% subsamples measure partition stability; they are not independent datasets, confidence intervals, or propagation of photometric uncertainty."""
    )
    code(source[source.index("def squared_matrix") : source.index("def run_pipeline")])
    md(
        """## 5. Validation before the full run

These executable checks compare Equation 1 with an independent classical Lomb–Scargle implementation, test scalar/vector spectral agreement and the zero-frequency limit, recover known Fourier amplitude ratios/relative phases from a synthetic signal, verify shape-feature invariance to magnitude offset/positive scaling, check imputation/scaling, and verify K-medoids finds the global best objective in a small enumerable example. Synthetic data are used only for tests, never added to the 1,000-star experiment."""
    )
    code((COMP / "comparison_checks.py").read_text())
    md(
        """## 6. End-to-end pipeline

Every run rereads and verifies the data; extracts features; computes JDR at 1,025, 2,049 and 4,097 angular frequencies; checks every pair and nearest-neighbor identity; recomputes a permuted embedding; performs three scalar-reference adaptive integral audits; fits all 15 representation/model combinations; refits every policy on 30 shared 800-star subsets; and evaluates the square-root convention sensitivity. All outputs retain the same ordered object IDs. No previous result is loaded as an input."""
    )
    code(source[source.index("def run_pipeline") :])
    code(
        """run_pipeline(DATA,OUTPUT)
summary=json.loads((OUTPUT/'summary.json').read_text())
display(pd.DataFrame(summary['grid_convergence']))
display(pd.read_csv(OUTPUT/'spectral_integral_audit.csv'))
quality=pd.read_csv(OUTPUT/'feature_quality.csv')
display(quality[['fourier_condition','undefined_features']].describe())
print('Objects with at least one undefined feature:',int((quality.undefined_features>0).sum()))
display(pd.Series(json.loads((OUTPUT/'timing.json').read_text()),name='Measured local runtime').to_frame())"""
    )
    md(
        """## 7. Read the paired results

Compare rows for the **same clustering model** across representations. Full-minus-JDR includes the benefit of explicit period/amplitude; shape-minus-JDR is the closer normalized-shape comparison. Neither difference establishes a universal winner or statistical significance. All configurations remain reported even if they perform poorly."""
    )
    code(
        """scores=pd.read_csv(OUTPUT/'model_comparison.csv')
display(scores[['representation','model','ari_all','ami_all','coverage','n_clusters','n_noise','purity_assigned','ari_assigned']].round(4))
paired=pd.read_csv(OUTPUT/'paired_ari_comparison.csv').set_index('model')
display(paired.round(4))
stability=pd.read_csv(OUTPUT/'subsampling_stability.csv')
display(stability.groupby(['representation','model']).ari_stability_all.quantile([.1,.5,.9]).unstack().round(3))
display(pd.read_csv(OUTPUT/'distance_convention_sensitivity.csv')[['representation','model','ari_all','coverage']].round(4))"""
    )
    md(
        """## 8. Repeat the complete analysis independently

A second full computation starts from the original files in a temporary directory. Numeric arrays and tables must match exactly; all scaling parameters, subset memberships and provenance must match. Wall times and compressed-file timestamps are excluded. This confirms repeatability in this recorded environment, not across arbitrary numerical-library versions or independent scientific data."""
    )
    code(
        """with tempfile.TemporaryDirectory(prefix='jdr-features-repeat-') as temporary:
    repeat=run_pipeline(DATA,Path(temporary))
    repeat_timing=json.loads((repeat/'timing.json').read_text())
    reproducibility=compare_runs(OUTPUT,repeat)
json_write(OUTPUT/'repeat_timing.json',repeat_timing)
display(pd.Series(reproducibility,name='Independent recomputation').to_frame())"""
    )
    md(
        """## 9. Publication figures and their source data

Four figure sets explain model agreement and stability, representation geometry, medoid-cluster composition and class retention, and the additional information/sampling/convention diagnostics. All tables and full arrays are saved. Figures are 183 mm wide, with embedded PDF fonts, editable SVG text and 600-dpi PNGs. Dense scatter layers are rasterized at 600 dpi inside vector files. The lower-resolution notebook previews are not submission files. Final journal dimensions and legibility still require review."""
    )
    code((COMP / "comparison_figures.py").read_text())
    code("""figure_dir=draw_figures(OUTPUT)
for stem,caption in json.loads((figure_dir/'captions.json').read_text()).items():
    display(Image(filename=str(figure_dir/f'{stem}_preview.png')))
    display(Markdown(caption))""")
    md(
        """## 10. Interpretation and reproducibility checklist

The comparison is descriptive and in-sample. Labels are reserved for evaluation, although selecting RR Lyrae subtypes and assuming K=3 are explicit prior knowledge. Catalog periods and epochs come from an existing curated catalog. Eight RRd objects cannot support generalization about rare-class recovery. Both representations can respond to observation cadence/noise; PC1 associations are descriptive and do not establish or rule out confounding. Fixed feature selection, robust scaling, correlated predictors, phase alignment and the narrow JDR band all affect the result. A future independent cohort and controlled cadence/noise simulations are needed for a scientific-superiority claim.

`classical_features.csv` stores unscaled features, `scalers.json` stores full-cohort transformations, and `subset_scalers.json` stores every subset's separately learned transformation. `cluster_labels.csv` retains every object and noise label. `comparison_arrays.npz` contains full embeddings, matrices, PCA coordinates and raw folded measurements indexed by numeric offsets; `allow_pickle=False` is sufficient. `subsample_membership.json` contains all shared subsets. The model, retention, composition, integral, sampling and convention-sensitivity CSVs are figure source data. Provenance records input/catalog hashes and numerical-library versions. The packaged notebook and code-cell hashes identify the actual implementation used.

To reproduce elsewhere, extract the bundle, create a Python 3.12 environment, install `environment-lock.txt`, then open this notebook and run every cell. Alternatively run the supplied `run_notebook.py` with that interpreter; it explicitly launches the same Python as its kernel. No repository `src/` files, earlier notebooks, network download or previous results are required. The source catalogs' credits remain applicable."""
    )
    code("""assert len(scores)==15 and len(stability)==3*5*30
assert (pd.read_csv(OUTPUT/'cluster_composition.csv').groupby(['representation','model'])['count'].sum()==1000).all()
assert json.loads((OUTPUT/'reproducibility.json').read_text())['arrays_identical']
json_write(OUTPUT/'notebook_checks.json',dict(self_tests_passed=SELF_TEST_RESULTS,all_primary_results=15,stability_rows=len(stability),all_cluster_totals=1000,completed=True))
display(pd.Series(json.loads((OUTPUT/'provenance.json').read_text())['versions'],name='Recorded environment').to_frame())
print('Completed self-contained comparison:',OUTPUT)""")
    notebook = nb.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nb.write(notebook, ROOT / "notebooks/jdr_vs_classical_features_1000.ipynb")


if __name__ == "__main__":
    main()
