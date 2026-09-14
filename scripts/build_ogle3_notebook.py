from pathlib import Path
import ast, nbformat as nb

R = Path(__file__).resolve().parents[1]
C = R / "scripts/notebook_components"
old = nb.read(R / "notebooks/jdr_vs_classical_features_1000.ipynb", 4)


def defs(path, name):
    s = path.read_text()
    return "\n\n".join(
        ast.get_source_segment(s, n)
        for n in ast.parse(s).body
        if isinstance(n, ast.FunctionDef) and n.name in name
    )


reader = (
    """REGIONS = ['blg','lmc','smc']
TYPES = ['RRab','RRc','RRd','RRe']
"""
    + defs(R / "scripts/prepare_ogle3_inventory.py", ["catalog_records"])
    + """

def resource_check(n):
    # Conservative budget for simultaneous dense matrices, sorting, PCA and library copies.
    estimate = max(192 * 2**30, 64 * n * n + 8 * n * 8194 * 5)
    free_disk = shutil.disk_usage(OUTPUT.parent).free
    if RUN_MODE == 'full':
        if psutil.virtual_memory().available < estimate:
            raise MemoryError(f'Exact full run requires at least {estimate/2**30:.1f} GiB available RAM under this conservative policy. Use a larger machine; no approximations are substituted.')
        if free_disk < 200 * 2**30:
            raise OSError('Keep at least 200 GiB free for exact results and the independent repeat.')
    return {'n':n,'one_float64_matrix_GiB':8*n*n/2**30,'conservative_available_RAM_GiB':estimate/2**30,'free_disk_GiB':free_disk/2**30}

def read_inputs(data_dir):
    data_dir=Path(data_dir)
    table=pd.read_csv(data_dir/'cohort.csv')
    if len(table)!=43057 or not table.object_id.is_unique or not table.sha256.is_unique:
        raise ValueError('Frozen complete eligible cohort changed.')
    expected=catalog_records(data_dir).set_index('object_id')
    if len(expected)!=44217: raise ValueError('Incomplete source catalog inventory.')
    hashes={}
    for rec in json.loads((data_dir/'sources_complete.json').read_text())['files']:
        h=sha256()
        with (data_dir/rec['file']).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        if h.hexdigest()!=rec['sha256']: raise ValueError('Source hash changed: '+rec['file'])
        hashes[rec['file']]=h.hexdigest()
    for row in table.itertuples():
        ref=expected.loc[row.object_id]
        if ref.catalog_type!=row.catalog_type or not np.allclose([ref.catalog_period,ref.catalog_epoch],[row.catalog_period,row.catalog_epoch],rtol=0,atol=1e-10):
            raise ValueError('Catalog mismatch: '+row.object_id)
        if sha256((data_dir/row.file).read_bytes()).hexdigest()!=row.sha256:
            raise ValueError('Raw photometry changed: '+row.object_id)
    resource_check(len(table))
    if RUN_MODE=='pilot':
        selected=np.sort(np.random.default_rng(PROTOCOL['seed']).choice(len(table),512,replace=False))
        table=table.iloc[selected].reset_index(drop=True)
    elif RUN_MODE!='full': raise ValueError('RUN_MODE must be pilot or full.')
    PROTOCOL['n_objects']=len(table)
    PROTOCOL['run_mode']=RUN_MODE
    curves=[]
    for row in table.itertuples():
        raw=np.loadtxt(data_dir/row.file)
        if raw.shape!=(row.n_observations,3) or len(raw)<30 or not np.isfinite(raw).all(): raise ValueError('Invalid observations.')
        t,x,e=raw.T
        phase=((t-row.catalog_epoch)/row.catalog_period)%1
        if np.any(e<=0) or len(np.unique(t))!=len(t) or len(np.unique(phase))!=len(t) or np.std(x)<=0: raise ValueError('Invalid signal.')
        order=np.argsort(phase,kind='stable');curves.append((phase[order],x[order],e[order]))
    return table,curves,hashes
"""
)
s = (C / "comparison_core.py").read_text()
start = s.index("def read_inputs")
end = s.index("def classical_features")
s = s[:start] + reader + "\n\n" + s[end:]
s = s.replace("n_objects=1000", "n_objects=None").replace(
    "n_clusters=3", "n_clusters=4", 1
)
s = (
    s.replace("choice(len(D), 3,", 'choice(len(D), PROTOCOL["n_clusters"],')
    .replace("np.arange(3)", 'np.arange(PROTOCOL["n_clusters"])')
    .replace("for k in range(3):", 'for k in range(PROTOCOL["n_clusters"]):')
    .replace("n_clusters=3,", 'n_clusters=PROTOCOL["n_clusters"],')
)
s = (
    s.replace("permutation(1000)", "permutation(len(table))")
    .replace("np.triu_indices(1000, 1)", "np.triu_indices(len(table), 1)")
    .replace(
        "rng.choice(1000, size=800, replace=False)",
        'rng.choice(len(table), size=int(PROTOCOL["subset_fraction"]*len(table)), replace=False)',
    )
    .replace('["RRab", "RRc", "RRd"]', "TYPES")
    .replace(
        "Eight RRd stars are insufficient for rare-class generalization.",
        "Scope is recorded in protocol.run_mode; pilot results are not full-catalog evidence.",
    )
    .replace(' / "acquisition.json"', ' / "sources_complete.json"')
)
(C / "ogle3_core.py").write_text(s)
f = (
    (C / "comparison_figures.py")
    .read_text()
    .replace('"RRd": "#009E73"}', '"RRd": "#009E73", "RRe": "#CC79A7"}')
)
f = f.replace('["RRab", "RRc", "RRd"]', "TYPES").replace(
    '["o", "^", "D"]', '["o", "^", "D", "v"]'
)
f = (
    f.replace("left = np.zeros(3)", 'left = np.zeros(PROTOCOL["n_clusters"])')
    .replace(
        ".reindex(range(3), fill_value=0)",
        '.reindex(range(PROTOCOL["n_clusters"]), fill_value=0)',
    )
    .replace(
        "range(3),\n                    counts",
        'range(PROTOCOL["n_clusters"]),\n                    counts',
    )
    .replace(
        'ax.set_yticks(range(3), ["C0", "C1", "C2"])',
        'ax.set_yticks(range(PROTOCOL["n_clusters"]), [f"C{k}" for k in range(PROTOCOL["n_clusters"])])',
    )
    .replace("ax.set_xlim(0, 1050)", "ax.set_xlim(0, 1.08*len(cohort))")
    .replace(
        "ax.set_xticks([0, 500, 1000])",
        "ax.set_xticks([0, len(cohort)//2, len(cohort)])",
    )
    .replace(
        'ax.set_xticks(range(3), ["RRab\\n(n=778)", "RRc\\n(n=214)", "RRd\\n(n=8)"])',
        'ax.set_xticks(range(4), [f"{k}\\n(n={sum(cohort.catalog_type==k)})" for k in TYPES])',
    )
    .replace("for j in range(3):", "for j in range(4):")
)
f = f.replace(
    "    directory.mkdir",
    "    fig.suptitle(f\"OGLE-III {RUN_MODE.upper()} | N={PROTOCOL['n_objects']:,}\", fontsize=8)\n    directory.mkdir",
)
f = (
    f.replace("1,000 original light curves", "the selected original light curves")
    .replace("30 shared 800-star subsets", "30 shared 80%-size subsets")
    .replace("All 1,000 objects", "All selected objects")
    .replace(
        "; eight RRd stars cannot support rare-class inference",
        "; pilot results cannot establish full-catalog performance",
    )
    .replace("all three medoid clusters", "all four medoid clusters")
)
f = f.replace(
    '    json_write(folder / "captions.json", captions)',
    '    captions = {k: f"OGLE-III {RUN_MODE.upper()}, N={len(cohort):,}. " + v for k,v in captions.items()}\n    json_write(folder / "captions.json", captions)',
)
(C / "ogle3_figures.py").write_text(f)
checks = (
    (C / "comparison_checks.py")
    .read_text()
    .replace(
        "[[0], [0.1], [4], [4.1], [9], [9.1]]",
        "[[0], [0.1], [4], [4.1], [9], [9.1], [16], [16.1]]",
    )
    .replace("combinations(range(6), 3)", "combinations(range(8), 4)")
    .replace("np.repeat(range(3), 2)", "np.repeat(range(4), 2)")
)
(C / "ogle3_checks.py").write_text(checks)
cells = []
for i, c in enumerate(old.cells):
    if i in [5, 6, 7, 27]:
        continue
    c = nb.from_dict(dict(c))
    c.pop("id", None)
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None
    if i == 0:
        c.source = """# JDR versus classical features: complete OGLE-III RR Lyrae pipeline

This self-contained notebook prepares the **exact** five-model, three-representation comparison for every eligible OGLE-III RR Lyrae in the bulge, LMC and SMC. All analysis code is visible below. Original catalog periods are used, including first-overtone periods for RRd; historical RRe labels are retained.

**Execution status:** default `RUN_MODE="pilot"` computes a seeded 512-star validation, not the full catalog. Change it to `"full"` on a larger-memory machine and Restart Kernel → Run All. No approximate clustering or landmark representation is substituted. The full cohort contains **43,057 eligible stars from 44,217 catalog entries**. The remaining 1,160 entries have explicit reasons in `exclusions.csv`; no individual measurements are repaired or clipped.

All matrix methods retain the original squared-dissimilarity convention. K-medoids, K-means and average linkage use **K=4** because this catalog has four historical subtypes (the previous three-class experiment used K=3). This is a declared subtype-informed assumption, not unsupervised model selection. K-medoids remains the original exact-distance alternating algorithm with 20 starts; it is not a guarantee of a globally optimal partition.

The notebook performs two independent computations in the selected mode, 30 paired 80% subsamples, spectral-integral checks and four figure sets. Every figure identifies PILOT or FULL and its actual N. Pilot findings must not be reported as full-catalog scientific results."""
    if i == 1:
        c.source = c.source[: c.source.index("candidates=")] + """import psutil, shutil
RUN_MODE = "pilot"  # Set to "full" only on the larger-memory machine.
candidates=[Path.cwd(),*Path.cwd().parents]
portable=next((p for p in candidates if (p/'inputs/sources_complete.json').exists()),None)
repository=next((p for p in candidates if (p/'data/ogle3_rrlyrae_all/cohort.csv').exists()),None)
if portable is not None:
    DATA=portable/'inputs'; BASE_OUTPUT=portable/'outputs'
elif repository is not None:
    DATA=repository/'data/ogle3_rrlyrae_all'; BASE_OUTPUT=repository/'results/representation_comparison_ogle3_all'
else: raise FileNotFoundError('Place the supplied inputs folder beside this notebook.')
OUTPUT=BASE_OUTPUT/RUN_MODE
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
print('Execution mode:',RUN_MODE,' | Input:',DATA,' | Output:',OUTPUT)
"""
    if i == 2:
        c.source = """## 1. Complete inventory, data management and resources

The frozen official sources are [OGLE-III OIII-CVS](https://ftp.astrouw.edu.pl/ogle/ogle3/OIII-CVS/): `blg/rrlyr`, `lmc/rrlyr`, and `smc/rrlyr`. Catalog counts are BLG 16,836, LMC 24,906 and SMC 2,475. Cite Soszyński et al. (2009, 2010, 2011) and the source READMEs. Catalog photometry may include OGLE-II supplements; we retain the published files without imposing an undocumented date cut.

`source/` retains original catalogs, READMEs and archives. `sources_complete.json` records URLs and SHA-256 checksums. `inventory.csv` covers all catalog IDs, `cohort.csv` contains eligible objects, and `exclusions.csv` records failed requirements. Original I-band bytes are kept in `photometry/`. Eligibility requires at least 30 finite observations, positive errors, unique times and phases, nonconstant magnitudes, valid catalog period/epoch and no byte-identical duplicate curves. These quality exclusions define the population studied; they may bias coverage and are not missing-at-random assumptions.

Fixed-width parsing is region-specific and appears below. Every execution verifies all source hashes, all eligible photometry hashes and catalog periods/epochs/subtypes before selection. The pilot is one seeded uniform sample of 512 eligible objects; the full mode performs no downsampling. All outputs are separated by mode.

Phase is `((time - catalog_epoch) / catalog_period) % 1`. No period search, interpolation, clipping or error weighting occurs. RRd uses the catalog first-overtone mode. Measurements and errors remain available in the output arrays.

For 43,057 objects, a float64 square matrix occupies about **13.81 GiB**; three primary matrices alone need about 41.4 GiB. Sorting, convergence checks, PCA, subsets and library copies require substantially more. The conservative full-mode guard requires **192 GiB available RAM and 200 GiB free disk**; use a machine with at least 256 GiB installed RAM and check available memory. This is a planning margin, not a peak-memory guarantee. Computation runs wherever this notebook kernel runs; nothing automatically uploads data to a cloud service. Full execution time has not been measured and may be substantial, especially the exact subsampling and repeat.

Five models retain their previous settings: 20 starts for K-medoids/K-means; average linkage; DBSCAN with minimum samples 5 and the 70th percentile of the fifth distance including self; HDBSCAN minimum cluster size/minimum samples 5. The three fixed-K models use K=4. Primary matrix inputs are squared Euclidean distances (J for JDR), with a separate square-root sensitivity."""
    if i == 3:
        c.source = s[: s.index("def classical_features")]
    if i == 8:
        c.source += "\ndisplay(resource_check(len(pd.read_csv(DATA/'cohort.csv'))))\ndisplay(pd.read_csv(DATA/'exclusions.csv').exclusion_reason.value_counts().to_frame('objects'))"
    if i == 14:
        c.source = s[s.index("def squared_matrix") : s.index("def run_pipeline")]
    if i == 16:
        c.source = s[s.index("def run_pipeline") :]
    if i == 21:
        c.source = c.source.replace(
            "prefix='jdr-features-repeat-'",
            "prefix='jdr-features-repeat-', dir=OUTPUT.parent",
        )
    if i == 23:
        c.source = f
    if i == 26:
        c.source = c.source.replace("==1000", "==len(cohort)")
    if i in [13, 15, 18, 20, 22, 25]:
        c.source = (
            c.source.replace("800-star", "80%-size")
            .replace("1,000", "selected")
            .replace("three clusters", "four clusters")
            .replace("$K=3$", "$K=4$")
            .replace("eight RRd", "pilot RRd")
            .replace("8 RRd", "pilot RRd")
        )
    if i == 25:
        c.source = """## 10. Interpretation and completion status

ARI and AMI measure chance-adjusted agreement with the four historical catalog subtypes, not predictive accuracy or discovery validity. Noise label −1 is retained for all-object scores; coverage and assigned-only scores are reported separately. Catalog labels do not enter the representations, but K=4 is subtype-informed. Do not select models or features from these scores and then present the same scores as independent confirmation.

JDR retains the manuscript-aligned unit-normalized spectral projections, the approved complex-amplitude clarification and the 0–2π phase-frequency band. Raw irregular sampling and observation count can influence it. Classical full additionally retains period/amplitude; its four-harmonic fit has different bandwidth. Region differences, exclusions, cadence, historical RRe interpretation and RRd single-mode folding limit astrophysical conclusions.

A successful pilot validates code paths and reproducibility on 512 stars only. It does not establish peak memory, numerical convergence, runtime or scientific performance for the full cohort. Full-mode checks must pass on the larger machine before full-catalog conclusions are written. Two runs in one recorded environment test repeatability; numerical results across different library/BLAS/platform combinations may differ. Retain provenance, dependency versions, cohort hashes and all outputs."""
    cells.append(c)
    if i == 14:
        cells.append(
            nb.v4.new_markdown_cell(
                "## 5. Independent numerical self-checks\nSynthetic Fourier recovery, independent Astropy powers, integral identities, preprocessing isolation and an enumerated four-medoid optimum are checked before the experiment."
            )
        )
        cells.append(nb.v4.new_code_cell(checks))
notebook = nb.v4.new_notebook(cells=cells, metadata=old.metadata)
nb.write(notebook, R / "notebooks/jdr_vs_classical_features_ogle3_all.ipynb")
