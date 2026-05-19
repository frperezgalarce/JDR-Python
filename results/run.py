import sys
from pathlib import Path
import os
import time
import glob
import itertools
import shutil
import logging

import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np

from astropy.timeseries import LombScargle
from collections import Counter

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "jdr_dbscan.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path.cwd().parents[0]  # adjust if needed
sys.path.append(str(PROJECT_ROOT))

from src.jdr import jdr, jdr_parallel
from src.JDRKMedoids import JDRKMedoids
from src.JDRDBSCAN import JDRDBSCAN

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

data_dir = "data"

n_files = 1000
min_period = 0.01
max_period = 100.0

files = sorted(glob.glob(os.path.join(data_dir, "*RRLYR*.dat")))[:n_files]

logger.info(f"Found {len(files)} files")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def read_light_curve(filepath, n_sample=300):
    """
    Read a light curve with columns:
    time, magnitude, and optionally error.
    """
    lc = pd.read_csv(filepath, sep=r"\s+", header=None, comment="#")

    if lc.shape[0] > n_sample:
        lc = lc.sample(n_sample)

    t = lc.iloc[:, 0].to_numpy(dtype=float)
    mag = lc.iloc[:, 1].to_numpy(dtype=float)

    err = (
        lc.iloc[:, 2].to_numpy(dtype=float)
        if lc.shape[1] > 2
        else None
    )

    return t, mag, err


def estimate_period(
    dat_file,
    min_period=min_period,
    max_period=max_period,
    n_freq=20000,
):
    """
    Estimate the dominant period using Lomb-Scargle.
    """
    t, mag, _ = read_light_curve(dat_file)

    mask = np.isfinite(t) & np.isfinite(mag)

    t = t[mask]
    mag = mag[mask]

    if len(t) < 5:
        raise ValueError(
            f"Not enough finite observations to estimate period: {dat_file}"
        )

    freq = np.linspace(1 / max_period, 1 / min_period, n_freq)

    power = LombScargle(t, mag).power(freq)

    best_freq = freq[np.argmax(power)]

    return 1 / best_freq


def fold_time(t, period, t0=None):
    """
    Convert observation times to phase in [0, 1).
    """
    if t0 is None:
        t0 = np.nanmin(t)

    return ((t - t0) / period) % 1.0


def write_folded_light_curve(
    input_file,
    output_dir,
    period=None,
    min_period=min_period,
    max_period=max_period,
):
    """
    Fold a light curve and write a JDR-compatible phase-space .dat file.
    """

    t, mag, err = read_light_curve(input_file)

    if period is None:
        period = estimate_period(
            input_file,
            min_period=min_period,
            max_period=max_period,
        )

    phase = fold_time(t, period)

    mask = np.isfinite(phase) & np.isfinite(mag)

    if err is not None:
        mask &= np.isfinite(err)

    phase = phase[mask]
    mag = mag[mask]

    err = err[mask] if err is not None else None

    order = np.argsort(phase)

    phase = phase[order]
    mag = mag[order]

    if err is not None:
        err = err[order]
        folded = np.column_stack([phase, mag, err])
    else:
        folded = np.column_stack([phase, mag])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / Path(input_file).name

    np.savetxt(output_file, folded, fmt="%.10f")

    logger.info(
        f"Folded light curve saved: {output_file} | period={period:.6f}"
    )

    return str(output_file), period


def build_folded_dataset(
    files,
    folded_dir="data/folded_rrlyr",
    min_period=min_period,
    max_period=max_period,
    overwrite=True,
):
    """
    Create folded files and return their paths plus the estimated periods.
    """

    folded_dir = Path(folded_dir)

    if overwrite and folded_dir.exists():
        shutil.rmtree(folded_dir)

    folded_dir.mkdir(parents=True, exist_ok=True)

    folded_files = []
    periods = {}

    for f in files:

        logger.info(f"Processing file: {f}")

        folded_f, period = write_folded_light_curve(
            f,
            output_dir=folded_dir,
            period=None,
            min_period=min_period,
            max_period=max_period,
        )

        folded_files.append(folded_f)

        periods[Path(f).stem] = period

    return folded_files, periods


# -----------------------------------------------------------------------------
# Build folded dataset
# -----------------------------------------------------------------------------

folded_files, periods = build_folded_dataset(
    files,
    folded_dir="data/folded_rrlyr",
    min_period=min_period,
    max_period=max_period,
    overwrite=True,
)

logger.info(f"Created {len(folded_files)} folded files")

logger.info(
    f"First folded file: "
    f"{folded_files[0] if folded_files else None}"
)

# -----------------------------------------------------------------------------
# Distance matrix
# -----------------------------------------------------------------------------

logger.info("Building JDR distance matrix...")

D = JDRKMedoids.build_distance_matrix(
    files=folded_files,
    distance_func=jdr_parallel,
    verbose=True,
)

logger.info("Distance matrix computed successfully")

# -----------------------------------------------------------------------------
# Save distance matrix
# -----------------------------------------------------------------------------

np.save("D_folded.npy", D)

logger.info("Saved distance matrix to D_folded.npy")

# -----------------------------------------------------------------------------
# Load distance matrix
# -----------------------------------------------------------------------------

D = np.load("D_folded.npy")

logger.info("Loaded distance matrix from D_folded.npy")

# -----------------------------------------------------------------------------
# DBSCAN parameter sweep
# -----------------------------------------------------------------------------

eps_values = np.linspace(0.01, 1, 10)

min_samples_values = [2, 3, 4]

logger.info("Starting DBSCAN parameter sweep")

for eps in eps_values:

    for min_samples in min_samples_values:

        model = JDRDBSCAN(
            eps=eps,
            min_samples=min_samples,
            verbose=False,
        )

        model.fit(D)

        logger.info(
            f"eps={eps:.4f}, "
            f"min_samples={min_samples}, "
            f"clusters={model.n_clusters_}, "
            f"noise={model.n_noise_}"
        )

# -----------------------------------------------------------------------------
# Final clustering
# -----------------------------------------------------------------------------

logger.info("Running final DBSCAN clustering")

model = JDRDBSCAN(
    eps=1.2,
    min_samples=3,
    verbose=False,
)

labels = model.fit_predict(D)

logger.info(f"Labels: {labels}")

logger.info(f"Number of clusters: {model.n_clusters_}")

logger.info(f"Number of noise points: {model.n_noise_}")

logger.info(f"Clusters: {model.clusters_}")

# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------

cols = ["id", "field", "star_id", "type", "ra", "dec"]

df = pd.read_csv(
    os.path.join(
        "notebooks/ident.dat"
    ),
    sep=r"\s+",
    header=None,
    usecols=[0, 1, 2, 3, 4, 5],
    names=cols,
    engine="python",
)

logger.info("Metadata file loaded successfully")

# -----------------------------------------------------------------------------
# Analyze clusters
# -----------------------------------------------------------------------------

cluster_files = [
    [files[idx] for idx in cluster]
    for cluster in model.clusters_
]

objects_by_class = {}

logger.info("Files in each cluster:")

for k, cluster in enumerate(cluster_files):

    logger.info(f"Cluster {k}:")

    objects = []

    for f in cluster:

        logger.info(f"  {f}")

        objects.append(
            os.path.splitext(os.path.basename(f))[0]
        )

    objects_by_class[str(k)] = objects

# -----------------------------------------------------------------------------
# Print object types by cluster
# -----------------------------------------------------------------------------

for cluster in objects_by_class.keys():

    logger.info(f"Cluster: {cluster}")

    for light_curve in objects_by_class[cluster]:

        lc_type = df[df.id == light_curve]["type"].values[0]

        logger.info(f"  {light_curve} -> {lc_type}")

logger.info("Execution completed successfully")


plt.hist(D.reshape(-1), bins=20)
plt.savefig('results/distance_histogram.png')

labels = []
for light_curve in files: 
    labels.append(df[df.id==os.path.splitext(os.path.basename(light_curve))[0]]['type'].values[0])

plt.figure(figsize=(8, 6))
im = plt.imshow(D, aspect="auto")
plt.colorbar(im, label="Distance")
plt.title("Distance Matrix Heatmap")
plt.xlabel("Observation index")
plt.ylabel("Observation index")

plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90)
plt.yticks(ticks=range(len(labels)), labels=labels)

plt.tight_layout()
plt.savefig('results/heatmap.png')

def plot_original_and_folded(filepath, period=None, star_name=None):
    # Plot original observations and the folded phase-space representation.
    t, mag, err = read_light_curve(filepath)
    if period is None:
        period = periods.get(Path(filepath).stem, None)
    if period is None:
        period = estimate_period(filepath)

    phase = fold_time(t, period)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if err is not None:
        axes[0].errorbar(t, mag, yerr=err, fmt=".", alpha=0.7)
    else:
        axes[0].plot(t, mag, ".", alpha=0.7)

    axes[0].set_title(f"Original: {star_name or Path(filepath).stem}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Magnitude")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3)

    if err is not None:
        axes[1].errorbar(phase, mag, yerr=err, fmt=".", alpha=0.7)
        axes[1].errorbar(phase + 1, mag, yerr=err, fmt=".", alpha=0.7)
    else:
        axes[1].plot(phase, mag, ".", alpha=0.7)
        axes[1].plot(phase + 1, mag, ".", alpha=0.7)

    axes[1].set_title(f"Folded: {star_name or Path(filepath).stem}\nP = {period:.6f}")
    axes[1].set_xlabel("Phase")
    axes[1].set_ylabel("Magnitude")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/"+filepath+".png")

for cluster in objects_by_class.keys():
    print(f"\nCluster {cluster}")
    for star in objects_by_class[cluster]:
        filepath = f"../data/{star}.dat"
        period = periods.get(star, None)
        plot_original_and_folded(filepath, period=period, star_name=star)

def purity_score(y_true, y_pred, ignore_noise=True):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ignore_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0.0

    total_correct = 0

    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        true_labels_cluster = y_true[mask]

        if len(true_labels_cluster) > 0:
            total_correct += Counter(true_labels_cluster).most_common(1)[0][1]

    return total_correct / len(y_true)


def get_true_labels_from_clusters(clusters, df, id_col="id", label_col="type"):
    y_true = []
    y_pred = []
    object_ids = []

    df_ids = set(df[id_col].values)

    for cluster_id, cluster_objects in enumerate(clusters):
        for light_curve in cluster_objects:

            # Case 1: cluster stores object IDs
            if light_curve in df_ids:
                label = df.loc[df[id_col] == light_curve, label_col].values[0]
                object_id = light_curve

            # Case 2: cluster stores row indices
            elif isinstance(light_curve, (int, np.integer)) and light_curve < len(df):
                label = df.iloc[light_curve][label_col]
                object_id = df.iloc[light_curve][id_col]

            else:
                continue

            y_true.append(label)
            y_pred.append(cluster_id)
            object_ids.append(object_id)

    return np.array(y_true), np.array(y_pred), np.array(object_ids)

def search_dbscan_by_purity(
    D,
    df,
    eps_values=None,
    min_samples_values=None,
    id_col="id",
    label_col="type",
    ignore_noise=True,
    verbose=True
):


    results = []

    best_model = None
    best_result = None
    best_purity = -np.inf

    for eps in eps_values:
        for min_samples in min_samples_values:

            model = JDRDBSCAN(
                eps=eps,
                min_samples=min_samples,
                verbose=False
            )

            model.fit(D)

            y_true, y_pred, object_ids = get_true_labels_from_clusters(
                clusters=model.clusters_,
                df=df,
                id_col=id_col,
                label_col=label_col
            )

            purity = purity_score(
                y_true=y_true,
                y_pred=y_pred,
                ignore_noise=ignore_noise
            )

            result = {
                "eps": eps,
                "min_samples": min_samples,
                "purity": purity,
                "n_clusters": model.n_clusters_,
                "n_noise": model.n_noise_
            }

            results.append(result)

            if verbose:
                print(
                    f"eps={eps:.4f}, min_samples={min_samples}, "
                    f"purity={purity:.4f}, "
                    f"clusters={model.n_clusters_}, noise={model.n_noise_}"
                )

            if purity > best_purity:
                best_purity = purity
                best_model = model
                best_result = result

    results_df = pd.DataFrame(results).sort_values(
        by=["purity", "n_clusters"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return best_model, best_result, results_df


eps_values = np.linspace(0.2, 100, 1000)
min_samples_values = [2, 3, 4, 5, 6]

best_model, best_result, results_df = search_dbscan_by_purity(
    D=D,
    df=df,
    eps_values=eps_values,
    min_samples_values=min_samples_values,
    id_col="id",
    label_col="type",
    ignore_noise=True
)

results_df.to_csv('results/table.csv')

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    results_df["n_noise"],
    results_df["purity"],
    c=results_df["min_samples"],
    s=80,
    alpha=0.8
)

plt.colorbar(scatter, label="min_samples")

plt.xlabel("Number of noisy objects")
plt.ylabel("Purity")
plt.title("Purity vs Number of Noisy Objects")

plt.grid(alpha=0.3)
plt.savefig('results/purity_number_noise.png')

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    results_df["n_noise"],
    results_df["purity"],
    c=results_df["min_samples"],
    s=30 + 30 * results_df["n_clusters"],
    alpha=0.7
)

plt.colorbar(scatter, label="min_samples")

plt.xlabel("Number of noisy objects")
plt.ylabel("Purity")
plt.title("Purity vs Number of Noisy Objects")

plt.grid(alpha=0.3)
plt.savefig('results/purity_noise_minsamples.png')


