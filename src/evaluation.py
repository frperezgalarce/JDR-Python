"""ID-aligned evaluation; rejected stars remain in the accounting."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def load_metadata(paths):
    frames = [
        pd.read_csv(
            p,
            sep=r"\s+",
            header=None,
            usecols=range(6),
            names=["id", "field", "star_id", "type", "ra", "dec"],
        )
        for p in paths
    ]
    df = pd.concat(frames, ignore_index=True)
    if df.id.duplicated().any():
        raise ValueError("Duplicate metadata IDs; provide each catalog only once.")
    if df[["id", "type"]].isna().any().any():
        raise ValueError("Missing metadata IDs or types.")
    return df


def aligned_labels(files, df):
    ids = [Path(f).stem for f in files]
    if len(set(ids)) != len(ids) or df.id.duplicated().any():
        raise ValueError("Object IDs must be unique.")
    indexed = df.set_index("id")
    missing = set(ids) - set(indexed.index)
    if missing:
        raise ValueError(f"Missing metadata for {sorted(missing)}")
    labels = indexed.loc[ids, "type"]
    if labels.isna().any():
        raise ValueError("Missing class labels.")
    return labels.to_numpy()


def purity_score(y_true, y_pred, ignore_noise=True):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("Labels must be aligned 1D arrays.")
    if ignore_noise:
        mask = y_pred != -1
        y_true, y_pred = y_true[mask], y_pred[mask]
    if not len(y_true):
        return None
    return float(
        sum(
            np.unique(y_true[y_pred == k], return_counts=True)[1].max()
            for k in np.unique(y_pred)
        )
        / len(y_true)
    )


def evaluate_clustering(y_true, labels):
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    if y_true.ndim != 1 or y_true.shape != labels.shape or not len(labels):
        raise ValueError("Nonempty, aligned labels required.")
    assigned = labels != -1
    groups, counts = np.unique(labels[assigned], return_counts=True)
    return {
        "coverage": float(assigned.mean()),
        "n_noise": int((~assigned).sum()),
        "n_clusters": len(groups),
        "cluster_sizes": dict(zip(map(str, groups), map(int, counts))),
        "purity_assigned": purity_score(y_true, labels),
        "ari_all": float(adjusted_rand_score(y_true, labels)),
        "ami_all": float(adjusted_mutual_info_score(y_true, labels)),
        "ari_assigned": (
            float(adjusted_rand_score(y_true[assigned], labels[assigned]))
            if assigned.sum() > 1
            else None
        ),
        "class_retention": {
            str(k): float(assigned[y_true == k].mean()) for k in np.unique(y_true)
        },
    }
