from __future__ import annotations
from src.validation import validate_distance_matrix, positive_integer
import numpy as np
from typing import Callable, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os


def _compute_pair(args):
    i, j, file_i, file_j, distance_func, symmetric = args
    forward = float(distance_func(file_i, file_j))
    reverse = float(distance_func(file_j, file_i))
    if not np.isfinite([forward, reverse]).all() or min(forward, reverse) < 0:
        raise ValueError("Distance callback returned an invalid value.")
    if symmetric and not np.isclose(forward, reverse, rtol=1e-10, atol=1e-12):
        raise ValueError("Distance callback is asymmetric; refusing to mirror it.")
    return i, j, forward, reverse


class JDRKMedoids:
    """
    Iterative JDR-based k-medoids.

    This implementation follows the pseudocode:

    1. Initialize K medoids
    2. Assign each observation to its nearest medoid
    3. Update each medoid as the point minimizing total within-cluster dissimilarity
    4. Repeat until no medoid changes

    Parameters
    ----------
    n_clusters : int
        Number of clusters K.
    max_iter : int, default=100
        Maximum number of iterations.
    random_state : int or None, default=None
        Seed for reproducibility.
    init_medoids : list[int] or None, default=None
        Optional list of initial medoid indices.
    verbose : bool, default=False
        Whether to print progress.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        init_medoids: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init_medoids = list(init_medoids) if init_medoids is not None else None
        self.verbose = verbose

        # learned attributes
        self.medoids_ = None
        self.labels_ = None
        self.clusters_ = None
        self.distance_matrix_ = None
        self.n_iter_ = 0
        self.inertia_ = None

    def _validate_distance_matrix(self, D):
        return validate_distance_matrix(D)

    def _initialize_medoids(self, n: int) -> List[int]:
        if self.init_medoids is not None:
            if any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer))
                for i in self.init_medoids
            ):
                raise ValueError("Medoid indices must be integers.")
            if len(self.init_medoids) != self.n_clusters:
                raise ValueError("init_medoids must have length equal to n_clusters.")
            if len(set(self.init_medoids)) != self.n_clusters:
                raise ValueError("init_medoids must contain unique indices.")
            if min(self.init_medoids) < 0 or max(self.init_medoids) >= n:
                raise ValueError("init_medoids contains invalid indices.")
            return list(self.init_medoids)

        return self._rng.choice(n, self.n_clusters, replace=False).tolist()

    def _assign_clusters(self, D: np.ndarray, medoids: List[int]) -> np.ndarray:
        """
        Assign each point to the nearest medoid.
        """
        distances_to_medoids = D[:, medoids]  # shape (n, K)
        labels = np.argmin(distances_to_medoids, axis=1)
        labels[np.asarray(medoids)] = np.arange(len(medoids))
        return labels

    def _build_clusters(self, labels: np.ndarray) -> List[List[int]]:
        clusters = [np.where(labels == k)[0].tolist() for k in range(self.n_clusters)]
        return clusters

    def _update_medoids(
        self, D: np.ndarray, clusters: List[List[int]], medoids: List[int]
    ) -> List[int]:
        """
        For each cluster C_k, choose the j in C_k minimizing sum_i in C_k d_ij.
        """
        new_medoids = medoids.copy()

        for k, cluster in enumerate(clusters):
            if len(cluster) == 0:
                # Empty cluster: keep previous medoid
                if self.verbose:
                    print(
                        f"Cluster {k} is empty. Keeping previous medoid {medoids[k]}."
                    )
                continue

            cluster_idx = np.array(cluster)
            subD = D[np.ix_(cluster_idx, cluster_idx)]
            total_distances = np.sum(subD, axis=0)
            best_local_idx = np.argmin(total_distances)
            new_medoids[k] = cluster_idx[best_local_idx]

        return new_medoids

    def _compute_inertia(
        self, D: np.ndarray, labels: np.ndarray, medoids: List[int]
    ) -> float:
        """
        Sum of distances from each point to its assigned medoid.
        """
        return float(sum(D[i, medoids[labels[i]]] for i in range(D.shape[0])))

    def fit(self, D: Union[np.ndarray, List[List[float]]]):
        """
        Fit k-medoids using a precomputed dissimilarity matrix.

        Parameters
        ----------
        D : array-like of shape (n, n)
            Precomputed dissimilarity matrix.

        Returns
        -------
        self
        """
        positive_integer(self.n_clusters, "n_clusters")
        positive_integer(self.max_iter, "max_iter")
        self._rng = np.random.default_rng(self.random_state)
        D = self._validate_distance_matrix(D)
        n = D.shape[0]

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if self.n_clusters > n:
            raise ValueError(
                "n_clusters cannot be greater than number of observations."
            )

        medoids = self._initialize_medoids(n)

        if self.verbose:
            print(f"Initial medoids: {medoids}")

        changed = True
        it = 0

        while changed and it < self.max_iter:
            it += 1
            changed = False

            # Assignment step
            labels = self._assign_clusters(D, medoids)
            clusters = self._build_clusters(labels)

            # Update step
            new_medoids = self._update_medoids(D, clusters, medoids)

            if new_medoids != medoids:
                changed = True
                medoids = new_medoids

            if self.verbose:
                print(f"Iteration {it}: medoids = {medoids}")

        # final assignment with final medoids
        labels = self._assign_clusters(D, medoids)
        clusters = self._build_clusters(labels)

        self.distance_matrix_ = D
        self.medoids_ = medoids
        self.labels_ = labels
        self.clusters_ = clusters
        self.n_iter_ = it
        self.inertia_ = self._compute_inertia(D, labels, medoids)

        return self

    def fit_predict(self, D: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        self.fit(D)
        return self.labels_

    def predict(self, D_new_to_medoids: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new observations.

        Parameters
        ----------
        D_new_to_medoids : np.ndarray of shape (n_new, K)
            Distances from each new observation to the fitted medoids.

        Returns
        -------
        labels : np.ndarray
            Cluster index for each new observation.
        """
        if self.medoids_ is None:
            raise ValueError("Model has not been fitted yet.")

        D_new_to_medoids = np.asarray(D_new_to_medoids)
        if D_new_to_medoids.ndim != 2 or D_new_to_medoids.shape[1] != self.n_clusters:
            raise ValueError("D_new_to_medoids must have shape (n_new, n_clusters).")

        if not np.isfinite(D_new_to_medoids).all() or np.any(D_new_to_medoids < 0):
            raise ValueError("Prediction distances must be finite and nonnegative.")
        return np.argmin(D_new_to_medoids, axis=1)

    @staticmethod
    def build_distance_matrix_parallel(
        files, distance_func, symmetric=True, verbose=True, n_jobs=None
    ):
        """Shared-grid built-in JDR, or bounded batches for arbitrary callbacks.

        Both callback directions are checked. Threads avoid nested process pools
        and notebook pickling restrictions. At most 128 pair tasks are queued.
        """
        from src.jdr import jdr, jdr_parallel, build_distance_matrix as shared_matrix

        workers = min(4, os.cpu_count() or 1) if n_jobs is None else n_jobs
        positive_integer(workers, "n_jobs")
        if distance_func in (jdr, jdr_parallel):
            return shared_matrix(files, n_jobs=workers)
        files = list(files)
        if not files:
            raise ValueError("No files supplied.")
        D = np.zeros((len(files), len(files)))
        tasks = (
            (i, j, files[i], files[j], distance_func, symmetric)
            for i in range(len(files))
            for j in range(i + 1, len(files))
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while batch := list(islice(tasks, 128)):
                for i, j, forward, reverse in pool.map(_compute_pair, batch):
                    D[i, j], D[j, i] = forward, reverse
        if verbose:
            print(f"Computed matrix for {len(files)} objects")
        return D

    @staticmethod
    def build_distance_matrix(files, distance_func, symmetric=True, verbose=True):
        return JDRKMedoids.build_distance_matrix_parallel(
            files, distance_func, symmetric=symmetric, verbose=verbose, n_jobs=1
        )

    def get_medoid_files(self, files: Sequence[str]) -> List[str]:
        """
        Return the file paths corresponding to the learned medoids.
        """
        if self.medoids_ is None:
            raise ValueError("Model has not been fitted yet.")
        return [files[idx] for idx in self.medoids_]

    def get_cluster_files(self, files: Sequence[str]) -> List[List[str]]:
        """
        Return the file paths assigned to each cluster.
        """
        if self.clusters_ is None:
            raise ValueError("Model has not been fitted yet.")
        return [[files[idx] for idx in cluster] for cluster in self.clusters_]
