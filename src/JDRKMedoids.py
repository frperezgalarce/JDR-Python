import numpy as np
import random
from typing import Callable, List, Optional, Sequence, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _compute_pair(args):
    """
    Helper function for multiprocessing.
    Must be defined at top level to be picklable.
    """
    i, j, file_i, file_j, distance_func = args
    d = distance_func(file_i, file_j)
    return i, j, d

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
        verbose: bool = False
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

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

    def _validate_distance_matrix(self, D: np.ndarray):
        if not isinstance(D, np.ndarray):
            D = np.asarray(D)

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("Distance matrix D must be square.")

        if np.any(np.isnan(D)):
            raise ValueError("Distance matrix D contains NaN values.")

        return D

    def _initialize_medoids(self, n: int) -> List[int]:
        if self.init_medoids is not None:
            if len(self.init_medoids) != self.n_clusters:
                raise ValueError("init_medoids must have length equal to n_clusters.")
            if len(set(self.init_medoids)) != self.n_clusters:
                raise ValueError("init_medoids must contain unique indices.")
            if min(self.init_medoids) < 0 or max(self.init_medoids) >= n:
                raise ValueError("init_medoids contains invalid indices.")
            return list(self.init_medoids)

        return random.sample(range(n), self.n_clusters)

    def _assign_clusters(self, D: np.ndarray, medoids: List[int]) -> np.ndarray:
        """
        Assign each point to the nearest medoid.
        """
        distances_to_medoids = D[:, medoids]   # shape (n, K)
        labels = np.argmin(distances_to_medoids, axis=1)
        return labels

    def _build_clusters(self, labels: np.ndarray) -> List[List[int]]:
        clusters = [np.where(labels == k)[0].tolist() for k in range(self.n_clusters)]
        return clusters

    def _update_medoids(self, D: np.ndarray, clusters: List[List[int]], medoids: List[int]) -> List[int]:
        """
        For each cluster C_k, choose the j in C_k minimizing sum_i in C_k d_ij.
        """
        new_medoids = medoids.copy()

        for k, cluster in enumerate(clusters):
            if len(cluster) == 0:
                # Empty cluster: keep previous medoid
                if self.verbose:
                    print(f"Cluster {k} is empty. Keeping previous medoid {medoids[k]}.")
                continue

            cluster_idx = np.array(cluster)
            subD = D[np.ix_(cluster_idx, cluster_idx)]
            total_distances = np.sum(subD, axis=0)
            best_local_idx = np.argmin(total_distances)
            new_medoids[k] = cluster_idx[best_local_idx]

        return new_medoids

    def _compute_inertia(self, D: np.ndarray, labels: np.ndarray, medoids: List[int]) -> float:
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
        D = self._validate_distance_matrix(D)
        n = D.shape[0]

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if self.n_clusters > n:
            raise ValueError("n_clusters cannot be greater than number of observations.")

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
                print(changed, medoids)

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

        return np.argmin(D_new_to_medoids, axis=1)


    @staticmethod
    def build_distance_matrix_parallel(
        files: Sequence[str],
        distance_func: Callable[[str, str], float],
        symmetric: bool = True,
        verbose: bool = True,
        n_jobs: int | None = None
    ) -> np.ndarray:
        """
        Build a pairwise distance matrix from a list of files in parallel.

        Parameters
        ----------
        files : sequence of str
            Paths to light-curve files.
        distance_func : callable
            Function like jdr(file1, file2).
        symmetric : bool, default=True
            Whether distance is symmetric.
        verbose : bool, default=True
            Whether to print progress.
        n_jobs : int or None, default=None
            Number of parallel workers. If None, uses all available CPUs.

        Returns
        -------
        D : np.ndarray of shape (n, n)
            Pairwise distance matrix.
        """

        n = len(files)
        D = np.zeros((n, n), dtype=float)

        if n_jobs is None:
            n_jobs = os.cpu_count()

        tasks = []

        if symmetric:
            for i in range(n):
                for j in range(i + 1, n):
                    tasks.append((i, j, files[i], files[j], distance_func))
        else:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        tasks.append((i, j, files[i], files[j], distance_func))

        total = len(tasks)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_compute_pair, task) for task in tasks]

            for k, future in enumerate(as_completed(futures), start=1):
                i, j, d = future.result()

                D[i, j] = d

                if symmetric:
                    D[j, i] = d

                if verbose and (k % 100 == 0 or k == total):
                    print(f"Computed {k}/{total} distances")

        return D

    @staticmethod
    def build_distance_matrix(
        files: Sequence[str],
        distance_func: Callable[[str, str], float],
        symmetric: bool = True,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Build a pairwise distance matrix from a list of files and a distance function.

        Parameters
        ----------
        files : sequence of str
            Paths to light-curve files.
        distance_func : callable
            Function like jdr(file1, file2) or jdr_parallel(file1, file2).
        symmetric : bool, default=True
            Whether distance is symmetric.
        verbose : bool, default=False
            Whether to print progress.

        Returns
        -------
        D : np.ndarray of shape (n, n)
        """
        n = len(files)
        D = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                d = distance_func(files[i], files[j])
                D[i, j] = d
                if symmetric:
                    D[j, i] = d
                else:
                    D[j, i] = distance_func(files[j], files[i])

                if verbose and (j % 20 == 0 or j == n - 1):
                    print(f"Computed distances for pair ({i}, {j})")

        return D

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