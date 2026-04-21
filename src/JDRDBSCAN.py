import numpy as np
from typing import List, Optional, Sequence, Union


class JDRDBSCAN:
    """
    DBSCAN using a precomputed distance matrix.

    Parameters
    ----------
    eps : float
        Maximum neighborhood radius.
    min_samples : int, default=5
        Minimum number of points in the eps-neighborhood
        (including the point itself) for a point to be a core point.
    verbose : bool, default=False
        Whether to print progress.
    """

    def __init__(
        self,
        eps: float,
        min_samples: int = 5,
        verbose: bool = False
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.verbose = verbose

        # learned attributes
        self.labels_ = None          # cluster labels, -1 means noise
        self.clusters_ = None        # list of lists of observation indices
        self.core_samples_ = None    # indices of core points
        self.distance_matrix_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def _validate_distance_matrix(self, D: np.ndarray) -> np.ndarray:
        if not isinstance(D, np.ndarray):
            D = np.asarray(D)

        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("Distance matrix D must be square.")

        if np.any(np.isnan(D)):
            raise ValueError("Distance matrix D contains NaN values.")

        if np.any(D < 0):
            raise ValueError("Distance matrix D contains negative values.")

        return D

    def _region_query(self, D: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Return indices of all points within eps of point_idx,
        including point_idx itself.
        """
        return np.where(D[point_idx] <= self.eps)[0]

    def _expand_cluster(
        self,
        D: np.ndarray,
        labels: np.ndarray,
        visited: np.ndarray,
        point_idx: int,
        neighbors: np.ndarray,
        cluster_id: int,
        core_mask: np.ndarray
    ) -> None:
        """
        Expand a cluster starting from a core point.
        """
        labels[point_idx] = cluster_id
        i = 0

        neighbors = list(neighbors)

        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._region_query(D, neighbor_idx)

                if len(neighbor_neighbors) >= self.min_samples:
                    core_mask[neighbor_idx] = True

                    # Add new reachable points
                    for nn in neighbor_neighbors:
                        if nn not in neighbors:
                            neighbors.append(nn)

            # Assign to cluster if not yet assigned or previously marked as noise
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id

            i += 1

    def _build_clusters(self, labels: np.ndarray) -> List[List[int]]:
        cluster_ids = sorted([c for c in np.unique(labels) if c != -1])
        return [np.where(labels == c)[0].tolist() for c in cluster_ids]

    def fit(self, D: Union[np.ndarray, List[List[float]]]):
        """
        Fit DBSCAN using a precomputed distance matrix.

        Parameters
        ----------
        D : array-like of shape (n, n)
            Precomputed distance matrix.

        Returns
        -------
        self
        """
        D = self._validate_distance_matrix(D)
        n = D.shape[0]

        if self.eps <= 0:
            raise ValueError("eps must be positive.")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive.")

        visited = np.zeros(n, dtype=bool)
        labels = -1 * np.ones(n, dtype=int)   # -1 means noise/unassigned
        core_mask = np.zeros(n, dtype=bool)

        cluster_id = 0

        for point_idx in range(n):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._region_query(D, point_idx)

            if len(neighbors) < self.min_samples:
                labels[point_idx] = -1  # noise
            else:
                core_mask[point_idx] = True

                if self.verbose:
                    print(f"Expanding cluster {cluster_id} from point {point_idx}")

                self._expand_cluster(
                    D=D,
                    labels=labels,
                    visited=visited,
                    point_idx=point_idx,
                    neighbors=neighbors,
                    cluster_id=cluster_id,
                    core_mask=core_mask
                )
                cluster_id += 1

        self.distance_matrix_ = D
        self.labels_ = labels
        self.clusters_ = self._build_clusters(labels)
        self.core_samples_ = np.where(core_mask)[0].tolist()
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))

        return self

    def fit_predict(self, D: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        self.fit(D)
        return self.labels_

    def get_cluster_files(self, files: Sequence[str]) -> List[List[str]]:
        """
        Return the file paths assigned to each non-noise cluster.
        """
        if self.clusters_ is None:
            raise ValueError("Model has not been fitted yet.")
        return [[files[idx] for idx in cluster] for cluster in self.clusters_]

    def get_noise_files(self, files: Sequence[str]) -> List[str]:
        """
        Return the file paths labeled as noise (-1).
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        return [files[i] for i in range(len(files)) if self.labels_[i] == -1]