"""
diversity.py - Measures diversity between generated counterion configurations.

Uses Hungarian algorithm for optimal ion matching (ions are indistinguishable)
and PBC-aware distances for all RMSD calculations.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from pbc_utils import minimum_image_distance


class DiversityAnalyzer:
    """Analyse structural diversity among counterion configurations.

    Parameters
    ----------
    lattice : np.ndarray
        3x3 array whose rows are the lattice vectors (Angstroms).
    """

    def __init__(self, lattice: np.ndarray) -> None:
        self.lattice: np.ndarray = np.array(lattice, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core RMSD
    # ------------------------------------------------------------------

    def _matched_rmsd(
        self, config_a: np.ndarray, config_b: np.ndarray
    ) -> float:
        """Compute the optimal RMSD between two ion configurations.

        Because counterions are indistinguishable, the Hungarian algorithm
        is used to find the assignment that minimises the RMSD.

        Parameters
        ----------
        config_a, config_b : np.ndarray
            Each is an (N, 3) array of Cartesian ion positions.

        Returns
        -------
        float
            The RMSD (Angstroms) under the optimal assignment.
        """
        n = len(config_a)
        if n == 0:
            return 0.0
        if n != len(config_b):
            raise ValueError(
                f"Configs must have the same number of ions "
                f"({len(config_a)} vs {len(config_b)})"
            )

        # Build cost matrix of squared PBC distances
        inv_lattice = np.linalg.inv(self.lattice)
        diff = config_b[np.newaxis, :, :] - config_a[:, np.newaxis, :]
        diff_frac = np.einsum("ijk,lk->ijl", diff, inv_lattice)
        diff_frac -= np.round(diff_frac)
        diff_cart = np.einsum("ijk,kl->ijl", diff_frac, self.lattice)
        dist_mat = np.linalg.norm(diff_cart, axis=-1)
        cost = dist_mat ** 2

        row_ind, col_ind = linear_sum_assignment(cost)
        msd = cost[row_ind, col_ind].sum() / n
        return float(np.sqrt(msd))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_rmsd_matrix(self, configs: List[np.ndarray]) -> np.ndarray:
        """Compute the NxN pairwise RMSD matrix for a list of configurations.

        Parameters
        ----------
        configs : list of np.ndarray
            Each element is an (M, 3) array of ion positions.  All arrays
            must have the same first dimension M.

        Returns
        -------
        np.ndarray
            Symmetric (N, N) matrix of pairwise RMSD values (Angstroms).
        """
        n = len(configs)
        mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                r = self._matched_rmsd(configs[i], configs[j])
                mat[i, j] = r
                mat[j, i] = r
        return mat

    def compute_diversity_score(self, configs: List[np.ndarray]) -> float:
        """Return the mean pairwise RMSD across all configuration pairs.

        Parameters
        ----------
        configs : list of np.ndarray
            List of (M, 3) ion-position arrays.

        Returns
        -------
        float
            Mean pairwise RMSD (Angstroms).  Returns 0.0 for fewer than
            2 configurations.
        """
        n = len(configs)
        if n < 2:
            return 0.0
        mat = self.compute_rmsd_matrix(configs)
        # Upper triangle mean
        upper = mat[np.triu_indices(n, k=1)]
        return float(np.mean(upper))

    def select_diverse_subset(
        self,
        configs: List[np.ndarray],
        n: int,
        scores: Optional[List[float]] = None,
    ) -> List[int]:
        """Select *n* maximally diverse configurations via greedy farthest-point sampling.

        Parameters
        ----------
        configs : list of np.ndarray
            Full pool of (M, 3) ion-position arrays.
        n : int
            Number of configurations to select.
        scores : list of float or None
            Optional energy scores (lower = better).  Used to break ties
            when two candidates have equal maximum distance to the
            already-selected set.

        Returns
        -------
        list of int
            Indices (into *configs*) of the selected subset.
        """
        total = len(configs)
        if n >= total:
            return list(range(total))

        rmsd_mat = self.compute_rmsd_matrix(configs)

        # Start with the config that has the largest sum of distances to all
        # others (most "central outlier").
        selected: List[int] = [int(np.argmax(rmsd_mat.sum(axis=1)))]

        for _ in range(n - 1):
            # For each candidate, find its minimum distance to the selected set
            remaining = [i for i in range(total) if i not in selected]
            min_dists = np.array(
                [rmsd_mat[i, selected].min() for i in remaining],
                dtype=np.float64,
            )
            # Find the maximum of those minimum distances (farthest point)
            max_min = min_dists.max()
            # Candidates tied at the max
            tied = [
                remaining[k]
                for k in range(len(remaining))
                if np.isclose(min_dists[k], max_min, atol=1e-8)
            ]
            if scores is not None and len(tied) > 1:
                # Break tie by preferring lower energy
                best = min(tied, key=lambda idx: scores[idx])
            else:
                best = tied[0]
            selected.append(best)

        return selected

    def get_summary(self, configs: List[np.ndarray]) -> Dict[str, float]:
        """Return summary statistics of the RMSD distribution.

        Parameters
        ----------
        configs : list of np.ndarray
            List of (M, 3) ion-position arrays.

        Returns
        -------
        dict
            Keys: 'min_rmsd', 'max_rmsd', 'mean_rmsd', 'std_rmsd'.
        """
        n = len(configs)
        if n < 2:
            return {
                "min_rmsd": 0.0,
                "max_rmsd": 0.0,
                "mean_rmsd": 0.0,
                "std_rmsd": 0.0,
            }
        mat = self.compute_rmsd_matrix(configs)
        upper = mat[np.triu_indices(n, k=1)]
        return {
            "min_rmsd": float(np.min(upper)),
            "max_rmsd": float(np.max(upper)),
            "mean_rmsd": float(np.mean(upper)),
            "std_rmsd": float(np.std(upper)),
        }
