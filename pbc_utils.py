"""
pbc_utils.py - Periodic boundary condition utilities for CC-Plocation.

Provides coordinate transformations, minimum-image distance calculations,
pairwise PBC distance matrices, and a periodic-aware KDTree wrapper.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def cart_to_frac(cart_positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to fractional coordinates.

    Parameters
    ----------
    cart_positions : np.ndarray
        (N, 3) array of Cartesian positions in Angstroms.
    lattice : np.ndarray
        (3, 3) matrix whose *rows* are the lattice vectors.

    Returns
    -------
    np.ndarray
        (N, 3) array of fractional coordinates.
    """
    inv_lattice = np.linalg.inv(lattice)  # (3, 3)
    return cart_positions @ inv_lattice


def frac_to_cart(frac_positions: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates.

    Parameters
    ----------
    frac_positions : np.ndarray
        (N, 3) array of fractional coordinates.
    lattice : np.ndarray
        (3, 3) matrix whose *rows* are the lattice vectors.

    Returns
    -------
    np.ndarray
        (N, 3) array of Cartesian positions in Angstroms.
    """
    return frac_positions @ lattice


def wrap_to_cell(positions: np.ndarray, lattice: Optional[np.ndarray] = None) -> np.ndarray:
    """Wrap fractional coordinates into the unit cell [0, 1).

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) or (3,) array of fractional coordinates.
    lattice : np.ndarray, optional
        (3, 3) lattice matrix (unused; kept for API consistency).

    Returns
    -------
    np.ndarray
        Array of wrapped fractional coordinates in [0, 1).
    """
    return positions % 1.0


# ---------------------------------------------------------------------------
# Minimum-image convention distances
# ---------------------------------------------------------------------------

def minimum_image_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice: np.ndarray,
) -> float:
    """Compute the minimum-image distance between two Cartesian positions.

    Parameters
    ----------
    pos1 : np.ndarray
        (3,) Cartesian position of point 1.
    pos2 : np.ndarray
        (3,) Cartesian position of point 2.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).

    Returns
    -------
    float
        Minimum-image distance in Angstroms.
    """
    inv_lattice = np.linalg.inv(lattice)
    diff_cart = pos2 - pos1
    diff_frac = diff_cart @ inv_lattice
    diff_frac -= np.round(diff_frac)
    diff_cart_mic = diff_frac @ lattice
    return float(np.linalg.norm(diff_cart_mic))


def get_all_pbc_distances(
    positions: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Compute the pairwise minimum-image distance matrix.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) array of Cartesian positions.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).

    Returns
    -------
    np.ndarray
        (N, N) symmetric distance matrix.
    """
    n = len(positions)
    inv_lattice = np.linalg.inv(lattice)

    # Vectorised pairwise differences
    # diff[i, j] = positions[j] - positions[i]  shape (N, N, 3)
    diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    diff_frac = np.einsum("ijk,lk->ijl", diff, inv_lattice)
    diff_frac -= np.round(diff_frac)
    diff_cart = np.einsum("ijk,kl->ijl", diff_frac, lattice)
    dist = np.linalg.norm(diff_cart, axis=-1)
    return dist


# ---------------------------------------------------------------------------
# Cross-set PBC distance matrix
# ---------------------------------------------------------------------------

def cross_pbc_distance_matrix(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Compute PBC-aware distance matrix between two sets of positions.

    Parameters
    ----------
    positions_a : np.ndarray
        (M, 3) array of Cartesian positions.
    positions_b : np.ndarray
        (N, 3) array of Cartesian positions.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).

    Returns
    -------
    np.ndarray
        (M, N) distance matrix using minimum image convention.
    """
    inv_lattice = np.linalg.inv(lattice)
    # diff[i, j] = positions_b[j] - positions_a[i], shape (M, N, 3)
    diff = positions_b[np.newaxis, :, :] - positions_a[:, np.newaxis, :]
    diff_frac = np.einsum("ijk,lk->ijl", diff, inv_lattice)
    diff_frac -= np.round(diff_frac)
    diff_cart = np.einsum("ijk,kl->ijl", diff_frac, lattice)
    return np.linalg.norm(diff_cart, axis=-1)


# ---------------------------------------------------------------------------
# Periodic KDTree
# ---------------------------------------------------------------------------

class PeriodicKDTree:
    """KDTree that handles periodic boundary conditions via image replication.

    The original points plus their 26 nearest periodic images are inserted
    into a ``scipy.spatial.cKDTree``.  Query results are mapped back to
    original atom indices.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) Cartesian coordinates of the points.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).
    """

    def __init__(self, positions: np.ndarray, lattice: np.ndarray) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)
        self.lattice = np.asarray(lattice, dtype=np.float64)
        self.n_atoms = len(self.positions)

        # Build replicated point set: 27 images (original + 26 neighbours)
        shifts = np.array(
            [[i, j, k] for i in (-1, 0, 1)
                        for j in (-1, 0, 1)
                        for k in (-1, 0, 1)],
            dtype=np.float64,
        )  # (27, 3)

        # Cartesian shift vectors
        cart_shifts = shifts @ self.lattice  # (27, 3)

        # Replicate positions
        all_positions = (
            self.positions[np.newaxis, :, :]  # (1, N, 3)
            + cart_shifts[:, np.newaxis, :]   # (27, 1, 3)
        )  # (27, N, 3)

        self._all_positions = all_positions.reshape(-1, 3)  # (27*N, 3)
        self._index_map = np.tile(np.arange(self.n_atoms), 27)  # maps back
        self._tree = cKDTree(self._all_positions)

    def query_radius(
        self,
        point: np.ndarray,
        r: float,
    ) -> List[Tuple[int, float]]:
        """Find all original-atom indices within radius *r* of *point*.

        Parameters
        ----------
        point : np.ndarray
            (3,) Cartesian query point.
        r : float
            Search radius in Angstroms.

        Returns
        -------
        list of (int, float)
            Pairs of (original_atom_index, distance), sorted by distance.
            Duplicate original indices (from multiple images) are reduced to
            the closest distance only.
        """
        point = np.asarray(point, dtype=np.float64)
        raw_indices = self._tree.query_ball_point(point, r)
        if not raw_indices:
            return []

        raw_positions = self._all_positions[raw_indices]
        dists = np.linalg.norm(raw_positions - point, axis=1)
        orig_indices = self._index_map[raw_indices]

        # Keep closest distance per original index
        best: dict = {}
        for oidx, d in zip(orig_indices, dists):
            oidx = int(oidx)
            if oidx not in best or d < best[oidx]:
                best[oidx] = float(d)

        results = sorted(best.items(), key=lambda x: x[1])
        return results

    def query_nearest(
        self,
        point: np.ndarray,
        k: int = 1,
    ) -> List[Tuple[int, float]]:
        """Find the *k* nearest original atoms to *point*.

        Parameters
        ----------
        point : np.ndarray
            (3,) Cartesian query point.
        k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list of (int, float)
            Pairs of (original_atom_index, distance), sorted by distance.
        """
        point = np.asarray(point, dtype=np.float64)
        # Query more than k to account for duplicate images
        query_k = min(k * 27, len(self._all_positions))
        dists, indices = self._tree.query(point, k=query_k)

        if query_k == 1:
            dists = [dists]
            indices = [indices]

        best: dict = {}
        for d, idx in zip(dists, indices):
            oidx = int(self._index_map[idx])
            if oidx not in best or d < best[oidx]:
                best[oidx] = float(d)

        results = sorted(best.items(), key=lambda x: x[1])
        return results[:k]


# ---------------------------------------------------------------------------
# Minimum-image displacement vector
# ---------------------------------------------------------------------------

def min_image_displacement(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Compute the minimum-image displacement vector from pos1 to pos2.

    Parameters
    ----------
    pos1 : np.ndarray
        (3,) Cartesian position of point 1.
    pos2 : np.ndarray
        (3,) Cartesian position of point 2.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).

    Returns
    -------
    np.ndarray
        (3,) Cartesian displacement vector (minimum image).
    """
    inv_lattice = np.linalg.inv(lattice)
    diff_cart = pos2 - pos1
    diff_frac = diff_cart @ inv_lattice
    diff_frac -= np.round(diff_frac)
    return diff_frac @ lattice


# ---------------------------------------------------------------------------
# Aliases (used by other modules)
# ---------------------------------------------------------------------------
fractional_to_cartesian = frac_to_cart
cartesian_to_fractional = cart_to_frac
wrap_fractional = wrap_to_cell
min_image_distance = minimum_image_distance
