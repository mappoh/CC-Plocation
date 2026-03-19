"""
Framework Analyzer for CC-Plocation.

Analyzes a POM framework structure to build exclusion zones that prevent
counterion placement inside the framework or too close to transition-metal sites.
Provides ExclusionGrid (3D boolean voxel grid) and FrameworkInfo (structural
metadata) classes.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from defaults import VDW_RADII, IONIC_RADII, TM_ELEMENTS as DEFAULT_TM_ELEMENTS
from pbc_utils import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    min_image_distance,
    min_image_displacement,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_GRID_POINTS = 500          # per axis – safety cap
_COARSEN_THRESHOLD = 200_000_000  # total voxel count that triggers auto-coarsen


# ===================================================================
# ExclusionGrid
# ===================================================================

class ExclusionGrid:
    """3D boolean grid over the unit cell marking forbidden voxels for
    counterion placement.

    A voxel is *forbidden* when a counterion centred there would overlap
    with any framework atom (van-der-Waals contact) or intrude into the
    buffer zone around a transition-metal site.

    Parameters
    ----------
    structure : dict
        Parsed POSCAR dict from ``structure_parser.parse_poscar()``.
        Expected keys: ``'lattice'``, ``'positions'``, ``'atom_labels'``,
        ``'species'``, ``'counts'``.
    counterion_element : str
        Element symbol for the counterion (e.g. ``'K'``).
    tm_elements : list[str] or None
        Transition-metal element symbols to buffer.  Falls back to
        ``defaults.DEFAULT_TM_ELEMENTS`` when *None*.
    tm_buffer : float
        Extra exclusion radius (Angstroms) around each TM site beyond the
        normal vdW overlap zone.
    grid_resolution : float
        Target voxel edge length in Angstroms.
    surface_aware : bool
        When *True*, under-coordinated (surface-exposed) TM sites receive
        reduced or no extra buffer.
    """

    def __init__(
        self,
        structure: dict,
        counterion_element: str,
        tm_elements: Optional[List[str]] = None,
        tm_buffer: float = 3.5,
        grid_resolution: float = 0.3,
        surface_aware: bool = True,
        max_framework_distance: Optional[float] = None,
    ) -> None:
        self.structure = structure
        self.counterion_element = counterion_element
        self.tm_elements: List[str] = (
            list(tm_elements) if tm_elements is not None else list(DEFAULT_TM_ELEMENTS)
        )
        self.tm_buffer = tm_buffer
        self.grid_resolution = grid_resolution
        self.surface_aware = surface_aware
        if max_framework_distance is None:
            raise ValueError(
                "max_framework_distance must be provided."
            )
        self.max_framework_distance = float(max_framework_distance)

        # Unpack structure ------------------------------------------------
        self.lattice: NDArray[np.float64] = np.array(
            structure["lattice"], dtype=np.float64
        )  # (3, 3) row-vectors
        self.positions: NDArray[np.float64] = np.array(
            structure["positions"], dtype=np.float64
        )  # (N, 3) Cartesian
        self.atom_labels: List[str] = list(structure["atom_labels"])

        # Counterion vdW radius -------------------------------------------
        self._r_counter: float = IONIC_RADII.get(
            counterion_element, VDW_RADII.get(counterion_element, 2.0)
        )

        # Grid state (populated by build()) --------------------------------
        self._grid: Optional[NDArray[np.bool_]] = None
        self._grid_shape: Optional[Tuple[int, int, int]] = None
        self._origin: NDArray[np.float64] = np.zeros(3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> "ExclusionGrid":
        """Construct the exclusion grid.

        Returns *self* so callers can chain: ``grid = ExclusionGrid(...).build()``.
        """
        na, nb, nc = self._determine_grid_shape()
        self._grid_shape = (na, nb, nc)

        # Fractional coordinates of every grid point ----------------------
        fa = np.linspace(0.0, 1.0, na, endpoint=False)
        fb = np.linspace(0.0, 1.0, nb, endpoint=False)
        fc = np.linspace(0.0, 1.0, nc, endpoint=False)
        frac_grid = np.stack(
            np.meshgrid(fa, fb, fc, indexing="ij"), axis=-1
        )  # (na, nb, nc, 3)

        # Cartesian coordinates of grid points ----------------------------
        cart_grid = frac_grid @ self.lattice  # broadcast (na,nb,nc,3) @ (3,3)

        # Start with all voxels allowed (False = not forbidden) -----------
        forbidden = np.zeros((na, nb, nc), dtype=np.bool_)

        # Pre-compute TM info if surface_aware ----------------------------
        tm_coord: Dict[int, int] = {}
        if self.surface_aware:
            tm_coord = self._compute_tm_coordination()

        # Mark forbidden voxels per framework atom ------------------------
        for idx, label in enumerate(self.atom_labels):
            pos = self.positions[idx]
            r_fw = VDW_RADII.get(label, 1.5)
            cutoff = r_fw + self._r_counter  # basic overlap exclusion

            # Extra buffer for TM sites -----------------------------------
            extra = 0.0
            if label in self.tm_elements:
                if self.surface_aware:
                    coord_num = tm_coord.get(idx, 0)
                    if coord_num >= 6:
                        extra = self.tm_buffer        # buried – full buffer
                    elif coord_num >= 4:
                        extra = self.tm_buffer * 0.5  # partially exposed
                    # coord < 4: surface-exposed → no extra buffer
                else:
                    extra = self.tm_buffer

            total_cutoff = cutoff + extra

            # Determine which grid points are within total_cutoff ---------
            mask = self._points_within_cutoff_pbc(
                cart_grid, pos, total_cutoff
            )
            forbidden |= mask

        # Mark voxels too far from ALL framework atoms as forbidden -----
        # Uses direct Euclidean distance (no PBC) so that ions near cell
        # boundaries are measured against the real framework, not periodic
        # images on the opposite side.
        near_any = np.zeros((na, nb, nc), dtype=np.bool_)
        cutoff_sq = self.max_framework_distance ** 2
        for idx in range(len(self.positions)):
            diff = cart_grid - self.positions[idx]  # (..., 3)
            dist_sq = np.sum(diff ** 2, axis=-1)
            near_any |= dist_sq <= cutoff_sq
        forbidden |= ~near_any

        self._grid = forbidden
        return self

    def is_allowed(self, position: NDArray[np.float64]) -> bool:
        """Check whether a Cartesian position falls in an allowed voxel.

        Parameters
        ----------
        position : array-like, shape (3,)
            Cartesian coordinates (Angstroms).

        Returns
        -------
        bool
            *True* when the voxel is **not** forbidden.
        """
        if self._grid is None:
            raise RuntimeError("Call build() before querying the grid.")

        position = np.asarray(position, dtype=np.float64)
        # Map to fractional, wrap into [0, 1)
        frac = cartesian_to_fractional(position, self.lattice) % 1.0
        na, nb, nc = self._grid_shape  # type: ignore[misc]
        ia = int(frac[0] * na) % na
        ib = int(frac[1] * nb) % nb
        ic = int(frac[2] * nc) % nc
        return not self._grid[ia, ib, ic]

    def get_allowed_positions(self) -> NDArray[np.float64]:
        """Return Cartesian coordinates of all allowed grid points.

        Returns
        -------
        np.ndarray, shape (M, 3)
            Each row is a Cartesian position in Angstroms.
        """
        if self._grid is None:
            raise RuntimeError("Call build() before querying the grid.")

        na, nb, nc = self._grid_shape  # type: ignore[misc]
        allowed_idx = np.argwhere(~self._grid)  # (M, 3)  int indices
        if allowed_idx.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        frac = allowed_idx.astype(np.float64)
        frac[:, 0] /= na
        frac[:, 1] /= nb
        frac[:, 2] /= nc
        return frac @ self.lattice  # (M, 3) Cartesian

    def get_allowed_fraction(self) -> float:
        """Fraction of grid voxels that are allowed (not forbidden).

        Useful for diagnostics – a very small allowed fraction may indicate
        that the grid resolution is too coarse or the buffers too large.
        """
        if self._grid is None:
            raise RuntimeError("Call build() before querying the grid.")
        return float(np.count_nonzero(~self._grid) / self._grid.size)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _determine_grid_shape(self) -> Tuple[int, int, int]:
        """Compute the number of grid points along each lattice vector.

        Auto-coarsens if the total voxel count exceeds the safety threshold
        and issues a warning.
        """
        lengths = np.linalg.norm(self.lattice, axis=1)  # a, b, c lengths
        n = np.maximum(np.round(lengths / self.grid_resolution).astype(int), 1)
        n = np.minimum(n, _MAX_GRID_POINTS)

        total = int(np.prod(n))
        if total > _COARSEN_THRESHOLD:
            scale = (total / _COARSEN_THRESHOLD) ** (1.0 / 3.0)
            n = np.maximum((n / scale).astype(int), 1)
            effective_res = lengths / n
            warnings.warn(
                f"Grid resolution {self.grid_resolution:.3f} A is too fine for "
                f"this cell ({lengths[0]:.1f} x {lengths[1]:.1f} x {lengths[2]:.1f} A). "
                f"Auto-coarsened to ~{effective_res.mean():.3f} A "
                f"({n[0]}x{n[1]}x{n[2]} = {int(np.prod(n)):,} voxels).",
                stacklevel=2,
            )

        return int(n[0]), int(n[1]), int(n[2])

    def _points_within_cutoff_pbc(
        self,
        cart_grid: NDArray[np.float64],
        site: NDArray[np.float64],
        cutoff: float,
    ) -> NDArray[np.bool_]:
        """Return a boolean mask over *cart_grid* for points within *cutoff*
        of *site*, respecting periodic boundary conditions.

        For efficiency, we convert grid and site to fractional space,
        compute the minimum-image fractional displacement, convert to
        Cartesian, and compare distances.

        Parameters
        ----------
        cart_grid : ndarray, shape (..., 3)
            Cartesian grid points.
        site : ndarray, shape (3,)
            Cartesian position of the framework atom.
        cutoff : float
            Exclusion radius in Angstroms.

        Returns
        -------
        ndarray of bool, same leading shape as *cart_grid* minus last axis.
        """
        # Fractional displacement, minimum image
        inv_lattice = np.linalg.inv(self.lattice)  # (3,3)
        frac_grid = cart_grid @ inv_lattice          # (..., 3)
        frac_site = site @ inv_lattice               # (3,)

        diff_frac = frac_grid - frac_site            # (..., 3)
        # Minimum-image convention: wrap to [-0.5, 0.5)
        diff_frac -= np.round(diff_frac)

        # Back to Cartesian
        diff_cart = diff_frac @ self.lattice          # (..., 3)
        dist_sq = np.sum(diff_cart ** 2, axis=-1)
        return dist_sq <= cutoff * cutoff

    def _compute_tm_coordination(self) -> Dict[int, int]:
        """Compute coordination numbers for all TM sites.

        A neighbour is any framework atom within 2.5 A (first coordination
        shell for typical TM-O bonds).

        Returns
        -------
        dict[int, int]
            Mapping from atom index to coordination number.
        """
        coord_cutoff = 2.5  # Angstroms
        tm_indices = [
            i for i, lab in enumerate(self.atom_labels)
            if lab in self.tm_elements
        ]
        coordination: Dict[int, int] = {}
        for ti in tm_indices:
            count = 0
            pos_tm = self.positions[ti]
            for j, lab_j in enumerate(self.atom_labels):
                if j == ti:
                    continue
                d = min_image_distance(pos_tm, self.positions[j], self.lattice)
                if d <= coord_cutoff:
                    count += 1
            coordination[ti] = count
        return coordination


# ===================================================================
# FrameworkInfo
# ===================================================================

class FrameworkInfo:
    """Extract structural metadata from a parsed POM framework.

    Parameters
    ----------
    structure : dict
        Parsed POSCAR dict from ``structure_parser.parse_poscar()``.
    tm_elements : list[str] or None
        Transition-metal symbols. Defaults to ``defaults.DEFAULT_TM_ELEMENTS``.
    """

    def __init__(
        self,
        structure: dict,
        tm_elements: Optional[List[str]] = None,
    ) -> None:
        self.structure = structure
        self.tm_elements: List[str] = (
            list(tm_elements) if tm_elements is not None else list(DEFAULT_TM_ELEMENTS)
        )

        self.lattice: NDArray[np.float64] = np.array(
            structure["lattice"], dtype=np.float64
        )
        self.positions: NDArray[np.float64] = np.array(
            structure["positions"], dtype=np.float64
        )
        self.atom_labels: List[str] = list(structure["atom_labels"])

        # Lazily cached properties
        self._tm_sites: Optional[List[Tuple[int, str, NDArray[np.float64]]]] = None
        self._tm_coordination: Optional[Dict[int, int]] = None
        self._centroid: Optional[NDArray[np.float64]] = None
        self._max_radius: Optional[float] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tm_sites(self) -> List[Tuple[int, str, NDArray[np.float64]]]:
        """List of ``(index, element, position)`` for every TM atom."""
        if self._tm_sites is None:
            self._tm_sites = [
                (i, lab, self.positions[i].copy())
                for i, lab in enumerate(self.atom_labels)
                if lab in self.tm_elements
            ]
        return self._tm_sites

    @property
    def tm_coordination(self) -> Dict[int, int]:
        """Coordination number for each TM site (index -> count).

        Uses a 2.5 A cutoff for the first coordination shell.
        """
        if self._tm_coordination is None:
            coord_cutoff = 2.5
            self._tm_coordination = {}
            for idx, _elem, pos_tm in self.tm_sites:
                count = 0
                for j in range(len(self.atom_labels)):
                    if j == idx:
                        continue
                    d = min_image_distance(
                        pos_tm, self.positions[j], self.lattice
                    )
                    if d <= coord_cutoff:
                        count += 1
                self._tm_coordination[idx] = count
        return self._tm_coordination

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Centre of mass of the framework (equal atomic weights assumed).

        Computed in fractional space to handle PBC correctly, then
        converted back to Cartesian.
        """
        if self._centroid is None:
            inv_lat = np.linalg.inv(self.lattice)
            frac = self.positions @ inv_lat  # (N, 3)
            # Unwrap periodic jumps relative to the first atom
            ref = frac[0]
            diff = frac - ref
            diff -= np.round(diff)
            unwrapped = ref + diff
            mean_frac = unwrapped.mean(axis=0) % 1.0
            self._centroid = mean_frac @ self.lattice
        return self._centroid

    @property
    def max_radius(self) -> float:
        """Maximum distance from the centroid to any framework atom,
        accounting for periodic boundary conditions."""
        if self._max_radius is None:
            dists = np.array([
                min_image_distance(self.centroid, pos, self.lattice)
                for pos in self.positions
            ])
            self._max_radius = float(dists.max())
        return self._max_radius

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_charge_centers(
        self,
        charges: NDArray[np.float64],
        threshold_fraction: float = 0.25,
    ) -> NDArray[np.float64]:
        """Identify positions of high negative charge density.

        Useful for guiding clustered counterion placement toward the most
        negatively charged regions of the POM.

        Parameters
        ----------
        charges : array-like, shape (N,)
            Partial charges for each atom in the structure (same ordering as
            ``structure['positions']``).  Negative values indicate electron-
            rich sites.
        threshold_fraction : float
            Fraction of the most-negative charge range used as the selection
            threshold.  For example, 0.25 selects atoms whose charge is
            within the bottom 25 % of the charge range (i.e. the most
            negative quarter).

        Returns
        -------
        np.ndarray, shape (M, 3)
            Cartesian positions of the atoms satisfying the charge criterion.
        """
        charges = np.asarray(charges, dtype=np.float64)
        if charges.shape[0] != self.positions.shape[0]:
            raise ValueError(
                f"charges length ({charges.shape[0]}) does not match number "
                f"of atoms ({self.positions.shape[0]})."
            )

        q_min = charges.min()
        q_max = charges.max()
        if q_min == q_max:
            # All charges identical – return centroid as sole charge centre.
            return self.centroid.reshape(1, 3)

        cutoff = q_min + threshold_fraction * (q_max - q_min)
        mask = charges <= cutoff  # most negative
        return self.positions[mask].copy()
