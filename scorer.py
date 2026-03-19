"""
scorer.py - Validates and scores counterion configurations for CC-Plocation.

Performs hard constraint checks (overlap, TM proximity, charge neutrality)
and computes soft Coulomb energy scores for ranking valid configurations.
Uses only numpy/scipy with PBC-aware distances via pbc_utils.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from defaults import K_COULOMB, TM_COORDINATION_CUTOFF, VDW_RADII, TM_ELEMENTS
from pbc_utils import cross_pbc_distance_matrix, cross_direct_distance_matrix, minimum_image_distance


class ConfigurationScorer:
    """Validate and score a counterion configuration around a POM framework.

    Parameters
    ----------
    structure : dict
        Must contain keys:
        - 'lattice': 3x3 numpy array (row vectors are lattice vectors)
        - 'positions': Nx3 numpy array of Cartesian positions
        - 'atom_labels': list[str] of element symbols, length N
    counterion_element : str
        Element symbol for the counterion (e.g. "K", "Na", "Cs").
    counterion_charges : dict[str, float]
        Map element symbol -> formal charge for counterion(s).
    framework_charges : dict[str, float]
        Map element symbol -> formal charge for every framework species.
    tm_elements : set[str] or None
        Set of transition-metal element symbols present in the framework.
        If None, auto-detected from TM_ELEMENTS.
    tm_buffer : float
        Minimum allowed distance (A) between a counterion and a *buried* TM.
    energy_threshold : float or None
        Optional upper bound on Coulomb energy (eV) for validity.
    min_ion_spacing : float
        Minimum allowed distance (A) between any two counterions.
    """

    def __init__(
        self,
        structure: Dict[str, Any],
        counterion_element: str,
        counterion_charges: Dict[str, float],
        framework_charges: Dict[str, float],
        tm_elements: Optional[set] = None,
        tm_buffer: float = 3.5,
        energy_threshold: Optional[float] = None,
        min_ion_spacing: float = 2.0,
        max_framework_distance: Optional[float] = None,
    ) -> None:
        self.structure = structure
        self.lattice: np.ndarray = np.array(structure["lattice"], dtype=np.float64)
        self.cart_coords: np.ndarray = np.array(structure["positions"], dtype=np.float64)
        self.species: List[str] = list(structure["atom_labels"])
        self.counterion_element = counterion_element
        self.counterion_charges = counterion_charges
        self.framework_charges = framework_charges
        self.tm_buffer = tm_buffer
        self.energy_threshold = energy_threshold
        self.min_ion_spacing = min_ion_spacing
        if max_framework_distance is None:
            raise ValueError(
                "max_framework_distance must be provided."
            )
        self.max_framework_distance = float(max_framework_distance)

        # Identify TM sites
        if tm_elements is None:
            self.tm_elements = set(TM_ELEMENTS)
        else:
            self.tm_elements = set(tm_elements)
        self.tm_indices: List[int] = [
            i for i, s in enumerate(self.species) if s in self.tm_elements
        ]
        self.tm_coords: np.ndarray = (
            self.cart_coords[self.tm_indices] if self.tm_indices else np.empty((0, 3))
        )

        # Pre-compute vdW radii for framework atoms
        self.framework_vdw: np.ndarray = np.array(
            [VDW_RADII.get(s, 2.0) for s in self.species], dtype=np.float64
        )
        self.counterion_vdw: float = VDW_RADII.get(counterion_element, 2.0)

        # Identify which TM atoms are "buried" (not surface-exposed)
        self._buried_tm_mask = self._compute_buried_tm_mask()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_buried_tm_mask(self) -> np.ndarray:
        """Return boolean array (len = len(tm_indices)) marking buried TMs.

        A TM is considered *buried* if it is surrounded by oxygen atoms on
        all sides (approximated by having >= 4 O neighbours within 2.5 A).
        """
        if len(self.tm_indices) == 0:
            return np.array([], dtype=bool)

        o_indices = [i for i, s in enumerate(self.species) if s == "O"]
        if len(o_indices) == 0:
            return np.ones(len(self.tm_indices), dtype=bool)

        o_coords = self.cart_coords[o_indices]
        buried = np.zeros(len(self.tm_indices), dtype=bool)
        for k, idx in enumerate(self.tm_indices):
            tm_pos = self.cart_coords[idx]
            dists = np.array([
                minimum_image_distance(tm_pos, oc, self.lattice) for oc in o_coords
            ])
            n_close = np.sum(dists < TM_COORDINATION_CUTOFF)
            buried[k] = n_close >= 4
        return buried

    def _check_framework_overlap(
        self, ion_positions: np.ndarray, fw_dist_mat: np.ndarray = None
    ) -> Dict[str, Any]:
        """Check that no counterion overlaps with a framework atom."""
        if len(ion_positions) == 0:
            return {"passed": True, "value": np.inf, "threshold": 0.0,
                    "detail": "No ions to check"}

        if fw_dist_mat is None:
            dist_mat = cross_pbc_distance_matrix(
                ion_positions, self.cart_coords, self.lattice
            )
        else:
            dist_mat = fw_dist_mat
        # vdW sum for each (ion, framework_atom) pair
        vdw_sums = self.counterion_vdw + self.framework_vdw  # shape (N_fw,)
        # Broadcast: dist_mat shape (N_ion, N_fw), vdw_sums shape (N_fw,)
        violations = dist_mat < vdw_sums[np.newaxis, :]
        min_dist = float(np.min(dist_mat))
        min_vdw = float(vdw_sums[np.unravel_index(np.argmin(dist_mat), dist_mat.shape)[1]])
        passed = not np.any(violations)
        return {
            "passed": passed,
            "value": min_dist,
            "threshold": min_vdw,
            "detail": (
                f"Min ion-framework dist {min_dist:.3f} A "
                f"(vdW sum threshold {min_vdw:.3f} A)"
            ),
        }

    def _check_ion_ion_overlap(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that no two counterions overlap (vdW criterion)."""
        n = len(ion_positions)
        if n < 2:
            return {"passed": True, "value": np.inf, "threshold": 0.0,
                    "detail": "Fewer than 2 ions"}

        dist_mat = cross_pbc_distance_matrix(
            ion_positions, ion_positions, self.lattice
        )
        np.fill_diagonal(dist_mat, np.inf)
        min_dist = float(np.min(dist_mat))
        vdw_sum = 2.0 * self.counterion_vdw
        passed = min_dist > vdw_sum
        return {
            "passed": passed,
            "value": min_dist,
            "threshold": vdw_sum,
            "detail": (
                f"Min ion-ion dist {min_dist:.3f} A "
                f"(vdW sum threshold {vdw_sum:.3f} A)"
            ),
        }

    def _check_tm_proximity(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that no counterion is too close to a buried TM site."""
        if len(self.tm_indices) == 0 or len(ion_positions) == 0:
            return {"passed": True, "value": np.inf,
                    "threshold": self.tm_buffer,
                    "detail": "No TM sites or no ions"}

        buried_coords = self.tm_coords[self._buried_tm_mask]
        if len(buried_coords) == 0:
            return {"passed": True, "value": np.inf,
                    "threshold": self.tm_buffer,
                    "detail": "No buried TM sites"}

        dist_mat = cross_pbc_distance_matrix(
            ion_positions, buried_coords, self.lattice
        )
        min_dist = float(np.min(dist_mat))
        passed = min_dist > self.tm_buffer
        return {
            "passed": passed,
            "value": min_dist,
            "threshold": self.tm_buffer,
            "detail": (
                f"Min ion-buried_TM dist {min_dist:.3f} A "
                f"(buffer {self.tm_buffer:.3f} A)"
            ),
        }

    def _check_min_ion_spacing(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that all ion-ion distances exceed min_ion_spacing."""
        n = len(ion_positions)
        if n < 2:
            return {"passed": True, "value": np.inf,
                    "threshold": self.min_ion_spacing,
                    "detail": "Fewer than 2 ions"}

        dist_mat = cross_pbc_distance_matrix(
            ion_positions, ion_positions, self.lattice
        )
        np.fill_diagonal(dist_mat, np.inf)
        min_dist = float(np.min(dist_mat))
        passed = min_dist > self.min_ion_spacing
        return {
            "passed": passed,
            "value": min_dist,
            "threshold": self.min_ion_spacing,
            "detail": (
                f"Min ion-ion dist {min_dist:.3f} A "
                f"(min spacing {self.min_ion_spacing:.3f} A)"
            ),
        }

    def _check_max_framework_distance(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that every counterion is within max_framework_distance of
        at least one framework atom.

        Uses direct Euclidean distance (no PBC wrapping) so that ions
        near cell boundaries are measured against the real framework
        positions, not periodic images on the opposite side of the cell.
        """
        if len(ion_positions) == 0:
            return {"passed": True, "value": 0.0,
                    "threshold": self.max_framework_distance,
                    "detail": "No ions to check"}

        dist_mat = cross_direct_distance_matrix(
            ion_positions, self.cart_coords
        )
        # For each ion, find the distance to its nearest framework atom
        min_fw_per_ion = np.min(dist_mat, axis=1)
        max_of_mins = float(np.max(min_fw_per_ion))
        passed = max_of_mins <= self.max_framework_distance
        return {
            "passed": passed,
            "value": max_of_mins,
            "threshold": self.max_framework_distance,
            "detail": (
                f"Max nearest-framework dist {max_of_mins:.3f} A "
                f"(limit {self.max_framework_distance:.3f} A)"
            ),
        }

    def _check_charge_neutrality(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that total charge (framework + counterions) sums to zero."""
        fw_charge = sum(
            self.framework_charges.get(s, 0.0) for s in self.species
        )
        ion_charge = len(ion_positions) * self.counterion_charges.get(
            self.counterion_element, 0.0
        )
        total = fw_charge + ion_charge
        passed = abs(total) < 1e-6
        return {
            "passed": passed,
            "value": total,
            "threshold": 0.0,
            "detail": (
                f"Framework charge {fw_charge:.2f}, "
                f"ion charge {ion_charge:.2f}, total {total:.2f}"
            ),
        }

    def _compute_coulomb_energy(
        self, ion_positions: np.ndarray
    ) -> float:
        """Compute total Coulomb energy (eV) via pairwise sum with PBC.

        Includes ion-ion, ion-framework, and framework-framework interactions.
        Uses minimum image convention for PBC.
        """
        n_fw = len(self.cart_coords)
        n_ion = len(ion_positions)

        # Build combined coordinate and charge arrays
        all_coords = np.vstack([self.cart_coords, ion_positions])
        fw_charges_arr = np.array(
            [self.framework_charges.get(s, 0.0) for s in self.species],
            dtype=np.float64,
        )
        ion_charges_arr = np.full(
            n_ion,
            self.counterion_charges.get(self.counterion_element, 0.0),
            dtype=np.float64,
        )
        all_charges = np.concatenate([fw_charges_arr, ion_charges_arr])

        n_total = n_fw + n_ion

        # Vectorised pairwise Coulomb sum using minimum image convention
        # Build full distance matrix via cross_pbc_distance_matrix
        dist_mat = cross_pbc_distance_matrix(all_coords, all_coords, self.lattice)

        # Charge product matrix (upper triangle)
        charge_prod = np.outer(all_charges, all_charges)

        # Mask diagonal and lower triangle
        mask = np.triu(np.ones((n_total, n_total), dtype=bool), k=1)
        # Avoid division by zero
        safe_dist = np.where(dist_mat > 1e-10, dist_mat, np.inf)

        energy = np.sum(K_COULOMB * charge_prod[mask] / safe_dist[mask])
        return float(energy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, counterion_positions: np.ndarray) -> Dict[str, Any]:
        """Validate and score a counterion configuration.

        Parameters
        ----------
        counterion_positions : np.ndarray
            (N_ions, 3) array of Cartesian coordinates for counterions.

        Returns
        -------
        dict
            'valid': bool - True if all hard checks pass.
            'checks': dict of check_name -> {passed, value, threshold, detail}.
            'coulomb_energy_eV': float - total Coulomb energy.
            'min_ion_framework_dist': float
            'min_ion_ion_dist': float
            'min_ion_tm_dist': float
            'total_score': float - lower is better (for ranking).
        """
        ions = np.atleast_2d(np.array(counterion_positions, dtype=np.float64))

        # Pre-compute distance matrices to avoid redundant work
        fw_dist_mat = (
            cross_pbc_distance_matrix(ions, self.cart_coords, self.lattice)
            if len(ions) > 0 else None
        )

        checks: Dict[str, Dict[str, Any]] = {}
        checks["framework_overlap"] = self._check_framework_overlap(ions, fw_dist_mat)
        checks["ion_ion_overlap"] = self._check_ion_ion_overlap(ions)
        checks["tm_proximity"] = self._check_tm_proximity(ions)
        checks["min_ion_spacing"] = self._check_min_ion_spacing(ions)
        checks["max_framework_distance"] = self._check_max_framework_distance(ions)
        checks["charge_neutrality"] = self._check_charge_neutrality(ions)

        valid = all(c["passed"] for c in checks.values())

        # Coulomb energy
        coulomb_energy = self._compute_coulomb_energy(ions)

        # If an energy threshold is set, treat it as a soft invalidation
        if self.energy_threshold is not None and coulomb_energy > self.energy_threshold:
            valid = False

        # Distance summaries (reuse pre-computed matrix)
        min_ion_fw = float(np.min(fw_dist_mat)) if fw_dist_mat is not None else np.inf

        if len(ions) >= 2:
            ion_dists = cross_pbc_distance_matrix(
                ions, ions, self.lattice
            )
            np.fill_diagonal(ion_dists, np.inf)
            min_ion_ion = float(np.min(ion_dists))
        else:
            min_ion_ion = np.inf

        if len(self.tm_coords) > 0 and len(ions) > 0:
            tm_dists = cross_pbc_distance_matrix(
                ions, self.tm_coords, self.lattice
            )
            min_ion_tm = float(np.min(tm_dists))
        else:
            min_ion_tm = np.inf

        return {
            "valid": valid,
            "checks": checks,
            "coulomb_energy_eV": coulomb_energy,
            "min_ion_framework_dist": min_ion_fw,
            "min_ion_ion_dist": min_ion_ion,
            "min_ion_tm_dist": min_ion_tm,
            "total_score": coulomb_energy,
        }
