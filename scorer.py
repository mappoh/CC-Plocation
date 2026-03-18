"""
scorer.py - Validates and scores counterion configurations for CC-Plocation.

Performs hard constraint checks (overlap, TM proximity, charge neutrality)
and computes soft Coulomb energy scores for ranking valid configurations.
Uses only numpy/scipy with PBC-aware distances via pbc_utils.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pbc_utils import minimum_image_distance

# Coulomb constant in eV*Ang/e^2
K_COULOMB: float = 14.3996

# Default vdW radii (Angstroms) for common elements
VDW_RADII: Dict[str, float] = {
    "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92,
    "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
    "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
    "Sc": 2.11, "Ti": 1.87, "V": 1.79, "Cr": 1.89, "Mn": 1.97,
    "Fe": 1.94, "Co": 1.92, "Ni": 1.63, "Cu": 1.40, "Zn": 1.39,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85,
    "Kr": 2.02, "Rb": 3.03, "Sr": 2.49, "Y": 2.19, "Zr": 1.86,
    "Nb": 2.07, "Mo": 2.09, "Tc": 2.09, "Ru": 2.07, "Rh": 1.95,
    "Pd": 1.63, "Ag": 1.72, "Cd": 1.58, "In": 1.93, "Sn": 2.17,
    "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16, "Cs": 3.43,
    "Ba": 2.68, "La": 2.43, "W": 2.10, "Re": 2.05, "Os": 2.00,
    "Ir": 2.02, "Pt": 1.75, "Au": 1.66, "Hg": 1.55, "Tl": 1.96,
    "Pb": 2.02, "Bi": 2.07,
}

# Default set of transition metal element symbols
DEFAULT_TM_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def _get_vdw_radius(element: str) -> float:
    """Return vdW radius for *element*, defaulting to 2.0 A if unknown."""
    return VDW_RADII.get(element, 2.0)


def _cross_pbc_distance_matrix(
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


def _minimum_image_vector(
    pos1: np.ndarray,
    pos2: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Return the minimum-image displacement vector from pos1 to pos2.

    Parameters
    ----------
    pos1, pos2 : np.ndarray
        (3,) Cartesian positions.
    lattice : np.ndarray
        (3, 3) lattice matrix (rows = lattice vectors).

    Returns
    -------
    np.ndarray
        (3,) Cartesian displacement vector under minimum image convention.
    """
    inv_lattice = np.linalg.inv(lattice)
    diff_cart = pos2 - pos1
    diff_frac = diff_cart @ inv_lattice
    diff_frac -= np.round(diff_frac)
    return diff_frac @ lattice


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
        If None, auto-detected from DEFAULT_TM_ELEMENTS.
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

        # Identify TM sites
        if tm_elements is None:
            self.tm_elements = DEFAULT_TM_ELEMENTS
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
            [_get_vdw_radius(s) for s in self.species], dtype=np.float64
        )
        self.counterion_vdw: float = _get_vdw_radius(counterion_element)

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
            n_close = np.sum(dists < 2.5)
            buried[k] = n_close >= 4
        return buried

    def _check_framework_overlap(
        self, ion_positions: np.ndarray
    ) -> Dict[str, Any]:
        """Check that no counterion overlaps with a framework atom."""
        if len(ion_positions) == 0:
            return {"passed": True, "value": np.inf, "threshold": 0.0,
                    "detail": "No ions to check"}

        dist_mat = _cross_pbc_distance_matrix(
            ion_positions, self.cart_coords, self.lattice
        )
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

        dist_mat = _cross_pbc_distance_matrix(
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

        dist_mat = _cross_pbc_distance_matrix(
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

        dist_mat = _cross_pbc_distance_matrix(
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
        # Build full distance matrix via _cross_pbc_distance_matrix
        dist_mat = _cross_pbc_distance_matrix(all_coords, all_coords, self.lattice)

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

        checks: Dict[str, Dict[str, Any]] = {}
        checks["framework_overlap"] = self._check_framework_overlap(ions)
        checks["ion_ion_overlap"] = self._check_ion_ion_overlap(ions)
        checks["tm_proximity"] = self._check_tm_proximity(ions)
        checks["min_ion_spacing"] = self._check_min_ion_spacing(ions)
        checks["charge_neutrality"] = self._check_charge_neutrality(ions)

        valid = all(c["passed"] for c in checks.values())

        # Coulomb energy
        coulomb_energy = self._compute_coulomb_energy(ions)

        # If an energy threshold is set, treat it as a soft invalidation
        if self.energy_threshold is not None and coulomb_energy > self.energy_threshold:
            valid = False

        # Distance summaries
        if len(ions) > 0:
            fw_dists = _cross_pbc_distance_matrix(
                ions, self.cart_coords, self.lattice
            )
            min_ion_fw = float(np.min(fw_dists))
        else:
            min_ion_fw = np.inf

        if len(ions) >= 2:
            ion_dists = _cross_pbc_distance_matrix(
                ions, ions, self.lattice
            )
            np.fill_diagonal(ion_dists, np.inf)
            min_ion_ion = float(np.min(ion_dists))
        else:
            min_ion_ion = np.inf

        if len(self.tm_coords) > 0 and len(ions) > 0:
            tm_dists = _cross_pbc_distance_matrix(
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
