"""
charge_analyzer.py - Formal charge analysis for POM frameworks.

Assigns formal oxidation states to every atom in a parsed POSCAR structure
and computes the net framework charge, which determines how many
counterions are needed.
"""

from typing import Dict, Any, Optional, List
import warnings
import numpy as np

from defaults import OXIDATION_STATES


def analyze_charges(
    structure: Dict[str, Any],
    oxidation_states: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Analyse formal charges of a POM framework.

    Parameters
    ----------
    structure : dict
        Parsed POSCAR dictionary as returned by ``parse_poscar``.
    oxidation_states : dict, optional
        Element → formal oxidation state overrides.  Any element not
        provided here (and not in the built-in defaults) will be assigned
        charge 0 with a warning.

    Returns
    -------
    dict
        - 'per_atom_charges'  : list[int]   - charge for every atom
        - 'net_charge'        : int          - total framework charge
        - 'counterion_count'  : dict         - {valence: count} e.g. {1: 6}
        - 'species_charges'   : dict[str, int] - element → oxidation state used
    """
    # Merge default oxidation states with user overrides
    ox: Dict[str, int] = dict(OXIDATION_STATES)
    if oxidation_states is not None:
        ox.update(oxidation_states)

    atom_labels: List[str] = structure["atom_labels"]
    per_atom_charges: List[int] = []
    species_charges: Dict[str, int] = {}

    for label in atom_labels:
        if label in ox:
            charge = ox[label]
        else:
            warnings.warn(
                f"No oxidation state defined for element '{label}'. "
                "Assuming 0."
            )
            charge = 0
        per_atom_charges.append(charge)
        species_charges[label] = charge

    net_charge: int = sum(per_atom_charges)

    # Validate
    if net_charge > 0:
        warnings.warn(
            f"Net framework charge is positive ({net_charge:+d}). "
            "POMs are expected to have a negative net charge."
        )
    elif net_charge == 0:
        warnings.warn(
            "Net framework charge is zero. No counterions needed."
        )

    # Calculate counterion counts for common valences
    abs_charge = abs(net_charge)
    counterion_count: Dict[int, int] = {}
    for valence in (1, 2, 3):
        if abs_charge % valence == 0:
            counterion_count[valence] = abs_charge // valence

    return {
        "per_atom_charges": per_atom_charges,
        "net_charge": net_charge,
        "counterion_count": counterion_count,
        "species_charges": species_charges,
    }
