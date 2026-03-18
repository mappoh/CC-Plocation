"""
writer.py - VASP POSCAR output writers for CC-Plocation.

Writes combined framework + counterion structures to POSCAR format,
with support for batch output of multiple configurations.
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np


def write_poscar(
    filepath: str,
    structure: Dict[str, Any],
    counterion_positions: np.ndarray,
    counterion_element: str,
    comment: Optional[str] = None,
) -> None:
    """Write a VASP POSCAR file combining framework atoms and counterions.

    Parameters
    ----------
    filepath : str
        Output file path.
    structure : dict
        Parsed POSCAR dictionary (from ``parse_poscar``).
    counterion_positions : np.ndarray
        (M, 3) Cartesian positions of counterions.
    counterion_element : str
        Element symbol for the counterions (e.g. 'K', 'Na').
    comment : str, optional
        Comment line.  Defaults to the original comment with a suffix.
    """
    counterion_positions = np.atleast_2d(counterion_positions)
    n_counterions = len(counterion_positions)
    lattice = structure["lattice"]

    # Build combined species list and counts
    species = list(structure["species"])
    counts = list(structure["counts"])

    if counterion_element in species:
        # Append counterions to existing species entry
        ci_idx = species.index(counterion_element)
        counts[ci_idx] += n_counterions
        # Insert counterion positions right after existing atoms of that species
        # Calculate insertion point: sum of counts up to and including ci_idx
        insert_at = sum(structure["counts"][:ci_idx + 1])
        framework_pos = structure["positions"]
        combined_positions = np.vstack([
            framework_pos[:insert_at],
            counterion_positions,
            framework_pos[insert_at:],
        ])
    else:
        species.append(counterion_element)
        counts.append(n_counterions)
        combined_positions = np.vstack([
            structure["positions"],
            counterion_positions,
        ])

    # Convert to fractional for output (Direct format is standard)
    inv_lattice = np.linalg.inv(lattice)
    frac_positions = combined_positions @ inv_lattice

    # Build comment line
    if comment is None:
        comment = f"{structure['comment']} + {n_counterions} {counterion_element}"

    # Write
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as fh:
        # Line 0: comment
        fh.write(f"{comment}\n")

        # Line 1: scale factor (always 1.0 since lattice is pre-scaled)
        fh.write("   1.00000000000000\n")

        # Lines 2-4: lattice vectors
        for vec in lattice:
            fh.write(f"  {vec[0]:20.14f}  {vec[1]:20.14f}  {vec[2]:20.14f}\n")

        # Line 5: species names
        fh.write("   " + "   ".join(species) + "\n")

        # Line 6: counts
        fh.write("   " + "   ".join(str(c) for c in counts) + "\n")

        # Coordinate type
        fh.write("Direct\n")

        # Positions
        for pos in frac_positions:
            fh.write(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}\n")


def write_batch(
    output_dir: str,
    structure: Dict[str, Any],
    configs: List[np.ndarray],
    counterion_element: str,
    strategy_names: List[str],
) -> List[str]:
    """Write multiple counterion configurations to separate POSCAR files.

    Parameters
    ----------
    output_dir : str
        Directory for output files (created if it does not exist).
    structure : dict
        Parsed POSCAR dictionary.
    configs : list of np.ndarray
        Each entry is an (M, 3) array of counterion Cartesian positions.
    counterion_element : str
        Element symbol for the counterions.
    strategy_names : list of str
        Strategy label for each configuration (must be same length as configs).

    Returns
    -------
    list of str
        Paths to all written files.
    """
    if len(configs) != len(strategy_names):
        raise ValueError(
            f"Number of configs ({len(configs)}) does not match "
            f"number of strategy names ({len(strategy_names)})."
        )

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    for idx, (positions, name) in enumerate(zip(configs, strategy_names)):
        filename = f"config_{name}_{idx}.vasp"
        filepath = os.path.join(output_dir, filename)
        comment = (
            f"{structure['comment']} | {name} config {idx} "
            f"| {len(np.atleast_2d(positions))} {counterion_element}"
        )
        write_poscar(filepath, structure, positions, counterion_element, comment=comment)
        written.append(filepath)

    return written
