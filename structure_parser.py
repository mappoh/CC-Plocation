"""
structure_parser.py - VASP POSCAR file parser for CC-Plocation.

Parses POSCAR/CONTCAR files into a structured dictionary without
external dependencies beyond numpy.
"""

from typing import Dict, List, Any, Optional
import numpy as np


def parse_poscar(filepath: str) -> Dict[str, Any]:
    """Parse a VASP POSCAR file.

    Handles both Direct (fractional) and Cartesian coordinate formats,
    optional Selective Dynamics lines, and the optional velocities block.

    Parameters
    ----------
    filepath : str
        Path to the POSCAR file.

    Returns
    -------
    dict
        Keys:
        - 'comment'     : str   - first line (comment / system name)
        - 'scale'       : float - universal scaling factor
        - 'lattice'     : np.ndarray (3, 3) - lattice vectors (rows), scaled
        - 'species'     : list[str] - element symbols in order
        - 'counts'      : list[int] - atom count per species
        - 'coord_type'  : str   - 'Cartesian' or 'Direct'
        - 'positions'   : np.ndarray (N, 3) - atomic positions in Cartesian
        - 'atom_labels' : list[str] - element symbol for every atom
        - 'selective_dynamics' : bool - whether selective dynamics was present
    """
    with open(filepath, "r") as fh:
        lines = fh.readlines()

    # Strip trailing whitespace / newlines
    lines = [line.rstrip() for line in lines]

    idx = 0

    # --- Line 0: comment ---
    comment = lines[idx]
    idx += 1

    # --- Line 1: scaling factor ---
    scale = float(lines[idx].split()[0])
    idx += 1

    # --- Lines 2-4: lattice vectors ---
    lattice = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        lattice[i] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
    lattice *= scale

    # --- Line 5: species names (VASP 5+ format) ---
    # Determine if this line contains element symbols or counts.
    tokens = lines[idx].split()
    try:
        # If all tokens are integers, this is VASP 4 format (no species line).
        [int(t) for t in tokens]
        raise ValueError(
            "VASP 4 format (no element symbols line) is not supported. "
            "Please add a species-name line to the POSCAR file."
        )
    except ValueError as e:
        if "VASP 4" in str(e):
            raise
        species: List[str] = tokens
        idx += 1

    # --- Line 6: atom counts ---
    counts: List[int] = [int(t) for t in lines[idx].split()]
    idx += 1
    total_atoms = sum(counts)

    if len(counts) != len(species):
        raise ValueError(
            f"Mismatch: {len(species)} species names but {len(counts)} count entries."
        )

    # --- Optional: Selective Dynamics ---
    selective_dynamics = False
    next_line = lines[idx].strip()
    if next_line and next_line[0] in ("S", "s"):
        selective_dynamics = True
        idx += 1

    # --- Coordinate type ---
    coord_line = lines[idx].strip()
    idx += 1
    if coord_line[0] in ("C", "c", "K", "k"):
        coord_type = "Cartesian"
    elif coord_line[0] in ("D", "d"):
        coord_type = "Direct"
    else:
        raise ValueError(f"Unrecognised coordinate type: '{coord_line}'")

    # --- Positions ---
    positions = np.zeros((total_atoms, 3), dtype=np.float64)
    for i in range(total_atoms):
        tokens = lines[idx].split()
        positions[i] = [float(tokens[j]) for j in range(3)]
        idx += 1

    # Convert Direct → Cartesian
    if coord_type == "Direct":
        positions = positions @ lattice  # frac_to_cart

    # If scaling factor is applied and coords are Cartesian, scale them too
    # (lattice already scaled above; Cartesian coords need separate scaling)
    if coord_type == "Cartesian" and scale != 1.0:
        positions *= scale

    # --- Build per-atom labels ---
    atom_labels: List[str] = []
    for sp, cnt in zip(species, counts):
        atom_labels.extend([sp] * cnt)

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "species": species,
        "counts": counts,
        "coord_type": coord_type,
        "positions": positions,
        "atom_labels": atom_labels,
        "selective_dynamics": selective_dynamics,
    }
