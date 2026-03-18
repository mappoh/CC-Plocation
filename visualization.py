"""
visualization.py - Generate VESTA-compatible files and matplotlib scripts
for visualising counterion configurations around POM structures.

Outputs:
  1. Extended XYZ file (readable by VESTA, ASE, Ovito, etc.)
  2. Python script that uses matplotlib to create a 3D scatter plot.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional, Set

import numpy as np

# Default vdW radii (A) for sizing atoms in visualisation
_VDW_RADII: Dict[str, float] = {
    "H": 1.20, "Li": 1.82, "Na": 2.27, "K": 2.75, "Rb": 3.03, "Cs": 3.43,
    "O": 1.52, "N": 1.55, "C": 1.70, "P": 1.80, "S": 1.80,
    "W": 2.10, "Mo": 2.09, "V": 1.79, "Nb": 2.07,
    "Ni": 1.63, "Co": 1.92, "Fe": 1.94, "Mn": 1.97, "Cu": 1.40, "Zn": 1.39,
    "Ti": 1.87, "Cr": 1.89, "Si": 2.10, "Al": 1.84,
}

# Default TM element symbols for colouring
DEFAULT_TM_ELEMENTS: Set[str] = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}


def _lattice_string(lattice: np.ndarray) -> str:
    """Format a 3x3 lattice as a flat string for extended XYZ Lattice field."""
    flat = lattice.flatten()
    return " ".join(f"{v:.10f}" for v in flat)


def write_extended_xyz(
    filepath: str,
    structure: Dict[str, Any],
    counterion_positions: np.ndarray,
    counterion_element: str,
) -> None:
    """Write an extended XYZ file containing framework + counterions.

    Parameters
    ----------
    filepath : str
        Output file path (should end with .xyz).
    structure : dict
        Keys: 'lattice' (3x3), 'positions' (Nx3), 'atom_labels' (list[str]).
    counterion_positions : np.ndarray
        (M, 3) Cartesian coordinates of counterions.
    counterion_element : str
        Element symbol for counterions.
    """
    lattice = np.array(structure["lattice"], dtype=np.float64)
    fw_coords = np.array(structure["positions"], dtype=np.float64)
    fw_species: List[str] = list(structure["atom_labels"])
    ions = np.atleast_2d(np.array(counterion_positions, dtype=np.float64))

    n_total = len(fw_species) + len(ions)

    with open(filepath, "w") as fh:
        fh.write(f"{n_total}\n")
        fh.write(
            f'Lattice="{_lattice_string(lattice)}" '
            f'Properties=species:S:1:pos:R:3\n'
        )
        for sp, coord in zip(fw_species, fw_coords):
            fh.write(f"{sp:>4s}  {coord[0]:14.8f}  {coord[1]:14.8f}  {coord[2]:14.8f}\n")
        for coord in ions:
            fh.write(
                f"{counterion_element:>4s}  {coord[0]:14.8f}  "
                f"{coord[1]:14.8f}  {coord[2]:14.8f}\n"
            )


def _write_matplotlib_script(
    filepath: str,
    xyz_path: str,
    structure: Dict[str, Any],
    counterion_positions: np.ndarray,
    counterion_element: str,
    config_name: str,
    tm_elements: Optional[Set[str]] = None,
) -> None:
    """Write a standalone Python/matplotlib script for 3D visualisation.

    The script reads data embedded directly (no external dependencies beyond
    matplotlib and numpy) so it can be run anywhere.

    Parameters
    ----------
    filepath : str
        Output .py file path.
    xyz_path : str
        Path to the companion XYZ file (for reference in comments).
    structure : dict
        Structure dictionary.
    counterion_positions : np.ndarray
        (M, 3) counterion positions.
    counterion_element : str
        Counterion element symbol.
    config_name : str
        Human-readable label for the configuration.
    tm_elements : set[str] or None
        TM element symbols; defaults to DEFAULT_TM_ELEMENTS.
    """
    if tm_elements is None:
        tm_elements = DEFAULT_TM_ELEMENTS

    lattice = np.array(structure["lattice"], dtype=np.float64)
    fw_coords = np.array(structure["positions"], dtype=np.float64)
    fw_species: List[str] = list(structure["atom_labels"])
    ions = np.atleast_2d(np.array(counterion_positions, dtype=np.float64))

    # Partition framework atoms
    fw_normal_x, fw_normal_y, fw_normal_z = [], [], []
    tm_x, tm_y, tm_z = [], [], []
    for sp, c in zip(fw_species, fw_coords):
        if sp in tm_elements:
            tm_x.append(c[0]); tm_y.append(c[1]); tm_z.append(c[2])
        else:
            fw_normal_x.append(c[0]); fw_normal_y.append(c[1]); fw_normal_z.append(c[2])

    ion_x = [c[0] for c in ions]
    ion_y = [c[1] for c in ions]
    ion_z = [c[2] for c in ions]

    # Unit cell edges: 12 edges of the parallelepiped
    a, b, c_vec = lattice[0], lattice[1], lattice[2]
    origin = np.zeros(3)
    corners = [
        origin, a, b, c_vec,
        a + b, a + c_vec, b + c_vec,
        a + b + c_vec,
    ]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    edge_lines = []
    for i0, i1 in edges:
        p0, p1 = corners[i0], corners[i1]
        edge_lines.append(
            f"    ax.plot3D([{p0[0]:.6f},{p1[0]:.6f}], "
            f"[{p0[1]:.6f},{p1[1]:.6f}], "
            f"[{p0[2]:.6f},{p1[2]:.6f}], 'k-', lw=0.5)"
        )
    edge_block = "\n".join(edge_lines)

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"3D scatter-plot visualisation for {config_name}.

        Generated by CC-Plocation.
        Companion XYZ file: {os.path.basename(xyz_path)}
        \"\"\"
        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Framework atoms (gray, small)
        fw_x = {fw_normal_x}
        fw_y = {fw_normal_y}
        fw_z = {fw_normal_z}
        if fw_x:
            ax.scatter(fw_x, fw_y, fw_z, c='gray', s=15, alpha=0.6, label='Framework')

        # TM sites (green for Ni, red for other TM)
        tm_x = {tm_x}
        tm_y = {tm_y}
        tm_z = {tm_z}
        if tm_x:
            ax.scatter(tm_x, tm_y, tm_z, c='green', s=60, alpha=0.9, label='TM sites')

        # Counterions (blue, large)
        ion_x = {ion_x}
        ion_y = {ion_y}
        ion_z = {ion_z}
        if ion_x:
            ax.scatter(ion_x, ion_y, ion_z, c='blue', s=120, alpha=0.9,
                       label='{counterion_element} counterions')

        # Unit cell wireframe
        {edge_block.lstrip()}

        ax.set_xlabel('x (A)')
        ax.set_ylabel('y (A)')
        ax.set_zlabel('z (A)')
        ax.set_title('{config_name}')
        ax.legend(loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig('{config_name}.png', dpi=150)
        plt.show()
    """)

    with open(filepath, "w") as fh:
        fh.write(script)


def write_vesta_file(
    filepath: str,
    structure: Dict[str, Any],
    counterion_positions: np.ndarray,
    counterion_element: str,
    tm_elements: Optional[Set[str]] = None,
) -> None:
    """Write an extended XYZ file that VESTA can open directly.

    This is a convenience wrapper around :func:`write_extended_xyz`.

    Parameters
    ----------
    filepath : str
        Output file path (.xyz).
    structure : dict
        Structure dictionary with 'lattice', 'positions', 'atom_labels'.
    counterion_positions : np.ndarray
        (M, 3) counterion Cartesian coordinates.
    counterion_element : str
        Element symbol for counterions.
    tm_elements : set[str] or None
        Transition-metal elements (unused here, kept for API consistency).
    """
    write_extended_xyz(filepath, structure, counterion_positions, counterion_element)


def write_visualization(
    output_dir: str,
    structure: Dict[str, Any],
    counterion_positions: np.ndarray,
    counterion_element: str,
    config_name: str,
    tm_elements: Optional[Set[str]] = None,
) -> None:
    """Write both an extended XYZ file and a matplotlib visualisation script.

    Parameters
    ----------
    output_dir : str
        Directory in which to write output files.
    structure : dict
        Structure dictionary with 'lattice', 'positions', 'atom_labels'.
    counterion_positions : np.ndarray
        (M, 3) counterion Cartesian coordinates.
    counterion_element : str
        Counterion element symbol.
    config_name : str
        Base name for the output files (no extension).
    tm_elements : set[str] or None
        TM element symbols for colouring.
    """
    os.makedirs(output_dir, exist_ok=True)

    xyz_path = os.path.join(output_dir, f"{config_name}.xyz")
    write_extended_xyz(xyz_path, structure, counterion_positions, counterion_element)

    script_path = os.path.join(output_dir, f"view_{config_name}.py")
    _write_matplotlib_script(
        script_path,
        xyz_path,
        structure,
        counterion_positions,
        counterion_element,
        config_name,
        tm_elements=tm_elements,
    )
