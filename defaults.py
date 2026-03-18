"""
defaults.py - Default parameters for CC-Plocation counterion placement tool.

Contains van der Waals radii, ionic radii, oxidation states, transition metal
definitions, scoring thresholds, and VESTA color mappings.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Van der Waals radii (Angstroms)
# Sources: Bondi (1964), Mantina et al. (2009), CRC Handbook
# ---------------------------------------------------------------------------
VDW_RADII: Dict[str, float] = {
    "W":  2.10,
    "Mo": 2.09,
    "V":  2.07,
    "Nb": 2.18,
    "Ta": 2.22,
    "Ni": 1.63,
    "Co": 2.00,
    "Fe": 2.04,
    "Mn": 2.05,
    "Cu": 1.40,
    "Zn": 1.39,
    "P":  1.80,
    "Si": 2.10,
    "Ge": 2.11,
    "As": 1.85,
    "O":  1.52,
    "K":  2.75,
    "Na": 2.27,
    "Li": 1.82,
    "Cs": 3.43,
    "Rb": 3.03,
    "Ca": 2.31,
    "Mg": 1.73,
    "Ba": 2.68,
    "Sr": 2.49,
    "N":  1.55,
    "S":  1.80,
    "H":  1.20,
    "C":  1.70,
}

# ---------------------------------------------------------------------------
# Shannon ionic radii for common counterions (Angstroms)
# Coordination number VI unless noted
# ---------------------------------------------------------------------------
IONIC_RADII: Dict[str, float] = {
    "K":  1.38,
    "Na": 1.02,
    "Li": 0.76,
    "Cs": 1.67,
    "Rb": 1.52,
    "Ca": 1.00,
    "Mg": 0.72,
    "Ba": 1.35,
    "Sr": 1.18,
    "NH4": 1.48,  # ammonium, treated as pseudo-atom
}

# ---------------------------------------------------------------------------
# Default formal oxidation states for POM framework elements
# ---------------------------------------------------------------------------
OXIDATION_STATES: Dict[str, int] = {
    "W":  +6,
    "Mo": +6,
    "V":  +5,
    "Nb": +5,
    "Ta": +5,
    "Ni": +2,
    "Co": +2,
    "Fe": +3,
    "Mn": +2,
    "Cu": +2,
    "Zn": +2,
    "P":  +5,
    "Si": +4,
    "Ge": +4,
    "As": +5,
    "O":  -2,
}

# ---------------------------------------------------------------------------
# Transition-metal elements commonly found as heteroatoms in POMs
# ---------------------------------------------------------------------------
TM_ELEMENTS: List[str] = [
    "Ni", "Co", "Fe", "Mn", "Cu", "Zn",
    "Ti", "Cr", "Sc",
]

# ---------------------------------------------------------------------------
# Buffer distance (Angstroms) around TM heteroatoms.
# Counterions should not be placed within this distance of a TM center.
# ---------------------------------------------------------------------------
TM_BUFFER_DISTANCE: float = 3.5

# ---------------------------------------------------------------------------
# Scoring thresholds used during candidate-site evaluation
# ---------------------------------------------------------------------------
SCORING: Dict[str, float] = {
    "min_o_distance": 2.5,       # minimum acceptable distance to nearest O (A)
    "max_o_distance": 4.5,       # maximum distance still considered "near" O (A)
    "ideal_o_distance": 2.8,     # ideal counterion-O distance (A)
    "clash_distance": 1.8,       # hard clash cutoff (A) to any framework atom
    "min_counterion_sep": 3.0,   # minimum separation between counterions (A)
    "o_coordination_weight": 1.0,  # weight for oxygen coordination score
    "symmetry_weight": 0.5,      # weight for symmetry-equivalence score
    "electrostatic_weight": 0.8, # weight for simple electrostatic score
}

# ---------------------------------------------------------------------------
# VESTA RGB colour mappings for visualisation (0-255 per channel)
# ---------------------------------------------------------------------------
VESTA_COLORS: Dict[str, Tuple[int, int, int]] = {
    "W":  (38,  89, 140),
    "Mo": (84, 181, 226),
    "V":  (166, 166, 171),
    "Nb": (115, 194, 201),
    "Ta": (77, 166, 255),
    "Ni": (80, 208,  80),
    "Co": (0,   0,  175),
    "Fe": (178, 102,   0),
    "Mn": (156,  75, 211),
    "Cu": (144, 115,  38),
    "Zn": (125, 128, 176),
    "P":  (255, 128,   0),
    "Si": (27,  59, 250),
    "Ge": (102, 143, 143),
    "As": (189, 128, 227),
    "O":  (255,  25,  25),
    "K":  (143,  64, 212),
    "Na": (249, 220,  60),
    "Li": (134, 224,  51),
    "Cs": (87, 253, 222),
    "Rb": (112,  46, 176),
    "Ca": (90, 150,  50),
    "Mg": (0,  150,  25),
    "Ba": (0, 201,   0),
    "Sr": (0, 255,   0),
    "N":  (48,  80, 248),
    "S":  (255, 200,  50),
    "H":  (255, 255, 255),
    "C":  (127, 127, 127),
}
