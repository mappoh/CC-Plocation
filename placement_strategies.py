"""
Placement strategies for counterion positioning around POM frameworks.

Implements six strategies using the Template Method pattern:
  1. RandomUniform       - Uniform random sampling in the unit cell
  2. PoissonDisk         - Bridson's algorithm for blue-noise sampling
  3. ClusteredGaussian   - Gaussian clusters near oxygen-dense regions
  4. ShellBased          - Concentric shells around POM centroid
  5. ElectrostaticGuided - Boltzmann-weighted sampling from Coulomb potential
  6. BoltzmannMC         - Metropolis-Hastings Monte Carlo refinement

All strategies respect periodic boundary conditions, honour the exclusion grid,
and enforce a minimum spacing between placed counterions.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import cKDTree

from pbc_utils import (
    fractional_to_cartesian,
    cartesian_to_fractional,
    minimum_image_distance,
    wrap_fractional,
)
from defaults import (
    OXIDATION_STATES as FORMAL_CHARGES,
    K_COULOMB as KE,
    KB_EV,
    TM_ELEMENTS,
    VDW_RADII,
    get_counterion_radius,
)


# ===================================================================
#  Base class
# ===================================================================
class PlacementStrategy(ABC):
    """Abstract base for all counterion-placement strategies.

    Uses the **Template Method** pattern: the public :meth:`place` method
    contains the shared rejection-sampling loop, while subclasses override
    :meth:`_propose_position` (and optionally :meth:`place` itself) to
    implement strategy-specific logic.

    Parameters
    ----------
    structure : dict
        Parsed structure dictionary with keys:
        - ``'lattice'``     : 3x3 numpy array (rows = lattice vectors, Angstroms)
        - ``'positions'``   : Nx3 numpy array of Cartesian coordinates
        - ``'atom_labels'`` : list[str] of element symbols (length N)
    exclusion_grid : ExclusionGrid
        Pre-built exclusion grid whose :meth:`is_allowed` accepts a
        Cartesian position and returns *True* when the site is valid.
    counterion_element : str
        Element symbol of the counterion to place (e.g. ``'K'``).
    min_ion_spacing : float
        Minimum allowed distance (Angstroms) between any two placed
        counterions.  Default 2.0.
    max_attempts : int
        Maximum rejection-sampling attempts per ion.  Default 10 000.
    """

    def __init__(
        self,
        structure,
        exclusion_grid,
        counterion_element,
        min_ion_spacing=None,
        max_attempts=10000,
        max_framework_distance=None,
    ):
        self.structure = structure
        self.lattice = np.array(structure["lattice"], dtype=np.float64)
        self.positions = np.array(structure["positions"], dtype=np.float64)
        self.atom_labels = list(structure["atom_labels"])
        self.exclusion_grid = exclusion_grid
        self.counterion_element = counterion_element
        self._ion_r: float = get_counterion_radius(counterion_element)
        # Default min_ion_spacing = 2 * ionic radius of counterion
        if min_ion_spacing is None:
            self.min_ion_spacing = 2.0 * self._ion_r
        else:
            self.min_ion_spacing = float(min_ion_spacing)
        self.max_attempts = int(max_attempts)
        if max_framework_distance is None:
            raise ValueError(
                "max_framework_distance must be provided."
            )
        self.max_framework_distance = float(max_framework_distance)

    # ---- helpers used by multiple strategies ----------------------------

    def _pbc_distance(self, pos_a, pos_b):
        """PBC-aware distance between two Cartesian positions."""
        return minimum_image_distance(pos_a, pos_b, self.lattice)

    def _check_min_spacing(self, candidate, placed):
        """Return True if *candidate* is at least min_ion_spacing from
        every position in *placed* (Nx3 array), respecting PBC."""
        for p in placed:
            if self._pbc_distance(candidate, p) < self.min_ion_spacing:
                return False
        return True

    def _check_framework_distance(self, candidate):
        """Return True if *candidate* passes overlap check against all
        framework atoms (supplements the grid check).  Uses ionic radius
        for the counterion since it is a charged species."""
        for i, pos in enumerate(self.positions):
            fw_r = VDW_RADII.get(self.atom_labels[i], 2.0)
            d = self._pbc_distance(candidate, pos)
            if d < (fw_r + self._ion_r):
                return False
        return True

    def _random_fractional_point(self, rng):
        """Return a uniformly random fractional coordinate in [0, 1)^3."""
        return rng.random(3)

    def _frac_to_cart(self, frac):
        """Fractional -> Cartesian using the lattice."""
        return fractional_to_cartesian(frac, self.lattice)

    def _cart_to_frac(self, cart):
        """Cartesian -> Fractional using the lattice."""
        return cartesian_to_fractional(cart, self.lattice)

    def _wrap_cart(self, cart):
        """Wrap a Cartesian position back into the unit cell."""
        frac = self._cart_to_frac(cart)
        frac = wrap_fractional(frac)
        return self._frac_to_cart(frac)

    # ---- template method ------------------------------------------------

    def place(self, n_ions, seed=None):
        """Place *n_ions* counterions using rejection sampling.

        Parameters
        ----------
        n_ions : int
            Number of counterions to place.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        numpy.ndarray or None
            (n_ions, 3) array of Cartesian positions, or *None* if the
            requested number could not be placed within *max_attempts*
            per ion.
        """
        rng = np.random.default_rng(seed)
        placed = []

        for _ in range(n_ions):
            success = False
            for _attempt in range(self.max_attempts):
                candidate = self._propose_position(rng, placed)
                candidate = self._wrap_cart(candidate)

                # 1. Exclusion-grid check (includes max-framework-distance)
                if not self.exclusion_grid.is_allowed(candidate):
                    continue

                # 2. Exact framework overlap check (radius contact)
                if not self._check_framework_distance(candidate):
                    continue

                # 3. Min spacing to already-placed ions
                if not self._check_min_spacing(candidate, placed):
                    continue

                placed.append(candidate)
                success = True
                break

            if not success:
                return None  # could not place this ion

        return np.array(placed, dtype=np.float64)

    @abstractmethod
    def _propose_position(self, rng, placed_positions):
        """Propose a single candidate Cartesian position.

        Parameters
        ----------
        rng : numpy.random.Generator
            PRNG instance (do **not** create a new one).
        placed_positions : list[numpy.ndarray]
            Positions already accepted (each shape ``(3,)``).

        Returns
        -------
        numpy.ndarray
            Shape ``(3,)`` candidate in Cartesian coordinates.
        """


# ===================================================================
#  1. RandomUniform
# ===================================================================
class RandomUniform(PlacementStrategy):
    """Place counterions at uniformly random positions in the unit cell.

    The simplest baseline strategy.  Each candidate is drawn from a
    uniform distribution over fractional coordinates [0, 1)^3 and then
    converted to Cartesian space.  Candidates that fall inside the
    exclusion zone or too close to an already-placed ion are rejected and
    a new candidate is drawn.
    """

    def _propose_position(self, rng, placed_positions):
        frac = self._random_fractional_point(rng)
        return self._frac_to_cart(frac)


# ===================================================================
#  2. PoissonDisk
# ===================================================================
class PoissonDisk(PlacementStrategy):
    """Blue-noise placement via Bridson's algorithm for 3D periodic cells.

    Guarantees a minimum spacing (``min_ion_spacing``) between all placed
    ions by construction, rather than relying on rejection alone.

    Algorithm
    ---------
    1. Seed with one random allowed point.
    2. Maintain an *active list* of points that can still spawn neighbours.
    3. For each active point, generate up to *k* candidates uniformly in
       the spherical annulus [min_spacing, 2 * min_spacing].
    4. Accept the first candidate that (a) is far enough from all existing
       points, (b) passes the exclusion-grid check.
    5. If none of the *k* candidates work, deactivate the point.
    6. Stop when *n_ions* have been placed or the active list is empty.

    Parameters
    ----------
    k : int
        Number of candidate samples per active point (default 30).
    """

    def __init__(self, *args, k=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)

    # Not used by the overridden place(), but required by the ABC.
    def _propose_position(self, rng, placed_positions):
        frac = self._random_fractional_point(rng)
        return self._frac_to_cart(frac)

    def _random_in_annulus(self, rng, centre, r_min, r_max):
        """Return a random point in the spherical annulus [r_min, r_max]
        around *centre*, wrapped into the unit cell."""
        # Uniform direction on the sphere
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        # Uniform radius in [r_min, r_max] with correct r^2 weighting
        r = (rng.uniform(r_min**3, r_max**3)) ** (1.0 / 3.0)
        candidate = centre + direction * r
        return self._wrap_cart(candidate)

    def place(self, n_ions, seed=None):
        """Run Bridson's algorithm to place *n_ions* counterions.

        Returns
        -------
        numpy.ndarray or None
            (n_ions, 3) array of Cartesian positions, or *None* on failure.
        """
        rng = np.random.default_rng(seed)
        r_min = self.min_ion_spacing
        r_max = 2.0 * r_min

        # --- seed point ---
        placed = []
        for _ in range(self.max_attempts):
            frac = self._random_fractional_point(rng)
            candidate = self._frac_to_cart(frac)
            candidate = self._wrap_cart(candidate)
            if (self.exclusion_grid.is_allowed(candidate)
                    and self._check_framework_distance(candidate)):
                placed.append(candidate)
                break
        else:
            return None

        if n_ions == 1:
            return np.array(placed, dtype=np.float64)

        active = [0]  # indices into placed

        while len(placed) < n_ions and active:
            # Pick a random active point
            idx = rng.integers(len(active))
            parent_idx = active[idx]
            parent = placed[parent_idx]

            found = False
            for _ in range(self.k):
                candidate = self._random_in_annulus(rng, parent, r_min, r_max)

                # Exclusion grid + distance checks
                if not self.exclusion_grid.is_allowed(candidate):
                    continue
                if not self._check_framework_distance(candidate):
                    continue

                # Min spacing to ALL placed points
                if not self._check_min_spacing(candidate, placed):
                    continue

                placed.append(candidate)
                active.append(len(placed) - 1)
                found = True
                break

            if not found:
                active.pop(idx)

        if len(placed) < n_ions:
            return None

        return np.array(placed[:n_ions], dtype=np.float64)


# ===================================================================
#  3. ClusteredGaussian
# ===================================================================
class ClusteredGaussian(PlacementStrategy):
    """Place counterions near Gaussian clusters in oxygen-dense regions.

    Cluster centres are identified as regions with high oxygen density
    that are far from transition-metal sites, approximating zones of
    high negative electrostatic potential.

    For each ion the algorithm:
    1. Picks a cluster centre with probability proportional to the
       number of nearby oxygen atoms.
    2. Adds a Gaussian-distributed offset (configurable *sigma*).

    Parameters
    ----------
    n_clusters : int or None
        Number of cluster centres.  If *None*, defaults to ``n_ions``
        (set during :meth:`place`).
    sigma : float
        Standard deviation (Angstroms) of the Gaussian perturbation
        applied to the chosen cluster centre.  Default 2.0.
    """

    def __init__(self, *args, n_clusters=None, sigma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.sigma = float(sigma)
        self._cluster_centres = None
        self._cluster_weights = None

    # -- cluster identification -------------------------------------------

    def _identify_clusters(self, n_clusters):
        """Find *n_clusters* centres near oxygen-dense, TM-free regions.

        Strategy:
        - Collect oxygen positions.
        - Exclude positions too close to transition-metal atoms.
        - Use k-means (Lloyd's algorithm, simplified) on oxygen positions
          to find cluster centres.
        - Weights = number of oxygen atoms assigned to each cluster.
        """
        # Transition metals (from defaults)
        tm_symbols = set(TM_ELEMENTS)
        oxy_pos = np.array(
            [p for p, l in zip(self.positions, self.atom_labels) if l == "O"],
            dtype=np.float64,
        )
        tm_pos = np.array(
            [p for p, l in zip(self.positions, self.atom_labels) if l in tm_symbols],
            dtype=np.float64,
        )

        if len(oxy_pos) == 0:
            # Fallback: use all framework positions
            oxy_pos = self.positions.copy()

        # Simple k-means on oxygen positions (no PBC, good enough for
        # cluster seeding inside one cell).
        k = min(n_clusters, len(oxy_pos))
        rng_km = np.random.default_rng(42)
        centres = oxy_pos[rng_km.choice(len(oxy_pos), size=k, replace=False)]

        for _ in range(50):  # iterations
            # Assign each oxygen to nearest centre
            dists = np.linalg.norm(
                oxy_pos[:, None, :] - centres[None, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)
            new_centres = np.empty_like(centres)
            for c in range(k):
                members = oxy_pos[labels == c]
                if len(members) > 0:
                    new_centres[c] = members.mean(axis=0)
                else:
                    new_centres[c] = centres[c]
            if np.allclose(centres, new_centres, atol=1e-6):
                break
            centres = new_centres

        # Shift centres away from TM atoms if any are too close
        if len(tm_pos) > 0:
            for i in range(k):
                for tm in tm_pos:
                    vec = centres[i] - tm
                    d = np.linalg.norm(vec)
                    # Push centre outward if within 2 Angstroms of a TM
                    if d < 2.0 and d > 1e-8:
                        centres[i] = tm + vec / d * 2.5

        # Weights proportional to oxygen count per cluster
        weights = np.zeros(k, dtype=np.float64)
        dists = np.linalg.norm(
            oxy_pos[:, None, :] - centres[None, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)
        for c in range(k):
            weights[c] = np.sum(labels == c)
        weights = np.maximum(weights, 1.0)  # avoid zero weight
        weights /= weights.sum()

        return centres, weights

    def place(self, n_ions, seed=None):
        """Place ions near Gaussian-blurred cluster centres."""
        nc = self.n_clusters if self.n_clusters is not None else n_ions
        nc = max(nc, 1)
        self._cluster_centres, self._cluster_weights = self._identify_clusters(nc)
        return super().place(n_ions, seed=seed)

    def _propose_position(self, rng, placed_positions):
        """Pick a weighted random cluster centre and add Gaussian noise."""
        idx = rng.choice(len(self._cluster_centres), p=self._cluster_weights)
        centre = self._cluster_centres[idx]
        offset = rng.normal(scale=self.sigma, size=3)
        return centre + offset


# ===================================================================
#  4. ShellBased
# ===================================================================
class ShellBased(PlacementStrategy):
    """Place counterions in concentric shells around the POM centroid.

    Shell radii start at ``max_framework_radius + buffer`` and increase
    outward in fixed increments.  Ions are distributed across shells as
    evenly as possible.

    Parameters
    ----------
    buffer : float
        Gap (Angstroms) between the outermost framework atom and the
        first shell.  Default 2.0.
    shell_spacing : float
        Radial distance (Angstroms) between successive shells.
        Default 1.5.
    """

    def __init__(self, *args, buffer=2.0, shell_spacing=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = float(buffer)
        self.shell_spacing = float(shell_spacing)

        # Compute centroid and maximum framework radius
        self._centroid = self.positions.mean(axis=0)
        radii = np.linalg.norm(self.positions - self._centroid, axis=1)
        self._max_radius = float(radii.max())

        # Shell assignments (filled during place())
        self._shell_radii = None
        self._shell_for_ion = None
        self._current_ion_index = 0

    def _propose_position(self, rng, placed_positions):
        """Pick the shell for the current ion and sample a random point
        on that shell's surface (centroid-centred sphere)."""
        shell_idx = self._shell_for_ion[
            min(self._current_ion_index, len(self._shell_for_ion) - 1)
        ]
        radius = self._shell_radii[shell_idx]

        # Random direction (uniform on sphere)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)

        candidate = self._centroid + direction * radius

        # Advance the ion counter only when called for a fresh ion
        # (the base class increments after acceptance; we track proposals)
        # We advance after each *ion* is placed; _propose may be called
        # many times for the same ion.  The base class loop handles this
        # implicitly: _current_ion_index is incremented in the overridden
        # place wrapper below.
        return candidate

    def place(self, n_ions, seed=None):
        """Wrapper that tracks which ion is being placed."""
        # Re-run shell setup (already done above, but safe for direct call)
        ions_per_shell_target = max(1, min(n_ions, 6))
        n_shells = max(1, int(np.ceil(n_ions / ions_per_shell_target)))
        self._shell_radii = np.array(
            [
                self._max_radius + self.buffer + i * self.shell_spacing
                for i in range(n_shells)
            ]
        )
        base = n_ions // n_shells
        extra = n_ions % n_shells
        self._shell_for_ion = []
        for s in range(n_shells):
            count = base + (1 if s < extra else 0)
            self._shell_for_ion.extend([s] * count)

        # Use custom rejection loop so we can track ion index
        rng = np.random.default_rng(seed)
        placed = []
        for ion_i in range(n_ions):
            self._current_ion_index = ion_i
            success = False
            for _ in range(self.max_attempts):
                candidate = self._propose_position(rng, placed)
                candidate = self._wrap_cart(candidate)
                if not self.exclusion_grid.is_allowed(candidate):
                    continue
                if not self._check_framework_distance(candidate):
                    continue
                if not self._check_min_spacing(candidate, placed):
                    continue
                placed.append(candidate)
                success = True
                break
            if not success:
                return None

        return np.array(placed, dtype=np.float64)


# ===================================================================
#  5. ElectrostaticGuided
# ===================================================================
class ElectrostaticGuided(PlacementStrategy):
    """Sample positions proportional to the Boltzmann weight of the
    Coulomb potential from framework formal charges.

    A coarse 3-D grid of the electrostatic potential is computed once at
    construction.  Candidate positions are drawn from the grid points
    with probability proportional to ``exp(-V / kT)`` (counterions are
    attracted to negative-potential regions).

    Parameters
    ----------
    grid_spacing : float
        Spacing (Angstroms) of the potential grid.  Default 1.0.
    temperature : float
        Temperature (K) for the Boltzmann weight.  Default 300.
    """

    def __init__(self, *args, grid_spacing=1.0, temperature=300.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_spacing = float(grid_spacing)
        self.temperature = float(temperature)

        # Build the potential grid
        self._grid_points, self._grid_probs = self._build_potential_grid()

    def _get_charge(self, element):
        """Return the formal charge for *element*."""
        return FORMAL_CHARGES.get(element, 0.0)

    def _build_potential_grid(self):
        """Compute Coulomb potential on a regular grid and derive
        sampling probabilities.

        Returns
        -------
        grid_points : ndarray, shape (M, 3)
            Cartesian coordinates of grid points.
        probs : ndarray, shape (M,)
            Normalised Boltzmann-weighted probabilities.
        """
        # Lattice lengths along each axis (orthorhombic approximation for
        # grid construction; exact for orthorhombic cells, reasonable
        # approximation otherwise).
        a_len = np.linalg.norm(self.lattice[0])
        b_len = np.linalg.norm(self.lattice[1])
        c_len = np.linalg.norm(self.lattice[2])

        na = max(2, int(np.round(a_len / self.grid_spacing)))
        nb = max(2, int(np.round(b_len / self.grid_spacing)))
        nc = max(2, int(np.round(c_len / self.grid_spacing)))

        # Fractional grid
        fa = np.linspace(0, 1, na, endpoint=False)
        fb = np.linspace(0, 1, nb, endpoint=False)
        fc = np.linspace(0, 1, nc, endpoint=False)
        frac_grid = np.array(
            np.meshgrid(fa, fb, fc, indexing="ij")
        ).reshape(3, -1).T  # (M, 3)

        # Cartesian grid
        cart_grid = frac_grid @ self.lattice  # (M, 3)

        # Framework charges
        charges = np.array(
            [self._get_charge(l) for l in self.atom_labels], dtype=np.float64
        )

        # Coulomb potential at each grid point (vectorised over framework
        # atoms, looped over grid chunks to limit memory).
        n_grid = len(cart_grid)
        potential = np.zeros(n_grid, dtype=np.float64)
        chunk = 5000  # grid points per chunk
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            pts = cart_grid[start:end]  # (C, 3)
            # Displacement vectors: (C, N_atoms, 3)
            diff = pts[:, None, :] - self.positions[None, :, :]
            # Minimum image (approximate for non-orthorhombic):
            frac_diff = diff @ np.linalg.inv(self.lattice)
            frac_diff -= np.round(frac_diff)
            cart_diff = frac_diff @ self.lattice
            dists = np.linalg.norm(cart_diff, axis=2)  # (C, N_atoms)
            dists = np.maximum(dists, 0.1)  # avoid singularity
            # V = sum_j  k_e * q_j / r_j
            potential[start:end] = np.sum(KE * charges[None, :] / dists, axis=1)

        # Boltzmann weights for a *positive* counterion being attracted to
        # negative potential:  P ~ exp(- q_ion * V / kT)
        q_ion = self._get_charge(self.counterion_element)
        if q_ion == 0.0:
            q_ion = 1.0  # default to +1 if unknown
        beta = 1.0 / (KB_EV * self.temperature)
        # Energy of ion at grid point:  E = q_ion * V
        # Shift exponent to avoid overflow: subtract max before exp
        exponent = -q_ion * potential * beta
        exponent -= np.max(exponent)  # numerical stability
        weights = np.exp(exponent)
        # Zero out excluded points
        for i, pt in enumerate(cart_grid):
            if not self.exclusion_grid.is_allowed(pt):
                weights[i] = 0.0

        total = weights.sum()
        if total < 1e-300:
            # Fallback: uniform over allowed points
            allowed_mask = np.array(
                [self.exclusion_grid.is_allowed(pt) for pt in cart_grid]
            )
            weights = allowed_mask.astype(np.float64)
            total = weights.sum()
            if total < 1e-300:
                # Nothing allowed – return uniform (will fail in place())
                weights = np.ones(n_grid, dtype=np.float64)
                total = weights.sum()
        probs = weights / total
        return cart_grid, probs

    def _propose_position(self, rng, placed_positions):
        """Draw a grid point weighted by the Boltzmann potential."""
        idx = rng.choice(len(self._grid_points), p=self._grid_probs)
        # Add a small random jitter (up to half grid_spacing) so
        # placements are not snapped to grid nodes.
        jitter = rng.uniform(-0.5 * self.grid_spacing, 0.5 * self.grid_spacing, size=3)
        return self._grid_points[idx] + jitter


# ===================================================================
#  6. BoltzmannMC (Metropolis-Hastings)
# ===================================================================
class BoltzmannMC(PlacementStrategy):
    """Refine counterion positions via Metropolis-Hastings Monte Carlo.

    Workflow
    --------
    1. Generate an initial configuration using :class:`RandomUniform`.
    2. Run *n_mc_steps* Metropolis-Hastings sweeps:
       a. Pick a random ion.
       b. Propose a Gaussian displacement.
       c. Compute the energy change ``Delta E`` (pairwise Coulomb with
          minimum-image convention).
       d. Accept with probability ``min(1, exp(-Delta E / kT))``.
    3. Return the final configuration.

    The total Coulomb energy includes **ion-framework** and **ion-ion**
    interactions:

    .. math::
        E = k_e \\sum_{i<j} \\frac{q_i q_j}{r_{ij}}

    Parameters
    ----------
    n_mc_steps : int
        Number of Monte Carlo trial moves.  Default 10 000.
    temperature : float
        Temperature (K).  Default 300.
    mc_step_sigma : float
        Standard deviation (Angstroms) of the Gaussian displacement
        proposal.  Default 0.5.
    """

    def __init__(self, *args, n_mc_steps=10000, temperature=300.0,
                 mc_step_sigma=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_mc_steps = int(n_mc_steps)
        self.temperature = float(temperature)
        self.mc_step_sigma = float(mc_step_sigma)

        # Precompute framework charges
        self._fw_charges = np.array(
            [FORMAL_CHARGES.get(l, 0.0) for l in self.atom_labels],
            dtype=np.float64,
        )
        self._ion_charge = FORMAL_CHARGES.get(self.counterion_element, 1.0)

    # Not used by overridden place(), required by ABC
    def _propose_position(self, rng, placed_positions):
        frac = self._random_fractional_point(rng)
        return self._frac_to_cart(frac)

    # ---- energy calculations -------------------------------------------

    def _ion_framework_energy(self, ion_pos):
        """Coulomb energy between one ion and all framework atoms (eV)."""
        diff = ion_pos - self.positions  # (N, 3)
        # Minimum image
        frac_diff = diff @ np.linalg.inv(self.lattice)
        frac_diff -= np.round(frac_diff)
        cart_diff = frac_diff @ self.lattice
        dists = np.linalg.norm(cart_diff, axis=1)
        dists = np.maximum(dists, 0.1)
        return float(np.sum(KE * self._ion_charge * self._fw_charges / dists))

    def _ion_ion_energy(self, ion_a, ion_b):
        """Coulomb energy between two counterions (eV)."""
        d = self._pbc_distance(ion_a, ion_b)
        d = max(d, 0.1)
        return KE * self._ion_charge * self._ion_charge / d

    def _total_energy(self, ions):
        """Total Coulomb energy of the ion configuration (eV).

        Includes ion-framework and ion-ion terms.
        """
        n = len(ions)
        e_total = 0.0
        # Ion-framework
        for ion in ions:
            e_total += self._ion_framework_energy(ion)
        # Ion-ion
        for i in range(n):
            for j in range(i + 1, n):
                e_total += self._ion_ion_energy(ions[i], ions[j])
        return e_total

    def _single_ion_energy(self, ion_pos, ions, idx):
        """Energy contribution of ion *idx* with all other ions and the
        framework.  Changing one ion only requires recomputing its
        interactions."""
        e = self._ion_framework_energy(ion_pos)
        for j, other in enumerate(ions):
            if j == idx:
                continue
            e += self._ion_ion_energy(ion_pos, other)
        return e

    # ---- main MC loop ---------------------------------------------------

    def place(self, n_ions, seed=None):
        """Place ions via RandomUniform, then refine with Metropolis MC.

        Returns
        -------
        numpy.ndarray or None
            (n_ions, 3) Cartesian positions, or *None* if initial
            placement fails.
        """
        rng = np.random.default_rng(seed)

        # --- initial placement using RandomUniform -----------------------
        init_strategy = RandomUniform(
            self.structure,
            self.exclusion_grid,
            self.counterion_element,
            min_ion_spacing=self.min_ion_spacing,
            max_attempts=self.max_attempts,
            max_framework_distance=self.max_framework_distance,
        )
        initial = init_strategy.place(n_ions, seed=seed)
        if initial is None:
            return None

        ions = [initial[i].copy() for i in range(n_ions)]

        beta = 1.0 / (KB_EV * self.temperature)

        # Pre-compute current energies per ion for efficiency
        energies = np.array(
            [self._single_ion_energy(ions[i], ions, i) for i in range(n_ions)],
            dtype=np.float64,
        )

        # --- Metropolis-Hastings loop ------------------------------------
        for _ in range(self.n_mc_steps):
            # Pick a random ion
            idx = rng.integers(n_ions)
            old_pos = ions[idx].copy()
            old_e = energies[idx]

            # Propose displacement
            displacement = rng.normal(scale=self.mc_step_sigma, size=3)
            new_pos = self._wrap_cart(old_pos + displacement)

            # Check exclusion grid (includes max-framework-distance) + overlap
            if not self.exclusion_grid.is_allowed(new_pos):
                continue
            if not self._check_framework_distance(new_pos):
                continue

            # Check min spacing with other ions
            too_close = False
            for j, other in enumerate(ions):
                if j == idx:
                    continue
                if self._pbc_distance(new_pos, other) < self.min_ion_spacing:
                    too_close = True
                    break
            if too_close:
                continue

            # Energy of ion at new position
            ions[idx] = new_pos
            new_e = self._single_ion_energy(new_pos, ions, idx)
            delta_e = new_e - old_e

            # Metropolis criterion
            if delta_e <= 0.0 or rng.random() < np.exp(-delta_e * beta):
                # Accept
                energies[idx] = new_e
            else:
                # Reject – revert
                ions[idx] = old_pos

        return np.array(ions, dtype=np.float64)
