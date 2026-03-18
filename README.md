# CC-Plocation

Counterion placement tool for Polyoxometalate (POM) structures. Generates diverse initial configurations of counterions dispersed around a POM framework, scored by physics-based rules, ready for downstream DFT optimization.

## Features

- **6 placement strategies**: random uniform, Poisson disk, clustered Gaussian, shell-based, electrostatic-guided, Boltzmann Monte Carlo
- **Automatic charge detection**: determines counterion count from formal oxidation states
- **Physics-based scoring**: vdW overlap rejection, TM proximity buffer, Coulomb energy ranking, charge neutrality enforcement
- **Surface-aware TM buffering**: distinguishes buried vs surface-exposed transition metal sites via coordination analysis
- **Diversity analysis**: pairwise RMSD between configurations using Hungarian algorithm for optimal ion matching
- **Periodic boundary conditions**: all distance calculations use minimum image convention
- **POSCAR output**: ready for VASP optimization with uniform naming for batch job submission

## Requirements

- Python 3.9+
- NumPy
- SciPy

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Auto-detect charge and place counterions with all strategies:
python main.py --input structure.vasp --counterion K

# Specify ion count and number of samples:
python main.py --input structure.vasp --counterion K --n-ions 5 --samples 10

# Use specific strategies only:
python main.py --input structure.vasp --counterion K --strategies boltzmann electrostatic

# Divalent counterion:
python main.py --input structure.vasp --counterion Ca --counterion-charge 2

# Custom TM buffer and output directory:
python main.py --input structure.vasp --counterion Na --tm-buffer 4.0 --output-dir ./my_configs

# Override oxidation states:
python main.py --input structure.vasp --counterion K --oxidation-states Ni:+3

# Verbose logging:
python main.py --input structure.vasp --counterion K -v
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--input`, `-i` | Input structure file (VASP POSCAR format) | required |
| `--counterion`, `-c` | Counterion element symbol (K, Na, Li, Cs, etc.) | required |
| `--n-ions` | Number of counterions to place | auto from charge |
| `--counterion-charge` | Charge of the counterion | +1 |
| `--strategies`, `-s` | Placement strategies to run | all |
| `--samples`, `-n` | Samples per strategy | 5 |
| `--tm-elements` | Transition metal elements to buffer | auto-detect |
| `--tm-buffer` | Exclusion buffer around TM sites (A) | 3.5 |
| `--grid-resolution` | Exclusion grid voxel size (A) | 0.5 |
| `--min-ion-spacing` | Minimum distance between counterions (A) | 2 x vdW radius |
| `--seed` | Random seed for reproducibility | 42 |
| `--output-dir`, `-o` | Output directory | ./configs |
| `--no-json` | Skip JSON report | false |
| `--oxidation-states` | Override oxidation states (e.g. Ni:+3) | built-in defaults |

## Placement Strategies

| Strategy | Description |
|----------|-------------|
| `random` | Uniform random sampling with rejection |
| `poisson` | Bridson's algorithm for blue-noise spacing |
| `clustered` | Gaussian clusters near oxygen-dense regions |
| `shell` | Concentric shells around the POM centroid |
| `electrostatic` | Boltzmann-weighted sampling from Coulomb potential |
| `boltzmann` | Metropolis-Hastings MC refinement (typically finds lowest energy) |

## Output

Output files are named `{input_prefix}_{NNN}.vasp` (e.g. `mono_001.vasp`, `mono_002.vasp`) for easy batch job submission. A `report.json` file contains per-configuration scores, distances, and metadata.

## Scoring Rules

Configurations are rejected if any of these checks fail:

- Ion overlaps with framework atom (vdW radii sum)
- Ion-ion overlap (vdW radii sum)
- Ion too close to buried TM site (configurable buffer)
- Charge neutrality violation

Valid configurations are ranked by Coulomb energy (lower is better).
