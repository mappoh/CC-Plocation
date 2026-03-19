#!/usr/bin/env python3
"""
CC-Plocation — Counterion placement for Polyoxometalate (POM) structures.

Generates diverse initial configurations of counterions dispersed around
a POM framework, scored by physics-based rules, and ready for downstream
DFT optimisation.

Usage examples
--------------
  # Auto-detect charge, place K+ with all strategies, 5 samples each:
  cc-plocation --input mono.vasp --counterion K --samples 5

  # Specify ion count manually, single strategy:
  cc-plocation --input tri.vasp --counterion K --n-ions 7 \
               --strategies boltzmann --samples 10

  # Custom TM buffer and seed:
  cc-plocation --input mono.vasp --counterion Na --tm-buffer 4.0 --seed 123
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

from structure_parser import parse_poscar
from charge_analyzer import analyze_charges
from framework_analyzer import ExclusionGrid, FrameworkInfo
from placement_strategies import (
    RandomUniform,
    PoissonDisk,
    ClusteredGaussian,
    ShellBased,
    ElectrostaticGuided,
    BoltzmannMC,
)
from defaults import MAX_FRAMEWORK_DISTANCE, VDW_RADII
from scorer import ConfigurationScorer
from diversity import DiversityAnalyzer
from writer import write_poscar, write_batch
from reporter import Reporter

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
STRATEGY_MAP = {
    "random": RandomUniform,
    "poisson": PoissonDisk,
    "clustered": ClusteredGaussian,
    "shell": ShellBased,
    "electrostatic": ElectrostaticGuided,
    "boltzmann": BoltzmannMC,
}

ALL_STRATEGIES = list(STRATEGY_MAP.keys())

log = logging.getLogger("cc-plocation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cc-plocation",
        description="Place counterions around a POM framework structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to input structure file (VASP POSCAR format).",
    )
    p.add_argument(
        "--counterion", "-c", required=True,
        help="Counterion element symbol (e.g. K, Na, Li, Cs).",
    )
    p.add_argument(
        "--n-ions", type=int, default=None,
        help="Number of counterions to place.  Auto-detected from charge if omitted.",
    )
    p.add_argument(
        "--counterion-charge", type=int, default=1,
        help="Charge of the counterion (default: +1).",
    )
    p.add_argument(
        "--strategies", "-s", nargs="+", default=["all"],
        choices=ALL_STRATEGIES + ["all"],
        help="Placement strategies to run (default: all).",
    )
    p.add_argument(
        "--samples", "-n", type=int, default=5,
        help="Number of samples per strategy (default: 5).",
    )
    p.add_argument(
        "--tm-elements", nargs="+", default=None,
        help="Transition-metal elements to buffer (default: auto-detect).",
    )
    p.add_argument(
        "--tm-buffer", type=float, default=3.5,
        help="Exclusion buffer around TM sites in Angstroms (default: 3.5).",
    )
    p.add_argument(
        "--grid-resolution", type=float, default=0.5,
        help="Exclusion grid voxel size in Angstroms (default: 0.5).",
    )
    p.add_argument(
        "--min-ion-spacing", type=float, default=None,
        help="Minimum distance between counterions (default: 2 * vdW radius).",
    )
    p.add_argument(
        "--max-framework-dist", type=float, default=None,
        help="Maximum distance from nearest framework atom (default: adaptive based on cell geometry, capped at 6.0 A).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--output-dir", "-o", default="./configs",
        help="Output directory for generated structures (default: ./configs).",
    )
    p.add_argument(
        "--no-json", action="store_true",
        help="Skip JSON report output.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    p.add_argument(
        "--oxidation-states", nargs="+", default=None,
        metavar="EL:CHARGE",
        help="Override oxidation states, e.g. Ni:+3 Fe:+2.",
    )
    return p


def parse_oxidation_overrides(overrides):
    """Parse 'EL:CHARGE' strings into a dict."""
    if overrides is None:
        return None
    result = {}
    for item in overrides:
        parts = item.split(":")
        if len(parts) != 2:
            print(f"Warning: ignoring malformed oxidation override '{item}' (expected EL:CHARGE)")
            continue
        el, charge_str = parts
        try:
            result[el] = int(charge_str)
        except ValueError:
            print(f"Warning: ignoring non-integer charge in '{item}'")
    return result if result else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args):
    t0 = time.time()

    # 1. Parse structure
    log.info("Parsing %s ...", args.input)
    structure = parse_poscar(args.input)
    formula = Reporter.get_formula(structure)
    log.info("Structure: %s (%d atoms)", formula, len(structure["positions"]))

    # 2. Charge analysis
    ox_overrides = parse_oxidation_overrides(args.oxidation_states)
    charges = analyze_charges(structure, oxidation_states=ox_overrides)
    net_charge = charges["net_charge"]
    log.info("Net framework charge: %d", net_charge)

    if args.n_ions is not None:
        n_ions = args.n_ions
    else:
        if args.counterion_charge == 0:
            print("Error: counterion charge is 0, cannot auto-detect ion count.")
            sys.exit(1)
        n_ions = int(abs(net_charge) / abs(args.counterion_charge))
        if n_ions == 0:
            print("Framework is charge-neutral — no counterions needed.")
            sys.exit(0)

    print(f"Structure: {formula} ({len(structure['positions'])} atoms)")
    print(f"Net charge: {net_charge}, placing {n_ions} × {args.counterion}{'%+d' % args.counterion_charge}")

    # 3. Compute adaptive max-framework-distance if not set by user
    if args.max_framework_dist is not None:
        max_fw_dist = args.max_framework_dist
    else:
        fw_info = FrameworkInfo(structure, tm_elements=args.tm_elements)
        cell_lengths = np.linalg.norm(fw_info.lattice, axis=1)
        half_min_cell = float(cell_lengths.min()) / 2.0

        # Geometry-based limit: keep ions inside cell
        geometry_limit = half_min_cell - fw_info.max_radius - 1.0

        # vdW-based floor: must exceed the smallest vdW contact distance
        # so there is actually room between vdW exclusion and max distance
        r_counter = VDW_RADII.get(args.counterion, 2.0)
        fw_species = {s for s in structure["species"] if s != args.counterion}
        if not fw_species:
            print("Error: no framework atoms found after excluding counterion species.")
            sys.exit(1)
        min_vdw_sum = min(
            VDW_RADII.get(s, 1.5) + r_counter for s in fw_species
        )
        min_viable = min_vdw_sum + 0.5  # 0.5 A shell beyond closest vdW contact

        if geometry_limit >= min_viable:
            max_fw_dist = geometry_limit
        else:
            max_fw_dist = min_viable
            log.warning(
                "Cell too small for geometry-based limit (%.2f A < vdW floor %.2f A); "
                "ions may extend near cell edges.",
                geometry_limit, min_viable,
            )
        max_fw_dist = min(max_fw_dist, MAX_FRAMEWORK_DISTANCE)  # cap at 6.0 A

        log.info(
            "Adaptive max-framework-distance: %.2f A "
            "(half_min_cell=%.2f, max_radius=%.2f, min_viable=%.2f)",
            max_fw_dist, half_min_cell, fw_info.max_radius, min_viable,
        )
        print(f"Max framework distance (adaptive): {max_fw_dist:.2f} A")

    # 4. Build exclusion grid
    log.info("Building exclusion grid (resolution=%.2f A) ...", args.grid_resolution)
    grid = ExclusionGrid(
        structure,
        args.counterion,
        tm_elements=args.tm_elements,
        tm_buffer=args.tm_buffer,
        grid_resolution=args.grid_resolution,
        max_framework_distance=max_fw_dist,
    )
    grid.build()
    allowed_frac = grid.get_allowed_fraction()
    log.info("Allowed volume fraction: %.1f%%", allowed_frac * 100)

    if allowed_frac < 0.01:
        print("Error: less than 1% of the cell is available for counterion placement.")
        print("Try reducing --tm-buffer or --grid-resolution.")
        sys.exit(1)

    # 5. Determine strategies
    if "all" in args.strategies:
        strategy_names = ALL_STRATEGIES
    else:
        strategy_names = args.strategies

    # 6. Create scorer
    scorer = ConfigurationScorer(
        structure,
        args.counterion,
        counterion_charges={args.counterion: args.counterion_charge},
        framework_charges=charges["species_charges"],
        tm_elements=set(args.tm_elements) if args.tm_elements else None,
        tm_buffer=args.tm_buffer,
        max_framework_distance=max_fw_dist,
    )

    # 7. Create reporter
    reporter = Reporter(structure, args.counterion)

    # 8. Run strategies
    all_valid_configs = []
    all_valid_scores = []
    all_strategy_labels = []
    config_counter = 0  # sequential counter for output filenames

    # Derive prefix from input filename (e.g. "mono.vasp" -> "mono")
    input_prefix = os.path.splitext(os.path.basename(args.input))[0]

    os.makedirs(args.output_dir, exist_ok=True)

    for strat_name in strategy_names:
        strat_class = STRATEGY_MAP[strat_name]
        print(f"\n  Strategy: {strat_name}")

        n_generated = 0
        n_passed = 0

        # Create strategy instance once per strategy (avoids rebuilding
        # expensive internal data structures, e.g. electrostatic potential grid)
        kwargs = {"max_framework_distance": max_fw_dist}
        if args.min_ion_spacing is not None:
            kwargs["min_ion_spacing"] = args.min_ion_spacing

        strategy = strat_class(structure, grid, args.counterion, **kwargs)

        for sample_i in range(args.samples):
            seed = args.seed + sample_i * 1000 + hash(strat_name) % 10000

            positions = strategy.place(n_ions, seed=seed)
            n_generated += 1

            if positions is None:
                log.debug("  %s sample %d: placement failed", strat_name, sample_i)
                reporter.add_result(strat_name, seed, None, None, False)
                continue

            # Score
            result = scorer.score(positions)
            is_valid = result["valid"]
            reporter.add_result(strat_name, seed, positions, result, is_valid)

            if is_valid:
                n_passed += 1
                all_valid_configs.append(positions)
                all_valid_scores.append(result["coulomb_energy_eV"])
                all_strategy_labels.append(f"{strat_name}_{sample_i}")

                # Write POSCAR
                config_counter += 1
                out_path = os.path.join(
                    args.output_dir,
                    f"{input_prefix}_{config_counter:03d}.vasp",
                )
                write_poscar(out_path, structure, positions, args.counterion)


                log.debug(
                    "  %s sample %d: VALID  E=%.2f eV",
                    strat_name, sample_i, result["coulomb_energy_eV"],
                )
            else:
                log.debug(
                    "  %s sample %d: INVALID (rejected)",
                    strat_name, sample_i,
                )

        print(f"    Generated: {n_generated}  Passed: {n_passed}")

    # 9. Diversity analysis
    diversity_summary = None
    if len(all_valid_configs) >= 2:
        da = DiversityAnalyzer(structure["lattice"])
        diversity_summary = da.get_summary(all_valid_configs)
        log.info(
            "Diversity: avg RMSD = %.2f A (min=%.2f, max=%.2f)",
            diversity_summary["mean_rmsd"],
            diversity_summary["min_rmsd"],
            diversity_summary["max_rmsd"],
        )

    # 10. Output reports
    reporter.print_summary_table(
        diversity_rmsd=diversity_summary["mean_rmsd"] if diversity_summary else None,
        output_dir=args.output_dir,
        n_output_files=len(all_valid_configs),
    )

    if not args.no_json:
        json_path = os.path.join(args.output_dir, "report.json")
        reporter.generate_json_report(json_path)
        print(f"\nJSON report: {json_path}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s — {len(all_valid_configs)} valid configurations generated.")

    if len(all_valid_configs) == 0:
        print("\nWarning: No valid configurations generated!")
        print("Try adjusting --tm-buffer, --grid-resolution, or --min-ion-spacing.")
        sys.exit(1)


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s: %(message)s",
    )

    run(args)


if __name__ == "__main__":
    main()
