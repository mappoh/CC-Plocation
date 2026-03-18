"""
reporter.py - Generate JSON reports and formatted stdout summary tables
for CC-Plocation counterion placement results.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np


class Reporter:
    """Collect, summarise, and export counterion placement results.

    Parameters
    ----------
    structure : dict
        Structure dictionary with keys 'lattice', 'positions', 'atom_labels'.
    counterion_element : str
        Element symbol for the counterion (e.g. "K").
    """

    def __init__(self, structure: Dict[str, Any], counterion_element: str) -> None:
        self.structure = structure
        self.counterion_element = counterion_element
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_formula(structure: Dict[str, Any]) -> str:
        """Return a compact chemical formula string from the structure.

        Elements are ordered alphabetically with counts appended, e.g.
        ``"Ni1O39P1W11"``.

        Parameters
        ----------
        structure : dict
            Must contain 'atom_labels' (list of element symbols).

        Returns
        -------
        str
            Chemical formula string.
        """
        counts: Dict[str, int] = {}
        for sp in structure["atom_labels"]:
            counts[sp] = counts.get(sp, 0) + 1

        # Order: metals first (by convention, heaviest first), then non-metals
        # A simple approach: sort alphabetically
        parts = []
        for elem in sorted(counts.keys()):
            parts.append(f"{elem}{counts[elem]}")
        return "".join(parts)

    def _strategy_summary(self) -> Dict[str, Dict[str, Any]]:
        """Group results by strategy and compute per-strategy statistics."""
        strategies: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._results:
            name = r["strategy"]
            strategies.setdefault(name, []).append(r)

        summary: Dict[str, Dict[str, Any]] = {}
        for name, entries in strategies.items():
            n_generated = len(entries)
            n_passed = sum(1 for e in entries if e["valid"])
            energies = [
                e["score_result"]["coulomb_energy_eV"]
                for e in entries
                if e["valid"]
            ]
            best_energy = min(energies) if energies else None
            summary[name] = {
                "generated": n_generated,
                "passed": n_passed,
                "best_energy_eV": best_energy,
                "entries": entries,
            }
        return summary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_result(
        self,
        strategy_name: str,
        seed: int,
        counterion_positions: np.ndarray,
        score_result: Dict[str, Any],
        valid: bool,
    ) -> None:
        """Store one configuration's results.

        Parameters
        ----------
        strategy_name : str
            Name of the placement strategy (e.g. "random", "electrostatic").
        seed : int
            Random seed used to generate this configuration.
        counterion_positions : np.ndarray
            (N, 3) Cartesian positions of counterions.
        score_result : dict
            Output of :meth:`ConfigurationScorer.score`.
        valid : bool
            Whether the configuration passed all hard checks.
        """
        # Make positions serialisable
        positions = np.array(counterion_positions, dtype=np.float64).tolist()

        # Make score_result JSON-serialisable (convert numpy types)
        clean_score = _make_serialisable(score_result)

        self._results.append(
            {
                "strategy": strategy_name,
                "seed": seed,
                "counterion_positions": positions,
                "score_result": clean_score,
                "valid": valid,
            }
        )

    def generate_json_report(self, filepath: str) -> None:
        """Write a comprehensive JSON report to *filepath*.

        The report contains:
        - metadata: input structure formula, counterion element, count, timestamp
        - per_strategy: list of configs with scores, distances, pass/fail
        - summary: total generated, total passed, best energy per strategy

        Parameters
        ----------
        filepath : str
            Path for the output JSON file.
        """
        formula = self.get_formula(self.structure)
        n_ions = (
            len(self._results[0]["counterion_positions"])
            if self._results
            else 0
        )
        strat_summary = self._strategy_summary()

        report: Dict[str, Any] = {
            "metadata": {
                "formula": formula,
                "counterion_element": self.counterion_element,
                "counterion_count": n_ions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_generated": len(self._results),
                "total_passed": sum(1 for r in self._results if r["valid"]),
            },
            "per_strategy": {},
            "summary": {},
        }

        for name, info in strat_summary.items():
            configs = []
            for entry in info["entries"]:
                configs.append(
                    {
                        "seed": entry["seed"],
                        "valid": entry["valid"],
                        "counterion_positions": entry["counterion_positions"],
                        "coulomb_energy_eV": entry["score_result"].get(
                            "coulomb_energy_eV"
                        ),
                        "min_ion_framework_dist": entry["score_result"].get(
                            "min_ion_framework_dist"
                        ),
                        "min_ion_ion_dist": entry["score_result"].get(
                            "min_ion_ion_dist"
                        ),
                        "min_ion_tm_dist": entry["score_result"].get(
                            "min_ion_tm_dist"
                        ),
                        "checks": entry["score_result"].get("checks", {}),
                    }
                )
            report["per_strategy"][name] = configs
            report["summary"][name] = {
                "generated": info["generated"],
                "passed": info["passed"],
                "best_energy_eV": info["best_energy_eV"],
            }

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as fh:
            json.dump(report, fh, indent=2)

    def print_summary_table(
        self,
        output_dir: Optional[str] = None,
        diversity_rmsd: Optional[float] = None,
        n_output_files: Optional[int] = None,
    ) -> None:
        """Print a formatted summary table to stdout.

        Parameters
        ----------
        output_dir : str or None
            Path to the output directory (shown in the footer).
        diversity_rmsd : float or None
            Average pairwise RMSD across valid configs (shown if provided).
        n_output_files : int or None
            Number of output files written.
        """
        formula = self.get_formula(self.structure)
        n_ions = (
            len(self._results[0]["counterion_positions"])
            if self._results
            else 0
        )
        strat_summary = self._strategy_summary()

        sep = "\u2500" * 55
        print(sep)
        print(
            f" CC-Plocation Results: {formula} + "
            f"{n_ions}\u00d7{self.counterion_element}"
        )
        print(sep)
        header = f" {'Strategy':<16}| {'Generated':^11}| {'Passed':^8}| {'Best E (eV)':>12}"
        print(header)
        print(" " + "\u2500" * 54)

        for name, info in strat_summary.items():
            best_e = (
                f"{info['best_energy_eV']:.1f}"
                if info["best_energy_eV"] is not None
                else "N/A"
            )
            print(
                f" {name:<16}| {info['generated']:^11}| "
                f"{info['passed']:^8}| {best_e:>12}"
            )

        if diversity_rmsd is not None:
            print(f" Diversity: avg RMSD = {diversity_rmsd:.1f} \u00c5")

        if output_dir is not None:
            n_files = n_output_files if n_output_files is not None else "?"
            print(f" Output: {output_dir} ({n_files} files)")

        print(sep)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
