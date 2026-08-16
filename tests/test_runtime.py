from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from math_solution.math_model import allocate_budget
from math_solution.select_sound_border_cells import select_border_cells_for_sound
from simulation import edge_travel_time_hours, precompute_sector_assignment, simulate_patrol
from single_patrol_demo import simulate_single_sortie_softmax


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_GRAPH_PATH = ROOT / "examples" / "synthetic" / "small_graph.json"
SYNTHETIC_PRIORITY_PATH = ROOT / "examples" / "synthetic" / "priority.json"
SYNTHETIC_DIST_PATH = ROOT / "examples" / "synthetic" / "dist.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CoreSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = load_json(SYNTHETIC_GRAPH_PATH)
        cls.priority = load_json(SYNTHETIC_PRIORITY_PATH)

    def test_edge_times_and_sector_assignment(self) -> None:
        self.assertAlmostEqual(
            edge_travel_time_hours({"distance_m": 1000, "road_between_cells": True}),
            1.0 / 60.0,
        )
        self.assertAlmostEqual(
            edge_travel_time_hours({"distance_m": 1000, "road_between_cells": False}),
            1.0 / 30.0,
        )

        owner, travel_time = precompute_sector_assignment(self.graph, ["r0_c0"])
        self.assertEqual(set(owner), set(self.graph["node_features"]))
        self.assertTrue(all(base == "r0_c0" for base in owner.values()))
        self.assertAlmostEqual(travel_time["r0_c1"], 1.0 / 60.0)

    def test_patrol_simulation_returns_to_base_within_budget(self) -> None:
        result = simulate_patrol(
            self.graph,
            ["r0_c0"],
            days=1,
            patrols_per_day=2,
            max_patrol_hours=0.25,
            softmax_tau=0.8,
            random_seed=7,
            node_priority=self.priority,
            include_paths=True,
            max_steps_per_sortie=50,
        )
        self.assertEqual(result["summary"]["total_sorties"], 2)
        self.assertEqual(result["summary"]["returns_to_base_count"], 2)
        for path in result["patrol_units"][0]["sample_sortie_paths"]:
            self.assertEqual(path[0], "r0_c0")
            self.assertEqual(path[-1], "r0_c0")
        for hours in result["patrol_units"][0]["sortie_hours_sample_first10"]:
            self.assertLessEqual(hours, 0.25 + 1e-12)

    def test_single_sortie_returns_to_base_within_budget(self) -> None:
        result = simulate_single_sortie_softmax(
            self.graph,
            "r0_c0",
            priority_by_node=self.priority,
            max_patrol_hours=0.25,
            tau=0.8,
            random_seed=7,
            max_steps=50,
        )
        self.assertTrue(result["ended_at_base"])
        self.assertEqual(result["path"][0], "r0_c0")
        self.assertEqual(result["path"][-1], "r0_c0")
        self.assertLessEqual(result["total_time_h"], 0.25 + 1e-12)

    def test_sound_selection_uses_only_boundary_cells(self) -> None:
        result = select_border_cells_for_sound(self.graph, self.priority, sound_km=2.0)
        selected = result["selected_border_cells"]
        self.assertEqual([row["cell_id"] for row in selected], ["r0_c1", "r1_c1"])
        self.assertTrue(result["meta"]["all_selected_are_border"])

    def test_budget_allocation(self) -> None:
        plan = allocate_budget(350000, 65000, 2, 90000)
        self.assertTrue(plan.minimum_met)
        self.assertTrue(plan.gps_bought)
        self.assertEqual(plan.k_min, 2)
        self.assertEqual(plan.k_max_by_budget, 4)


class IntegrationTests(unittest.TestCase):
    def test_top_level_annual_model_on_synthetic_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            report = Path(tmp) / "report.md"
            command = [
                sys.executable,
                str(ROOT / "math_solution" / "math_model.py"),
                "--total-budget",
                "0",
                "--dist",
                str(SYNTHETIC_DIST_PATH),
                "--sound-graph",
                str(SYNTHETIC_GRAPH_PATH),
                "--sound-priority",
                str(SYNTHETIC_PRIORITY_PATH),
                "--out",
                str(out),
                "--report-out",
                str(report),
                "--skip-plot",
                "--no-progress",
            ]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            payload = load_json(out)
            self.assertEqual(payload["result"]["chosen_K"], 0)
            self.assertTrue(math.isfinite(payload["result"]["annual_total_loss"]))
            self.assertEqual(sum(payload["result"]["events_by_risk"].values()), 365 * 24)
            self.assertTrue(report.exists())

    def test_restored_etosha_patrol_allocator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "patrol.json"
            command = [
                sys.executable,
                str(ROOT / "math_solution" / "patrol_alloc_greedy_unique.py"),
                "--Kmin",
                "2",
                "--Kmax",
                "2",
                "--topL",
                "20",
                "--Tlim",
                "12",
                "--seed",
                "1",
                "--out",
                str(out),
            ]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
            best = load_json(out)["best"]
            self.assertEqual(best["K"], 2)
            self.assertEqual(len(best["patrols"]), 2)
            self.assertEqual(len(best["assigned_cells"]), len(set(best["assigned_cells"])))
            self.assertGreater(best["total_priority"], 0.0)
            for patrol in best["patrols"]:
                self.assertEqual(patrol["way"][0], patrol["base"])
                self.assertEqual(patrol["way"][-1], patrol["base"])
                self.assertLessEqual(patrol["time_h"], 12.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
