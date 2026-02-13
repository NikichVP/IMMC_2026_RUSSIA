from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

try:
    import patrol_alloc_greedy_unique as patrol
except ModuleNotFoundError:
    from math_solution import patrol_alloc_greedy_unique as patrol


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Defaults and model constants
DEFAULT_PATROL_COST = 50000.0
DEFAULT_MIN_RANGERS_REQUIRED = 2
DEFAULT_GPS_FULL_COST = 60000.0
DEFAULT_SOUND_TRACKER_COST_PER_KM = 750.0
DEFAULT_TOTAL_BORDER_KM = 820.0

DEFAULT_DIST_PATH = "solution/big_dist_with_portals_time_priority.json"
DEFAULT_PATROL_PATH = "solution/patrol_house_to_big_cell.json"
DEFAULT_BIGGRAPH_PATH = "solution/etosha_big_square_graph_14x14.json"
DEFAULT_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_SOUND_GRAPH_PATH = "solution/etosha_grid_graph_with_big_squares.json"
DEFAULT_SOUND_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_SOUND_SCORE_FIELD = "priority_P_i"

DEFAULT_TLIM_H = 12.0
DEFAULT_TOPL = 180
DEFAULT_SEED = 1
DEFAULT_SCORE_GAIN_POW = 1.35
DEFAULT_SCORE_TIME_POW = 0.75
DEFAULT_OUT_PATH = "math_solution/math_model_result.json"


@dataclass
class BudgetPlan:
    total_budget: float
    patrol_cost: float
    min_rangers_required: int
    mandatory_rangers: int
    spent_on_mandatory: float
    minimum_met: bool
    gps_full_cost: float
    gps_bought: bool
    spent_on_gps: float
    remaining_after_gps: float
    k_min: int
    k_max_by_budget: int


def _resolve_write_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if path.startswith("math_solution/") or path.startswith("solution/"):
        return os.path.join(PROJECT_DIR, path)
    return os.path.abspath(path)


def allocate_budget(
    total_budget: float,
    patrol_cost: float,
    min_rangers_required: int,
    gps_full_cost: float,
) -> BudgetPlan:
    if total_budget < 0.0:
        raise ValueError("total_budget must be >= 0")
    if patrol_cost <= 0.0:
        raise ValueError("patrol_cost must be > 0")
    if min_rangers_required < 1:
        raise ValueError("min_rangers_required must be >= 1")
    if gps_full_cost < 0.0:
        raise ValueError("gps_full_cost must be >= 0")

    affordable_total = int(total_budget // patrol_cost)
    mandatory = min(min_rangers_required, affordable_total)
    spent_mandatory = mandatory * patrol_cost
    minimum_met = mandatory >= min_rangers_required

    remaining = total_budget - spent_mandatory
    gps_bought = minimum_met and (remaining >= gps_full_cost)
    spent_gps = gps_full_cost if gps_bought else 0.0
    remaining_after_gps = remaining - spent_gps

    if minimum_met:
        k_max_by_budget = int((total_budget - spent_gps) // patrol_cost)
    else:
        k_max_by_budget = affordable_total

    return BudgetPlan(
        total_budget=total_budget,
        patrol_cost=patrol_cost,
        min_rangers_required=min_rangers_required,
        mandatory_rangers=mandatory,
        spent_on_mandatory=spent_mandatory,
        minimum_met=minimum_met,
        gps_full_cost=gps_full_cost,
        gps_bought=gps_bought,
        spent_on_gps=spent_gps,
        remaining_after_gps=remaining_after_gps,
        k_min=mandatory,
        k_max_by_budget=k_max_by_budget,
    )


def _prepare_patrol_inputs(
    dist_path: str,
    patrol_path: str,
    biggraph_path: str,
    priority_path: str,
    topL: Optional[int],
):
    model, dist_big_ids = patrol.load_side_transition_model(dist_path, biggraph_path)
    houses_big_ids = patrol.load_houses_from_patrol_json(patrol_path)
    S = patrol.build_S_big(biggraph_path, priority_path)

    base_set = set(houses_big_ids)
    cands = [bid for bid in dist_big_ids if S.get(bid, 0.0) > 0.0 and bid not in base_set]
    cands_sorted = sorted(cands, key=lambda bid: S.get(bid, 0.0), reverse=True)
    if topL is not None:
        cands_sorted = cands_sorted[: max(0, topL)]

    return model, houses_big_ids, S, cands_sorted


def run_math_model(args: argparse.Namespace) -> dict:
    plan = allocate_budget(
        total_budget=args.total_budget,
        patrol_cost=args.patrol_cost,
        min_rangers_required=args.min_rangers_required,
        gps_full_cost=args.gps_full_cost,
    )

    if args.score_gain_pow <= 0.0 or args.score_time_pow <= 0.0:
        raise ValueError("score_gain_pow and score_time_pow must be > 0")

    best_patrol: Optional[dict] = None
    per_k_best: List[dict] = []
    per_k_full: List[Optional[dict]] = []

    if plan.minimum_met:
        model, houses_big_ids, S, cands_sorted = _prepare_patrol_inputs(
            dist_path=args.dist,
            patrol_path=args.patrol,
            biggraph_path=args.biggraph,
            priority_path=args.priority,
            topL=args.topL,
        )

        best_patrol, per_k_best, per_k_full = patrol.search_best(
            Kmin=plan.k_min,
            Kmax=plan.k_max_by_budget,
            Tlim=args.Tlim,
            houses_big_ids=houses_big_ids,
            cands_sorted=cands_sorted,
            S=S,
            model=model,
            seed=args.seed,
            score_gain_pow=args.score_gain_pow,
            score_time_pow=args.score_time_pow,
            budget_total_for_patrol_sound=(plan.total_budget - plan.spent_on_gps),
            patrol_cost=plan.patrol_cost,
            sound_tracker_cost_per_km=args.sound_tracker_cost_per_km,
            total_border_km=args.total_border_km,
            sound_graph_path=args.sound_graph,
            sound_priority_path=args.sound_priority,
            sound_score_field=args.sound_score_field,
        )

    if best_patrol is None:
        chosen_K = plan.mandatory_rangers if plan.minimum_met else 0
        patrol_priority = 0.0
        total_time_h = 0.0
        sound_priority = 0.0
        total_coverage_priority = 0.0
        patrol_spent_total = chosen_K * plan.patrol_cost
        sound_budget = plan.total_budget - plan.spent_on_gps - patrol_spent_total
        if args.sound_tracker_cost_per_km <= 0.0:
            raise ValueError("sound_tracker_cost_per_km must be > 0")
        if args.total_border_km < 0.0:
            raise ValueError("total_border_km must be >= 0")
        sound_covered_km = min(
            args.total_border_km,
            max(0.0, sound_budget) / args.sound_tracker_cost_per_km,
        )
        sound_selection = None
    else:
        chosen_K = int(best_patrol["K"])
        patrol_priority = float(best_patrol.get("total_priority", 0.0))
        total_time_h = float(best_patrol.get("total_time_h", 0.0))
        sound_priority = float(best_patrol.get("sound_priority", 0.0))
        total_coverage_priority = float(best_patrol.get("total_coverage_priority", patrol_priority + sound_priority))
        patrol_spent_total = float(best_patrol.get("patrol_spent_total", chosen_K * plan.patrol_cost))
        sound_budget = float(best_patrol.get("sound_budget", plan.total_budget - plan.spent_on_gps - patrol_spent_total))
        sound_covered_km = float(best_patrol.get("sound_covered_km", 0.0))
        sound_selection = best_patrol.get("sound_selection")

    payload = {
        "meta": {
            "total_budget": plan.total_budget,
            "patrol_cost": plan.patrol_cost,
            "min_rangers_required": plan.min_rangers_required,
            "gps_full_cost": plan.gps_full_cost,
            "sound_tracker_cost_per_km": args.sound_tracker_cost_per_km,
            "total_border_km": args.total_border_km,
            "k_min_used": plan.k_min,
            "k_max_by_budget": plan.k_max_by_budget,
            "Tlim": args.Tlim,
            "topL": args.topL,
            "seed": args.seed,
            "score_gain_pow": args.score_gain_pow,
            "score_time_pow": args.score_time_pow,
            "dist_path": args.dist,
            "patrol_path": args.patrol,
            "biggraph_path": args.biggraph,
            "priority_path": args.priority,
            "sound_graph_path": args.sound_graph,
            "sound_priority_path": args.sound_priority,
            "sound_score_field": args.sound_score_field,
        },
        "budget_plan": {
            "mandatory_rangers": plan.mandatory_rangers,
            "spent_on_mandatory": plan.spent_on_mandatory,
            "minimum_met": plan.minimum_met,
            "gps_bought": plan.gps_bought,
            "spent_on_gps": plan.spent_on_gps,
            "remaining_after_gps": plan.remaining_after_gps,
        },
        "result": {
            "chosen_K": chosen_K,
            "patrol_spent_total": patrol_spent_total,
            "sound_budget": sound_budget,
            "sound_covered_km": sound_covered_km,
            "patrol_priority": patrol_priority,
            "sound_priority": sound_priority,
            "total_coverage_priority": total_coverage_priority,
            "total_time_h": total_time_h,
        },
        "patrol_best": best_patrol,
        "sound_selection_best": sound_selection,
        "per_K_best": per_k_best,
        "per_K_analysis": per_k_best,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Math model: enforce mandatory rangers, optional GPS, then pass remaining budget to patrol allocator"
    )

    parser.add_argument("--total-budget", type=float, required=True)
    parser.add_argument("--patrol-cost", type=float, default=DEFAULT_PATROL_COST)
    parser.add_argument("--min-rangers-required", type=int, default=DEFAULT_MIN_RANGERS_REQUIRED)
    parser.add_argument("--gps-full-cost", type=float, default=DEFAULT_GPS_FULL_COST)
    parser.add_argument("--sound-tracker-cost-per-km", type=float, default=DEFAULT_SOUND_TRACKER_COST_PER_KM)
    parser.add_argument("--total-border-km", type=float, default=DEFAULT_TOTAL_BORDER_KM)

    parser.add_argument("--dist", default=DEFAULT_DIST_PATH)
    parser.add_argument("--patrol", default=DEFAULT_PATROL_PATH)
    parser.add_argument("--biggraph", default=DEFAULT_BIGGRAPH_PATH)
    parser.add_argument("--priority", default=DEFAULT_PRIORITY_PATH)
    parser.add_argument("--sound-graph", default=DEFAULT_SOUND_GRAPH_PATH)
    parser.add_argument("--sound-priority", default=DEFAULT_SOUND_PRIORITY_PATH)
    parser.add_argument("--sound-score-field", default=DEFAULT_SOUND_SCORE_FIELD)

    parser.add_argument("--Tlim", type=float, default=DEFAULT_TLIM_H)
    parser.add_argument("--topL", type=int, default=DEFAULT_TOPL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--score-gain-pow", type=float, default=DEFAULT_SCORE_GAIN_POW)
    parser.add_argument("--score-time-pow", type=float, default=DEFAULT_SCORE_TIME_POW)

    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_math_model(args)

    out_path = _resolve_write_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if not payload["budget_plan"]["minimum_met"]:
        print("BEST K=None total_priority=0.0")
    else:
        r = payload["result"]
        print(
            f"BEST K={r['chosen_K']} total_priority={r['total_coverage_priority']} "
            f"(patrol={r['patrol_priority']}, sound={r['sound_priority']}) "
            f"sound_km={r['sound_covered_km']:.2f}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
