from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    import patrol_alloc_greedy_unique as patrol
except ModuleNotFoundError:
    from math_solution import patrol_alloc_greedy_unique as patrol

try:
    import select_sound_border_cells as sound_select
except ModuleNotFoundError:
    from math_solution import select_sound_border_cells as sound_select

try:
    import risk_year_simulation as risk_sim
except ModuleNotFoundError:
    from math_solution import risk_year_simulation as risk_sim


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
DEFAULT_REPORT_PATH = "math_solution/math_model_report.md"
DEFAULT_PLOT_PATH = "math_solution/math_model_best_plan_map.png"
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_SCREEN_DAYS = 30
DEFAULT_FINAL_TOP_N = 100
PATROL_POSTS_COUNT = 8

LOSS_EPS = 1e-12


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


def _resolve_read_path(path: str) -> str:
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


def _is_better_loss(cur: Optional[dict], best: Optional[dict]) -> bool:
    if cur is None:
        return False
    if best is None:
        return True

    cur_loss = float(cur.get("annual_total_loss", float("inf")))
    best_loss = float(best.get("annual_total_loss", float("inf")))
    if cur_loss < best_loss - LOSS_EPS:
        return True
    if abs(cur_loss - best_loss) <= LOSS_EPS:
        cur_cov = float(cur.get("total_coverage_priority", 0.0))
        best_cov = float(best.get("total_coverage_priority", 0.0))
        if cur_cov > best_cov + LOSS_EPS:
            return True
        if abs(cur_cov - best_cov) <= LOSS_EPS:
            return float(cur.get("total_time_h", float("inf"))) < float(best.get("total_time_h", float("inf"))) - LOSS_EPS
    return False


def _screen_sort_key(cand: dict) -> tuple:
    return (
        float(cand.get("screen_total_loss", float("inf"))),
        -float(cand.get("total_coverage_priority", 0.0)),
        float(cand.get("total_time_h", float("inf"))),
    )


def _is_better_screen(cur: Optional[dict], best: Optional[dict]) -> bool:
    if cur is None:
        return False
    if best is None:
        return True
    return _screen_sort_key(cur) < _screen_sort_key(best)


def _add_candidate_to_screen_finalists(finalists: List[dict], cand: dict, top_n: int) -> None:
    if top_n <= 0:
        return
    if len(finalists) < top_n:
        finalists.append(cand)
        return
    worst_idx = max(range(len(finalists)), key=lambda i: _screen_sort_key(finalists[i]))
    if _screen_sort_key(cand) < _screen_sort_key(finalists[worst_idx]):
        finalists[worst_idx] = cand


def _composition_count_for_k(K: int, H: int = PATROL_POSTS_COUNT) -> int:
    if K < 0:
        return 0
    if K <= H:
        return math.comb(H, K)
    return math.comb(K + H - 1, H - 1)


def _gen_feasible_compositions(K: int, H: int = PATROL_POSTS_COUNT):
    if K < 0:
        return
    if K <= H:
        if K == 0:
            yield [0] * H
            return
        for idxs in itertools.combinations(range(H), K):
            m = [0] * H
            for i in idxs:
                m[i] = 1
            yield m
        return
    for m in patrol.gen_compositions(K, H):
        yield m


def _validate_sound_selection_on_park_border(sound_selection: dict, park_border_cells: set) -> None:
    rows = sound_selection.get("selected_border_cells", [])
    if not isinstance(rows, list):
        raise ValueError("sound selection format is invalid: selected_border_cells must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sound selection format is invalid: row must be an object")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str):
            raise ValueError("sound selection format is invalid: missing cell_id")
        if cell_id not in park_border_cells:
            raise ValueError(
                f"sound selection contains non-park-border cell {cell_id}; expected only is_boarder/is_border park boundary cells"
            )


def _write_report(payload: dict, report_path: str) -> None:
    meta = payload.get("meta", {})
    result = payload.get("result", {})
    budget = payload.get("budget_plan", {})

    lines: List[str] = []
    lines.append("# Math Model Annual Risk Simulation Report")
    lines.append("")
    lines.append("## Budget Setup")
    lines.append(f"- Total budget: {meta.get('total_budget')}")
    lines.append(f"- Patrol cost: {meta.get('patrol_cost')}")
    lines.append(f"- GPS full cost: {meta.get('gps_full_cost')}")
    lines.append(f"- Sound tracker cost per km: {meta.get('sound_tracker_cost_per_km')}")
    lines.append(f"- Minimum ranger groups required: {meta.get('min_rangers_required')}")
    lines.append(f"- Mandatory minimum met: {budget.get('minimum_met')}")
    lines.append(f"- GPS bought: {budget.get('gps_bought')}")
    lines.append(f"- Stage-1 screen days: {meta.get('screen_days')}")
    lines.append(f"- Stage-2 top-N: {meta.get('final_top_n')}")
    lines.append("")
    lines.append("## Best Plan (Min Annual Loss)")
    lines.append(f"- Chosen ranger groups K: {result.get('chosen_K')}")
    lines.append(f"- Patrol spent total: {result.get('patrol_spent_total')}")
    lines.append(f"- Sound budget: {result.get('sound_budget')}")
    lines.append(f"- Sound covered km: {result.get('sound_covered_km')}")
    lines.append(f"- Annual total loss: {result.get('annual_total_loss')}")
    lines.append(f"- Annual loss by risk: {result.get('annual_loss_by_risk')}")
    lines.append(f"- Events by risk: {result.get('events_by_risk')}")
    lines.append(f"- Detected by risk: {result.get('detected_by_risk')}")
    lines.append(f"- Undetected by risk: {result.get('undetected_by_risk')}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Result JSON: {payload.get('artifacts', {}).get('result_json_path')}")
    lines.append(f"- Plot image: {payload.get('artifacts', {}).get('plot_path')}")
    lines.append(f"- Report file: {payload.get('artifacts', {}).get('report_path')}")
    lines.append("")
    lines.append("## Risk Constants (TODO update)")
    lines.append(f"- C_OP={risk_sim.C_OP}, C_IH={risk_sim.C_IH}, C_SN={risk_sim.C_SN}, C_TD={risk_sim.C_TD}")
    lines.append(
        f"- MAX_LOSS_OP={risk_sim.MAX_LOSS_OP}, MAX_LOSS_IH={risk_sim.MAX_LOSS_IH}, "
        f"MAX_LOSS_SN={risk_sim.MAX_LOSS_SN}, MAX_LOSS_TD={risk_sim.MAX_LOSS_TD}"
    )
    lines.append(
        f"- T_SLA_H={risk_sim.T_SLA_H}, T_KILL_OP_H={risk_sim.T_KILL_OP_H}, "
        f"T_KILL_IH_H={risk_sim.T_KILL_IH_H}, T_KILL_SN_H={risk_sim.T_KILL_SN_H}, T_KILL_TD_H={risk_sim.T_KILL_TD_H}"
    )

    out_dir = os.path.dirname(report_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _maybe_build_plot(args: argparse.Namespace, result_json_path: str) -> Optional[str]:
    if args.skip_plot:
        return None

    plot_path = _resolve_write_path(args.plot_out)
    cmd = [
        "python3",
        os.path.join("math_solution", "viz_patrol_alloc_k2.py"),
        "--result",
        result_json_path,
        "--smallgraph",
        args.sound_graph,
        "--biggraph",
        args.biggraph,
        "--priority",
        args.sound_priority,
        "--routes",
        "all",
        "--out",
        plot_path,
    ]

    try:
        subprocess.run(cmd, cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
        return plot_path
    except Exception:
        return None


def run_math_model(args: argparse.Namespace) -> dict:
    plan = allocate_budget(
        total_budget=args.total_budget,
        patrol_cost=args.patrol_cost,
        min_rangers_required=args.min_rangers_required,
        gps_full_cost=args.gps_full_cost,
    )

    if args.score_gain_pow <= 0.0 or args.score_time_pow <= 0.0:
        raise ValueError("score_gain_pow and score_time_pow must be > 0")
    if args.sound_tracker_cost_per_km <= 0.0:
        raise ValueError("sound_tracker_cost_per_km must be > 0")
    if args.total_border_km < 0.0:
        raise ValueError("total_border_km must be >= 0")
    if args.screen_days <= 0:
        raise ValueError("screen_days must be > 0")
    if args.final_top_n < 0:
        raise ValueError("final_top_n must be >= 0")

    best_patrol: Optional[dict] = None
    per_k_best: List[dict] = []

    sim_ctx = risk_sim.build_simulation_context(
        small_graph_path=args.sound_graph,
        dist_path=args.dist,
        priority_path=args.sound_priority,
    )
    park_border_cells = set(sim_ctx.border_nodes)
    if args.progress:
        print(
            f"[progress] Simulation context ready: park_border_cells={len(park_border_cells)}, "
            f"photo_traps={len(sim_ctx.photo_nodes)}, road_cells={len(sim_ctx.road_nodes)}",
            flush=True,
        )

    if plan.minimum_met:
        model, houses_big_ids, S, cands_sorted = _prepare_patrol_inputs(
            dist_path=args.dist,
            patrol_path=args.patrol,
            biggraph_path=args.biggraph,
            priority_path=args.priority,
            topL=args.topL,
        )

        k_values = list(range(plan.k_min, plan.k_max_by_budget + 1))
        total_compositions_planned = sum(_composition_count_for_k(K, PATROL_POSTS_COUNT) for K in k_values)
        global_compositions_done = 0
        screen_finalists: List[dict] = []
        screen_best_by_k: dict = {}
        comp_count_by_k: dict = {}
        final_best_by_k: dict = {}

        if args.progress:
            print(
                f"[progress] Start K-loop: K={plan.k_min}..{plan.k_max_by_budget}, "
                f"total_compositions={total_compositions_planned}, "
                f"screen_days={args.screen_days}, final_top_n={args.final_top_n}",
                flush=True,
            )
            print(
                "[progress] Rule active: for K<=8 only unique-post compositions are evaluated (m_i in {0,1})",
                flush=True,
            )

        # Stage 1: evaluate all candidates with short simulation and keep only global top-N.
        for K in k_values:
            patrol_spent_total = K * plan.patrol_cost
            sound_budget = plan.total_budget - plan.spent_on_gps - patrol_spent_total
            sound_covered_km_requested = min(args.total_border_km, max(0.0, sound_budget) / args.sound_tracker_cost_per_km)
            sound_selection = sound_select.build_sound_selection(
                sound_km=sound_covered_km_requested,
                graph_path=args.sound_graph,
                priority_path=args.sound_priority,
                score_field=args.sound_score_field,
            )
            _validate_sound_selection_on_park_border(sound_selection=sound_selection, park_border_cells=park_border_cells)
            sound_covered_km = float(sound_selection.get("meta", {}).get("covered_km", sound_covered_km_requested))
            sound_priority = float(sound_selection.get("meta", {}).get("selected_priority_sum", 0.0))

            best_k_screen: Optional[dict] = None
            comp_count = 0
            k_compositions_total = _composition_count_for_k(K, PATROL_POSTS_COUNT)
            k_t0 = time.perf_counter()
            if args.progress:
                print(
                    f"[progress] K={K} screen start: compositions={k_compositions_total}, "
                    f"sound_km={sound_covered_km:.2f}, gps={plan.gps_bought}, days={args.screen_days}",
                    flush=True,
                )
            for comp_idx, m in enumerate(_gen_feasible_compositions(K, PATROL_POSTS_COUNT)):
                comp_count += 1
                global_compositions_done += 1
                local_seed = (args.seed * patrol.SEED_MUL_A) ^ (K * patrol.SEED_MUL_B) ^ comp_idx
                rng_patrol = random.Random(local_seed)

                patrol_res = patrol.solve_one_distribution(
                    K=K,
                    m=m,
                    houses_big_ids=houses_big_ids,
                    cands_sorted=cands_sorted,
                    S=S,
                    model=model,
                    Tlim=args.Tlim,
                    score_gain_pow=args.score_gain_pow,
                    score_time_pow=args.score_time_pow,
                    rng=rng_patrol,
                )

                sim_res_screen = risk_sim.simulate_period(
                    ctx=sim_ctx,
                    patrol_plan=patrol_res,
                    sound_selection=sound_selection,
                    gps_enabled=plan.gps_bought,
                    seed=(local_seed ^ 0x5DEECE66D),
                    days=args.screen_days,
                )

                cand = dict(patrol_res)
                cand["patrol_spent_total"] = float(patrol_spent_total)
                cand["sound_budget"] = float(sound_budget)
                cand["sound_covered_km"] = float(sound_covered_km)
                cand["sound_priority"] = float(sound_priority)
                cand["total_coverage_priority"] = float(cand.get("total_priority", 0.0)) + float(sound_priority)
                cand["sound_selection"] = sound_selection
                cand["seed"] = int(local_seed)
                cand["screen_total_loss"] = float(sim_res_screen.get("total_loss", sim_res_screen.get("annual_total_loss", float("inf"))))
                cand["screen_loss_by_risk"] = dict(
                    sim_res_screen.get("loss_by_risk", sim_res_screen.get("annual_loss_by_risk", {}))
                )
                cand["screen_events_by_risk"] = dict(sim_res_screen.get("events_by_risk", {}))
                cand["screen_days"] = int(args.screen_days)

                if _is_better_screen(cand, best_k_screen):
                    best_k_screen = cand

                _add_candidate_to_screen_finalists(screen_finalists, cand, args.final_top_n)
                if args.progress and args.progress_every > 0 and (comp_count % args.progress_every == 0):
                    best_k_loss = float(best_k_screen["screen_total_loss"]) if best_k_screen is not None else float("inf")
                    elapsed_k = time.perf_counter() - k_t0
                    print(
                        f"[progress] K={K} screen progress {comp_count}/{k_compositions_total} "
                        f"(global {global_compositions_done}/{total_compositions_planned}), "
                        f"best_K_screen_loss={best_k_loss:.4f}, finalists={len(screen_finalists)}, elapsed_K={elapsed_k:.1f}s",
                        flush=True,
                    )

            comp_count_by_k[K] = comp_count
            screen_best_by_k[K] = best_k_screen

            if args.progress:
                k_elapsed = time.perf_counter() - k_t0
                k_loss = float(best_k_screen["screen_total_loss"]) if best_k_screen is not None else float("inf")
                global_best_screen = min(screen_finalists, key=_screen_sort_key) if screen_finalists else None
                global_best_k = int(global_best_screen["K"]) if global_best_screen is not None else None
                global_best_loss = (
                    float(global_best_screen["screen_total_loss"]) if global_best_screen is not None else float("inf")
                )
                print(
                    f"[progress] K={K} screen done in {k_elapsed:.1f}s, "
                    f"best_for_K_screen_loss={k_loss:.4f}; global_best_screen_K={global_best_k}, "
                    f"global_best_screen_loss={global_best_loss:.4f}",
                    flush=True,
                )

        finalists_sorted = sorted(screen_finalists, key=_screen_sort_key)
        if args.progress:
            print(
                f"[progress] Stage-1 complete: finalists_kept={len(finalists_sorted)}",
                flush=True,
            )

        # Stage 2: full-year simulation only for top-N finalists from stage 1.
        if args.progress and finalists_sorted:
            print(
                f"[progress] Stage-2 start: full-year evaluation for top {len(finalists_sorted)} candidates",
                flush=True,
            )

        for idx, cand in enumerate(finalists_sorted, start=1):
            K = int(cand["K"])
            local_seed = int(cand.get("seed", 0))
            sim_res_year = risk_sim.simulate_year(
                ctx=sim_ctx,
                patrol_plan=cand,
                sound_selection=cand.get("sound_selection"),
                gps_enabled=plan.gps_bought,
                seed=(local_seed ^ 0x5DEECE66D),
            )
            full_cand = dict(cand)
            full_cand.update(sim_res_year)

            if _is_better_loss(full_cand, final_best_by_k.get(K)):
                final_best_by_k[K] = full_cand
            if _is_better_loss(full_cand, best_patrol):
                best_patrol = full_cand

            if args.progress and (idx == len(finalists_sorted) or (args.progress_every > 0 and idx % args.progress_every == 0)):
                global_best_k = int(best_patrol["K"]) if best_patrol is not None else None
                global_best_loss = float(best_patrol["annual_total_loss"]) if best_patrol is not None else float("inf")
                print(
                    f"[progress] Stage-2 progress {idx}/{len(finalists_sorted)}: "
                    f"global_best_K={global_best_k}, global_best_annual_loss={global_best_loss:.4f}",
                    flush=True,
                )

        for K in k_values:
            best_year = final_best_by_k.get(K)
            best_screen = screen_best_by_k.get(K)
            src = best_year if best_year is not None else best_screen
            comp_count = int(comp_count_by_k.get(K, 0))

            if src is None:
                row = {
                    "K": int(K),
                    "m": [0] * PATROL_POSTS_COUNT,
                    "total_priority": 0.0,
                    "total_time_h": 0.0,
                    "patrol_spent_total": float(K * plan.patrol_cost),
                    "sound_budget": 0.0,
                    "sound_covered_km": 0.0,
                    "sound_priority": 0.0,
                    "total_coverage_priority": 0.0,
                    "screen_total_loss": float("inf"),
                    "screen_loss_by_risk": {k: 0.0 for k in risk_sim.RISK_TYPES},
                    "annual_total_loss": float("inf"),
                    "annual_loss_by_risk": {k: 0.0 for k in risk_sim.RISK_TYPES},
                    "events_by_risk": {k: 0 for k in risk_sim.RISK_TYPES},
                    "detected_by_risk": {k: 0 for k in risk_sim.RISK_TYPES},
                    "undetected_by_risk": {k: 0 for k in risk_sim.RISK_TYPES},
                    "compositions_evaluated": comp_count,
                    "final_year_evaluated": False,
                }
            else:
                row = {
                    "K": int(src["K"]),
                    "m": list(src["m"]),
                    "total_priority": float(src.get("total_priority", 0.0)),
                    "total_time_h": float(src.get("total_time_h", 0.0)),
                    "patrol_spent_total": float(src.get("patrol_spent_total", K * plan.patrol_cost)),
                    "sound_budget": float(src.get("sound_budget", 0.0)),
                    "sound_covered_km": float(src.get("sound_covered_km", 0.0)),
                    "sound_priority": float(src.get("sound_priority", 0.0)),
                    "total_coverage_priority": float(src.get("total_coverage_priority", 0.0)),
                    "screen_total_loss": float(src.get("screen_total_loss", float("inf"))),
                    "screen_loss_by_risk": dict(src.get("screen_loss_by_risk", {})),
                    "annual_total_loss": float(src.get("annual_total_loss", float("inf"))),
                    "annual_loss_by_risk": dict(src.get("annual_loss_by_risk", {k: 0.0 for k in risk_sim.RISK_TYPES})),
                    "events_by_risk": dict(src.get("events_by_risk", {k: 0 for k in risk_sim.RISK_TYPES})),
                    "detected_by_risk": dict(src.get("detected_by_risk", {k: 0 for k in risk_sim.RISK_TYPES})),
                    "undetected_by_risk": dict(src.get("undetected_by_risk", {k: 0 for k in risk_sim.RISK_TYPES})),
                    "compositions_evaluated": comp_count,
                    "final_year_evaluated": best_year is not None,
                    "sound_selection": src.get("sound_selection"),
                }
            per_k_best.append(row)

    if best_patrol is None:
        chosen_K = 0
        patrol_priority = 0.0
        total_time_h = 0.0
        sound_priority = 0.0
        total_coverage_priority = 0.0
        patrol_spent_total = 0.0
        sound_budget = 0.0
        sound_covered_km = 0.0
        sound_selection = None

        sim_res = risk_sim.simulate_year(
            ctx=sim_ctx,
            patrol_plan={"patrols": []},
            sound_selection=None,
            gps_enabled=plan.gps_bought,
            seed=args.seed,
        )
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
        sim_res = {
            "annual_total_loss": float(best_patrol.get("annual_total_loss", 0.0)),
            "annual_loss_by_risk": dict(best_patrol.get("annual_loss_by_risk", {})),
            "events_by_risk": dict(best_patrol.get("events_by_risk", {})),
            "detected_by_risk": dict(best_patrol.get("detected_by_risk", {})),
            "undetected_by_risk": dict(best_patrol.get("undetected_by_risk", {})),
        }

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
            "simulation_hours": 365 * 24,
            "simulation_start_date": "2025-01-01",
            "simulation_rule": "1 event per hour, sampled by season-specific normalized OP/SN/IH/TD probabilities",
            "two_stage_screening_enabled": True,
            "screen_days": int(args.screen_days),
            "screen_hours": int(args.screen_days * 24),
            "final_top_n": int(args.final_top_n),
            "composition_rule_k_le_8": "m_i in {0,1} (no duplicate post allocations)",
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
            "annual_total_loss": sim_res["annual_total_loss"],
            "annual_loss_by_risk": sim_res["annual_loss_by_risk"],
            "events_by_risk": sim_res["events_by_risk"],
            "detected_by_risk": sim_res["detected_by_risk"],
            "undetected_by_risk": sim_res["undetected_by_risk"],
        },
        "patrol_best": best_patrol,
        "sound_selection_best": sound_selection,
        "per_K_best": per_k_best,
        "per_K_analysis": per_k_best,
        "risk_constants_todo_update": {
            "C_OP": risk_sim.C_OP,
            "C_IH": risk_sim.C_IH,
            "C_SN": risk_sim.C_SN,
            "C_TD": risk_sim.C_TD,
            "MAX_LOSS_OP": risk_sim.MAX_LOSS_OP,
            "MAX_LOSS_IH": risk_sim.MAX_LOSS_IH,
            "MAX_LOSS_SN": risk_sim.MAX_LOSS_SN,
            "MAX_LOSS_TD": risk_sim.MAX_LOSS_TD,
            "T_SLA_H": risk_sim.T_SLA_H,
            "T_KILL_OP_H": risk_sim.T_KILL_OP_H,
            "T_KILL_IH_H": risk_sim.T_KILL_IH_H,
            "T_KILL_SN_H": risk_sim.T_KILL_SN_H,
            "T_KILL_TD_H": risk_sim.T_KILL_TD_H,
            "OP_IH_MAX_DURATION_H": risk_sim.OP_IH_MAX_DURATION_H,
            "SN_MAX_DURATION_H": risk_sim.SN_MAX_DURATION_H,
            "TD_MAX_DURATION_H": risk_sim.TD_MAX_DURATION_H,
            "RANGER_SPEED_KMH": risk_sim.RANGER_SPEED_KMH,
            "INTRUDER_SPEED_KMH": risk_sim.INTRUDER_SPEED_KMH,
            "RANGER_DETECT_RADIUS_M": risk_sim.RANGER_DETECT_RADIUS_M,
            "PHOTO_DETECT_RADIUS_M": risk_sim.PHOTO_DETECT_RADIUS_M,
        },
        "artifacts": {
            "result_json_path": None,
            "report_path": None,
            "plot_path": None,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Math model: enforce mandatory rangers, optional GPS, then evaluate all ranger compositions "
            "via 2-stage simulation (short screen + full year on top-N); choose plan by minimum annual loss"
        )
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
    parser.add_argument("--screen-days", type=int, default=DEFAULT_SCREEN_DAYS)
    parser.add_argument("--final-top-n", type=int, default=DEFAULT_FINAL_TOP_N)

    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    parser.add_argument("--report-out", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plot-out", default=DEFAULT_PLOT_PATH)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_math_model(args)

    out_path = _resolve_write_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload["artifacts"]["result_json_path"] = out_path

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    plot_path = _maybe_build_plot(args=args, result_json_path=out_path)
    payload["artifacts"]["plot_path"] = plot_path

    report_path = _resolve_write_path(args.report_out)
    payload["artifacts"]["report_path"] = report_path
    _write_report(payload, report_path=report_path)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    r = payload["result"]
    print(
        f"BEST K={r['chosen_K']} annual_loss={r['annual_total_loss']:.6f} "
        f"(OP={r['annual_loss_by_risk'].get('OP', 0.0):.4f}, "
        f"SN={r['annual_loss_by_risk'].get('SN', 0.0):.4f}, "
        f"IH={r['annual_loss_by_risk'].get('IH', 0.0):.4f}, "
        f"TD={r['annual_loss_by_risk'].get('TD', 0.0):.4f})"
    )
    print(f"REPORT: {report_path}")
    print(f"PLOT: {plot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
