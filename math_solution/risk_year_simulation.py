from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple


# TODO(USER): tune all constants below with real field calibration data.
# Cost/value scales
C_OP = 10.0
C_IH = 10.0
C_SN = 10.0
C_TD = 10.0

# TODO(USER): hard caps for loss if an event is not intercepted in time.
MAX_LOSS_OP = 10.0
MAX_LOSS_IH = 10.0
MAX_LOSS_SN = 10.0
MAX_LOSS_TD = 10.0

# TODO(USER): SLA and kill/exposure constants (hours).
T_SLA_H = 2.0
T_KILL_OP_H = 6.0
T_KILL_IH_H = 6.0
T_KILL_SN_H = 24.0
T_KILL_TD_H = 24.0

# TODO(USER): event duration limits.
OP_IH_MAX_DURATION_H = 8
SN_MAX_DURATION_H = 24
TD_MAX_DURATION_H = 24 * 14

# Movement and detection assumptions
RANGER_SPEED_KMH = 60.0
INTRUDER_SPEED_KMH = 5.0
RANGER_DETECT_RADIUS_M = 1000.0
PHOTO_DETECT_RADIUS_M = 300.0
RANGER_STOP_RADIUS_M = 1000.0

# Daily patrol randomization assumptions
PATROL_ACTIVE_HOURS_PER_DAY = 12
PATROL_LOOPS_PER_DAY = 2
DAILY_KEYPOINT_LIMIT = 8

# Target weights from user request
TARGET_WEIGHT_ANIMALS = 0.2
TARGET_WEIGHT_PLANTS = 0.1
TARGET_WEIGHT_WATER = 0.7

# Disease process assumptions
TD_INITIAL_INFECTED = 1.0
TD_DAILY_GROWTH_MULT = 2.0

# Seasons and normalized risk probabilities after filtering to OP/SN/IH/TD.
RISK_TYPES = ("OP", "SN", "IH", "TD")
SEASON_PROBS: Dict[str, Dict[str, float]] = {
    "G1": {"OP": 0.4818, "SN": 0.2878, "IH": 0.1673, "TD": 0.0631},
    "G2": {"OP": 0.2980, "SN": 0.4849, "IH": 0.1780, "TD": 0.0390},
    "G3": {"OP": 0.0747, "SN": 0.1284, "IH": 0.2150, "TD": 0.5819},
    "G4": {"OP": 0.5240, "SN": 0.3220, "IH": 0.1118, "TD": 0.0422},
}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def _resolve_read_path(path: str) -> str:
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"File not found: {path}")

    candidates = [path, os.path.join(PROJECT_DIR, path), os.path.join(SCRIPT_DIR, path)]
    if path.startswith("solution/"):
        tail = path[len("solution/") :]
        candidates.extend([os.path.join(PROJECT_DIR, tail), os.path.join(SCRIPT_DIR, tail)])

    seen = set()
    for c in candidates:
        ac = os.path.abspath(c)
        if ac in seen:
            continue
        seen.add(ac)
        if os.path.exists(ac):
            return ac
    raise FileNotFoundError(f"File not found: {path}")


def _load_json(path: str) -> dict:
    with open(_resolve_read_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class SimulationContext:
    node_xy: Dict[str, Tuple[float, float]]
    node_big: Dict[str, str]
    rep_node_by_big: Dict[str, str]
    border_nodes: Tuple[str, ...]
    border_animal_nodes: Tuple[str, ...]
    road_nodes: Tuple[str, ...]
    animal_nodes: Tuple[str, ...]
    photo_nodes: Tuple[str, ...]
    target_nodes_by_big: Dict[str, Tuple[str, ...]]
    target_node_score: Dict[str, float]
    node_priority: Dict[str, float]
    nearest_target_bigs_by_big: Dict[str, Tuple[str, ...]]
    big_distance_m: Dict[str, Dict[str, float]]


def _month_to_season(month: int) -> str:
    # G1 Dry+popular - Jul-Sep
    # G2 Wet+popular - Oct-Nov
    # G3 Wet - Dec-Feb
    # G4 Dry - Mar-Jun
    if month in (7, 8, 9):
        return "G1"
    if month in (10, 11):
        return "G2"
    if month in (12, 1, 2):
        return "G3"
    return "G4"


def _season_for_hour(hour_idx: int) -> str:
    d = date(2025, 1, 1) + timedelta(days=(hour_idx // 24))
    return _month_to_season(d.month)


def _sample_event_type(rng: random.Random, season: str) -> str:
    probs = SEASON_PROBS[season]
    u = rng.random()
    c = 0.0
    for rt in RISK_TYPES:
        c += probs[rt]
        if u <= c:
            return rt
    return RISK_TYPES[-1]


def _euclid_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe_distance_m(
    node_a: str,
    node_b: str,
    node_xy: Dict[str, Tuple[float, float]],
    node_big: Dict[str, str],
    big_distance_m: Dict[str, Dict[str, float]],
) -> float:
    if node_a == node_b:
        return 0.0
    ba = node_big.get(node_a)
    bb = node_big.get(node_b)
    if ba is not None and bb is not None:
        d = big_distance_m.get(ba, {}).get(bb, float("inf"))
        if math.isfinite(d) and d > 0.0:
            return d
    xa = node_xy.get(node_a)
    xb = node_xy.get(node_b)
    if xa is None or xb is None:
        return float("inf")
    return _euclid_m(xa, xb)


def _build_target_structures(
    inside_nodes: Sequence[str],
    node_big: Dict[str, str],
    animals: Set[str],
    plants: Set[str],
    waters: Set[str],
    big_distance_m: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
    target_node_score: Dict[str, float] = {}
    target_nodes_by_big_mut: Dict[str, List[str]] = {}
    target_big_score: Dict[str, float] = {}

    for nid in inside_nodes:
        w = 0.0
        if nid in animals:
            w += TARGET_WEIGHT_ANIMALS
        if nid in plants:
            w += TARGET_WEIGHT_PLANTS
        if nid in waters:
            w += TARGET_WEIGHT_WATER
        if w <= 0.0:
            continue
        target_node_score[nid] = w
        bid = node_big.get(nid)
        if bid is None:
            continue
        target_nodes_by_big_mut.setdefault(bid, []).append(nid)
        target_big_score[bid] = target_big_score.get(bid, 0.0) + w

    target_nodes_by_big = {k: tuple(v) for k, v in target_nodes_by_big_mut.items()}

    nearest_target_bigs_by_big: Dict[str, Tuple[str, ...]] = {}
    target_bigs = list(target_nodes_by_big.keys())
    for src_big in big_distance_m.keys():
        scored: List[Tuple[float, str]] = []
        for tb in target_bigs:
            d = big_distance_m.get(src_big, {}).get(tb, float("inf"))
            if not math.isfinite(d):
                continue
            inv = target_big_score.get(tb, 0.0) / max(1.0, d)
            if inv <= 0.0:
                continue
            scored.append((-inv, tb))
        scored.sort()
        nearest_target_bigs_by_big[src_big] = tuple(tb for _, tb in scored[:16])

    return target_node_score, target_nodes_by_big, nearest_target_bigs_by_big


def _load_node_priority_map(priority_path: str) -> Dict[str, float]:
    payload = _load_json(priority_path)
    out: Dict[str, float] = {}
    if isinstance(payload.get("node_priority"), dict):
        src = payload["node_priority"]
    else:
        src = payload
    if not isinstance(src, dict):
        return out
    for k, v in src.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
            continue
        if isinstance(v, dict):
            if "priority_P_i" in v and isinstance(v["priority_P_i"], (int, float)):
                out[k] = float(v["priority_P_i"])
            elif "priority" in v and isinstance(v["priority"], (int, float)):
                out[k] = float(v["priority"])
    return out


def build_simulation_context(
    small_graph_path: str,
    dist_path: str,
    priority_path: str = "solution/etosha_node_priority_compact_clamped.json",
) -> SimulationContext:
    small_graph = _load_json(small_graph_path)
    dist_payload = _load_json(dist_path)

    node_features = small_graph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError(f"Invalid small graph format: {small_graph_path}")

    dist = dist_payload.get("dist")
    if not isinstance(dist, dict):
        raise ValueError(f"Invalid dist format: {dist_path}")

    node_xy: Dict[str, Tuple[float, float]] = {}
    node_big: Dict[str, str] = {}
    by_rc: Dict[Tuple[int, int], str] = {}
    big_to_nodes: Dict[str, List[str]] = {}
    border_nodes: List[str] = []
    border_animal_nodes: List[str] = []
    road_nodes: List[str] = []
    animal_nodes: List[str] = []
    photo_nodes: List[str] = []
    plant_nodes: Set[str] = set()
    water_nodes: Set[str] = set()
    inside_nodes: List[str] = []

    for nid, nf in node_features.items():
        if not isinstance(nf, dict):
            continue
        if nf.get("median_elevation_m") is None:
            continue

        centroid = nf.get("centroid_m")
        if not (isinstance(centroid, list) and len(centroid) == 2):
            continue

        inside_nodes.append(nid)
        node_xy[nid] = (float(centroid[0]), float(centroid[1]))
        bid = str(nf.get("big_square_id", ""))
        node_big[nid] = bid
        big_to_nodes.setdefault(bid, []).append(nid)
        rr = int(nf.get("row", -1))
        cc = int(nf.get("col", -1))
        if rr >= 0 and cc >= 0:
            by_rc[(rr, cc)] = nid

        if bool(nf.get("animals_present", False)):
            animal_nodes.append(nid)

        if float(nf.get("road_total_length_m", 0.0) or 0.0) > 0.0:
            road_nodes.append(nid)

        if bool(nf.get("has_plant", False)):
            plant_nodes.add(nid)

        poi = nf.get("poi_type_counts", {})
        if isinstance(poi, dict):
            if float(poi.get("photo_trap", 0.0) or 0.0) > 0.0:
                photo_nodes.append(nid)
            if float(poi.get("waterhole", 0.0) or 0.0) > 0.0 or float(poi.get("waterhole_dry", 0.0) or 0.0) > 0.0:
                water_nodes.add(nid)

    rep_node_by_big: Dict[str, str] = {}
    for bid, nodes in big_to_nodes.items():
        if not nodes:
            continue
        rep_node_by_big[bid] = nodes[0]

    big_distance_m: Dict[str, Dict[str, float]] = {}
    for a, row in dist.items():
        if not isinstance(row, dict):
            continue
        inner: Dict[str, float] = {}
        for b, rec in row.items():
            if isinstance(rec, dict):
                try:
                    x = float(rec.get("distance_m", float("inf")))
                except (TypeError, ValueError):
                    x = float("inf")
                inner[b] = x
        big_distance_m[a] = inner

    target_node_score, target_nodes_by_big, nearest_target_bigs_by_big = _build_target_structures(
        inside_nodes=inside_nodes,
        node_big=node_big,
        animals=set(animal_nodes),
        plants=plant_nodes,
        waters=water_nodes,
        big_distance_m=big_distance_m,
    )
    node_priority = _load_node_priority_map(priority_path=priority_path)

    # Derived park boundary: inside cells with at least one outside (or missing) 4-neighbor.
    inside_set = set(inside_nodes)
    animal_set = set(animal_nodes)
    for nid in inside_nodes:
        nf = node_features.get(nid, {})
        rr = int(nf.get("row", -1)) if isinstance(nf, dict) else -1
        cc = int(nf.get("col", -1)) if isinstance(nf, dict) else -1
        if rr < 0 or cc < 0:
            continue
        has_outside_neighbor = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = by_rc.get((rr + dr, cc + dc))
            if nb is None or nb not in inside_set:
                has_outside_neighbor = True
                break
        if not has_outside_neighbor:
            continue
        border_nodes.append(nid)
        if nid in animal_set:
            border_animal_nodes.append(nid)

    return SimulationContext(
        node_xy=node_xy,
        node_big=node_big,
        rep_node_by_big=rep_node_by_big,
        border_nodes=tuple(border_nodes),
        border_animal_nodes=tuple(border_animal_nodes),
        road_nodes=tuple(road_nodes),
        animal_nodes=tuple(animal_nodes),
        photo_nodes=tuple(photo_nodes),
        target_nodes_by_big=target_nodes_by_big,
        target_node_score=target_node_score,
        node_priority=node_priority,
        nearest_target_bigs_by_big=nearest_target_bigs_by_big,
        big_distance_m=big_distance_m,
    )


def _pick_target_node(
    rng: random.Random,
    ctx: SimulationContext,
    spawn_node: str,
) -> str:
    spawn_big = ctx.node_big.get(spawn_node)
    if spawn_big is None:
        return spawn_node

    target_bigs = list(ctx.nearest_target_bigs_by_big.get(spawn_big, ()))
    if not target_bigs:
        return spawn_node

    big_weights: List[float] = []
    for tb in target_bigs:
        d = ctx.big_distance_m.get(spawn_big, {}).get(tb, float("inf"))
        if not math.isfinite(d):
            big_weights.append(0.0)
            continue
        s = 0.0
        for nid in ctx.target_nodes_by_big.get(tb, ()): 
            s += ctx.target_node_score.get(nid, 0.0)
        big_weights.append(s / max(1.0, d))

    if not any(w > 0.0 for w in big_weights):
        return spawn_node

    chosen_big = rng.choices(target_bigs, weights=big_weights, k=1)[0]
    nodes = list(ctx.target_nodes_by_big.get(chosen_big, ()))
    if not nodes:
        return spawn_node

    nw = [ctx.target_node_score.get(n, 0.0) for n in nodes]
    if not any(w > 0.0 for w in nw):
        return nodes[0]
    return rng.choices(nodes, weights=nw, k=1)[0]


def _sample_border_node_priority_quadratic(
    rng: random.Random,
    border_nodes: Sequence[str],
    node_priority: Dict[str, float],
) -> str:
    if not border_nodes:
        raise ValueError("border_nodes must be non-empty")
    weights: List[float] = []
    for nid in border_nodes:
        p = float(node_priority.get(nid, 0.0))
        if p < 0.0:
            p = 0.0
        weights.append(p * p)
    if not any(w > 0.0 for w in weights):
        return rng.choice(list(border_nodes))
    return rng.choices(list(border_nodes), weights=weights, k=1)[0]


def _build_hourly_patrol_positions(
    ctx: SimulationContext,
    patrol_plan: dict,
    seed: int,
    n_days: int = 365,
) -> Tuple[List[List[str]], List[List[str]]]:
    rng = random.Random(seed)
    patrols = patrol_plan.get("patrols", [])
    if not isinstance(patrols, list):
        return [], []

    n_hours = n_days * 24
    hourly_big: List[List[str]] = []
    hourly_nodes: List[List[str]] = []

    for p in patrols:
        base_big = str(p.get("base", ""))
        if base_big not in ctx.rep_node_by_big:
            continue

        assigned = p.get("assigned", [])
        way = p.get("way", [])

        key_bigs: List[str] = []
        for b in assigned:
            sb = str(b)
            if sb != base_big and sb in ctx.rep_node_by_big and sb not in key_bigs:
                key_bigs.append(sb)
        if not key_bigs:
            for b in way:
                sb = str(b)
                if sb != base_big and sb in ctx.rep_node_by_big and sb not in key_bigs:
                    key_bigs.append(sb)

        one_big = [base_big for _ in range(n_hours)]
        one_node = [ctx.rep_node_by_big[base_big] for _ in range(n_hours)]

        for day in range(n_days):
            day_start = day * 24
            if not key_bigs:
                continue

            # Randomized daily route over key points.
            shuffled = list(key_bigs)
            rng.shuffle(shuffled)
            shuffled = shuffled[:DAILY_KEYPOINT_LIMIT]

            route = [base_big]
            cur = base_big
            remain = set(shuffled)
            while remain:
                nxt = min(
                    remain,
                    key=lambda b: ctx.big_distance_m.get(cur, {}).get(b, float("inf")),
                )
                route.append(nxt)
                remain.remove(nxt)
                cur = nxt
            route.append(base_big)

            seg_durations_h: List[float] = []
            total_h = 0.0
            for a, b in zip(route, route[1:]):
                da = ctx.rep_node_by_big[a]
                db = ctx.rep_node_by_big[b]
                d_m = _safe_distance_m(da, db, ctx.node_xy, ctx.node_big, ctx.big_distance_m)
                if not math.isfinite(d_m):
                    d_m = 0.0
                dt_h = d_m / (RANGER_SPEED_KMH * 1000.0)
                seg_durations_h.append(dt_h)
                total_h += dt_h

            for hh in range(24):
                global_h = day_start + hh
                if total_h <= 1e-9:
                    one_big[global_h] = base_big
                    one_node[global_h] = ctx.rep_node_by_big[base_big]
                    continue

                # Business rule: patrol completes one route in each 12-hour shift
                # (two full route loops per day).
                shift_h = float(hh % PATROL_ACTIVE_HOURS_PER_DAY)
                t_h = (shift_h / float(PATROL_ACTIVE_HOURS_PER_DAY)) * total_h

                acc = 0.0
                chosen_big = base_big
                for seg_i, dt_h in enumerate(seg_durations_h):
                    if t_h <= acc + dt_h or seg_i == len(seg_durations_h) - 1:
                        a = route[seg_i]
                        b = route[seg_i + 1]
                        # Approximate current block by nearest segment endpoint.
                        if dt_h <= 1e-9:
                            chosen_big = b
                        else:
                            frac = max(0.0, min(1.0, (t_h - acc) / dt_h))
                            chosen_big = a if frac < 0.5 else b
                        break
                    acc += dt_h

                one_big[global_h] = chosen_big
                one_node[global_h] = ctx.rep_node_by_big.get(chosen_big, ctx.rep_node_by_big[base_big])

        hourly_big.append(one_big)
        hourly_nodes.append(one_node)

    return hourly_big, hourly_nodes


def _nearest_ranger_distance_m_at_hour(
    hour: int,
    node_id: str,
    hourly_patrol_nodes: List[List[str]],
    ctx: SimulationContext,
) -> float:
    if not hourly_patrol_nodes:
        return float("inf")

    best = float("inf")
    for route in hourly_patrol_nodes:
        if hour < 0 or hour >= len(route):
            continue
        ranger_node = route[hour]
        if ranger_node not in ctx.node_xy or node_id not in ctx.node_xy:
            continue
        d = _euclid_m(ctx.node_xy[ranger_node], ctx.node_xy[node_id])
        if d < best:
            best = d
    return best


def _nearest_response_time_h_at_hour(
    hour: int,
    node_id: str,
    hourly_patrol_nodes: List[List[str]],
    ctx: SimulationContext,
) -> float:
    if not hourly_patrol_nodes:
        return float("inf")

    best_m = float("inf")
    for route in hourly_patrol_nodes:
        if hour < 0 or hour >= len(route):
            continue
        ranger_node = route[hour]
        d_m = _safe_distance_m(ranger_node, node_id, ctx.node_xy, ctx.node_big, ctx.big_distance_m)
        if d_m < best_m:
            best_m = d_m

    if not math.isfinite(best_m):
        return float("inf")
    return best_m / (RANGER_SPEED_KMH * 1000.0)


def _photo_near(node_id: str, ctx: SimulationContext, cache: Dict[str, bool]) -> bool:
    if node_id in cache:
        return cache[node_id]

    xy = ctx.node_xy.get(node_id)
    if xy is None:
        cache[node_id] = False
        return False

    for pn in ctx.photo_nodes:
        pxy = ctx.node_xy.get(pn)
        if pxy is None:
            continue
        if _euclid_m(xy, pxy) <= PHOTO_DETECT_RADIUS_M:
            cache[node_id] = True
            return True

    cache[node_id] = False
    return False


def _loss_from_detection(
    event_type: str,
    t_det_h: float,
    t_hab_h: float,
    detect_node: str,
    hourly_patrol_nodes: List[List[str]],
    ctx: SimulationContext,
    max_event_end_h: float,
) -> float:
    t_resp_h = _nearest_response_time_h_at_hour(int(t_det_h), detect_node, hourly_patrol_nodes, ctx)
    if not math.isfinite(t_resp_h):
        return {
            "OP": MAX_LOSS_OP,
            "IH": MAX_LOSS_IH,
            "SN": MAX_LOSS_SN,
            "TD": MAX_LOSS_TD,
        }[event_type]

    if t_resp_h > T_SLA_H:
        return {
            "OP": MAX_LOSS_OP,
            "IH": MAX_LOSS_IH,
            "SN": MAX_LOSS_SN,
            "TD": MAX_LOSS_TD,
        }[event_type]

    t_arr_h = t_det_h + t_resp_h
    if t_arr_h >= max_event_end_h:
        return {
            "OP": MAX_LOSS_OP,
            "IH": MAX_LOSS_IH,
            "SN": MAX_LOSS_SN,
            "TD": MAX_LOSS_TD,
        }[event_type]

    if event_type == "OP":
        exposed = max(0.0, t_arr_h - t_hab_h)
        d = min(1.0, exposed / max(1e-9, T_KILL_OP_H))
        return min(MAX_LOSS_OP, C_OP * d)
    if event_type == "IH":
        exposed = max(0.0, t_arr_h - t_hab_h)
        d = min(1.0, exposed / max(1e-9, T_KILL_IH_H))
        return min(MAX_LOSS_IH, C_IH * d)
    if event_type == "SN":
        # Business rule from user: saved if detected before T (24h).
        if (t_arr_h - t_hab_h) <= T_KILL_SN_H:
            return 0.0
        return MAX_LOSS_SN

    exposed = max(0.0, t_arr_h - t_hab_h)
    growth_steps = int(max(0.0, math.floor(exposed / 24.0)))
    growth = TD_DAILY_GROWTH_MULT ** growth_steps
    d = min(1.0, (exposed / max(1e-9, T_KILL_TD_H)) * growth)
    return min(MAX_LOSS_TD, C_TD * d)


def simulate_period(
    ctx: SimulationContext,
    patrol_plan: dict,
    sound_selection: Optional[dict],
    gps_enabled: bool,
    seed: int,
    days: int = 365,
) -> dict:
    if days <= 0:
        raise ValueError("days must be > 0")
    rng = random.Random(seed)
    hourly_patrol_big, hourly_patrol_nodes = _build_hourly_patrol_positions(
        ctx=ctx,
        patrol_plan=patrol_plan,
        seed=seed,
        n_days=days,
    )

    sound_nodes: Set[str] = set()
    if isinstance(sound_selection, dict):
        for row in sound_selection.get("selected_border_cells", []):
            cid = row.get("cell_id") if isinstance(row, dict) else None
            if isinstance(cid, str):
                sound_nodes.add(cid)

    total_loss = 0.0
    loss_by_type = {k: 0.0 for k in RISK_TYPES}
    events_by_type = {k: 0 for k in RISK_TYPES}
    detected_by_type = {k: 0 for k in RISK_TYPES}
    undetected_by_type = {k: 0 for k in RISK_TYPES}

    photo_cache: Dict[str, bool] = {}
    active_events: List[dict] = []

    n_hours = days * 24
    for hour in range(n_hours):
        season = _season_for_hour(hour)
        e_type = _sample_event_type(rng, season)

        if e_type in ("OP", "IH"):
            spawn_mode = "border" if rng.random() < 0.5 else "road"
            if spawn_mode == "border":
                spawn_pool = ctx.border_nodes if ctx.border_nodes else ctx.animal_nodes
            else:
                spawn_pool = ctx.road_nodes if ctx.road_nodes else ctx.animal_nodes

            if not spawn_pool:
                spawn_pool = tuple(ctx.node_xy.keys())

            if spawn_mode == "border" and ctx.border_nodes:
                # Border spawn probability is quadratic by cell priority: higher priority -> much higher chance.
                spawn_node = _sample_border_node_priority_quadratic(
                    rng=rng,
                    border_nodes=spawn_pool,
                    node_priority=ctx.node_priority,
                )
            else:
                spawn_node = rng.choice(spawn_pool)
            target_node = _pick_target_node(rng, ctx, spawn_node)

            d_to_target_m = _safe_distance_m(spawn_node, target_node, ctx.node_xy, ctx.node_big, ctx.big_distance_m)
            if not math.isfinite(d_to_target_m):
                d_to_target_m = 0.0
            t_to_hab_h = d_to_target_m / (INTRUDER_SPEED_KMH * 1000.0)
            t_hab_h = hour + t_to_hab_h

            ev = {
                "type": e_type,
                "start_h": float(hour),
                "spawn_node": spawn_node,
                "target_node": target_node,
                "spawn_mode": spawn_mode,
                "t_hab_h": float(t_hab_h),
                "end_h": float(hour + OP_IH_MAX_DURATION_H),
            }

            auto_detect = False
            detect_node = spawn_node
            if spawn_mode == "road" and gps_enabled:
                auto_detect = True
            elif spawn_mode == "border" and spawn_node in sound_nodes:
                auto_detect = True
            elif _nearest_ranger_distance_m_at_hour(hour, spawn_node, hourly_patrol_nodes, ctx) <= RANGER_DETECT_RADIUS_M:
                auto_detect = True
            elif _photo_near(spawn_node, ctx, photo_cache):
                auto_detect = True

            if auto_detect:
                loss = _loss_from_detection(
                    event_type=e_type,
                    t_det_h=float(hour),
                    t_hab_h=float(t_hab_h),
                    detect_node=detect_node,
                    hourly_patrol_nodes=hourly_patrol_nodes,
                    ctx=ctx,
                    max_event_end_h=float(hour + OP_IH_MAX_DURATION_H),
                )
                total_loss += loss
                loss_by_type[e_type] += loss
                events_by_type[e_type] += 1
                detected_by_type[e_type] += 1
            else:
                active_events.append(ev)

        elif e_type == "SN":
            pool = ctx.animal_nodes if ctx.animal_nodes else tuple(ctx.node_xy.keys())
            if not pool:
                continue
            node = rng.choice(pool)
            ev = {
                "type": "SN",
                "start_h": float(hour),
                "node": node,
                "end_h": float(hour + SN_MAX_DURATION_H),
            }
            if _nearest_ranger_distance_m_at_hour(hour, node, hourly_patrol_nodes, ctx) <= RANGER_DETECT_RADIUS_M:
                loss = 0.0
                total_loss += loss
                loss_by_type["SN"] += loss
                events_by_type["SN"] += 1
                detected_by_type["SN"] += 1
            else:
                active_events.append(ev)

        else:  # TD
            pool = ctx.border_animal_nodes if ctx.border_animal_nodes else ctx.border_nodes
            if not pool:
                pool = ctx.animal_nodes if ctx.animal_nodes else tuple(ctx.node_xy.keys())
            if not pool:
                continue
            node = rng.choice(pool)
            ev = {
                "type": "TD",
                "start_h": float(hour),
                "node": node,
                "end_h": float(min(n_hours, hour + TD_MAX_DURATION_H)),
            }
            if _nearest_ranger_distance_m_at_hour(hour, node, hourly_patrol_nodes, ctx) <= RANGER_STOP_RADIUS_M:
                loss = _loss_from_detection(
                    event_type="TD",
                    t_det_h=float(hour),
                    t_hab_h=float(hour),
                    detect_node=node,
                    hourly_patrol_nodes=hourly_patrol_nodes,
                    ctx=ctx,
                    max_event_end_h=float(min(n_hours, hour + TD_MAX_DURATION_H)),
                )
                total_loss += loss
                loss_by_type["TD"] += loss
                events_by_type["TD"] += 1
                detected_by_type["TD"] += 1
            else:
                active_events.append(ev)

        next_events: List[dict] = []
        for ev in active_events:
            et = ev["type"]
            start_h = float(ev["start_h"])
            end_h = float(ev["end_h"])

            if et in ("OP", "IH"):
                cur_node = ev["target_node"] if float(hour) >= ev["t_hab_h"] else ev["spawn_node"]
                detected = False
                if _nearest_ranger_distance_m_at_hour(hour, cur_node, hourly_patrol_nodes, ctx) <= RANGER_DETECT_RADIUS_M:
                    detected = True
                elif _photo_near(cur_node, ctx, photo_cache):
                    detected = True

                if detected:
                    loss = _loss_from_detection(
                        event_type=et,
                        t_det_h=float(hour),
                        t_hab_h=float(ev["t_hab_h"]),
                        detect_node=cur_node,
                        hourly_patrol_nodes=hourly_patrol_nodes,
                        ctx=ctx,
                        max_event_end_h=end_h,
                    )
                    total_loss += loss
                    loss_by_type[et] += loss
                    events_by_type[et] += 1
                    detected_by_type[et] += 1
                    continue

                if float(hour + 1) >= end_h:
                    loss = MAX_LOSS_OP if et == "OP" else MAX_LOSS_IH
                    total_loss += loss
                    loss_by_type[et] += loss
                    events_by_type[et] += 1
                    undetected_by_type[et] += 1
                    continue

                next_events.append(ev)
                continue

            if et == "SN":
                node = ev["node"]
                if _nearest_ranger_distance_m_at_hour(hour, node, hourly_patrol_nodes, ctx) <= RANGER_DETECT_RADIUS_M:
                    total_loss += 0.0
                    loss_by_type["SN"] += 0.0
                    events_by_type["SN"] += 1
                    detected_by_type["SN"] += 1
                    continue

                if float(hour + 1) >= end_h:
                    loss = MAX_LOSS_SN
                    total_loss += loss
                    loss_by_type["SN"] += loss
                    events_by_type["SN"] += 1
                    undetected_by_type["SN"] += 1
                    continue

                next_events.append(ev)
                continue

            # TD
            node = ev["node"]
            if _nearest_ranger_distance_m_at_hour(hour, node, hourly_patrol_nodes, ctx) <= RANGER_STOP_RADIUS_M:
                loss = _loss_from_detection(
                    event_type="TD",
                    t_det_h=float(hour),
                    t_hab_h=start_h,
                    detect_node=node,
                    hourly_patrol_nodes=hourly_patrol_nodes,
                    ctx=ctx,
                    max_event_end_h=end_h,
                )
                total_loss += loss
                loss_by_type["TD"] += loss
                events_by_type["TD"] += 1
                detected_by_type["TD"] += 1
                continue

            if float(hour + 1) >= end_h:
                loss = MAX_LOSS_TD
                total_loss += loss
                loss_by_type["TD"] += loss
                events_by_type["TD"] += 1
                undetected_by_type["TD"] += 1
                continue

            next_events.append(ev)

        active_events = next_events

    # Force-close any leftovers at year end.
    for ev in active_events:
        et = ev["type"]
        if et == "OP":
            loss = MAX_LOSS_OP
        elif et == "IH":
            loss = MAX_LOSS_IH
        elif et == "SN":
            loss = MAX_LOSS_SN
        else:
            loss = MAX_LOSS_TD
        total_loss += loss
        loss_by_type[et] += loss
        events_by_type[et] += 1
        undetected_by_type[et] += 1

    return {
        "period_days": int(days),
        "period_hours": int(n_hours),
        "total_loss": float(total_loss),
        "loss_by_risk": {k: float(v) for k, v in loss_by_type.items()},
        "annual_total_loss": float(total_loss),
        "annual_loss_by_risk": {k: float(v) for k, v in loss_by_type.items()},
        "events_by_risk": events_by_type,
        "detected_by_risk": detected_by_type,
        "undetected_by_risk": undetected_by_type,
        "assumptions": {
            "start_date": "2025-01-01",
            "hours": n_hours,
            "days": int(days),
            "one_event_per_hour": True,
            "gps_auto_detect_road_entry": bool(gps_enabled),
            "op_ih_spawn_border_vs_road": "50/50",
            "op_ih_border_spawn_weighting": "priority^2 on derived park boundary cells (inside + outside 4-neighbor)",
            "ranger_detection_radius_m": RANGER_DETECT_RADIUS_M,
            "photo_detection_radius_m": PHOTO_DETECT_RADIUS_M,
            "ranger_speed_kmh": RANGER_SPEED_KMH,
            "intruder_speed_kmh": INTRUDER_SPEED_KMH,
            "sla_h": T_SLA_H,
            "patrol_shift_hours": PATROL_ACTIVE_HOURS_PER_DAY,
            "patrol_loops_per_day": PATROL_LOOPS_PER_DAY,
            "patrol_loop_policy": "one full route per 12-hour shift (2 loops per day)",
        },
    }


def simulate_year(
    ctx: SimulationContext,
    patrol_plan: dict,
    sound_selection: Optional[dict],
    gps_enabled: bool,
    seed: int,
) -> dict:
    return simulate_period(
        ctx=ctx,
        patrol_plan=patrol_plan,
        sound_selection=sound_selection,
        gps_enabled=gps_enabled,
        seed=seed,
        days=365,
    )
