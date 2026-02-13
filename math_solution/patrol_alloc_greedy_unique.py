import argparse
import heapq
import json
import math
import os
import random
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import select_sound_border_cells as sound_select
except ModuleNotFoundError:
    try:
        from math_solution import select_sound_border_cells as sound_select
    except ModuleNotFoundError:
        sound_select = None


INF = float("inf")
EPS = 1e-12
SIDES = ("N", "S", "E", "W")
OVERLAP_ALPHA = 1.0
OVERLAP_HARD_MAX_USE = 1
DEFAULT_SPEED_KMH = 40.0
RANDOM_TIE_LOW = 0.9
RANDOM_TIE_HIGH = 1.1
SCORE_DENOM_EPS = 1e-12
PROGRESS_PRINT_STEP = 32
SEED_MUL_A = 1000003
SEED_MUL_B = 911382323

DEFAULT_DIST_PATH = "solution/big_dist_with_portals_time_priority.json"
DEFAULT_PATROL_PATH = "solution/patrol_house_to_big_cell.json"
DEFAULT_BIGGRAPH_PATH = "solution/etosha_big_square_graph_14x14.json"
DEFAULT_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_KMIN = 1
DEFAULT_KMAX = 10
DEFAULT_TLIM_H = 12.0
DEFAULT_TOPL = None
DEFAULT_SEED = 1
DEFAULT_SCORE_GAIN_POW = 1.35
DEFAULT_SCORE_TIME_POW = 0.75
DEFAULT_OUT_PATH = "solution/patrol_alloc_greedy_unique_result.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def gen_compositions(K: int, H: int = 8) -> Iterable[List[int]]:
    if H <= 0:
        return
    if H == 1:
        yield [K]
        return
    for x in range(K + 1):
        for tail in gen_compositions(K - x, H - 1):
            yield [x] + tail


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


def _resolve_write_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if path.startswith("solution/"):
        return os.path.join(PROJECT_DIR, path)
    if path.startswith("math_solution/"):
        return os.path.join(PROJECT_DIR, path)
    return os.path.abspath(path)


def _load_json(path: str) -> dict:
    resolved = _resolve_read_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return INF
    if not math.isfinite(x):
        return INF
    return x


def _safe_priority(v) -> float:
    x = _safe_float(v)
    if not math.isfinite(x):
        return 0.0
    return x


def _parse_big_id(bid: str) -> Tuple[int, int]:
    if not isinstance(bid, str) or not bid.startswith("br") or "_bc" not in bid:
        raise ValueError(f"Bad big-square id format: {bid}")
    left, right = bid.split("_bc", 1)
    return int(left[2:]), int(right)


def _opposite(side: str) -> str:
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[side]


def _side_out_from_to(u: str, v: str, id_to_rc: Dict[str, Tuple[int, int]]) -> Optional[str]:
    ur, uc = id_to_rc[u]
    vr, vc = id_to_rc[v]
    dr = vr - ur
    dc = vc - uc
    if dr == 0 and dc == 1:
        return "E"
    if dr == 0 and dc == -1:
        return "W"
    if dr == 1 and dc == 0:
        return "S"
    if dr == -1 and dc == 0:
        return "N"
    return None


def _pair_better(t1: float, p1: float, t2: float, p2: float) -> bool:
    if t1 < t2 - EPS:
        return True
    if abs(t1 - t2) <= EPS and p1 > p2 + EPS:
        return True
    return False


def load_time_matrix(
    dist_path: str,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], List[str]]:
    payload = _load_json(dist_path)
    dist = payload.get("dist")
    if not isinstance(dist, dict):
        raise ValueError(f"Invalid dist format in: {dist_path}")

    time_h: Dict[str, Dict[str, float]] = {}
    pr_h: Dict[str, Dict[str, float]] = {}
    for a, row in dist.items():
        if not isinstance(row, dict):
            continue
        inner_t: Dict[str, float] = {}
        inner_p: Dict[str, float] = {}
        for b, feat in row.items():
            if isinstance(feat, dict):
                inner_t[b] = _safe_float(feat.get("time_h"))
                inner_p[b] = _safe_priority(feat.get("small_priority_sum"))
            else:
                inner_t[b] = INF
                inner_p[b] = 0.0
        time_h[a] = inner_t
        pr_h[a] = inner_p
    return time_h, pr_h, list(dist.keys())


def _transition_key(block: str, side_a: Optional[str], side_b: Optional[str]) -> Optional[Tuple[str, str, str]]:
    if side_a not in SIDES or side_b not in SIDES:
        return None
    if side_a <= side_b:
        return (block, side_a, side_b)
    return (block, side_b, side_a)


def _merge_ops(
    ops: Dict[Tuple[str, str, str], int],
    delta_map: Dict[Tuple[str, str, str], int],
    sign: int,
) -> None:
    for k, v in delta_map.items():
        nv = ops.get(k, 0) + sign * int(v)
        if nv == 0:
            ops.pop(k, None)
        else:
            ops[k] = nv


def load_side_transition_model(
    dist_path: str,
    biggraph_path: str,
) -> Tuple[dict, List[str]]:
    dist_payload = _load_json(dist_path)
    dist = dist_payload.get("dist")
    inside = dist_payload.get("inside_best_by_block_side")
    if not isinstance(dist, dict):
        raise ValueError(f"Invalid dist format in: {dist_path}")
    if not isinstance(inside, dict):
        raise ValueError(
            f"Missing inside_best_by_block_side in: {dist_path}. "
            f"Rebuild dist by running solution/build_big_dist_with_portals.py"
        )

    meta = dist_payload.get("meta", {}) if isinstance(dist_payload, dict) else {}
    speed_kmh = _safe_float(meta.get("speed_kmh_for_time"))
    if not math.isfinite(speed_kmh) or speed_kmh <= 0.0:
        speed_kmh = DEFAULT_SPEED_KMH
    speed_m_per_h = speed_kmh * 1000.0

    biggraph = _load_json(biggraph_path)
    node_features = biggraph.get("node_features")
    edge_features = biggraph.get("edge_features")
    if not isinstance(node_features, dict) or not isinstance(edge_features, dict):
        raise ValueError(f"Invalid biggraph format in: {biggraph_path}")

    big_ids = sorted([bid for bid in dist.keys() if bid in node_features])
    if not big_ids:
        raise ValueError("No common big ids between dist and biggraph")

    id_to_rc = {bid: _parse_big_id(bid) for bid in big_ids}

    inside_t: Dict[str, Dict[str, Dict[str, float]]] = {
        b: {si: {so: INF for so in SIDES} for si in SIDES} for b in big_ids
    }
    inside_p: Dict[str, Dict[str, Dict[str, float]]] = {
        b: {si: {so: 0.0 for so in SIDES} for si in SIDES} for b in big_ids
    }

    for b in big_ids:
        b_map = inside.get(b, {})
        if not isinstance(b_map, dict):
            continue
        for si in SIDES:
            row = b_map.get(si, {})
            if not isinstance(row, dict):
                continue
            for so in SIDES:
                rec = row.get(so, {})
                if isinstance(rec, dict):
                    inside_t[b][si][so] = _safe_float(rec.get("time_h"))
                    inside_p[b][si][so] = _safe_priority(rec.get("small_priority_sum"))

    adj: Dict[str, List[Tuple[str, str, float]]] = {b: [] for b in big_ids}
    for u in big_ids:
        row = edge_features.get(u, {})
        if not isinstance(row, dict):
            continue
        for v, ef in row.items():
            if v not in id_to_rc or not isinstance(ef, dict):
                continue
            if ef.get("is_diagonal"):
                continue
            side_out = _side_out_from_to(u, v, id_to_rc)
            if side_out is None:
                continue
            d_m = _safe_float(ef.get("distance_m"))
            if not math.isfinite(d_m):
                continue
            t_h = d_m / speed_m_per_h
            adj[u].append((v, side_out, t_h))

    leg_time: Dict[str, Dict[str, float]] = {a: {b: INF for b in big_ids} for a in big_ids}
    leg_pr: Dict[str, Dict[str, float]] = {a: {b: 0.0 for b in big_ids} for a in big_ids}
    leg_end_side: Dict[str, Dict[str, Optional[str]]] = {a: {b: None for b in big_ids} for a in big_ids}
    leg_first_side: Dict[str, Dict[str, Optional[str]]] = {a: {b: None for b in big_ids} for a in big_ids}
    leg_trans: Dict[str, Dict[str, Dict[Tuple[str, str, str], int]]] = {
        a: {b: {} for b in big_ids} for a in big_ids
    }

    for src_idx, src in enumerate(big_ids, start=1):
        state_best: Dict[Tuple[str, str], Tuple[float, float]] = {}
        state_parent: Dict[Tuple[str, str], Optional[Tuple[Tuple[str, str], str]]] = {}
        state_first_out: Dict[Tuple[str, str], str] = {}
        pq: List[Tuple[float, float, str, str]] = []

        for nxt, side_out, edge_t in adj.get(src, []):
            key = (nxt, _opposite(side_out))
            cand_t = edge_t
            cand_p = 0.0
            old = state_best.get(key, (INF, -INF))
            if _pair_better(cand_t, cand_p, old[0], old[1]):
                state_best[key] = (cand_t, cand_p)
                state_parent[key] = None
                state_first_out[key] = side_out
                heapq.heappush(pq, (cand_t, -cand_p, nxt, _opposite(side_out)))

        node_best_pair: Dict[str, Tuple[float, float]] = {src: (0.0, 0.0)}
        node_best_state: Dict[str, Optional[Tuple[str, str]]] = {src: None}

        while pq:
            cur_t, neg_cur_p, u, entry_side = heapq.heappop(pq)
            cur_p = -neg_cur_p
            cur_state = (u, entry_side)
            best_pair = state_best.get(cur_state, (INF, -INF))
            if abs(cur_t - best_pair[0]) > EPS or abs(cur_p - best_pair[1]) > EPS:
                continue

            old_node = node_best_pair.get(u, (INF, -INF))
            if _pair_better(cur_t, cur_p, old_node[0], old_node[1]):
                node_best_pair[u] = (cur_t, cur_p)
                node_best_state[u] = cur_state

            for v, side_out, edge_t in adj.get(u, []):
                in_t = inside_t[u][entry_side][side_out]
                if not math.isfinite(in_t):
                    continue
                in_p = inside_p[u][entry_side][side_out]

                cand_t = cur_t + in_t + edge_t
                cand_p = cur_p + in_p
                nxt_state = (v, _opposite(side_out))
                old = state_best.get(nxt_state, (INF, -INF))
                if _pair_better(cand_t, cand_p, old[0], old[1]):
                    state_best[nxt_state] = (cand_t, cand_p)
                    state_parent[nxt_state] = (cur_state, side_out)
                    state_first_out[nxt_state] = state_first_out[cur_state]
                    heapq.heappush(pq, (cand_t, -cand_p, v, _opposite(side_out)))

        for dst in big_ids:
            if src == dst:
                leg_time[src][dst] = 0.0
                leg_pr[src][dst] = 0.0
                leg_end_side[src][dst] = None
                leg_first_side[src][dst] = None
                leg_trans[src][dst] = {}
                continue

            end_state = node_best_state.get(dst)
            best = node_best_pair.get(dst)
            if end_state is None or best is None or not math.isfinite(best[0]):
                leg_time[src][dst] = INF
                leg_pr[src][dst] = 0.0
                leg_end_side[src][dst] = None
                leg_first_side[src][dst] = None
                leg_trans[src][dst] = {}
                continue

            leg_time[src][dst] = best[0]
            leg_pr[src][dst] = max(0.0, best[1])
            leg_end_side[src][dst] = end_state[1]
            leg_first_side[src][dst] = state_first_out.get(end_state)

            trans_counter: Dict[Tuple[str, str, str], int] = {}
            cur = end_state
            seen: Set[Tuple[str, str]] = set()
            while cur is not None:
                if cur in seen:
                    trans_counter = {}
                    break
                seen.add(cur)
                parent_rec = state_parent.get(cur)
                if parent_rec is None:
                    break
                prev_state, side_out = parent_rec
                blk = prev_state[0]
                side_in = prev_state[1]
                tk = _transition_key(blk, side_in, side_out)
                if tk is not None:
                    trans_counter[tk] = trans_counter.get(tk, 0) + 1
                cur = prev_state
            leg_trans[src][dst] = trans_counter

        if src_idx % PROGRESS_PRINT_STEP == 0 or src_idx == len(big_ids):
            print(f"[side_model] {src_idx}/{len(big_ids)} sources prepared")

    model = {
        "big_ids": big_ids,
        "inside_t": inside_t,
        "inside_p": inside_p,
        "leg_time": leg_time,
        "leg_pr": leg_pr,
        "leg_end_side": leg_end_side,
        "leg_first_side": leg_first_side,
        "leg_trans": leg_trans,
    }
    return model, big_ids


def load_houses_from_patrol_json(patrol_path: str) -> List[str]:
    payload = _load_json(patrol_path)
    patrol_houses = payload.get("patrol_houses")
    if not isinstance(patrol_houses, list):
        raise ValueError(f"Invalid patrol format in: {patrol_path}")

    houses_big_ids: List[str] = []
    for idx, row in enumerate(patrol_houses):
        to_big = row.get("to_big") if isinstance(row, dict) else None
        if not isinstance(to_big, dict) or not to_big:
            raise ValueError(f"Invalid patrol_houses[{idx}].to_big in: {patrol_path}")

        best_bid = None
        best_t = INF
        for bid, feat in to_big.items():
            t = _safe_float(feat.get("time_h") if isinstance(feat, dict) else None)
            if t < best_t:
                best_t = t
                best_bid = bid
        if best_bid is None or not math.isfinite(best_t):
            raise ValueError(f"No finite base candidate for patrol_houses[{idx}] in: {patrol_path}")
        houses_big_ids.append(best_bid)

    if len(houses_big_ids) != 8:
        raise ValueError(f"Expected 8 patrol houses, got {len(houses_big_ids)} in: {patrol_path}")
    return houses_big_ids


def build_S_big(biggraph_path: str, priority_path: str) -> Dict[str, float]:
    biggraph = _load_json(biggraph_path)
    node_features = biggraph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError(f"Invalid biggraph format in: {biggraph_path}")

    priority: Dict[str, float]
    resolved_priority_path: Optional[str] = None
    try:
        resolved_priority_path = _resolve_read_path(priority_path)
    except FileNotFoundError:
        resolved_priority_path = None

    if resolved_priority_path is not None:
        payload = _load_json(resolved_priority_path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid priority format in: {priority_path}")
        priority = {}
        for k, v in payload.items():
            p = _safe_float(v)
            priority[k] = p if math.isfinite(p) else 0.0
    else:
        priority = {}

    S: Dict[str, float] = {}
    for big_id, feat in node_features.items():
        contained = feat.get("contained_small_node_ids") if isinstance(feat, dict) else None
        if not isinstance(contained, list):
            contained = []
        s = 0.0
        for sid in contained:
            p = priority.get(sid, 0.0)
            if math.isfinite(p):
                s += p
        S[big_id] = s
    return S


def _is_corner_pair(side_in: str, side_out: str) -> bool:
    if side_in == side_out:
        return False
    return _opposite(side_in) != side_out


def _choose_visit_inside(
    block: str,
    side_in: Optional[str],
    side_out: Optional[str],
    model: dict,
    force_return: bool,
) -> Tuple[float, float, Optional[Tuple[str, str]]]:
    inside_t = model["inside_t"]
    inside_p = model["inside_p"]

    options: List[Tuple[str, str]] = []
    if force_return:
        if side_in in SIDES:
            options = [(side_in, side_in)]
        else:
            options = [(s, s) for s in SIDES]
    else:
        if side_in in SIDES and side_out in SIDES:
            options.append((side_in, side_out))
            if _is_corner_pair(side_in, side_out):
                options.append((side_out, side_in))
        elif side_in in SIDES:
            options.extend((side_in, s) for s in SIDES)
        elif side_out in SIDES:
            options.extend((s, side_out) for s in SIDES)
        else:
            for si in SIDES:
                for so in SIDES:
                    options.append((si, so))

    seen = set()
    uniq: List[Tuple[str, str]] = []
    for p in options:
        if p not in seen:
            seen.add(p)
            uniq.append(p)

    best_pair: Optional[Tuple[str, str]] = None
    best_t = INF
    best_p = 0.0
    best_ratio = -INF

    for si, so in uniq:
        t = inside_t.get(block, {}).get(si, {}).get(so, INF)
        p = inside_p.get(block, {}).get(si, {}).get(so, 0.0)
        if not math.isfinite(t):
            continue
        if t <= EPS:
            ratio = 1e30 if p > 0.0 else 0.0
        else:
            ratio = p / t
        if (
            ratio > best_ratio + EPS
            or (abs(ratio - best_ratio) <= EPS and t < best_t - EPS)
            or (abs(ratio - best_ratio) <= EPS and abs(t - best_t) <= EPS and p > best_p + EPS)
        ):
            best_ratio = ratio
            best_t = t
            best_p = p
            best_pair = (si, so)

    if best_pair is None:
        return INF, 0.0, None
    return best_t, max(0.0, best_p), best_pair


def _insert_metrics(
    u: str,
    v: str,
    x: str,
    model: dict,
) -> Tuple[float, float, Optional[Tuple[str, str, str]]]:
    leg_time = model["leg_time"]
    leg_pr = model["leg_pr"]
    leg_end_side = model["leg_end_side"]
    leg_first_side = model["leg_first_side"]

    t_ux = leg_time.get(u, {}).get(x, INF)
    t_xv = leg_time.get(x, {}).get(v, INF)
    t_uv = leg_time.get(u, {}).get(v, INF)
    if not (math.isfinite(t_ux) and math.isfinite(t_xv) and math.isfinite(t_uv)):
        return INF, 0.0, None

    p_ux = leg_pr.get(u, {}).get(x, 0.0)
    p_xv = leg_pr.get(x, {}).get(v, 0.0)
    p_uv = leg_pr.get(u, {}).get(v, 0.0)

    in_side = leg_end_side.get(u, {}).get(x)
    out_side = leg_first_side.get(x, {}).get(v)

    # For A->x->A style detours, force internal pass side_in->side_in in target block.
    visit_t, visit_p, visit_pair = _choose_visit_inside(
        block=x,
        side_in=in_side,
        side_out=out_side,
        model=model,
        force_return=(u == v),
    )
    if not math.isfinite(visit_t):
        return INF, 0.0, None

    dt = t_ux + t_xv - t_uv + visit_t
    dp = p_ux + p_xv - p_uv + visit_p
    if dt < 0.0:
        dt = 0.0
    if dp < 0.0:
        dp = 0.0

    visit_key = None
    if visit_pair is not None:
        visit_key = _transition_key(x, visit_pair[0], visit_pair[1])

    return dt, dp, visit_key


def dt_insert(u: str, v: str, x: str, model: dict) -> float:
    d, _, _ = _insert_metrics(u, v, x, model)
    return d


def _transition_ops_for_insert(
    u: str,
    v: str,
    x: str,
    model: dict,
    visit_key: Optional[Tuple[str, str, str]],
) -> Dict[Tuple[str, str, str], int]:
    leg_trans = model["leg_trans"]
    ops: Dict[Tuple[str, str, str], int] = {}
    _merge_ops(ops, leg_trans.get(u, {}).get(x, {}), +1)
    _merge_ops(ops, leg_trans.get(x, {}).get(v, {}), +1)
    _merge_ops(ops, leg_trans.get(u, {}).get(v, {}), -1)
    if visit_key is not None:
        ops[visit_key] = ops.get(visit_key, 0) + 1
        if ops[visit_key] == 0:
            ops.pop(visit_key, None)
    return ops


def _overlap_units(cnt: int) -> int:
    return cnt - 1 if cnt > 1 else 0


def _overlap_delta_from_ops(
    use_global: Dict[Tuple[str, str, str], int],
    ops: Dict[Tuple[str, str, str], int],
) -> int:
    delta = 0
    for e, dd in ops.items():
        c0 = use_global.get(e, 0)
        c1 = c0 + dd
        if c1 < 0:
            c1 = 0
        delta += _overlap_units(c1) - _overlap_units(c0)
    return delta


def _apply_ops(use_global: Dict[Tuple[str, str, str], int], ops: Dict[Tuple[str, str, str], int]) -> None:
    for e, dd in ops.items():
        c = use_global.get(e, 0) + dd
        if c <= 0:
            use_global.pop(e, None)
        else:
            use_global[e] = c


def _violates_hard_overlap_cap(
    use_global: Dict[Tuple[str, str, str], int],
    ops: Dict[Tuple[str, str, str], int],
    max_use: int,
) -> bool:
    if max_use <= 0:
        return False
    for e, dd in ops.items():
        c1 = use_global.get(e, 0) + dd
        if c1 > max_use:
            return True
    return False


class Patrol:
    def __init__(self, base: str, model: dict):
        self.base = base
        self.model = model
        self.way: List[str] = [base, base]
        self.t_used: float = 0.0
        self.assigned: List[str] = []
        self.best_dt: Dict[str, float] = {}
        self.best_pos: Dict[str, int] = {}
        self.edge_bucket: List[Set[str]] = [set()]

    def init_cache(self, cands: Iterable[str]) -> None:
        self.best_dt.clear()
        self.best_pos.clear()
        self.edge_bucket = [set()]
        for x in cands:
            d = dt_insert(self.base, self.base, x, self.model)
            self.best_dt[x] = d
            self.best_pos[x] = 0
            self.edge_bucket[0].add(x)

    def can_take(self, x: str, Tlim: float) -> bool:
        d = self.best_dt.get(x, INF)
        return math.isfinite(d) and (self.t_used + d <= Tlim)

    def remove_candidate(self, x: str) -> None:
        pos = self.best_pos.pop(x, None)
        self.best_dt.pop(x, None)
        if pos is not None and 0 <= pos < len(self.edge_bucket):
            self.edge_bucket[pos].discard(x)

    def _recompute_best_for_cell(self, x: str) -> Tuple[float, int]:
        best_d = INF
        best_j = -1
        for j in range(len(self.way) - 1):
            d = dt_insert(self.way[j], self.way[j + 1], x, self.model)
            if d < best_d:
                best_d = d
                best_j = j
        return best_d, best_j

    def insert_cell_and_update_cache(self, x: str) -> None:
        if x not in self.best_pos:
            raise ValueError(f"Cell {x} is not available in patrol cache")
        old_j = self.best_pos[x]
        if old_j < 0 or old_j >= len(self.way) - 1:
            raise ValueError(f"Invalid insertion position for {x}: {old_j}")

        d_add = self.best_dt.get(x, INF)
        if not math.isfinite(d_add):
            raise ValueError(f"Cannot insert {x}: infinite dt")

        u = self.way[old_j]
        v = self.way[old_j + 1]
        self.way.insert(old_j + 1, x)
        self.t_used += d_add
        self.assigned.append(x)

        self.remove_candidate(x)
        repair_set = set()
        if 0 <= old_j < len(self.edge_bucket):
            repair_set = set(self.edge_bucket[old_j])

        old_buckets = self.edge_bucket
        new_buckets: List[Set[str]] = [set() for _ in range(len(old_buckets) + 1)]
        for idx, bucket in enumerate(old_buckets):
            if idx < old_j:
                new_buckets[idx] = bucket
            elif idx > old_j:
                new_buckets[idx + 1] = bucket
                for y in bucket:
                    if y in self.best_pos:
                        self.best_pos[y] = idx + 1
        self.edge_bucket = new_buckets

        cur_cells = list(self.best_dt.keys())
        for y in cur_cells:
            d1 = dt_insert(u, x, y, self.model)
            d2 = dt_insert(x, v, y, self.model)
            cur_d = self.best_dt[y]

            upd = False
            new_d = cur_d
            new_j = self.best_pos[y]
            if d1 < new_d:
                new_d = d1
                new_j = old_j
                upd = True
            if d2 < new_d:
                new_d = d2
                new_j = old_j + 1
                upd = True

            if upd:
                old_pos = self.best_pos[y]
                if 0 <= old_pos < len(self.edge_bucket):
                    self.edge_bucket[old_pos].discard(y)
                self.best_dt[y] = new_d
                self.best_pos[y] = new_j
                if 0 <= new_j < len(self.edge_bucket):
                    self.edge_bucket[new_j].add(y)

        for y in repair_set:
            if y not in self.best_dt:
                continue
            old_pos = self.best_pos[y]
            if 0 <= old_pos < len(self.edge_bucket):
                self.edge_bucket[old_pos].discard(y)
            best_d, best_j = self._recompute_best_for_cell(y)
            self.best_dt[y] = best_d
            self.best_pos[y] = best_j
            if 0 <= best_j < len(self.edge_bucket):
                self.edge_bucket[best_j].add(y)


def _is_better(cur: Optional[dict], best: Optional[dict], eps: float = EPS) -> bool:
    if cur is None:
        return False
    if best is None:
        return True
    if cur["total_priority"] > best["total_priority"] + eps:
        return True
    if abs(cur["total_priority"] - best["total_priority"]) <= eps:
        return cur["total_time_h"] < best["total_time_h"] - eps
    return False


def _is_better_total_coverage(cur: Optional[dict], best: Optional[dict], eps: float = EPS) -> bool:
    if cur is None:
        return False
    if best is None:
        return True
    if cur["total_coverage_priority"] > best["total_coverage_priority"] + eps:
        return True
    if abs(cur["total_coverage_priority"] - best["total_coverage_priority"]) <= eps:
        if cur["total_priority"] > best["total_priority"] + eps:
            return True
        if abs(cur["total_priority"] - best["total_priority"]) <= eps:
            return cur["total_time_h"] < best["total_time_h"] - eps
    return False


def _sound_coverage_km(sound_budget: float, cost_per_km: float, border_km: float) -> float:
    if cost_per_km <= 0.0:
        raise ValueError("sound_tracker_cost_per_km must be > 0")
    if border_km < 0.0:
        raise ValueError("total_border_km must be >= 0")
    return min(border_km, max(0.0, sound_budget) / cost_per_km)


def solve_one_distribution(
    K: int,
    m: List[int],
    houses_big_ids: List[str],
    cands_sorted: List[str],
    S: Dict[str, float],
    model: dict,
    Tlim: float,
    score_gain_pow: float,
    score_time_pow: float,
    rng: random.Random,
) -> dict:
    if K <= 0:
        return {
            "K": K,
            "m": list(m),
            "total_priority": 0.0,
            "total_time_h": 0.0,
            "houses_big_ids": list(houses_big_ids),
            "patrols": [],
            "assigned_cells": [],
        }

    patrols: List[Patrol] = []
    for j, cnt in enumerate(m):
        base = houses_big_ids[j]
        for _ in range(cnt):
            patrols.append(Patrol(base, model))
    if len(patrols) != K:
        raise ValueError(f"Invalid composition for K={K}: m={m}")

    for p in patrols:
        p.init_cache(cands_sorted)

    trans_use_global: Dict[Tuple[str, str, str], int] = {}
    remaining = set(cands_sorted)
    assigned_cells: List[str] = []
    total_priority = 0.0

    for x in cands_sorted:
        if x not in remaining:
            continue

        best_idx = -1
        best_score = -INF
        best_gain = 0.0
        best_ops: Optional[Dict[Tuple[str, str, str], int]] = None

        for i, p in enumerate(patrols):
            if not p.can_take(x, Tlim):
                continue
            j = p.best_pos.get(x, -1)
            if j < 0 or j >= len(p.way) - 1:
                continue
            u = p.way[j]
            v = p.way[j + 1]

            d, gain, visit_key = _insert_metrics(u, v, x, model)
            if not math.isfinite(d) or d <= 0.0:
                continue
            if p.t_used + d > Tlim + EPS:
                continue
            if gain <= 0.0:
                continue

            ops = _transition_ops_for_insert(u, v, x, model, visit_key)
            if _violates_hard_overlap_cap(trans_use_global, ops, OVERLAP_HARD_MAX_USE):
                continue

            overlap_delta = _overlap_delta_from_ops(trans_use_global, ops)
            overlap_penalty = 1.0 + OVERLAP_ALPHA * max(0, overlap_delta)
            gain_term = gain if abs(score_gain_pow - 1.0) <= EPS else (gain ** score_gain_pow)
            time_term = d if abs(score_time_pow - 1.0) <= EPS else (d ** score_time_pow)
            denom = time_term if time_term > 0.0 else SCORE_DENOM_EPS
            score = (gain_term / denom) * rng.uniform(RANDOM_TIE_LOW, RANDOM_TIE_HIGH) / overlap_penalty
            if score > best_score:
                best_score = score
                best_idx = i
                best_gain = gain
                best_ops = ops

        if best_idx < 0:
            continue

        patrols[best_idx].insert_cell_and_update_cache(x)
        if best_ops is not None:
            _apply_ops(trans_use_global, best_ops)
        total_priority += best_gain
        assigned_cells.append(x)
        remaining.discard(x)
        for i, p in enumerate(patrols):
            if i != best_idx:
                p.remove_candidate(x)

    if len(assigned_cells) != len(set(assigned_cells)):
        raise AssertionError("assigned_cells must be unique")

    patrol_dicts = []
    total_time_h = 0.0
    for p in patrols:
        if not (len(p.way) >= 2 and p.way[0] == p.base and p.way[-1] == p.base):
            raise AssertionError("patrol.way must start and end at base")
        total_time_h += p.t_used
        patrol_dicts.append(
            {
                "base": p.base,
                "time_h": p.t_used,
                "way": list(p.way),
                "assigned": list(p.assigned),
            }
        )

    return {
        "K": K,
        "m": list(m),
        "total_priority": total_priority,
        "total_time_h": total_time_h,
        "houses_big_ids": list(houses_big_ids),
        "patrols": patrol_dicts,
        "assigned_cells": assigned_cells,
    }


def search_best(
    Kmin: int,
    Kmax: int,
    Tlim: float,
    houses_big_ids: List[str],
    cands_sorted: List[str],
    S: Dict[str, float],
    model: dict,
    seed: int,
    score_gain_pow: float,
    score_time_pow: float,
    budget_total_for_patrol_sound: Optional[float] = None,
    patrol_cost: Optional[float] = None,
    sound_tracker_cost_per_km: Optional[float] = None,
    total_border_km: Optional[float] = None,
    sound_graph_path: Optional[str] = None,
    sound_priority_path: Optional[str] = None,
    sound_score_field: str = "priority_P_i",
) -> Tuple[Optional[dict], List[dict], List[Optional[dict]]]:
    best_global: Optional[dict] = None
    per_K_best: List[dict] = []
    per_K_full: List[Optional[dict]] = []

    k_start = max(1, int(Kmin))
    if Kmax < k_start:
        return None, per_K_best, per_K_full

    sound_args = [
        budget_total_for_patrol_sound,
        patrol_cost,
        sound_tracker_cost_per_km,
        total_border_km,
        sound_graph_path,
        sound_priority_path,
    ]
    sound_enabled = all(x is not None for x in sound_args)
    if any(x is not None for x in sound_args) and not sound_enabled:
        raise ValueError("Sound budget integration requires all sound-related arguments")
    if sound_enabled and sound_select is None:
        raise ValueError("select_sound_border_cells module is unavailable")

    for K in range(k_start, Kmax + 1):
        best_k: Optional[dict] = None
        for comp_idx, m in enumerate(gen_compositions(K, 8)):
            local_seed = (seed * SEED_MUL_A) ^ (K * SEED_MUL_B) ^ comp_idx
            rng = random.Random(local_seed)
            res = solve_one_distribution(
                K=K,
                m=m,
                houses_big_ids=houses_big_ids,
                cands_sorted=cands_sorted,
                S=S,
                model=model,
                Tlim=Tlim,
                score_gain_pow=score_gain_pow,
                score_time_pow=score_time_pow,
                rng=rng,
            )
            if _is_better(res, best_k):
                best_k = res

        if best_k is None:
            base_row = {"K": K, "m": [0] * 8, "total_priority": 0.0, "total_time_h": 0.0}
            per_K_full.append(None)
        else:
            base_row = {
                "K": K,
                "m": list(best_k["m"]),
                "total_priority": float(best_k["total_priority"]),
                "total_time_h": float(best_k["total_time_h"]),
            }
            per_K_full.append(best_k)

        if sound_enabled:
            sound_budget = float(budget_total_for_patrol_sound) - float(K) * float(patrol_cost)
            sound_km = _sound_coverage_km(
                sound_budget=sound_budget,
                cost_per_km=float(sound_tracker_cost_per_km),
                border_km=float(total_border_km),
            )
            sound_result = sound_select.build_sound_selection(
                sound_km=sound_km,
                graph_path=str(sound_graph_path),
                priority_path=str(sound_priority_path),
                score_field=sound_score_field,
            )
            sound_priority = float(sound_result.get("meta", {}).get("selected_priority_sum", 0.0))
            row = dict(base_row)
            row["patrol_spent_total"] = float(K) * float(patrol_cost)
            row["sound_budget"] = sound_budget
            row["sound_covered_km"] = sound_km
            row["sound_priority"] = sound_priority
            row["total_coverage_priority"] = float(row["total_priority"]) + sound_priority
            row["sound_selection"] = sound_result
            per_K_best.append(row)

            if best_k is not None:
                cand = dict(best_k)
                cand["patrol_spent_total"] = row["patrol_spent_total"]
                cand["sound_budget"] = sound_budget
                cand["sound_covered_km"] = sound_km
                cand["sound_priority"] = sound_priority
                cand["total_coverage_priority"] = row["total_coverage_priority"]
                cand["sound_selection"] = sound_result
                if _is_better_total_coverage(cand, best_global):
                    best_global = cand
        else:
            per_K_best.append(base_row)
            if best_k is not None and _is_better(best_k, best_global):
                best_global = best_k

    return best_global, per_K_best, per_K_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy unique patrol allocation on big cells")
    parser.add_argument("--dist", default=DEFAULT_DIST_PATH)
    parser.add_argument("--patrol", default=DEFAULT_PATROL_PATH)
    parser.add_argument("--biggraph", default=DEFAULT_BIGGRAPH_PATH)
    parser.add_argument("--priority", default=DEFAULT_PRIORITY_PATH)
    parser.add_argument("--Kmin", type=int, default=DEFAULT_KMIN)
    parser.add_argument("--Kmax", type=int, default=DEFAULT_KMAX)
    parser.add_argument("--Tlim", type=float, default=DEFAULT_TLIM_H)
    parser.add_argument("--topL", type=int, default=DEFAULT_TOPL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--score-gain-pow", type=float, default=DEFAULT_SCORE_GAIN_POW)
    parser.add_argument("--score-time-pow", type=float, default=DEFAULT_SCORE_TIME_POW)
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.score_gain_pow <= 0.0 or args.score_time_pow <= 0.0:
        raise ValueError("--score-gain-pow and --score-time-pow must be > 0")

    model, dist_big_ids = load_side_transition_model(args.dist, args.biggraph)
    houses_big_ids = load_houses_from_patrol_json(args.patrol)
    S = build_S_big(args.biggraph, args.priority)

    base_set = set(houses_big_ids)
    cands = [bid for bid in dist_big_ids if S.get(bid, 0.0) > 0.0 and bid not in base_set]
    cands_sorted = sorted(cands, key=lambda bid: S.get(bid, 0.0), reverse=True)
    if args.topL is not None:
        cands_sorted = cands_sorted[: max(0, args.topL)]

    best, per_K_best, _ = search_best(
        Kmin=args.Kmin,
        Kmax=args.Kmax,
        Tlim=args.Tlim,
        houses_big_ids=houses_big_ids,
        cands_sorted=cands_sorted,
        S=S,
        model=model,
        seed=args.seed,
        score_gain_pow=args.score_gain_pow,
        score_time_pow=args.score_time_pow,
    )

    out_payload = {
        "meta": {
            "Kmax": args.Kmax,
            "Kmin": args.Kmin,
            "Tlim": args.Tlim,
            "topL": args.topL,
            "seed": args.seed,
            "score_gain_pow": args.score_gain_pow,
            "score_time_pow": args.score_time_pow,
            "dist_path": args.dist,
            "patrol_path": args.patrol,
            "biggraph_path": args.biggraph,
            "priority_path": args.priority,
        },
        "best": best,
        "per_K_best": per_K_best,
    }

    out_path = _resolve_write_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    if best is None:
        print("BEST K=None total_priority=0.0")
    else:
        print(f"BEST K={best['K']} total_priority={best['total_priority']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
