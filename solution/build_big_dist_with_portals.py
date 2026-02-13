from __future__ import annotations

import heapq
import json
import math
import os
import time


_SMALL_CACHE = None
_BIG_CACHE = None
_SMALL_CACHE_PATH = None
_BIG_CACHE_PATH = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


def _resolve_read_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    candidates = [
        path,
        os.path.join(_PROJECT_ROOT, path),
        os.path.join(_SCRIPT_DIR, path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_PROJECT_ROOT, path)


def _resolve_write_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def precompute_big_square_portal_routes_weight_over_distance(
    big_square_id: str,
    side_in: str,
    side_out: str,
    *,
    k_portals_per_side: int = 5,
    small_graph_path: str = "solution/etosha_grid_graph_with_big_squares.json",
    big_graph_path: str = "solution/etosha_big_square_graph_14x14.json",
    weight_keys: tuple = ("priority", "risk", "score", "heat", "value"),
    eps_dist: float = 1e-9,
):
    """
    ОДНА функция, которая:
      - сама загружает оба json из solution/
      - берёт заданный большой квадрат big_square_id = "br{row}_bc{col}"
      - выбирает по 5 "порталов" (входов) на стороне side_in и side_out (с углами)
      - для каждого входа на side_in считает оптимальные (по weight/distance) маршруты
        к каждому выходу на side_out, но с ПЕРВИЧНОЙ оптимизацией по distance:
          1) минимизируем distance (кратчайший путь)
          2) среди кратчайших максимизируем total_weight
          => ratio = total_weight / distance

    Почему так: иначе "weight/distance" без ограничений превращается в циклы/NP.
    Тут гарантированно быстро и стабильно.

    Возвращает dict:
      {
        "big_square_id": ...,
        "side_in": ..., "side_out": ...,
        "portals_in": [cell_id...],
        "portals_out": [cell_id...],
        "table": {
          in_portal: {
            out_portal: {
              "distance_m": float,
              "weight": float,
              "ratio": float,
              "path": [cell_id0, cell_id1, ...]  # from in_portal to out_portal
            }, ...
          }, ...
        },
        "best_out_for_in": { in_portal: {...best...}, ... }
      }

    side_in / side_out: "N","S","E","W" (или "north","south","east","west").
    """

    # ----------------- helpers (внутри функции, чтобы "одна функция") -----------------
    def _norm_side(s: str) -> str:
        s = s.strip().lower()
        if s in ("n", "north", "top", "up"): return "N"
        if s in ("s", "south", "bottom", "down"): return "S"
        if s in ("w", "west", "left"): return "W"
        if s in ("e", "east", "right"): return "E"
        raise ValueError(f"Bad side: {s}")

    def _get_rc(nf: dict):
        # пытаемся найти row/col у малой клетки
        for rk, ck in (("row", "col"), ("grid_row", "grid_col"), ("r", "c"), ("i", "j")):
            if rk in nf and ck in nf:
                return int(nf[rk]), int(nf[ck])
        return None

    def _get_weight(nf: dict) -> float:
        for k in weight_keys:
            if k in nf:
                try:
                    return float(nf[k])
                except Exception:
                    pass
        return 0.0

    def _pick_k_portals_on_side(side_nodes_sorted, k):
        # side_nodes_sorted: список cell_id отсортированных вдоль стороны
        if k <= 0:
            return []
        n = len(side_nodes_sorted)
        if n == 0:
            return []
        if k == 1:
            return [side_nodes_sorted[0]]
        idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
        out = []
        for i in idxs:
            ii = int(i)
            if ii < 0:
                ii = 0
            if ii >= n:
                ii = n - 1
            out.append(side_nodes_sorted[ii])
        return out

    def _dijkstra_shortest_then_maxweight(src, adj, node_w):
        # Dijkstra: минимизируем dist, при равенстве dist максимизируем weight
        INF = 10**30
        dist = {u: INF for u in adj}
        bestw = {u: -INF for u in adj}
        parent = {u: None for u in adj}

        dist[src] = 0.0
        bestw[src] = node_w.get(src, 0.0)
        pq = [(0.0, -bestw[src], src)]

        while pq:
            d, negw, u = heapq.heappop(pq)
            w = -negw
            if d > dist[u] + eps_dist:
                continue
            if abs(d - dist[u]) <= eps_dist and w < bestw[u] - 1e-12:
                continue

            for v, cost in adj[u]:
                nd = d + cost
                nw = w + node_w.get(v, 0.0)

                if nd + eps_dist < dist[v]:
                    dist[v] = nd
                    bestw[v] = nw
                    parent[v] = u
                    heapq.heappush(pq, (nd, -nw, v))
                elif abs(nd - dist[v]) <= eps_dist and nw > bestw[v] + 1e-12:
                    bestw[v] = nw
                    parent[v] = u
                    heapq.heappush(pq, (nd, -nw, v))

        return dist, bestw, parent

    def _reconstruct_path(parent, src, dst):
        if dst is None:
            return None
        cur = dst
        path = []
        seen = set()
        while cur is not None:
            path.append(cur)
            if cur == src:
                break
            if cur in seen:
                # защита от странностей (не должно происходить)
                return None
            seen.add(cur)
            cur = parent[cur]
        if not path or path[-1] != src:
            return None
        path.reverse()
        return path

    # ----------------- load graphs -----------------
    global _SMALL_CACHE, _BIG_CACHE, _SMALL_CACHE_PATH, _BIG_CACHE_PATH
    small_graph_path_resolved = _resolve_read_path(small_graph_path)
    big_graph_path_resolved = _resolve_read_path(big_graph_path)

    if _SMALL_CACHE is None or _SMALL_CACHE_PATH != small_graph_path_resolved:
        with open(small_graph_path_resolved, "r", encoding="utf-8") as f:
            _SMALL_CACHE = json.load(f)
        _SMALL_CACHE_PATH = small_graph_path_resolved
    if _BIG_CACHE is None or _BIG_CACHE_PATH != big_graph_path_resolved:
        with open(big_graph_path_resolved, "r", encoding="utf-8") as f:
            _BIG_CACHE = json.load(f)
        _BIG_CACHE_PATH = big_graph_path_resolved
    small = _SMALL_CACHE
    big = _BIG_CACHE

    side_in = _norm_side(side_in)
    side_out = _norm_side(side_out)

    big_nodes = big["node_features"]
    if big_square_id not in big_nodes:
        raise KeyError(f"big_square_id not found: {big_square_id}")

    contained = big_nodes[big_square_id]["contained_small_node_ids"]
    contained_set = set(contained)

    # ----------------- build subgraph adjacency for this big square -----------------
    # small["edge_features"] ожидаем как dict[src][dst] -> feat
    edge_feat = small.get("edge_features", {})
    node_feat = small.get("node_features", {})

    # веса узлов
    node_w = {}
    for cid in contained:
        nf = node_feat.get(cid, {})
        node_w[cid] = _get_weight(nf)

    # adjacency list внутри блока
    adj = {cid: [] for cid in contained}
    for u in contained:
        neighs = edge_feat.get(u, {})
        if not isinstance(neighs, dict):
            continue
        for v, feat in neighs.items():
            if v not in contained_set:
                continue
            # distance_m обязательно, иначе 1
            try:
                cost = float(feat.get("distance_m", 1.0))
            except Exception:
                cost = 1.0
            adj[u].append((v, cost))

    # ----------------- find side nodes (N/S/E/W) using row/col -----------------
    # берём row_range/col_range из big_graph как “окно”
    rr = big_nodes[big_square_id].get("row_range")
    cr = big_nodes[big_square_id].get("col_range")
    if not rr or not cr:
        # fallback: вычислим min/max по row/col в node_features
        rows, cols = [], []
        for cid in contained:
            rc = _get_rc(node_feat.get(cid, {}))
            if rc is not None:
                rows.append(rc[0]); cols.append(rc[1])
        if not rows:
            raise ValueError("Cannot determine row/col for small nodes (no row/col keys found)")
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
    else:
        min_r, max_r = int(rr[0]), int(rr[1])
        min_c, max_c = int(cr[0]), int(cr[1])

    side_nodes = {"N": [], "S": [], "W": [], "E": []}
    for cid in contained:
        rc = _get_rc(node_feat.get(cid, {}))
        if rc is None:
            continue
        r, c = rc
        if r == min_r: side_nodes["N"].append(cid)
        if r == max_r: side_nodes["S"].append(cid)
        if c == min_c: side_nodes["W"].append(cid)
        if c == max_c: side_nodes["E"].append(cid)

    # сортировка вдоль стороны
    def _sort_side(side):
        if side in ("N", "S"):
            return sorted(side_nodes[side], key=lambda cid: _get_rc(node_feat[cid])[1])  # по col
        else:
            return sorted(side_nodes[side], key=lambda cid: _get_rc(node_feat[cid])[0])  # по row

    in_side_sorted = _sort_side(side_in)
    out_side_sorted = _sort_side(side_out)

    portals_in = _pick_k_portals_on_side(in_side_sorted, k_portals_per_side)
    portals_out = _pick_k_portals_on_side(out_side_sorted, k_portals_per_side)

    if not portals_in or not portals_out:
        raise ValueError("No portals found on requested sides (check row/col keys and side names)")

    # ----------------- run Dijkstra from each вход-портал (5 раз) -----------------
    table = {}
    best_out_for_in = {}

    for pin in portals_in:
        dist, bw, parent = _dijkstra_shortest_then_maxweight(pin, adj, node_w)

        row = {}
        best = None

        for pout in portals_out:
            d = dist.get(pout, math.inf)
            if d == math.inf or d <= 0:
                item = {"distance_m": math.inf, "weight": 0.0, "ratio": -math.inf, "path": None}
            else:
                w = float(bw.get(pout, 0.0))
                r = w / d
                path = _reconstruct_path(parent, pin, pout)
                item = {"distance_m": float(d), "weight": w, "ratio": float(r), "path": path}

            row[pout] = item
            if best is None or item["ratio"] > best["ratio"]:
                best = {"out_portal": pout, **item}

        table[pin] = row
        best_out_for_in[pin] = best

    return {
        "big_square_id": big_square_id,
        "side_in": side_in,
        "side_out": side_out,
        "portals_in": portals_in,
        "portals_out": portals_out,
        "table": table,
        "best_out_for_in": best_out_for_in,
        "note": "Оптимизация: кратчайший distance, среди кратчайших max weight => max(weight/distance) на кратчайших путях.",
    }


def build_dist_big_cells_with_portals(
    out_path: str = "solution/big_dist_with_portals_time_priority.json",
    k: int = 5,
    speed_kmh: float = 40.0,
    priority_path: str = "solution/etosha_node_priority_compact_clamped.json",
):
    sides = ("N", "S", "E", "W")
    small_graph_path = _resolve_read_path("solution/etosha_grid_graph_with_big_squares.json")
    big_graph_path = _resolve_read_path("solution/etosha_big_square_graph_14x14.json")
    out_path_resolved = _resolve_write_path(out_path)
    inf_json = 1e30
    speed_m_per_h = speed_kmh * 1000.0
    weight_keys = ("priority", "risk", "score", "heat", "value")
    priority_path_resolved = _resolve_read_path(priority_path)

    with open(small_graph_path, "r", encoding="utf-8") as f:
        small = json.load(f)

    with open(big_graph_path, "r", encoding="utf-8") as f:
        big = json.load(f)

    small_nodes = small["node_features"]
    big_nodes = big["node_features"]
    big_edges = big["edge_features"]
    big_ids = sorted(big_nodes.keys())

    small_priority_by_cell = {}
    priority_source = "small_node_features_fallback"
    detected_priority_key = None

    if os.path.exists(priority_path_resolved):
        with open(priority_path_resolved, "r", encoding="utf-8") as f:
            raw_pr = json.load(f)
        if isinstance(raw_pr, dict):
            for cid in small_nodes.keys():
                try:
                    small_priority_by_cell[cid] = float(raw_pr.get(cid, 0.0))
                except Exception:
                    small_priority_by_cell[cid] = 0.0
            priority_source = priority_path_resolved

    if not small_priority_by_cell:
        candidate_keys = []
        sample_nf = next(iter(small_nodes.values())) if small_nodes else {}
        if isinstance(sample_nf, dict):
            for kk in sample_nf.keys():
                kl = str(kk).lower()
                if any(tok in kl for tok in ("prior", "risk", "score", "heat", "value")):
                    candidate_keys.append(kk)
        for wk in weight_keys:
            if wk not in candidate_keys:
                candidate_keys.append(wk)

        for kk in candidate_keys:
            nonzero = 0
            for cid, nf in small_nodes.items():
                try:
                    v = float(nf.get(kk, 0.0))
                except Exception:
                    v = 0.0
                small_priority_by_cell[cid] = v
                if abs(v) > 1e-12:
                    nonzero += 1
            if nonzero > 0:
                detected_priority_key = str(kk)
                priority_source = f"small_node_features[{kk}]"
                break
        else:
            for cid in small_nodes.keys():
                small_priority_by_cell[cid] = 0.0

    nonzero_priority_cells = sum(1 for v in small_priority_by_cell.values() if abs(float(v)) > 1e-12)
    print(
        f"[build_dist] priority source: {priority_source} | "
        f"nonzero cells: {nonzero_priority_cells}/{len(small_priority_by_cell)}"
    )

    def _parse_big_id(bid: str):
        if not isinstance(bid, str) or not bid.startswith("br") or "_bc" not in bid:
            raise ValueError(f"Bad big-square id format: {bid}")
        left, right = bid.split("_bc", 1)
        return int(left[2:]), int(right)

    id_to_rc = {bid: _parse_big_id(bid) for bid in big_ids}

    # Cache inside matrices (k x k) for each block and each pair (side_in, side_out)
    # inside_mat[bid][side_in][side_out][in_idx][out_idx] = distance_m
    inside_mat = {
        bid: {
            si: {so: [[math.inf for _ in range(k)] for _ in range(k)] for so in sides}
            for si in sides
        }
        for bid in big_ids
    }
    raw_cache = {}
    portals_side = {bid: {s: [] for s in sides} for bid in big_ids}
    portal_idx_map = {bid: {s: {} for s in sides} for bid in big_ids}

    small_meta = small.get("meta", {})
    small_cell_size = float(small_meta.get("cell_size_m", 1000.0))
    small_origin = small_meta.get("grid_origin_m", [0.0, 0.0])
    small_ox = float(small_origin[0])
    small_oy = float(small_origin[1])

    small_rc = {}
    small_xy = {}
    for cid, nf in small_nodes.items():
        rr = int(nf.get("row", 0))
        cc = int(nf.get("col", 0))
        small_rc[cid] = (rr, cc)
        c = nf.get("centroid_m")
        if isinstance(c, list) and len(c) == 2:
            small_xy[cid] = (float(c[0]), float(c[1]))
        else:
            small_xy[cid] = (
                small_ox + (cc + 0.5) * small_cell_size,
                small_oy + (rr + 0.5) * small_cell_size,
            )

    small_adj = {}
    for su, neis in small.get("edge_features", {}).items():
        lst = small_adj.setdefault(su, [])
        if not isinstance(neis, dict):
            continue
        for sv, ef in neis.items():
            try:
                sd = float(ef.get("distance_m", 1.0))
            except Exception:
                sd = 1.0
            lst.append((sv, sd))

    def _extract_kxk_matrix(res):
        table = res.get("table", {})
        pins = list(res.get("portals_in", []))
        pouts = list(res.get("portals_out", []))
        mat = [[math.inf for _ in range(k)] for _ in range(k)]
        for i in range(k):
            pin = pins[i] if i < len(pins) else None
            row = table.get(pin, {}) if pin is not None else {}
            if not isinstance(row, dict):
                row = {}
            for j in range(k):
                pout = pouts[j] if j < len(pouts) else None
                rec = row.get(pout, {}) if pout is not None else {}
                if not isinstance(rec, dict):
                    rec = {}
                d = rec.get("distance_m", math.inf)
                try:
                    d = float(d)
                except Exception:
                    d = math.inf
                mat[i][j] = d
        return mat

    for bid in big_ids:
        for side_in in sides:
            for side_out in sides:
                key = (bid, side_in, side_out)
                if key in raw_cache:
                    res = raw_cache[key]
                else:
                    res = precompute_big_square_portal_routes_weight_over_distance(
                        bid,
                        side_in,
                        side_out,
                        k_portals_per_side=k,
                        small_graph_path=small_graph_path,
                        big_graph_path=big_graph_path,
                    )
                    raw_cache[key] = res
                inside_mat[bid][side_in][side_out] = _extract_kxk_matrix(res)
                if side_in == side_out and not portals_side[bid][side_in]:
                    portals_side[bid][side_in] = list(res.get("portals_in", []))

    for bid in big_ids:
        for side in sides:
            plist = portals_side[bid][side]
            idx_map = portal_idx_map[bid][side]
            for idx, pid in enumerate(plist):
                idx_map.setdefault(pid, []).append(idx)

    def _opposite(side):
        return {"N": "S", "S": "N", "E": "W", "W": "E"}[side]

    def _side_out_from_to(u, v):
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

    def _lex_better(a_big, a_inside, b_big, b_inside):
        # True if (a_big, a_inside) < (b_big, b_inside) lexicographically.
        if a_big < b_big - 1e-12:
            return True
        if abs(a_big - b_big) <= 1e-12 and a_inside < b_inside - 1e-12:
            return True
        return False

    nearest_idx_cache = {}

    def _nearest_portal_idx_by_small_distance(src_portal, target_portals):
        key = (src_portal, tuple(target_portals))
        if key in nearest_idx_cache:
            return nearest_idx_cache[key]

        if not target_portals:
            nearest_idx_cache[key] = None
            return None

        target_set = set(target_portals)
        if src_portal in target_set:
            idx0 = target_portals.index(src_portal)
            nearest_idx_cache[key] = idx0
            return idx0

        dist = {src_portal: 0.0}
        pq = [(0.0, src_portal)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            if u in target_set:
                idx = target_portals.index(u)
                nearest_idx_cache[key] = idx
                return idx
            for v, w in small_adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        sx, sy = small_xy.get(src_portal, (0.0, 0.0))
        best_i = 0
        best_d2 = math.inf
        for i, pid in enumerate(target_portals):
            tx, ty = small_xy.get(pid, (0.0, 0.0))
            d2 = (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        nearest_idx_cache[key] = best_i
        return best_i

    def _matched_next_portal_id(u, v, side_out, p_out):
        side_in_v = _opposite(side_out)
        v_portals = portals_side[v][side_in_v]
        if not v_portals:
            return None

        rc = small_rc.get(p_out)
        if rc is not None:
            rr, cc = rc
            if side_out == "E":
                target_rc = (rr, cc + 1)
            elif side_out == "W":
                target_rc = (rr, cc - 1)
            elif side_out == "S":
                target_rc = (rr + 1, cc)
            elif side_out == "N":
                target_rc = (rr - 1, cc)
            else:
                target_rc = None
            if target_rc is not None:
                for pid in v_portals:
                    if small_rc.get(pid) == target_rc:
                        return pid

        idx = _nearest_portal_idx_by_small_distance(p_out, v_portals)
        if idx is None:
            return None
        if idx < 0 or idx >= len(v_portals):
            return None
        return v_portals[idx]

    boundary_match = {}
    for u in big_ids:
        for v, ef in big_edges.get(u, {}).items():
            if ef.get("is_diagonal"):
                continue
            side_out = _side_out_from_to(u, v)
            if side_out is None:
                continue
            for p_out in portals_side[u][side_out]:
                boundary_match[(u, v, p_out)] = _matched_next_portal_id(u, v, side_out, p_out)

    def _state_dijkstra_from(src):
        # State: (node, entry_side, entry_portal_cell_id)
        # entry_side in {"N","S","E","W"} for non-source states.
        state_best = {}
        state_parent = {}
        pq = []

        # Boundary-start: inside cost in source block is 0; choose any portal on outgoing side.
        for nxt, ef in big_edges.get(src, {}).items():
            if ef.get("is_diagonal"):
                continue
            side_out = _side_out_from_to(src, nxt)
            if side_out is None:
                continue
            side_in_next = _opposite(side_out)
            edge_d = float(ef.get("distance_m", 1.0))
            for p_out in portals_side[src][side_out]:
                p_in_next = boundary_match.get((src, nxt, p_out))
                if p_in_next is None:
                    continue
                key = (nxt, side_in_next, p_in_next)
                cand_big = edge_d
                cand_inside = 0.0
                old = state_best.get(key, (math.inf, math.inf))
                if _lex_better(cand_big, cand_inside, old[0], old[1]):
                    state_best[key] = (cand_big, cand_inside)
                    state_parent[key] = None
                    heapq.heappush(pq, (cand_big, cand_inside, nxt, side_in_next, p_in_next))

        # Best by destination node and corresponding best end state.
        node_best_pair = {src: (0.0, 0.0)}
        node_best_state = {src: None}

        while pq:
            cur_big, cur_inside, u, entry_side, entry_portal = heapq.heappop(pq)
            cur_key = (u, entry_side, entry_portal)
            best_pair = state_best.get(cur_key, (math.inf, math.inf))
            if abs(cur_big - best_pair[0]) > 1e-12 or abs(cur_inside - best_pair[1]) > 1e-12:
                continue

            old_node = node_best_pair.get(u, (math.inf, math.inf))
            if _lex_better(cur_big, cur_inside, old_node[0], old_node[1]):
                node_best_pair[u] = (cur_big, cur_inside)
                node_best_state[u] = cur_key

            for v, ef in big_edges.get(u, {}).items():
                if ef.get("is_diagonal"):
                    continue
                side_out = _side_out_from_to(u, v)
                if side_out is None:
                    continue
                side_in_next = _opposite(side_out)
                edge_d = float(ef.get("distance_m", 1.0))
                mat = inside_mat[u][entry_side][side_out]
                entry_idx_list = portal_idx_map[u][entry_side].get(entry_portal, [])
                if not entry_idx_list:
                    continue

                for idx_out, p_out in enumerate(portals_side[u][side_out]):
                    p_in_next = boundary_match.get((u, v, p_out))
                    if p_in_next is None:
                        continue
                    portal_cost = math.inf
                    for idx_in in entry_idx_list:
                        val = mat[idx_in][idx_out]
                        if val < portal_cost:
                            portal_cost = val
                    if not math.isfinite(portal_cost):
                        continue
                    cand_big = cur_big + edge_d
                    cand_inside = cur_inside + portal_cost
                    key = (v, side_in_next, p_in_next)
                    old = state_best.get(key, (math.inf, math.inf))
                    if _lex_better(cand_big, cand_inside, old[0], old[1]):
                        state_best[key] = (cand_big, cand_inside)
                        state_parent[key] = (cur_key, p_out)
                        heapq.heappush(pq, (cand_big, cand_inside, v, side_in_next, p_in_next))

        return state_best, state_parent, node_best_pair, node_best_state

    def _reconstruct_state_chain(end_state, state_parent):
        if end_state is None:
            return []
        chain = []
        cur = end_state
        seen = set()
        while cur is not None:
            if cur in seen:
                return []
            seen.add(cur)
            chain.append(cur)
            parent_rec = state_parent.get(cur)
            if parent_rec is None:
                cur = None
            else:
                cur = parent_rec[0]
        chain.reverse()
        return chain

    def _small_priority_sum_for_chain(src, dst, chain_states):
        # chain_states contains states for nodes after src, ending at dst.
        # boundary-end: no inside movement in dst.
        if src == dst or len(chain_states) <= 1:
            return 0.0

        used_cells = set()
        total_p = 0.0

        for t in range(len(chain_states) - 1):
            cur_state = chain_states[t]
            nxt_state = chain_states[t + 1]
            u = cur_state[0]
            if u == dst:
                break
            side_in = cur_state[1]
            pin = cur_state[2]
            side_out = _side_out_from_to(u, nxt_state[0])
            if side_out is None:
                continue
            parent_rec = state_parent.get(nxt_state)
            if parent_rec is None:
                continue
            prev_state, pout = parent_rec
            if prev_state != cur_state:
                continue

            res = raw_cache.get((u, side_in, side_out))
            if res is None:
                res = precompute_big_square_portal_routes_weight_over_distance(
                    u,
                    side_in,
                    side_out,
                    k_portals_per_side=k,
                    small_graph_path=small_graph_path,
                    big_graph_path=big_graph_path,
                )
                raw_cache[(u, side_in, side_out)] = res

            portals_in = list(res.get("portals_in", []))
            portals_out = list(res.get("portals_out", []))
            if pin not in portals_in or pout not in portals_out:
                continue
            rec = res.get("table", {}).get(pin, {}).get(pout, {})
            path = rec.get("path")
            if not isinstance(path, list):
                continue

            for cid in path:
                if cid not in used_cells:
                    used_cells.add(cid)
                    total_p += float(small_priority_by_cell.get(cid, 0.0))

        return float(total_p)

    def _small_priority_sum_for_path(path):
        if not isinstance(path, list):
            return 0.0
        used = set()
        s = 0.0
        for cid in path:
            if cid in used:
                continue
            used.add(cid)
            s += float(small_priority_by_cell.get(cid, 0.0))
        return float(s)

    inside_best_by_block_side = {bid: {si: {} for si in sides} for bid in big_ids}
    for bid in big_ids:
        for side_in in sides:
            for side_out in sides:
                res = raw_cache.get((bid, side_in, side_out))
                if res is None:
                    res = precompute_big_square_portal_routes_weight_over_distance(
                        bid,
                        side_in,
                        side_out,
                        k_portals_per_side=k,
                        small_graph_path=small_graph_path,
                        big_graph_path=big_graph_path,
                    )
                    raw_cache[(bid, side_in, side_out)] = res

                best_d = math.inf
                best_p = -math.inf
                table = res.get("table", {})
                if isinstance(table, dict):
                    for row in table.values():
                        if not isinstance(row, dict):
                            continue
                        for rec in row.values():
                            if not isinstance(rec, dict):
                                continue
                            try:
                                d = float(rec.get("distance_m", math.inf))
                            except Exception:
                                d = math.inf
                            if not math.isfinite(d):
                                continue
                            p = _small_priority_sum_for_path(rec.get("path"))
                            if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and p > best_p + 1e-12):
                                best_d = d
                                best_p = p

                if not math.isfinite(best_d):
                    inside_best_by_block_side[bid][side_in][side_out] = {
                        "distance_m": inf_json,
                        "time_h": inf_json,
                        "small_priority_sum": inf_json,
                    }
                else:
                    inside_best_by_block_side[bid][side_in][side_out] = {
                        "distance_m": float(best_d),
                        "time_h": float(best_d / speed_m_per_h),
                        "small_priority_sum": float(best_p if math.isfinite(best_p) else 0.0),
                    }

    dist_out = {i: {} for i in big_ids}
    t0 = time.time()
    total_sources = len(big_ids)
    for idx_src, i in enumerate(big_ids, start=1):
        state_best, state_parent, node_best_pair, node_best_state = _state_dijkstra_from(i)
        for j in big_ids:
            if i == j:
                dist_out[i][j] = {
                    "distance_m": 0.0,
                    "time_h": 0.0,
                    "small_priority_sum": 0.0,
                }
                continue

            pair = node_best_pair.get(j)
            end_state = node_best_state.get(j)
            if pair is None or end_state is None:
                dist_out[i][j] = {
                    "distance_m": inf_json,
                    "time_h": inf_json,
                    "small_priority_sum": inf_json,
                }
            else:
                dist_m = float(pair[0] + pair[1])
                if not math.isfinite(dist_m):
                    dist_m = inf_json
                    time_h = inf_json
                    pr_sum = inf_json
                else:
                    time_h = float(dist_m / speed_m_per_h)
                    chain = _reconstruct_state_chain(end_state, state_parent)
                    pr_sum = _small_priority_sum_for_chain(i, j, chain)
                dist_out[i][j] = {
                    "distance_m": dist_m,
                    "time_h": time_h,
                    "small_priority_sum": pr_sum,
                }

        elapsed = time.time() - t0
        done = idx_src
        left = total_sources - done
        eta = (elapsed / done) * left if done > 0 else math.inf
        print(
            f"[build_dist] {done}/{total_sources} sources done | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        )

    payload = {
        "meta": {
            "method": "state_dijkstra_lexicographic_big_then_inside_portal_dp_boundary_to_boundary_portal_cell_matched",
            "k_portals_per_side": int(k),
            "sides": list(sides),
            "big_cell_count": len(big_ids),
            "speed_kmh_for_time": float(speed_kmh),
            "small_graph_path": small_graph_path,
            "big_graph_path": big_graph_path,
            "inside_cost_model": "k_by_k_portal_matrix_dp",
            "inside_best_by_block_side_mode": "best_portal_pair_by_min_distance_tiebreak_max_small_priority_sum",
            "diagonal_big_edges_used": False,
            "small_priority_sum_mode": "unique_small_cell_ids_on_intermediate_big_blocks",
            "weight_keys": list(weight_keys),
            "priority_source": priority_source,
            "priority_path_requested": priority_path,
            "priority_path_resolved": priority_path_resolved,
            "detected_priority_key": detected_priority_key,
            "nonzero_priority_cells": int(nonzero_priority_cells),
            "inf_replacement": inf_json,
            "out_path": out_path_resolved,
        },
        "inside_best_by_block_side": inside_best_by_block_side,
        "dist": dist_out,
    }

    out_dir = os.path.dirname(out_path_resolved)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path_resolved, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return payload


def build_patrol_house_to_big_cell(
    out_path: str = "solution/patrol_house_to_big_cell.json",
    k: int = 5,
    speed_kmh: float = 40.0,
):
    sides = ("N", "S", "E", "W")
    small_graph_path = _resolve_read_path("solution/etosha_grid_graph_with_big_squares.json")
    big_graph_path = _resolve_read_path("solution/etosha_big_square_graph_14x14.json")
    out_path_resolved = _resolve_write_path(out_path)
    inf_json = 1e30
    speed_m_per_h = speed_kmh * 1000.0

    with open(small_graph_path, "r", encoding="utf-8") as f:
        small = json.load(f)
    with open(big_graph_path, "r", encoding="utf-8") as f:
        big = json.load(f)

    small_nodes = small.get("node_features", {})
    small_edges = small.get("edge_features", {})
    big_ids = sorted(big.get("node_features", {}).keys())

    def _is_cell_id(x):
        return isinstance(x, str) and x.startswith("r") and "_c" in x

    def _contains_patrol_word(text: str) -> bool:
        t = text.lower()
        return ("patrol" in t) or ("ranger" in t) or ("house" in t) or ("post" in t)

    def _collect_patrol_house_cell_ids():
        found = set()
        meta = small.get("meta", {})
        for key in ("patrol_house_cell_ids", "patrol_houses", "patrol_house_cells", "ranger_post_cell_ids"):
            val = meta.get(key)
            if isinstance(val, list):
                for x in val:
                    if _is_cell_id(x):
                        found.add(x)
            elif isinstance(val, dict):
                for x in val.keys():
                    if _is_cell_id(x):
                        found.add(x)
                for x in val.values():
                    if _is_cell_id(x):
                        found.add(x)
                    if isinstance(x, list):
                        for y in x:
                            if _is_cell_id(y):
                                found.add(y)

        if found:
            return sorted(found)

        for cid, nf in small_nodes.items():
            poi = nf.get("poi_type_counts")
            hit = False
            if isinstance(poi, dict):
                for k0, v0 in poi.items():
                    try:
                        cnt = float(v0)
                    except Exception:
                        cnt = 0.0
                    if cnt > 0 and _contains_patrol_word(str(k0)):
                        hit = True
                        break
            if not hit:
                for key in ("poi_type", "category", "type", "kind", "name"):
                    v = nf.get(key)
                    if isinstance(v, str) and _contains_patrol_word(v):
                        hit = True
                        break
                    if isinstance(v, list):
                        for vv in v:
                            if isinstance(vv, str) and _contains_patrol_word(vv):
                                hit = True
                                break
                        if hit:
                            break
            if hit:
                found.add(cid)
        return sorted(found)

    patrol_house_cell_ids = _collect_patrol_house_cell_ids()

    # Build small-graph adjacency once.
    small_adj = {}
    for u, neis in small_edges.items():
        lst = small_adj.setdefault(u, [])
        if not isinstance(neis, dict):
            continue
        for v, ef in neis.items():
            try:
                d = float(ef.get("distance_m", 1.0))
            except Exception:
                d = 1.0
            lst.append((v, d))

    # Gather portals on each side and all-side union for each big cell.
    side_portals = {}
    portals_all_by_big = {}
    for bid in big_ids:
        all_p = []
        for side in sides:
            res = precompute_big_square_portal_routes_weight_over_distance(
                bid,
                side,
                side,
                k_portals_per_side=k,
                small_graph_path=small_graph_path,
                big_graph_path=big_graph_path,
            )
            p = list(res.get("portals_in", []))
            side_portals[(bid, side)] = p
            all_p.extend(p)
        portals_all_by_big[bid] = all_p

    def _dijkstra_small_from(src):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in small_adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    out_rows = []
    t0 = time.time()
    total = len(patrol_house_cell_ids)
    for idx, ph in enumerate(patrol_house_cell_ids, start=1):
        d_small = _dijkstra_small_from(ph)
        to_big = {}
        for bid in big_ids:
            best = math.inf
            for p in portals_all_by_big.get(bid, []):
                dp = d_small.get(p, math.inf)
                if dp < best:
                    best = dp
            if not math.isfinite(best):
                dist_m = inf_json
                time_h = inf_json
            else:
                dist_m = float(best)
                time_h = float(dist_m / speed_m_per_h)
            to_big[bid] = {"distance_m": dist_m, "time_h": time_h}
        out_rows.append({"cell_id": ph, "to_big": to_big})

        elapsed = time.time() - t0
        left = total - idx
        eta = (elapsed / idx) * left if idx > 0 else math.inf
        print(
            f"[patrol_to_big] {idx}/{total} houses done | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        )

    payload = {
        "meta": {
            "method": "min_small_graph_distance_to_any_big_cell_portal",
            "k_portals_per_side": int(k),
            "sides": list(sides),
            "speed_kmh_for_time": float(speed_kmh),
            "small_graph_path": small_graph_path,
            "big_graph_path": big_graph_path,
            "patrol_house_count": len(patrol_house_cell_ids),
            "inf_replacement": inf_json,
            "out_path": out_path_resolved,
        },
        "patrol_house_cell_ids": patrol_house_cell_ids,
        "patrol_houses": out_rows,
    }

    out_dir = os.path.dirname(out_path_resolved)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path_resolved, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    payload_dist = build_dist_big_cells_with_portals()
    payload_patrol = build_patrol_house_to_big_cell()
    print("[saved] big dist file:", payload_dist.get("meta", {}).get("out_path"))
    print("[saved] patrol->big file:", payload_patrol.get("meta", {}).get("out_path"))
