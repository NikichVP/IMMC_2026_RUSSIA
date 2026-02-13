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
    out_path: str = "solution/big_dist_with_portals.json",
    k: int = 5,
):
    sides = ("N", "S", "E", "W")
    small_graph_path = _resolve_read_path("solution/etosha_grid_graph_with_big_squares.json")
    big_graph_path = _resolve_read_path("solution/etosha_big_square_graph_14x14.json")
    out_path_resolved = _resolve_write_path(out_path)

    with open(big_graph_path, "r", encoding="utf-8") as f:
        big = json.load(f)

    big_nodes = big["node_features"]
    big_edges = big["edge_features"]
    big_ids = sorted(big_nodes.keys())
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

    def _best_first_vector(first_block, side_out):
        # dp_out[j] = min_{entry_side, i} mat(entry_side, side_out)[i][j]
        dp = [math.inf for _ in range(k)]
        for side_in in sides:
            mat = inside_mat[first_block][side_in][side_out]
            for j in range(k):
                best = math.inf
                for i in range(k):
                    val = mat[i][j]
                    if val < best:
                        best = val
                if best < dp[j]:
                    dp[j] = best
        return dp

    def _lex_better(a_big, a_inside, b_big, b_inside):
        # True if (a_big, a_inside) < (b_big, b_inside) lexicographically.
        if a_big < b_big - 1e-12:
            return True
        if abs(a_big - b_big) <= 1e-12 and a_inside < b_inside - 1e-12:
            return True
        return False

    def _state_dijkstra_from(src):
        # State: (node, entry_side, entry_portal_idx)
        # entry_side in {"N","S","E","W"} for non-source states.
        state_best = {}
        pq = []

        # First step from source: choose neighbor side_out and start vector on source.
        for nxt, ef in big_edges.get(src, {}).items():
            if ef.get("is_diagonal"):
                continue
            side_out = _side_out_from_to(src, nxt)
            if side_out is None:
                continue
            side_in_next = _opposite(side_out)
            edge_d = float(ef.get("distance_m", 1.0))
            start_vec = _best_first_vector(src, side_out)
            for j in range(k):
                inside_part = start_vec[j]
                if not math.isfinite(inside_part):
                    continue
                key = (nxt, side_in_next, j)
                cand_big = edge_d
                cand_inside = inside_part
                old = state_best.get(key, (math.inf, math.inf))
                if _lex_better(cand_big, cand_inside, old[0], old[1]):
                    state_best[key] = (cand_big, cand_inside)
                    heapq.heappush(pq, (cand_big, cand_inside, nxt, side_in_next, j))

        # Best by destination node (include src itself as zero).
        node_best = {src: (0.0, 0.0)}

        while pq:
            cur_big, cur_inside, u, entry_side, entry_idx = heapq.heappop(pq)
            best_pair = state_best.get((u, entry_side, entry_idx), (math.inf, math.inf))
            if abs(cur_big - best_pair[0]) > 1e-12 or abs(cur_inside - best_pair[1]) > 1e-12:
                continue

            old_node = node_best.get(u, (math.inf, math.inf))
            if _lex_better(cur_big, cur_inside, old_node[0], old_node[1]):
                node_best[u] = (cur_big, cur_inside)

            for v, ef in big_edges.get(u, {}).items():
                if ef.get("is_diagonal"):
                    continue
                side_out = _side_out_from_to(u, v)
                if side_out is None:
                    continue
                side_in_next = _opposite(side_out)
                edge_d = float(ef.get("distance_m", 1.0))
                mat = inside_mat[u][entry_side][side_out]

                for j in range(k):
                    portal_cost = mat[entry_idx][j]
                    if not math.isfinite(portal_cost):
                        continue
                    cand_big = cur_big + edge_d
                    cand_inside = cur_inside + portal_cost
                    key = (v, side_in_next, j)
                    old = state_best.get(key, (math.inf, math.inf))
                    if _lex_better(cand_big, cand_inside, old[0], old[1]):
                        state_best[key] = (cand_big, cand_inside)
                        heapq.heappush(pq, (cand_big, cand_inside, v, side_in_next, j))

        return node_best

    dist_out = {i: {} for i in big_ids}
    t0 = time.time()
    total_sources = len(big_ids)
    for idx_src, i in enumerate(big_ids, start=1):
        node_best = _state_dijkstra_from(i)
        for j in big_ids:
            pair = node_best.get(j)
            if pair is None:
                dist_out[i][j] = math.inf
            else:
                dist_out[i][j] = float(pair[0] + pair[1])

        elapsed = time.time() - t0
        done = idx_src
        left = total_sources - done
        eta = (elapsed / done) * left if done > 0 else math.inf
        print(
            f"[build_dist] {done}/{total_sources} sources done | "
            f"elapsed={elapsed:.1f}s | eta={eta:.1f}s"
        )

    # JSON-safe conversion: replace non-finite distances.
    INF_JSON = 1e30
    for i in big_ids:
        for j in big_ids:
            v = dist_out[i][j]
            if not math.isfinite(v):
                dist_out[i][j] = INF_JSON

    payload = {
        "meta": {
            "method": "state_dijkstra_lexicographic_big_then_inside_portal_dp",
            "k_portals_per_side": int(k),
            "sides": list(sides),
            "big_cell_count": len(big_ids),
            "small_graph_path": small_graph_path,
            "big_graph_path": big_graph_path,
            "inside_cost_model": "k_by_k_portal_matrix_dp",
            "diagonal_big_edges_used": False,
            "json_inf_replacement": 1e30,
            "out_path": out_path_resolved,
        },
        "dist": dist_out,
    }

    out_dir = os.path.dirname(out_path_resolved)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path_resolved, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return payload


if __name__ == "__main__":
    build_dist_big_cells_with_portals()
