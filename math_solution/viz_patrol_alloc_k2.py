import argparse
import heapq
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from build_big_dist_with_portals import precompute_big_square_portal_routes_weight_over_distance
except ModuleNotFoundError:
    try:
        from solution.build_big_dist_with_portals import precompute_big_square_portal_routes_weight_over_distance
    except ModuleNotFoundError:
        precompute_big_square_portal_routes_weight_over_distance = None

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mpl-cache"
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = "/tmp"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
except ModuleNotFoundError as e:
    raise SystemExit(
        "ERROR: matplotlib/numpy are required for visualization. "
        "Install them in your environment (e.g. pip install matplotlib numpy)."
    ) from e


INF = float("inf")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def _safe_float(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return INF
    if not math.isfinite(x):
        return INF
    return x


def _resolve_read_path(path: str) -> str:
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"File not found: {path}")

    candidates = [path, os.path.join(SCRIPT_DIR, path), os.path.join(PROJECT_DIR, path)]
    if path.startswith("solution/"):
        tail = path[len("solution/") :]
        candidates.extend([os.path.join(SCRIPT_DIR, tail), os.path.join(PROJECT_DIR, tail)])

    seen = set()
    uniq = []
    for c in candidates:
        ac = os.path.abspath(c)
        if ac not in seen:
            uniq.append(ac)
            seen.add(ac)

    for c in uniq:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"File not found: {path}")


def _resolve_out_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if path.startswith("solution/"):
        return os.path.join(PROJECT_DIR, path)
    if "/" not in path and "\\" not in path:
        return os.path.join(SCRIPT_DIR, path)
    return os.path.abspath(path)


def _load_json(path: str) -> dict:
    resolved = _resolve_read_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_priority_map(priority_path: str) -> Dict[str, float]:
    try:
        _resolve_read_path(priority_path)
    except FileNotFoundError:
        return {}
    payload = _load_json(priority_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid priority JSON format: {priority_path}")
    out: Dict[str, float] = {}
    for k, v in payload.items():
        p = _safe_float(v)
        out[k] = p if math.isfinite(p) else 0.0
    return out


def _build_S_big(biggraph: dict, priority_small: Dict[str, float]) -> Dict[str, float]:
    node_features = biggraph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError("Invalid biggraph: missing node_features")

    S: Dict[str, float] = {}
    for bid, feat in node_features.items():
        contained = feat.get("contained_small_node_ids", []) if isinstance(feat, dict) else []
        if not isinstance(contained, list):
            contained = []
        s = 0.0
        for sid in contained:
            p = priority_small.get(sid, 0.0)
            if math.isfinite(p):
                s += p
        S[bid] = s
    return S


def _small_grid_layers(smallgraph: dict) -> dict:
    meta = smallgraph.get("meta", {})
    node_features = smallgraph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError("Invalid smallgraph: missing node_features")

    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    min_x, min_y = meta["grid_origin_m"]
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    road_m = np.zeros((n_rows, n_cols), dtype="float32")

    for nf in node_features.values():
        rr = int(nf["row"])
        cc = int(nf["col"])
        if nf.get("median_elevation_m") is not None:
            inside_mask[rr, cc] = True
        road_m[rr, cc] = float(nf.get("road_total_length_m", 0.0))

    return {
        "inside_mask": inside_mask,
        "road_m": road_m,
        "extent": [min_x, max_x, min_y, max_y],
        "meta": meta,
    }


def _small_priority_layers(smallgraph: dict, priority_small: Dict[str, float]) -> dict:
    meta = smallgraph.get("meta", {})
    node_features = smallgraph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError("Invalid smallgraph: missing node_features")

    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    min_x, min_y = meta["grid_origin_m"]
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    pr_arr = np.zeros((n_rows, n_cols), dtype="float32")

    for cid, nf in node_features.items():
        rr = int(nf["row"])
        cc = int(nf["col"])
        if nf.get("median_elevation_m") is None:
            outside_mask[rr, cc] = True
            pr_arr[rr, cc] = 0.0
        else:
            inside_mask[rr, cc] = True
            pr_arr[rr, cc] = float(priority_small.get(cid, 0.0))

    return {
        "inside_mask": inside_mask,
        "outside_mask": outside_mask,
        "priority_arr": pr_arr,
        "extent": [min_x, max_x, min_y, max_y],
    }


def _extract_big_cells(biggraph: dict) -> Dict[str, dict]:
    meta = biggraph.get("meta", {})
    node_features = biggraph.get("node_features")
    if not isinstance(node_features, dict):
        raise ValueError("Invalid biggraph: missing node_features")

    min_x, min_y = meta.get("grid_origin_m", [0.0, 0.0])
    cell_size = float(meta.get("small_cell_size_m", 1000.0))

    out: Dict[str, dict] = {}
    for bid, nf in node_features.items():
        bbox = nf.get("bbox_m")
        if isinstance(bbox, list) and len(bbox) == 4:
            x0, y0, x1, y1 = map(float, bbox)
        else:
            rr = nf.get("row_range", [0, 0])
            cc = nf.get("col_range", [0, 0])
            r0, r1 = int(rr[0]), int(rr[1])
            c0, c1 = int(cc[0]), int(cc[1])
            x0 = min_x + c0 * cell_size
            x1 = min_x + (c1 + 1) * cell_size
            y0 = min_y + r0 * cell_size
            y1 = min_y + (r1 + 1) * cell_size
        out[bid] = {
            "bbox": (x0, y0, x1, y1),
            "center": ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
        }
    return out


def _build_small_graph_index(smallgraph: dict) -> dict:
    meta = smallgraph.get("meta", {})
    node_features = smallgraph.get("node_features")
    edge_features = smallgraph.get("edge_features")
    if not isinstance(node_features, dict) or not isinstance(edge_features, dict):
        raise ValueError("Invalid smallgraph: missing node_features/edge_features")

    min_x, min_y = meta["grid_origin_m"]
    cell_size = float(meta["cell_size_m"])

    node_xy: Dict[str, Tuple[float, float]] = {}
    node_big: Dict[str, str] = {}
    big_to_nodes: Dict[str, List[str]] = defaultdict(list)

    for nid, nf in node_features.items():
        if nf.get("median_elevation_m") is None:
            continue
        centroid = nf.get("centroid_m")
        if isinstance(centroid, list) and len(centroid) == 2:
            x, y = float(centroid[0]), float(centroid[1])
        else:
            rr = int(nf["row"])
            cc = int(nf["col"])
            x = min_x + (cc + 0.5) * cell_size
            y = min_y + (rr + 0.5) * cell_size
        node_xy[nid] = (x, y)
        bid = str(nf.get("big_square_id", ""))
        node_big[nid] = bid
        big_to_nodes[bid].append(nid)

    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for src, row in edge_features.items():
        if src not in node_xy or not isinstance(row, dict):
            continue
        for dst, ef in row.items():
            if dst not in node_xy:
                continue
            w = _safe_float(ef.get("distance_m") if isinstance(ef, dict) else None)
            if math.isfinite(w) and w > 0.0:
                adj[src].append((dst, w))

    return {
        "node_xy": node_xy,
        "node_big": node_big,
        "big_to_nodes": big_to_nodes,
        "adj": adj,
    }


class _PortalReconstructor:
    def __init__(
        self,
        smallgraph: dict,
        biggraph: dict,
        smallgraph_path: str,
        biggraph_path: str,
        small_idx: dict,
        k_portals_per_side: int = 5,
    ):
        if precompute_big_square_portal_routes_weight_over_distance is None:
            raise ValueError("Portal route precompute module is unavailable")
        self.smallgraph = smallgraph
        self.biggraph = biggraph
        self.smallgraph_path = smallgraph_path
        self.biggraph_path = biggraph_path
        self.small_idx = small_idx
        self.k = int(k_portals_per_side)
        self.sides = ("N", "S", "E", "W")

        self.small_nodes = smallgraph.get("node_features", {})
        self.small_edges = smallgraph.get("edge_features", {})
        self.big_nodes = biggraph.get("node_features", {})
        self.big_edges = biggraph.get("edge_features", {})
        self.big_ids = sorted(self.big_nodes.keys())
        self.id_to_rc = {bid: self._parse_big_id(bid) for bid in self.big_ids}

        self.raw_cache: Dict[Tuple[str, str, str], dict] = {}
        self.portals_side: Dict[str, Dict[str, List[str]]] = {
            bid: {s: [] for s in self.sides} for bid in self.big_ids
        }
        self.portal_idx_map: Dict[str, Dict[str, Dict[str, List[int]]]] = {
            bid: {s: {} for s in self.sides} for bid in self.big_ids
        }
        self.nearest_idx_cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]] = {}
        self.boundary_match: Dict[Tuple[str, str, str], Optional[str]] = {}

        self.small_rc: Dict[str, Tuple[int, int]] = {}
        self.small_xy: Dict[str, Tuple[float, float]] = {}
        for cid, nf in self.small_nodes.items():
            try:
                rr = int(nf.get("row", 0))
                cc = int(nf.get("col", 0))
            except Exception:
                continue
            self.small_rc[cid] = (rr, cc)
            c = nf.get("centroid_m")
            if isinstance(c, list) and len(c) == 2:
                self.small_xy[cid] = (float(c[0]), float(c[1]))
            else:
                self.small_xy[cid] = small_idx["node_xy"].get(cid, (0.0, 0.0))

        self.small_adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for su, neis in self.small_edges.items():
            if not isinstance(neis, dict):
                continue
            for sv, ef in neis.items():
                d = _safe_float(ef.get("distance_m") if isinstance(ef, dict) else None)
                if math.isfinite(d) and d > 0.0:
                    self.small_adj[su].append((sv, d))

    @staticmethod
    def _parse_big_id(bid: str) -> Tuple[int, int]:
        if not isinstance(bid, str) or not bid.startswith("br") or "_bc" not in bid:
            raise ValueError(f"Bad big-square id format: {bid}")
        left, right = bid.split("_bc", 1)
        return int(left[2:]), int(right)

    @staticmethod
    def _lex_better(a_big: float, a_inside: float, b_big: float, b_inside: float) -> bool:
        if a_big < b_big - 1e-12:
            return True
        if abs(a_big - b_big) <= 1e-12 and a_inside < b_inside - 1e-12:
            return True
        return False

    @staticmethod
    def _opposite(side: str) -> str:
        return {"N": "S", "S": "N", "E": "W", "W": "E"}[side]

    def _side_out_from_to(self, u: str, v: str) -> Optional[str]:
        ur, uc = self.id_to_rc[u]
        vr, vc = self.id_to_rc[v]
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

    def _get_raw_res(self, bid: str, side_in: str, side_out: str) -> dict:
        key = (bid, side_in, side_out)
        if key not in self.raw_cache:
            self.raw_cache[key] = precompute_big_square_portal_routes_weight_over_distance(
                bid,
                side_in,
                side_out,
                k_portals_per_side=self.k,
                small_graph_path=self.smallgraph_path,
                big_graph_path=self.biggraph_path,
            )
        return self.raw_cache[key]

    def _ensure_portals_for(self, bid: str, side: str) -> None:
        if self.portals_side[bid][side]:
            return
        res = self._get_raw_res(bid, side, side)
        plist = list(res.get("portals_in", []))
        self.portals_side[bid][side] = plist
        idx_map: Dict[str, List[int]] = {}
        for idx, pid in enumerate(plist):
            idx_map.setdefault(pid, []).append(idx)
        self.portal_idx_map[bid][side] = idx_map

    def _nearest_portal_idx_by_small_distance(self, src_portal: str, target_portals: List[str]) -> Optional[int]:
        key = (src_portal, tuple(target_portals))
        if key in self.nearest_idx_cache:
            return self.nearest_idx_cache[key]

        if not target_portals:
            self.nearest_idx_cache[key] = None
            return None

        target_set = set(target_portals)
        if src_portal in target_set:
            idx0 = target_portals.index(src_portal)
            self.nearest_idx_cache[key] = idx0
            return idx0

        dist = {src_portal: 0.0}
        pq = [(0.0, src_portal)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, INF):
                continue
            if u in target_set:
                idx = target_portals.index(u)
                self.nearest_idx_cache[key] = idx
                return idx
            for v, w in self.small_adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        sx, sy = self.small_xy.get(src_portal, (0.0, 0.0))
        best_i = 0
        best_d2 = INF
        for i, pid in enumerate(target_portals):
            tx, ty = self.small_xy.get(pid, (0.0, 0.0))
            d2 = (sx - tx) * (sx - tx) + (sy - ty) * (sy - ty)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self.nearest_idx_cache[key] = best_i
        return best_i

    def _matched_next_portal_id(self, u: str, v: str, side_out: str, p_out: str) -> Optional[str]:
        key = (u, v, p_out)
        if key in self.boundary_match:
            return self.boundary_match[key]

        side_in_v = self._opposite(side_out)
        self._ensure_portals_for(v, side_in_v)
        v_portals = self.portals_side[v][side_in_v]
        if not v_portals:
            self.boundary_match[key] = None
            return None

        rc = self.small_rc.get(p_out)
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
                    if self.small_rc.get(pid) == target_rc:
                        self.boundary_match[key] = pid
                        return pid

        idx = self._nearest_portal_idx_by_small_distance(p_out, v_portals)
        if idx is None:
            self.boundary_match[key] = None
            return None
        if idx < 0 or idx >= len(v_portals):
            self.boundary_match[key] = None
            return None
        self.boundary_match[key] = v_portals[idx]
        return v_portals[idx]

    def _inside_mat(self, u: str, side_in: str, side_out: str) -> Tuple[List[List[float]], List[str], List[str]]:
        res = self._get_raw_res(u, side_in, side_out)
        pins = list(res.get("portals_in", []))
        pouts = list(res.get("portals_out", []))
        table = res.get("table", {})
        mat = [[INF for _ in range(len(pouts))] for _ in range(len(pins))]
        for i, pin in enumerate(pins):
            row = table.get(pin, {}) if isinstance(table, dict) else {}
            for j, pout in enumerate(pouts):
                rec = row.get(pout, {}) if isinstance(row, dict) else {}
                d = _safe_float(rec.get("distance_m") if isinstance(rec, dict) else None)
                mat[i][j] = d
        return mat, pins, pouts

    def _reconstruct_state_chain(
        self,
        end_state: Optional[Tuple[str, str, str]],
        state_parent: Dict[Tuple[str, str, str], Optional[Tuple[Tuple[str, str, str], str]]],
    ) -> List[Tuple[str, str, str]]:
        if end_state is None:
            return []
        chain: List[Tuple[str, str, str]] = []
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

    def _state_dijkstra_pair(
        self, src: str, dst: str
    ) -> Tuple[List[Tuple[str, str, str]], Dict[Tuple[str, str, str], Optional[Tuple[Tuple[str, str, str], str]]], Dict[Tuple[str, str, str], str]]:
        state_best: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        state_parent: Dict[Tuple[str, str, str], Optional[Tuple[Tuple[str, str, str], str]]] = {}
        state_first_out: Dict[Tuple[str, str, str], str] = {}
        pq: List[Tuple[float, float, str, str, str]] = []

        for nxt, ef in self.big_edges.get(src, {}).items():
            if ef.get("is_diagonal"):
                continue
            side_out = self._side_out_from_to(src, nxt)
            if side_out is None:
                continue
            side_in_next = self._opposite(side_out)
            edge_d = float(ef.get("distance_m", 1.0))
            self._ensure_portals_for(src, side_out)
            for p_out in self.portals_side[src][side_out]:
                p_in_next = self._matched_next_portal_id(src, nxt, side_out, p_out)
                if p_in_next is None:
                    continue
                key = (nxt, side_in_next, p_in_next)
                cand_big = edge_d
                cand_inside = 0.0
                old = state_best.get(key, (INF, INF))
                if self._lex_better(cand_big, cand_inside, old[0], old[1]):
                    state_best[key] = (cand_big, cand_inside)
                    state_parent[key] = None
                    state_first_out[key] = p_out
                    heapq.heappush(pq, (cand_big, cand_inside, nxt, side_in_next, p_in_next))

        node_best_pair: Dict[str, Tuple[float, float]] = {src: (0.0, 0.0)}
        node_best_state: Dict[str, Optional[Tuple[str, str, str]]] = {src: None}

        while pq:
            cur_big, cur_inside, u, entry_side, entry_portal = heapq.heappop(pq)
            cur_key = (u, entry_side, entry_portal)
            best_pair = state_best.get(cur_key, (INF, INF))
            if abs(cur_big - best_pair[0]) > 1e-12 or abs(cur_inside - best_pair[1]) > 1e-12:
                continue

            old_node = node_best_pair.get(u, (INF, INF))
            if self._lex_better(cur_big, cur_inside, old_node[0], old_node[1]):
                node_best_pair[u] = (cur_big, cur_inside)
                node_best_state[u] = cur_key

            for v, ef in self.big_edges.get(u, {}).items():
                if ef.get("is_diagonal"):
                    continue
                side_out = self._side_out_from_to(u, v)
                if side_out is None:
                    continue
                side_in_next = self._opposite(side_out)
                edge_d = float(ef.get("distance_m", 1.0))

                self._ensure_portals_for(u, entry_side)
                self._ensure_portals_for(u, side_out)
                mat, _, pouts = self._inside_mat(u, entry_side, side_out)
                entry_idx_list = self.portal_idx_map[u][entry_side].get(entry_portal, [])
                if not entry_idx_list:
                    continue

                for idx_out, p_out in enumerate(pouts):
                    p_in_next = self._matched_next_portal_id(u, v, side_out, p_out)
                    if p_in_next is None:
                        continue
                    portal_cost = INF
                    for idx_in in entry_idx_list:
                        if 0 <= idx_in < len(mat):
                            val = mat[idx_in][idx_out]
                            if val < portal_cost:
                                portal_cost = val
                    if not math.isfinite(portal_cost):
                        continue
                    cand_big = cur_big + edge_d
                    cand_inside = cur_inside + portal_cost
                    key = (v, side_in_next, p_in_next)
                    old = state_best.get(key, (INF, INF))
                    if self._lex_better(cand_big, cand_inside, old[0], old[1]):
                        state_best[key] = (cand_big, cand_inside)
                        state_parent[key] = (cur_key, p_out)
                        heapq.heappush(pq, (cand_big, cand_inside, v, side_in_next, p_in_next))

        end_state = node_best_state.get(dst)
        chain = self._reconstruct_state_chain(end_state, state_parent)
        return chain, state_parent, state_first_out

    def _connect_small_nodes(self, a: str, b: str) -> List[str]:
        if a == b:
            return [a]
        path, _ = _astar_path(a, b, self.small_idx)
        return path

    def path_big_to_big(self, src: str, dst: str) -> List[str]:
        if src == dst:
            return []
        chain, state_parent, state_first_out = self._state_dijkstra_pair(src, dst)
        if not chain:
            return []

        nodes: List[str] = []
        first_state = chain[0]
        first_in = first_state[2]
        first_out = state_first_out.get(first_state)
        if first_out is None:
            return []
        nodes.append(first_out)
        if first_in != first_out:
            nodes.append(first_in)

        for t in range(len(chain) - 1):
            cur_state = chain[t]
            nxt_state = chain[t + 1]
            u = cur_state[0]
            side_in = cur_state[1]
            pin = cur_state[2]
            side_out = self._side_out_from_to(u, nxt_state[0])
            if side_out is None:
                continue

            parent_rec = state_parent.get(nxt_state)
            if parent_rec is None:
                continue
            prev_state, pout = parent_rec
            if prev_state != cur_state:
                continue

            res = self._get_raw_res(u, side_in, side_out)
            rec = res.get("table", {}).get(pin, {}).get(pout, {})
            path_inside = rec.get("path")
            if isinstance(path_inside, list) and path_inside:
                if nodes[-1] == path_inside[0]:
                    nodes.extend(path_inside[1:])
                else:
                    conn = self._connect_small_nodes(nodes[-1], path_inside[0])
                    if conn:
                        nodes.extend(conn[1:])
                    nodes.extend(path_inside[1:] if nodes and nodes[-1] == path_inside[0] else path_inside)

            pin_next = nxt_state[2]
            if nodes and nodes[-1] != pin_next:
                conn = self._connect_small_nodes(nodes[-1], pin_next)
                if conn:
                    nodes.extend(conn[1:])
                else:
                    nodes.append(pin_next)

        return nodes


def _pick_node_in_big(big_id: str, target_xy: Tuple[float, float], small_idx: dict) -> str:
    nodes = small_idx["big_to_nodes"].get(big_id, [])
    if not nodes:
        raise ValueError(f"No valid small nodes for big cell: {big_id}")
    tx, ty = target_xy
    node_xy = small_idx["node_xy"]
    return min(nodes, key=lambda n: (node_xy[n][0] - tx) ** 2 + (node_xy[n][1] - ty) ** 2)


def _heuristic(a: str, b: str, node_xy: Dict[str, Tuple[float, float]]) -> float:
    ax, ay = node_xy[a]
    bx, by = node_xy[b]
    return math.hypot(ax - bx, ay - by)


def _astar_path(start: str, goal: str, small_idx: dict) -> Tuple[List[str], float]:
    if start == goal:
        return [start], 0.0

    adj = small_idx["adj"]
    node_xy = small_idx["node_xy"]
    open_heap: List[Tuple[float, float, str]] = []
    heapq.heappush(open_heap, (_heuristic(start, goal, node_xy), 0.0, start))

    best_g: Dict[str, float] = {start: 0.0}
    parent: Dict[str, str] = {}
    visited = set()

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = [goal]
            x = goal
            while x in parent:
                x = parent[x]
                path.append(x)
            path.reverse()
            return path, g

        for nxt, w in adj.get(cur, []):
            ng = g + w
            if ng + 1e-9 < best_g.get(nxt, INF):
                best_g[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + _heuristic(nxt, goal, node_xy), ng, nxt))

    return [], INF


def _build_detailed_route_xy_astar(
    way: List[str], big_cells: Dict[str, dict], small_idx: dict
) -> Tuple[List[Tuple[float, float]], float]:
    if not way:
        return [], 0.0
    first_big = way[0]
    first_center = big_cells.get(first_big, {}).get("center")
    if first_center is None:
        raise ValueError(f"Big cell not found in graph: {first_big}")

    current_node = _pick_node_in_big(first_big, first_center, small_idx)
    start_node = current_node
    route_nodes = [current_node]
    total_m = 0.0
    node_big = small_idx["node_big"]
    node_xy = small_idx["node_xy"]

    for idx, next_big in enumerate(way[1:], start=1):
        is_last = idx == (len(way) - 1)
        if next_big not in big_cells:
            raise ValueError(f"Big cell not found in graph: {next_big}")
        if is_last and next_big == first_big:
            if current_node != start_node:
                seg_path, seg_m = _astar_path(current_node, start_node, small_idx)
                if not seg_path or not math.isfinite(seg_m):
                    raise ValueError(f"No small-cell path found to close loop: {node_big.get(current_node)} -> {first_big}")
                route_nodes.extend(seg_path[1:])
                total_m += seg_m
            continue
        if node_big.get(current_node) == next_big:
            continue
        target_node = _pick_node_in_big(next_big, node_xy[current_node], small_idx)
        seg_path, seg_m = _astar_path(current_node, target_node, small_idx)
        if not seg_path or not math.isfinite(seg_m):
            raise ValueError(f"No small-cell path found between big cells: {node_big.get(current_node)} -> {next_big}")
        route_nodes.extend(seg_path[1:])
        total_m += seg_m
        current_node = target_node

    xy = [node_xy[n] for n in route_nodes]
    return xy, total_m


def _build_detailed_route_xy(
    way: List[str],
    big_cells: Dict[str, dict],
    small_idx: dict,
    portal_router: Optional[_PortalReconstructor],
) -> Tuple[List[Tuple[float, float]], float]:
    if not way:
        return [], 0.0
    if portal_router is None:
        return _build_detailed_route_xy_astar(way, big_cells, small_idx)

    node_xy = small_idx["node_xy"]
    route_nodes: List[str] = []

    for u, v in zip(way, way[1:]):
        seg_nodes = portal_router.path_big_to_big(u, v)
        if not seg_nodes:
            if not route_nodes:
                c0 = big_cells.get(u, {}).get("center")
                if c0 is None:
                    raise ValueError(f"Big cell not found in graph: {u}")
                route_nodes.append(_pick_node_in_big(u, c0, small_idx))
            start = route_nodes[-1]
            target = _pick_node_in_big(v, node_xy[start], small_idx)
            fb_path, _ = _astar_path(start, target, small_idx)
            if not fb_path:
                raise ValueError(f"No path found for pair {u}->{v}")
            seg_nodes = fb_path

        if not route_nodes:
            route_nodes.extend(seg_nodes)
        else:
            if route_nodes[-1] != seg_nodes[0]:
                conn, _ = _astar_path(route_nodes[-1], seg_nodes[0], small_idx)
                if conn:
                    route_nodes.extend(conn[1:])
            if route_nodes[-1] == seg_nodes[0]:
                route_nodes.extend(seg_nodes[1:])
            else:
                route_nodes.extend(seg_nodes)

    if len(way) >= 2 and way[0] == way[-1] and route_nodes and route_nodes[0] != route_nodes[-1]:
        close, _ = _astar_path(route_nodes[-1], route_nodes[0], small_idx)
        if close:
            route_nodes.extend(close[1:])

    xy = [node_xy[n] for n in route_nodes if n in node_xy]
    total_m = _polyline_length_m(xy)
    return xy, total_m


def _polyline_length_m(xy: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(xy) - 1):
        total += math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1])
    return total


def _draw_direction_arrows(
    ax,
    xy: List[Tuple[float, float]],
    arrow_step_km: float = 10.0,
    zorder: int = 10,
) -> None:
    if len(xy) < 2:
        return
    step_m = max(0.3, arrow_step_km) * 1000.0
    total_m = _polyline_length_m(xy)
    if total_m <= step_m:
        return

    arrow_len_m = 1900.0
    head_len_m = 520.0
    head_half_w_m = 290.0
    shaft_color = "#4f4f4f"
    shaft_lw = 1.25
    head_lw = 1.25

    target = step_m
    walked = 0.0
    seg_prev = 0
    for i in range(len(xy) - 1):
        x0, y0 = xy[i]
        x1, y1 = xy[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        while walked + seg >= target:
            ratio = (target - walked) / seg
            xm = x0 + ratio * (x1 - x0)
            ym = y0 + ratio * (y1 - y0)
            if i == seg_prev and seg < 1.0:
                target += step_m
                continue
            seg_prev = i

            # Direction unit vector and perpendicular vector for an open V-head arrow.
            ux = (x1 - x0) / seg
            uy = (y1 - y0) / seg
            px = -uy
            py = ux

            sx = xm - ux * arrow_len_m
            sy = ym - uy * arrow_len_m
            ax.plot([sx, xm], [sy, ym], color=shaft_color, linewidth=shaft_lw, alpha=0.95, zorder=zorder)

            hx1 = xm - ux * head_len_m + px * head_half_w_m
            hy1 = ym - uy * head_len_m + py * head_half_w_m
            hx2 = xm - ux * head_len_m - px * head_half_w_m
            hy2 = ym - uy * head_len_m - py * head_half_w_m
            ax.plot([xm, hx1], [ym, hy1], color=shaft_color, linewidth=head_lw, alpha=0.95, zorder=zorder)
            ax.plot([xm, hx2], [ym, hy2], color=shaft_color, linewidth=head_lw, alpha=0.95, zorder=zorder)
            target += step_m
        walked += seg


def _parse_routes(spec: str, patrol_count: int) -> List[int]:
    s = (spec or "").strip().lower()
    if s in ("", "all"):
        return list(range(patrol_count))

    out = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        idx1 = int(p)
        idx0 = idx1 - 1
        if idx0 < 0 or idx0 >= patrol_count:
            raise ValueError(f"Route index out of range: {idx1}, patrol_count={patrol_count}")
        out.append(idx0)
    if not out:
        raise ValueError("No routes selected")
    return out


def _draw(
    result: dict,
    smallgraph: dict,
    layers: dict,
    pr_layers: dict,
    big_cells: Dict[str, dict],
    S: Dict[str, float],
    small_idx: dict,
    portal_router: Optional[_PortalReconstructor],
    route_spec: str,
    out_png: str,
) -> None:
    best = result.get("best")
    if not isinstance(best, dict):
        best = result.get("patrol_best")
    if not isinstance(best, dict):
        raise ValueError("Invalid result JSON: missing best/patrol_best")
    patrols = best.get("patrols")
    if not isinstance(patrols, list) or not patrols:
        raise ValueError("Invalid result JSON: missing patrols")

    use_routes = _parse_routes(route_spec, len(patrols))

    inside_mask = pr_layers["inside_mask"]
    outside_mask = pr_layers["outside_mask"]
    priority_arr = pr_layers["priority_arr"]
    extent = pr_layers["extent"]

    fig, ax = plt.subplots(1, 1, figsize=(13, 10), dpi=220)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    outside_img = np.where(outside_mask, 1.0, np.nan)
    inside_img = np.where(inside_mask, 1.0, np.nan)
    ax.imshow(
        outside_img,
        extent=extent,
        origin="lower",
        cmap=colors.ListedColormap(["#efefef"]),
        interpolation="nearest",
        alpha=1.0,
        zorder=1,
    )
    ax.imshow(
        inside_img,
        extent=extent,
        origin="lower",
        cmap=colors.ListedColormap(["#dcdcdc"]),
        interpolation="nearest",
        alpha=1.0,
        zorder=2,
    )

    inside_pr = np.where(inside_mask, priority_arr, np.nan)
    vmax_pr = float(np.nanpercentile(inside_pr, 99.0)) if np.any(np.isfinite(inside_pr)) else 1.0
    if vmax_pr <= 0.0:
        vmax_pr = 1.0
    im_pr = ax.imshow(
        inside_pr,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
        alpha=0.94,
        zorder=3,
    )
    cbar_pr_small = fig.colorbar(im_pr, ax=ax, fraction=0.035, pad=0.02)
    cbar_pr_small.set_label("Small-cell priority heatmap (clamped)")

    for bid, info in big_cells.items():
        x0, y0, x1, y1 = info["bbox"]
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor="#3b3b3b",
                linewidth=0.45,
                alpha=0.65,
                zorder=5,
            )
        )

    route_colors = ["#0057ff", "#00b050", "#19a974", "#a66bff", "#ff8c00", "#00a7a7"]
    legend_handles: List[Line2D] = []

    for rank, p_idx in enumerate(use_routes):
        patrol = patrols[p_idx]
        way = patrol.get("way", [])
        if not isinstance(way, list) or len(way) < 2:
            continue
        xy, route_len_m = _build_detailed_route_xy(
            way=way,
            big_cells=big_cells,
            small_idx=small_idx,
            portal_router=portal_router,
        )
        if len(xy) < 2:
            continue
        col = route_colors[rank % len(route_colors)]
        xs = [q[0] for q in xy]
        ys = [q[1] for q in xy]

        ax.plot(xs, ys, color=col, linewidth=1.8, alpha=0.96, zorder=8)
        ax.plot(
            [xs[0]],
            [ys[0]],
            marker="s",
            markersize=3.5,
            markerfacecolor=col,
            markeredgecolor="white",
            markeredgewidth=0.45,
            linestyle="None",
            zorder=11,
        )
        ax.annotate(
            f"S{p_idx+1}",
            (xs[0], ys[0]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color=col,
            zorder=12,
        )
        _draw_direction_arrows(ax, xy, arrow_step_km=10.0, zorder=10)

        t_h = _safe_float(patrol.get("time_h"))
        label = f"Route #{p_idx+1}: {route_len_m/1000.0:.1f} km, {t_h:.2f} h"
        legend_handles.append(Line2D([0], [0], color=col, lw=3, label=label))

    # Draw selected sound-tracker border cells on top of map.
    sound_selection = best.get("sound_selection")
    if not isinstance(sound_selection, dict):
        sound_selection = result.get("sound_selection_best")
    sound_rows = sound_selection.get("selected_border_cells", []) if isinstance(sound_selection, dict) else []
    node_features = smallgraph.get("node_features", {}) if isinstance(smallgraph, dict) else {}
    sound_drawn = 0
    for row in sound_rows:
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        nf = node_features.get(cell_id) if isinstance(cell_id, str) else None
        if not isinstance(nf, dict):
            continue
        bbox = nf.get("bbox_m")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x0, y0, x1, y1 = map(float, bbox)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor="#00c2ff",
                linewidth=1.4,
                alpha=0.96,
                zorder=13,
            )
        )
        sound_drawn += 1

    if sound_drawn > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#00c2ff",
                lw=2.2,
                label=f"Sound tracker border cells: {sound_drawn}",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=8)

    k_val = best.get("K")
    ax.set_title(f"Priority map + big-square boundaries | patrol routes (K={k_val})")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed visualization of patrol routes on small-grid paths")
    parser.add_argument("--result", default="patrol_alloc_greedy_unique_result.json")
    parser.add_argument("--smallgraph", default="etosha_grid_graph_with_big_squares.json")
    parser.add_argument("--biggraph", default="etosha_big_square_graph_14x14.json")
    parser.add_argument("--priority", default="etosha_node_priority_compact_clamped.json")
    parser.add_argument("--routes", default="all", help="all or comma-separated route numbers, e.g. 1,2")
    parser.add_argument("--km_step", type=float, default=1.0, help="Deprecated: ignored (kept for CLI compatibility)")
    parser.add_argument("--out", default="patrol_alloc_k2_routes_detailed_map.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smallgraph_path = _resolve_read_path(args.smallgraph)
    biggraph_path = _resolve_read_path(args.biggraph)

    result = _load_json(args.result)
    smallgraph = _load_json(args.smallgraph)
    biggraph = _load_json(args.biggraph)
    priority_small = _load_priority_map(args.priority)

    S = _build_S_big(biggraph, priority_small)
    layers = _small_grid_layers(smallgraph)
    pr_layers = _small_priority_layers(smallgraph, priority_small)
    big_cells = _extract_big_cells(biggraph)
    small_idx = _build_small_graph_index(smallgraph)
    portal_router: Optional[_PortalReconstructor]
    try:
        portal_router = _PortalReconstructor(
            smallgraph=smallgraph,
            biggraph=biggraph,
            smallgraph_path=smallgraph_path,
            biggraph_path=biggraph_path,
            small_idx=small_idx,
            k_portals_per_side=5,
        )
    except Exception:
        portal_router = None

    out_path = _resolve_out_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _draw(
        result=result,
        smallgraph=smallgraph,
        layers=layers,
        pr_layers=pr_layers,
        big_cells=big_cells,
        S=S,
        small_idx=small_idx,
        portal_router=portal_router,
        route_spec=args.routes,
        out_png=out_path,
    )
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
