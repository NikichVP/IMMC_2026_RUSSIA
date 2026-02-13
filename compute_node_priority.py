from __future__ import annotations

import argparse
import heapq
import json
import math
from typing import Any, Dict, Iterable, List, Tuple


# =========================
# Tunable constants
# =========================
TOP_K_NEIGHBORS = 1000

W_ANIMAL = 1.00
W_WATER = 0.50
W_PLANT = 0.50
W_COVERAGE_PENALTY = 0.50

# Layer weights now come from a decay function and are normalized to 1.
# raw(h) = exp(-HOP_DECAY_ALPHA * h), h in {0(self), 1, 2}
HOP_DECAY_ALPHA = 1.0

DEFAULT_UNKNOWN_ANIMAL_WEIGHT = 0.0
ANIMAL_WEIGHTS: Dict[str, float] = {
    "all_lions": 1.00,
    "all_black_rhyno": 0.95,
    "all_white_rhino": 0.90,
    "all_elephants": 0.75,
    "all_cheetah": 0.80,
    "all_leopard": 0.85,
    "all_hyena": 0.50,
    "all_zebra": 0.35,
    "dry_zebra": 0.30,
    "wet_zebra": 0.30,
}


def edge_travel_time_hours(edge_feat: Dict[str, Any]) -> float:
    distance_m = float(edge_feat.get("distance_m", 1000.0))
    road_between_cells = bool(edge_feat.get("road_between_cells", False))
    speed_kmh = 60.0 if road_between_cells else 30.0
    return distance_m / (speed_kmh * 1000.0)


def avg(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        return 0.0
    return sum(data) / len(data)


def _normalized_layer_weights(use_hop1: bool, use_hop2: bool) -> Tuple[float, float, float]:
    raw_self = math.exp(-HOP_DECAY_ALPHA * 0.0)
    raw_hop1 = math.exp(-HOP_DECAY_ALPHA * 1.0) if use_hop1 else 0.0
    raw_hop2 = math.exp(-HOP_DECAY_ALPHA * 2.0) if use_hop2 else 0.0
    raw_sum = raw_self + raw_hop1 + raw_hop2
    if raw_sum <= 0.0:
        return 1.0, 0.0, 0.0
    return raw_self / raw_sum, raw_hop1 / raw_sum, raw_hop2 / raw_sum


def top_k_by_time(candidates: List[str], time_by_node: Dict[str, float], k: int) -> List[str]:
    return sorted(candidates, key=lambda cid: (time_by_node.get(cid, float("inf")), cid))[:k]


def compute_local_score(node_feat: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    animals_present = node_feat.get("animals_present") or []
    poi_counts = node_feat.get("poi_type_counts") or {}

    an_i = 0.0
    for animal in animals_present:
        an_i += float(ANIMAL_WEIGHTS.get(animal, DEFAULT_UNKNOWN_ANIMAL_WEIGHT))

    wa_i = 1.0 if int(poi_counts.get("waterhole", 0)) > 0 or int(poi_counts.get("waterhole_dry", 0)) > 0 else 0.0
    pl_i = 1.0 if bool(node_feat.get("has_plant", False)) else 0.0
    cov_i = 1.0 if int(poi_counts.get("patrol_house", 0)) > 0 or int(poi_counts.get("photo_trap", 0)) > 0 else 0.0

    local_raw = W_ANIMAL * an_i + W_WATER * wa_i + W_PLANT * pl_i
    s_i = local_raw * (1.0 - W_COVERAGE_PENALTY * cov_i)

    return s_i, {
        "AN_i": an_i,
        "WA_i": wa_i,
        "PL_i": pl_i,
        "COV_i": cov_i,
    }


def build_hop_sets_and_times(
    node_id: str,
    weighted_adj: Dict[str, List[Tuple[str, float]]],
    top_k_neighbors: int,
) -> Tuple[List[str], Dict[str, float], List[str], Dict[str, float]]:
    n1_time: Dict[str, float] = {}
    for n1, t_1 in weighted_adj.get(node_id, []):
        n1_time[n1] = t_1
    n1_nodes = sorted(n1_time.keys())

    n1_set = set(n1_nodes)
    forward_time: Dict[str, float] = {}
    best_time: Dict[str, float] = {node_id: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, node_id)]

    # Берем до top_k_neighbors ближайших узлов, исключая self и прямых соседей n1.
    while pq and len(forward_time) < top_k_neighbors:
        t_cur, cur = heapq.heappop(pq)
        if t_cur > best_time.get(cur, float("inf")):
            continue

        if cur != node_id and cur not in n1_set and cur not in forward_time:
            forward_time[cur] = t_cur
            if len(forward_time) >= top_k_neighbors:
                break

        for nxt, edge_t in weighted_adj.get(cur, []):
            cand = t_cur + edge_t
            old = best_time.get(nxt)
            if old is None or cand < old:
                best_time[nxt] = cand
                heapq.heappush(pq, (cand, nxt))

    forward_nodes = sorted(forward_time.keys(), key=lambda cid: (forward_time[cid], cid))
    return n1_nodes, n1_time, forward_nodes, forward_time


def compute_priorities(graph: Dict[str, Any], top_k_neighbors: int = TOP_K_NEIGHBORS) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]
    weighted_adj = {
        node_id: [(nxt, edge_travel_time_hours(edge_feat)) for nxt, edge_feat in neis.items()]
        for node_id, neis in edges.items()
    }

    local_score: Dict[str, float] = {}
    local_parts: Dict[str, Dict[str, float]] = {}

    for node_id, feat in nodes.items():
        s_i, parts = compute_local_score(feat)
        local_score[node_id] = s_i
        local_parts[node_id] = parts

    priority_out: Dict[str, Dict[str, Any]] = {}
    for node_id in nodes.keys():
        n1_nodes, n1_time, n2_nodes, n2_time = build_hop_sets_and_times(
            node_id=node_id,
            weighted_adj=weighted_adj,
            top_k_neighbors=top_k_neighbors,
        )

        n1_top = top_k_by_time(n1_nodes, n1_time, top_k_neighbors)
        n2_top = top_k_by_time(n2_nodes, n2_time, top_k_neighbors)

        n1_avg = avg(local_score[n] for n in n1_top)
        n2_avg = avg(local_score[n] for n in n2_top)
        w_self, w_n1, w_n2 = _normalized_layer_weights(bool(n1_top), bool(n2_top))
        p_i = w_self * local_score[node_id] + w_n1 * n1_avg + w_n2 * n2_avg

        priority_out[node_id] = {
            "local_score_S_i": local_score[node_id],
            "priority_P_i": p_i,
            **local_parts[node_id],
            "n1_used_count": len(n1_top),
            "n2_used_count": len(n2_top),
            "n1_used_nodes": n1_top,
            "n2_used_nodes": n2_top,
            "n1_avg_S": n1_avg,
            "n2_avg_S": n2_avg,
            "w_self": w_self,
            "w_n1": w_n1,
            "w_n2": w_n2,
        }

    return {
        "meta": {
            "method": "local_score_plus_neighbors",
            "formula_local": "S_i=(1.00*sum_{t in animals_present(i)}w(t)+0.50*WA_i+0.50*PL_i)*(1-0.50*COV_i)",
            "formula_priority": "P_i=w0*S_i+w1*avg(N1)+w2*avg(FWD_topK), "
            "raw(h)=exp(-alpha*h), h in {0,1,2}, then normalize to sum=1 over available layers",
            "neighbor_scope": "N1 = direct neighbors; FWD_topK = up to top_k closest nodes by shortest-path time excluding self and N1",
            "top_k_neighbors": int(top_k_neighbors),
            "hop_decay_alpha": HOP_DECAY_ALPHA,
            "animal_weights": ANIMAL_WEIGHTS,
            "unknown_animal_weight": DEFAULT_UNKNOWN_ANIMAL_WEIGHT,
        },
        "node_priority": priority_out,
    }


def build_compact_priority_map(full_result: Dict[str, Any]) -> Dict[str, float]:
    node_priority = full_result.get("node_priority", {})
    return {node_id: float(parts["priority_P_i"]) for node_id, parts in node_priority.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute node priorities for each graph cell")
    parser.add_argument("--graph", default="etosha_grid_graph.json", help="Path to graph JSON")
    parser.add_argument(
        "--out",
        default="etosha_node_priority_compact.json",
        help="Output JSON path with compact priorities (node_id -> priority)",
    )
    parser.add_argument(
        "--top-k-neighbors",
        type=int,
        default=TOP_K_NEIGHBORS,
        help="How many nearest forward nodes (excluding self and direct neighbors) to use",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Save full verbose payload instead of compact map",
    )
    args = parser.parse_args()

    if args.top_k_neighbors <= 0:
        raise ValueError("--top-k-neighbors must be > 0")

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)

    result_full = compute_priorities(graph, top_k_neighbors=args.top_k_neighbors)
    if args.full:
        payload: Any = result_full
        count = len(result_full["node_priority"])
    else:
        payload = build_compact_priority_map(result_full)
        count = len(payload)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    mode = "full" if args.full else "compact"
    print(f"Saved {mode} priorities for {count} nodes -> {args.out}")


if __name__ == "__main__":
    main()
