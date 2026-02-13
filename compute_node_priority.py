from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Tuple


# =========================
# Tunable constants
# =========================
TOP_K_NEIGHBORS = 5

W_WATER = 0.30
W_BORDER = 0.20
W_PLANT = 0.10
W_COVERAGE_PENALTY = 0.30

W_SELF = 0.70
W_NEIGH_HOP1 = 0.20
W_NEIGH_HOP2 = 0.10

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


def top_k_by_time(candidates: List[str], time_by_node: Dict[str, float], k: int) -> List[str]:
    return sorted(candidates, key=lambda cid: (time_by_node.get(cid, float("inf")), cid))[:k]


def compute_local_score(node_feat: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    animals_present = node_feat.get("animals_present") or []
    poi_counts = node_feat.get("poi_type_counts") or {}

    an_i = 0.0
    for animal in animals_present:
        an_i = max(an_i, float(ANIMAL_WEIGHTS.get(animal, DEFAULT_UNKNOWN_ANIMAL_WEIGHT)))

    wa_i = 1.0 if int(poi_counts.get("waterhole", 0)) > 0 or int(poi_counts.get("waterhole_dry", 0)) > 0 else 0.0
    br_i = 1.0 if bool(node_feat.get("is_boarder", False)) else 0.0
    pl_i = 1.0 if bool(node_feat.get("has_plant", False)) else 0.0
    cov_i = 1.0 if int(poi_counts.get("patrol_house", 0)) > 0 or int(poi_counts.get("photo_trap", 0)) > 0 else 0.0

    local_raw = an_i + W_WATER * wa_i + W_BORDER * br_i + W_PLANT * pl_i
    s_i = local_raw * (1.0 - W_COVERAGE_PENALTY * cov_i)

    return s_i, {
        "AN_i": an_i,
        "WA_i": wa_i,
        "BR_i": br_i,
        "PL_i": pl_i,
        "COV_i": cov_i,
    }


def build_hop_sets_and_times(
    node_id: str,
    edges: Dict[str, Dict[str, Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, float], List[str], Dict[str, float]]:
    n1_time: Dict[str, float] = {}
    for n1, ef_1 in edges.get(node_id, {}).items():
        n1_time[n1] = edge_travel_time_hours(ef_1)

    n1_nodes = sorted(n1_time.keys())

    n2_time: Dict[str, float] = {}
    n1_set = set(n1_nodes)
    for n1 in n1_nodes:
        t_1 = n1_time[n1]
        for n2, ef_2 in edges.get(n1, {}).items():
            if n2 == node_id or n2 in n1_set:
                continue
            t_2 = t_1 + edge_travel_time_hours(ef_2)
            old = n2_time.get(n2)
            if old is None or t_2 < old:
                n2_time[n2] = t_2

    n2_nodes = sorted(n2_time.keys())
    return n1_nodes, n1_time, n2_nodes, n2_time


def compute_priorities(graph: Dict[str, Any], top_k_neighbors: int = TOP_K_NEIGHBORS) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]

    local_score: Dict[str, float] = {}
    local_parts: Dict[str, Dict[str, float]] = {}

    for node_id, feat in nodes.items():
        s_i, parts = compute_local_score(feat)
        local_score[node_id] = s_i
        local_parts[node_id] = parts

    priority_out: Dict[str, Dict[str, Any]] = {}
    for node_id in nodes.keys():
        n1_nodes, n1_time, n2_nodes, n2_time = build_hop_sets_and_times(node_id, edges)

        n1_top = top_k_by_time(n1_nodes, n1_time, top_k_neighbors)
        n2_top = top_k_by_time(n2_nodes, n2_time, top_k_neighbors)

        n1_avg = avg(local_score[n] for n in n1_top)
        n2_avg = avg(local_score[n] for n in n2_top)
        p_i = W_SELF * local_score[node_id] + W_NEIGH_HOP1 * n1_avg + W_NEIGH_HOP2 * n2_avg

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
        }

    return {
        "meta": {
            "method": "local_score_plus_neighbors",
            "formula_local": "S_i=(AN_i+0.30*WA_i+0.20*BR_i+0.10*PL_i)*(1-0.30*COV_i)",
            "formula_priority": f"P_i=0.70*S_i+0.20*avg(N1_top{top_k_neighbors})+0.10*avg(N2_top{top_k_neighbors})",
            "top_k_neighbors": int(top_k_neighbors),
            "animal_weights": ANIMAL_WEIGHTS,
            "unknown_animal_weight": DEFAULT_UNKNOWN_ANIMAL_WEIGHT,
        },
        "node_priority": priority_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute node priorities for each graph cell")
    parser.add_argument("--graph", default="etosha_grid_graph.json", help="Path to graph JSON")
    parser.add_argument(
        "--out",
        default="etosha_node_priority.json",
        help="Output JSON path with local and final priority scores",
    )
    parser.add_argument(
        "--top-k-neighbors",
        type=int,
        default=TOP_K_NEIGHBORS,
        help="How many nearest neighbors to use per hop layer (N1 and N2)",
    )
    args = parser.parse_args()

    if args.top_k_neighbors <= 0:
        raise ValueError("--top-k-neighbors must be > 0")

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)

    result = compute_priorities(graph, top_k_neighbors=args.top_k_neighbors)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"Saved priorities for {len(result['node_priority'])} nodes -> {args.out}")


if __name__ == "__main__":
    main()
