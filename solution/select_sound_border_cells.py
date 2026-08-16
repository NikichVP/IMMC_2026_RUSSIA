from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List


CELL_ID_RE = re.compile(r"^r\d+_c\d+$")


@dataclass(frozen=True)
class BorderCellScore:
    cell_id: str
    priority: float
    row: int
    col: int
    centroid_m: List[float]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_priority_map(priority_data: Dict[str, Any], score_field: str) -> Dict[str, float]:
    # Old/verbose format: {"meta": ..., "node_priority": {cell_id: {"priority_P_i": ...}}}
    if isinstance(priority_data.get("node_priority"), dict):
        out: Dict[str, float] = {}
        for cell_id, rec in priority_data["node_priority"].items():
            if isinstance(rec, dict):
                if score_field in rec:
                    out[cell_id] = float(rec[score_field])
                elif "priority_P_i" in rec:
                    out[cell_id] = float(rec["priority_P_i"])
            elif isinstance(rec, (int, float)):
                out[cell_id] = float(rec)
        return out

    # Compact/new format: {cell_id: score} OR {cell_id: {"priority_P_i": score}}
    out: Dict[str, float] = {}
    for k, v in priority_data.items():
        if not isinstance(k, str) or not CELL_ID_RE.match(k):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
            continue
        if isinstance(v, dict):
            if score_field in v:
                out[k] = float(v[score_field])
            elif "priority_P_i" in v:
                out[k] = float(v["priority_P_i"])
    return out


def _clamp_priority_to_valid_map(
    graph: Dict[str, Any],
    priority_map: Dict[str, float],
) -> Dict[str, float]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    out: Dict[str, float] = {}
    for cid, nf in nodes.items():
        if nf.get("median_elevation_m") is None:
            out[cid] = 0.0
        else:
            out[cid] = float(priority_map.get(cid, 0.0))
    return out


def _resolve_priority_data(
    priority_path: str,
    graph: Dict[str, Any],
    score_field: str,
) -> Dict[str, Any]:
    candidate_paths = [
        priority_path,
        "etosha_node_priority_compact_clamped.json",
        "etosha_node_priority.json",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            data = _load_json(path)
            if _extract_priority_map(data, score_field=score_field):
                print(f"Using priority file: {path}")
                return data

    print("No valid priority file found, computing priorities on the fly...")
    from compute_node_priority import compute_priorities, build_compact_priority_map

    full = compute_priorities(graph)
    compact = build_compact_priority_map(full)
    clamped = _clamp_priority_to_valid_map(graph=graph, priority_map=compact)
    with open("etosha_node_priority_compact_clamped.json", "w", encoding="utf-8") as f:
        json.dump(clamped, f, ensure_ascii=False)
    print("Saved computed priorities to: etosha_node_priority_compact_clamped.json")
    return clamped


def _extract_border_cells(
    graph: Dict[str, Any],
    priority_data: Dict[str, Any],
    score_field: str = "priority_P_i",
) -> List[BorderCellScore]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    node_priority = _extract_priority_map(priority_data, score_field=score_field)
    if not node_priority:
        raise ValueError("No node priorities found in priority file (unsupported format)")

    out: List[BorderCellScore] = []
    for cell_id, node_feat in nodes.items():
        is_border = bool(node_feat.get("is_boarder", node_feat.get("is_border", False)))
        if not is_border:
            continue

        if cell_id not in node_priority:
            raise ValueError(f"Priority for border cell '{cell_id}' not found in priority file")

        centroid = node_feat.get("centroid_m") or [0.0, 0.0]
        if not isinstance(centroid, list) or len(centroid) != 2:
            centroid = [0.0, 0.0]

        out.append(
            BorderCellScore(
                cell_id=cell_id,
                priority=float(node_priority[cell_id]),
                row=int(node_feat.get("row", -1)),
                col=int(node_feat.get("col", -1)),
                centroid_m=[float(centroid[0]), float(centroid[1])],
            )
        )

    out.sort(key=lambda x: (-x.priority, x.cell_id))
    return out


def select_border_cells_for_sound(
    graph: Dict[str, Any],
    priority_data: Dict[str, Any],
    sound_km: float,
    *,
    score_field: str = "priority_P_i",
) -> Dict[str, Any]:
    if sound_km < 0:
        raise ValueError("sound_km must be non-negative")

    cell_size_m = float(graph.get("meta", {}).get("cell_size_m", 1000.0))
    if cell_size_m <= 0:
        raise ValueError("Invalid graph.meta.cell_size_m, expected > 0")

    km_per_border_cell = cell_size_m / 1000.0
    requested_cell_count = int(math.floor(sound_km / km_per_border_cell + 1e-12))

    border_cells = _extract_border_cells(graph, priority_data, score_field=score_field)
    selected_count = min(requested_cell_count, len(border_cells))
    selected = border_cells[:selected_count]

    covered_km = selected_count * km_per_border_cell
    unallocated_km = max(0.0, sound_km - covered_km)

    return {
        "meta": {
            "score_field": score_field,
            "sound_km_requested": sound_km,
            "km_per_border_cell": km_per_border_cell,
            "requested_cell_count": requested_cell_count,
            "selected_cell_count": selected_count,
            "total_border_cells_available": len(border_cells),
            "covered_km": covered_km,
            "unallocated_km": unallocated_km,
        },
        "selected_border_cells": [
            {
                "cell_id": c.cell_id,
                "priority": c.priority,
                "row": c.row,
                "col": c.col,
                "centroid_m": c.centroid_m,
            }
            for c in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select highest-priority border cells for sound coverage by requested km"
    )
    parser.add_argument("sound_km", type=float, help="How many km of border can be covered by sound")
    parser.add_argument("--graph", default="etosha_grid_graph.json", help="Path to graph JSON")
    parser.add_argument(
        "--priority",
        default="etosha_node_priority_compact_clamped.json",
        help="Path to node priority JSON (compact or verbose)",
    )
    parser.add_argument(
        "--score-field",
        default="priority_P_i",
        help="Score field for dict-based priority formats (ignored for compact float map)",
    )
    parser.add_argument(
        "--out",
        default="sound_border_cell_selection.json",
        help="Output JSON with selected border cells",
    )
    args = parser.parse_args()

    graph = _load_json(args.graph)
    priority_data = _resolve_priority_data(args.priority, graph=graph, score_field=args.score_field)
    result = select_border_cells_for_sound(
        graph,
        priority_data,
        sound_km=args.sound_km,
        score_field=args.score_field,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    meta = result["meta"]
    print(f"Saved {meta['selected_cell_count']} border cells -> {args.out}")
    print(
        f"Requested: {meta['sound_km_requested']:.3f} km, "
        f"Covered: {meta['covered_km']:.3f} km, "
        f"Unallocated: {meta['unallocated_km']:.3f} km"
    )


if __name__ == "__main__":
    main()
