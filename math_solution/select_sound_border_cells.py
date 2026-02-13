from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


CELL_ID_RE = re.compile(r"^r\d+_c\d+$")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_GRAPH_PATH = "solution/etosha_grid_graph_with_big_squares.json"
DEFAULT_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_SCORE_FIELD = "priority_P_i"


@dataclass(frozen=True)
class BorderCellScore:
    cell_id: str
    priority: float
    row: int
    col: int
    centroid_m: List[float]


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


def _load_json(path: str) -> Dict[str, Any]:
    resolved = _resolve_read_path(path)
    with open(resolved, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_priority_map(priority_data: Dict[str, Any], score_field: str) -> Dict[str, float]:
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

    out: Dict[str, float] = {}
    for k, v in priority_data.items():
        if not isinstance(k, str) or not CELL_ID_RE.match(k):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
        elif isinstance(v, dict):
            if score_field in v:
                out[k] = float(v[score_field])
            elif "priority_P_i" in v:
                out[k] = float(v["priority_P_i"])
    return out


def _resolve_priority_data(
    priority_path: str,
    graph: Dict[str, Any],
    score_field: str,
) -> Dict[str, Any]:
    candidate_paths = [
        priority_path,
        "solution/etosha_node_priority_compact_clamped.json",
        "solution/etosha_node_priority_compact.json",
        "solution/etosha_node_priority.json",
        "etosha_node_priority_compact_clamped.json",
        "etosha_node_priority_compact.json",
        "etosha_node_priority.json",
    ]

    for path in candidate_paths:
        try:
            data = _load_json(path)
        except FileNotFoundError:
            continue
        if _extract_priority_map(data, score_field=score_field):
            return data

    # Optional fallback: compute priorities if module exists.
    try:
        from solution.compute_node_priority import compute_priorities, build_compact_priority_map
    except ModuleNotFoundError:
        try:
            from compute_node_priority import compute_priorities, build_compact_priority_map
        except ModuleNotFoundError as e:
            raise ValueError("No valid priority file found and compute_node_priority module is unavailable") from e

    full = compute_priorities(graph)
    compact = build_compact_priority_map(full)
    return compact


def _extract_border_cells(
    graph: Dict[str, Any],
    priority_data: Dict[str, Any],
    score_field: str = DEFAULT_SCORE_FIELD,
) -> List[BorderCellScore]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    node_priority = _extract_priority_map(priority_data, score_field=score_field)
    if not node_priority:
        raise ValueError("No node priorities found in priority file (unsupported format)")

    by_rc: Dict[tuple, str] = {}
    inside_mask: Dict[str, bool] = {}
    for cid, nf in nodes.items():
        if not isinstance(nf, dict):
            continue
        rr = int(nf.get("row", -1))
        cc = int(nf.get("col", -1))
        if rr >= 0 and cc >= 0:
            by_rc[(rr, cc)] = cid
        inside_mask[cid] = nf.get("median_elevation_m") is not None

    out: List[BorderCellScore] = []
    for cell_id, node_feat in nodes.items():
        if not isinstance(node_feat, dict):
            continue
        # True park border: inside cell with at least one outside (or missing) 4-neighbor.
        if node_feat.get("median_elevation_m") is None:
            continue
        rr = int(node_feat.get("row", -1))
        cc = int(node_feat.get("col", -1))
        if rr < 0 or cc < 0:
            continue
        has_outside_neighbor = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb_id = by_rc.get((rr + dr, cc + dc))
            if nb_id is None:
                has_outside_neighbor = True
                break
            if not inside_mask.get(nb_id, False):
                has_outside_neighbor = True
                break
        if not has_outside_neighbor:
            continue

        if cell_id not in node_priority:
            continue

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
    score_field: str = DEFAULT_SCORE_FIELD,
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
    all_selected_are_border = True

    covered_km = selected_count * km_per_border_cell
    unallocated_km = max(0.0, sound_km - covered_km)
    selected_priority_sum = float(sum(c.priority for c in selected))

    return {
        "meta": {
            "selection_policy": (
                "top_priority_on_derived_park_boundary_desc_then_cell_id "
                "(derived boundary = inside cell with >=1 outside 4-neighbor)"
            ),
            "score_field": score_field,
            "sound_km_requested": sound_km,
            "km_per_border_cell": km_per_border_cell,
            "requested_cell_count": requested_cell_count,
            "selected_cell_count": selected_count,
            "total_border_cells_available": len(border_cells),
            "all_selected_are_border": all_selected_are_border,
            "covered_km": covered_km,
            "unallocated_km": unallocated_km,
            "selected_priority_sum": selected_priority_sum,
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


def build_sound_selection(
    sound_km: float,
    graph_path: str = DEFAULT_GRAPH_PATH,
    priority_path: str = DEFAULT_PRIORITY_PATH,
    score_field: str = DEFAULT_SCORE_FIELD,
) -> Dict[str, Any]:
    graph = _load_json(graph_path)
    priority_data = _resolve_priority_data(priority_path, graph=graph, score_field=score_field)
    return select_border_cells_for_sound(graph, priority_data, sound_km=sound_km, score_field=score_field)
