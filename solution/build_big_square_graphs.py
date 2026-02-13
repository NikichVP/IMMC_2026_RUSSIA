from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple


DEFAULT_INPUT_GRAPH = "etosha_grid_graph.json"
DEFAULT_OUT_SMALL_WITH_BIG = "etosha_grid_graph_with_big_squares.json"
DEFAULT_OUT_BIG_GRAPH = "etosha_big_square_graph_14x14.json"
DEFAULT_PRIORITY_PATH = "etosha_node_priority_compact_clamped.json"
DEFAULT_SQUARE_SIZE = 14


def big_square_id(big_row: int, big_col: int) -> str:
    return f"br{big_row}_bc{big_col}"


def _require_meta_field(meta: Dict[str, Any], key: str) -> Any:
    if key not in meta:
        raise ValueError(f"Required meta field '{key}' is missing in input graph")
    return meta[key]


def _window_ranges(
    big_row: int,
    big_col: int,
    square_size: int,
    n_rows: int,
    n_cols: int,
) -> Tuple[int, int, int, int]:
    row_start = big_row * square_size
    row_end_exclusive = min((big_row + 1) * square_size, n_rows)
    col_start = big_col * square_size
    col_end_exclusive = min((big_col + 1) * square_size, n_cols)
    return row_start, row_end_exclusive, col_start, col_end_exclusive


def _load_priority_map(priority_path: str) -> tuple[Dict[str, float], str]:
    candidate_paths = [Path(priority_path)]
    script_dir = Path(__file__).resolve().parent
    candidate_paths.append(script_dir / priority_path)

    chosen_path = None
    for p in candidate_paths:
        if p.exists():
            chosen_path = p
            break
    if chosen_path is None:
        raise FileNotFoundError(
            f"Priority file not found: {priority_path} "
            f"(also checked: {(script_dir / priority_path).as_posix()})"
        )

    with chosen_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, dict):
        raise ValueError(f"Unsupported priority JSON format in {chosen_path}")

    if "node_priority" in loaded and isinstance(loaded["node_priority"], dict):
        out = {
            str(k): float(v.get("priority_P_i", 0.0)) if isinstance(v, dict) else float(v)
            for k, v in loaded["node_priority"].items()
        }
    else:
        out = {str(k): float(v) for k, v in loaded.items()}
    return out, chosen_path.as_posix()


def build_graphs(
    input_graph: Dict[str, Any],
    square_size: int,
    priority_by_node: Dict[str, float],
    priority_source_path: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if square_size <= 0:
        raise ValueError("square_size must be > 0")

    if "meta" not in input_graph or "node_features" not in input_graph or "edge_features" not in input_graph:
        raise ValueError("Input graph must contain 'meta', 'node_features', and 'edge_features'")

    meta: Dict[str, Any] = input_graph["meta"]
    node_features: Dict[str, Dict[str, Any]] = input_graph["node_features"]
    edge_features: Dict[str, Dict[str, Dict[str, Any]]] = input_graph["edge_features"]

    n_rows, n_cols = _require_meta_field(meta, "grid_shape_rows_cols")
    min_x, min_y = _require_meta_field(meta, "grid_origin_m")
    cell_size_m = float(_require_meta_field(meta, "cell_size_m"))

    n_rows = int(n_rows)
    n_cols = int(n_cols)
    min_x = float(min_x)
    min_y = float(min_y)

    big_rows = (n_rows + square_size - 1) // square_size
    big_cols = (n_cols + square_size - 1) // square_size
    total_big_squares_in_grid = big_rows * big_cols

    # Map each small node -> big square.
    small_to_big: Dict[str, Tuple[str, int, int]] = {}
    big_to_small_nodes: Dict[str, list[str]] = {}
    big_index: Dict[str, Tuple[int, int]] = {}

    for node_id, nf in node_features.items():
        row = int(nf["row"])
        col = int(nf["col"])
        big_row = row // square_size
        big_col = col // square_size
        bid = big_square_id(big_row, big_col)

        small_to_big[node_id] = (bid, big_row, big_col)
        big_to_small_nodes.setdefault(bid, []).append(node_id)
        big_index[bid] = (big_row, big_col)

    # 1) Full small graph + membership.
    for node_id, nf in node_features.items():
        bid, big_row, big_col = small_to_big[node_id]
        nf["big_square_id"] = bid
        nf["big_square_row"] = int(big_row)
        nf["big_square_col"] = int(big_col)

    meta["big_square_partition"] = {
        "square_size_cells": [int(square_size), int(square_size)],
        "big_grid_shape_rows_cols": [int(big_rows), int(big_cols)],
        "big_square_count_total_in_grid": int(total_big_squares_in_grid),
        "big_square_count_with_small_nodes": int(len(big_to_small_nodes)),
        "source_node_count": int(len(node_features)),
    }

    small_graph_with_big = {
        "meta": meta,
        "node_features": node_features,
        "edge_features": edge_features,
    }

    # 2) Big-square graph.
    big_node_features: Dict[str, Dict[str, Any]] = {}
    bid_by_coord = {(r, c): big_square_id(r, c) for _, (r, c) in big_index.items()}

    for bid, (big_row, big_col) in sorted(big_index.items(), key=lambda x: x[0]):
        row_start, row_end_exclusive, col_start, col_end_exclusive = _window_ranges(
            big_row=big_row,
            big_col=big_col,
            square_size=square_size,
            n_rows=n_rows,
            n_cols=n_cols,
        )

        x0 = min_x + col_start * cell_size_m
        y0 = min_y + row_start * cell_size_m
        x1 = min_x + col_end_exclusive * cell_size_m
        y1 = min_y + row_end_exclusive * cell_size_m

        contained = sorted(big_to_small_nodes.get(bid, []))
        priority_sum_small_cells = float(sum(float(priority_by_node.get(cid, 0.0)) for cid in contained))
        height_cells = row_end_exclusive - row_start
        width_cells = col_end_exclusive - col_start

        big_node_features[bid] = {
            "big_row": int(big_row),
            "big_col": int(big_col),
            "row_range": [int(row_start), int(row_end_exclusive - 1)],
            "col_range": [int(col_start), int(col_end_exclusive - 1)],
            "window_shape_rows_cols": [int(height_cells), int(width_cells)],
            "is_full_square_window": bool(
                height_cells == square_size and width_cells == square_size
            ),
            "window_capacity_cells": int(height_cells * width_cells),
            "contained_small_node_count": int(len(contained)),
            "contained_small_node_ids": contained,
            "priority_sum_small_cells": priority_sum_small_cells,
            "bbox_m": [float(x0), float(y0), float(x1), float(y1)],
            "centroid_m": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
            "is_border": bool(
                big_row == 0 or big_col == 0 or big_row == big_rows - 1 or big_col == big_cols - 1
            ),
        }

    big_edge_features: Dict[str, Dict[str, Dict[str, Any]]] = {bid: {} for bid in big_node_features}
    for src_id, src_nf in big_node_features.items():
        src_row = int(src_nf["big_row"])
        src_col = int(src_nf["big_col"])
        src_cx, src_cy = src_nf["centroid_m"]
        src_h = int(src_nf["window_shape_rows_cols"][0])
        src_w = int(src_nf["window_shape_rows_cols"][1])

        for dr, dc in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            nbr_row = src_row + dr
            nbr_col = src_col + dc
            dst_id = bid_by_coord.get((nbr_row, nbr_col))
            if dst_id is None or dst_id not in big_node_features:
                continue

            dst_nf = big_node_features[dst_id]
            dst_cx, dst_cy = dst_nf["centroid_m"]
            dst_h = int(dst_nf["window_shape_rows_cols"][0])
            dst_w = int(dst_nf["window_shape_rows_cols"][1])

            dx = float(dst_cx - src_cx)
            dy = float(dst_cy - src_cy)
            distance_m = math.hypot(dx, dy)
            is_diagonal = dr != 0 and dc != 0
            distance_big_cells = math.sqrt(2.0) if is_diagonal else 1.0

            if is_diagonal:
                shared_border_m = 0.0
            elif dr == 0:
                shared_border_m = float(min(src_h, dst_h) * cell_size_m)
            else:
                shared_border_m = float(min(src_w, dst_w) * cell_size_m)

            big_edge_features[src_id][dst_id] = {
                "distance_m": float(distance_m),
                "distance_big_cells": float(distance_big_cells),
                "shared_border_m": float(shared_border_m),
                "is_diagonal": bool(is_diagonal),
            }

    big_graph = {
        "meta": {
            "source_meta": meta,
            "square_size_cells": [int(square_size), int(square_size)],
            "small_cell_size_m": float(cell_size_m),
            "big_grid_shape_rows_cols": [int(big_rows), int(big_cols)],
            "big_square_count_total_in_grid": int(total_big_squares_in_grid),
            "big_square_count_with_small_nodes": int(len(big_node_features)),
            "small_node_count": int(len(node_features)),
            "priority_aggregation": "sum_of_small_cell_priorities",
            "priority_source_path": priority_source_path,
            "metric_crs": meta.get("metric_crs"),
            "grid_origin_m": [float(min_x), float(min_y)],
        },
        "node_features": big_node_features,
        "edge_features": big_edge_features,
    }

    return small_graph_with_big, big_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build two graphs from an existing small-cell graph: (1) augmented small graph with 14x14 membership, (2) big-square graph with 8-neighbor adjacency."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_GRAPH, help="Input small-cell graph JSON")
    parser.add_argument(
        "--out-small-with-big",
        default=DEFAULT_OUT_SMALL_WITH_BIG,
        help="Output JSON: original graph + big square membership fields per node",
    )
    parser.add_argument(
        "--out-big-graph",
        default=DEFAULT_OUT_BIG_GRAPH,
        help="Output JSON: big-square graph",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=DEFAULT_SQUARE_SIZE,
        help="Big square size in small cells per side (default: 14)",
    )
    parser.add_argument(
        "--priority",
        default=DEFAULT_PRIORITY_PATH,
        help="Path to compact node priority JSON (clamped recommended)",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        input_graph = json.load(f)
    priority_by_node, priority_source_path = _load_priority_map(args.priority)

    small_with_big, big_graph = build_graphs(
        input_graph=input_graph,
        square_size=int(args.square_size),
        priority_by_node=priority_by_node,
        priority_source_path=priority_source_path,
    )

    with open(args.out_small_with_big, "w", encoding="utf-8") as f:
        json.dump(small_with_big, f, ensure_ascii=False)
    with open(args.out_big_graph, "w", encoding="utf-8") as f:
        json.dump(big_graph, f, ensure_ascii=False)

    print(f"Input graph: {args.input}")
    print(f"Priority source: {priority_source_path}")
    print(f"Output (small + big membership): {args.out_small_with_big}")
    print(f"Output (big-square graph): {args.out_big_graph}")
    print(
        "Summary: "
        f"small_nodes={len(small_with_big['node_features'])}, "
        f"big_nodes={len(big_graph['node_features'])}"
    )


if __name__ == "__main__":
    main()
