from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SMALL_GRAPH_PATH = BASE_DIR / "etosha_grid_graph_with_big_squares.json"
DEFAULT_BIG_GRAPH_PATH = BASE_DIR / "etosha_big_square_graph_14x14.json"
DEFAULT_BIG_DIST_PATH = BASE_DIR / "big_dist_with_portals_time_priority.json"
DEFAULT_PATROL_TO_BIG_PATH = BASE_DIR / "patrol_house_to_big_cell.json"
DEFAULT_OUT_PNG = BASE_DIR / "show" / "patrol_routes_simple.png"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def get_xy_from_small_node(nf: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[float, float]:
    c = nf.get("centroid_m")
    if isinstance(c, list) and len(c) == 2:
        return float(c[0]), float(c[1])
    row = int(nf["row"])
    col = int(nf["col"])
    cell_size = float(meta["cell_size_m"])
    x0, y0 = meta["grid_origin_m"]
    return float(x0) + (col + 0.5) * cell_size, float(y0) + (row + 0.5) * cell_size


def get_xy_from_big_node(bnf: Dict[str, Any]) -> Tuple[float, float]:
    c = bnf.get("centroid_m")
    if isinstance(c, list) and len(c) == 2:
        return float(c[0]), float(c[1])
    bbox = bnf.get("bbox_m")
    if isinstance(bbox, list) and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    raise ValueError("Big-node record has no centroid_m/bbox_m")


def dijkstra_big_path(
    start_id: str,
    end_id: str,
    big_edges: Dict[str, Dict[str, Dict[str, Any]]],
) -> Optional[List[str]]:
    if start_id == end_id:
        return [start_id]
    dist: Dict[str, float] = {start_id: 0.0}
    parent: Dict[str, Optional[str]] = {start_id: None}
    pq: List[Tuple[float, str]] = [(0.0, start_id)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == end_id:
            break
        for v, ef in big_edges.get(u, {}).items():
            if ef.get("is_diagonal"):
                continue
            try:
                w = float(ef.get("distance_m", 1.0))
            except Exception:
                w = 1.0
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if end_id not in dist:
        return None
    out: List[str] = []
    cur: Optional[str] = end_id
    seen = set()
    while cur is not None:
        if cur in seen:
            return None
        seen.add(cur)
        out.append(cur)
        if cur == start_id:
            break
        cur = parent.get(cur)
    if not out or out[-1] != start_id:
        return None
    out.reverse()
    return out


def choose_target_big_cell(
    start_big_id: str,
    to_big: Dict[str, Dict[str, Any]],
    priority_by_big: Dict[str, float],
    used_targets: set,
    inf_json: float,
) -> Optional[str]:
    ranked: List[Tuple[float, float, float, str]] = []
    for bid, rec in to_big.items():
        if bid == start_big_id:
            continue
        d = float(rec.get("distance_m", inf_json))
        if (not math.isfinite(d)) or d >= inf_json * 0.5:
            continue
        pr = float(priority_by_big.get(bid, 0.0))
        score = pr / (d + 1.0)
        ranked.append((score, pr, -d, bid))

    if not ranked:
        return None
    ranked.sort(reverse=True)
    for _, _, _, bid in ranked:
        if bid not in used_targets:
            return bid
    return ranked[0][3]


def draw_routes(
    small_graph: Dict[str, Any],
    big_graph: Dict[str, Any],
    big_dist_data: Dict[str, Any],
    patrol_to_big_data: Dict[str, Any],
    out_png: Path,
    n_patrols: int,
) -> None:
    small_meta = small_graph["meta"]
    small_nodes = small_graph["node_features"]
    big_nodes = big_graph["node_features"]
    big_edges = big_graph["edge_features"]
    inf_json = float(big_dist_data.get("meta", {}).get("inf_replacement", 1e30))

    priority_by_big = {
        bid: float(bnf.get("priority_sum_small_cells", 0.0))
        for bid, bnf in big_nodes.items()
    }
    big_xy = {bid: get_xy_from_big_node(bnf) for bid, bnf in big_nodes.items()}

    patrol_rows = list(patrol_to_big_data.get("patrol_houses", []))[: max(0, int(n_patrols))]
    used_targets: set = set()
    route_rows: List[Dict[str, Any]] = []

    for row in patrol_rows:
        ph_id = str(row.get("cell_id"))
        if ph_id not in small_nodes:
            continue
        ph_nf = small_nodes[ph_id]
        start_big = ph_nf.get("big_square_id")
        if not isinstance(start_big, str) or start_big not in big_nodes:
            continue
        target_big = choose_target_big_cell(
            start_big_id=start_big,
            to_big=row.get("to_big", {}),
            priority_by_big=priority_by_big,
            used_targets=used_targets,
            inf_json=inf_json,
        )
        if target_big is None:
            continue
        path_big = dijkstra_big_path(start_big, target_big, big_edges=big_edges)
        if not path_big:
            continue

        used_targets.add(target_big)
        metrics = big_dist_data.get("dist", {}).get(start_big, {}).get(target_big, {})
        dist_m = float(metrics.get("distance_m", inf_json))
        time_h = float(metrics.get("time_h", inf_json))
        cost = float(metrics.get("small_priority_sum", inf_json))
        ph_to_target = float(row.get("to_big", {}).get(target_big, {}).get("distance_m", inf_json))

        route_rows.append(
            {
                "patrol_cell_id": ph_id,
                "patrol_xy": get_xy_from_small_node(ph_nf, small_meta),
                "start_big": start_big,
                "target_big": target_big,
                "path_big": path_big,
                "distance_m": dist_m,
                "time_h": time_h,
                "small_priority_sum": cost,
                "patrol_to_target_boundary_m": ph_to_target,
            }
        )

    cell_size = float(small_meta["cell_size_m"])
    n_rows, n_cols = [int(x) for x in small_meta["grid_shape_rows_cols"]]
    x0, y0 = [float(v) for v in small_meta["grid_origin_m"]]
    extent = [x0, x0 + n_cols * cell_size, y0, y0 + n_rows * cell_size]

    fig, ax = plt.subplots(figsize=(13, 10), dpi=220)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("X (meters, projected CRS)")
    ax.set_ylabel("Y (meters, projected CRS)")
    ax.set_title("Simple patrol routes on big-grid graph (blue lines)")

    for bid, bnf in big_nodes.items():
        bbox = bnf.get("bbox_m")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bx0, by0, bx1, by1 = [float(v) for v in bbox]
        ax.add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                fill=False,
                edgecolor="#dddddd",
                linewidth=0.7,
                alpha=0.7,
                zorder=1,
            )
        )

    seen_edges = set()
    for u, neis in big_edges.items():
        if u not in big_xy:
            continue
        ux, uy = big_xy[u]
        for v, ef in neis.items():
            if ef.get("is_diagonal"):
                continue
            if v not in big_xy:
                continue
            key = tuple(sorted((u, v)))
            if key in seen_edges:
                continue
            seen_edges.add(key)
            vx, vy = big_xy[v]
            ax.plot([ux, vx], [uy, vy], color="#ececec", linewidth=0.5, alpha=0.9, zorder=2)

    for idx, r in enumerate(route_rows, start=1):
        path_xy = [big_xy[bid] for bid in r["path_big"] if bid in big_xy]
        if not path_xy:
            continue

        px, py = r["patrol_xy"]
        sx, sy = path_xy[0]
        tx, ty = path_xy[-1]

        route_x = [px, sx] + [x for x, _ in path_xy]
        route_y = [py, sy] + [y for _, y in path_xy]
        ax.plot(route_x, route_y, color="#0b61ff", linewidth=2.1, alpha=0.9, zorder=4)

        ax.scatter([px], [py], s=38, color="#222222", marker="o", zorder=6)
        ax.scatter([tx], [ty], s=58, color="#0b61ff", marker="s", edgecolors="white", linewidths=0.5, zorder=7)

        dist_km = r["distance_m"] / 1000.0 if r["distance_m"] < inf_json * 0.5 else math.inf
        time_h = r["time_h"] if r["time_h"] < inf_json * 0.5 else math.inf
        cost = r["small_priority_sum"] if r["small_priority_sum"] < inf_json * 0.5 else math.inf
        text = (
            f"P{idx}: {r['start_big']} -> {r['target_big']}\n"
            f"d={dist_km:.1f} km, t={time_h:.2f} h, cost={cost:.1f}"
        )
        ax.text(
            tx + cell_size * 0.6,
            ty + cell_size * 0.4,
            text,
            fontsize=7.4,
            color="#0d2a6a",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#5a8dff", "alpha": 0.92},
            zorder=8,
        )

    ax.text(
        0.01,
        0.01,
        "Blue routes: selected patrol examples. Cost = small_priority_sum from big-dist table.",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#666666", "alpha": 0.9},
        zorder=9,
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a simple patrol-route figure with blue lines on the big-cell grid."
    )
    parser.add_argument("--small-graph", type=Path, default=DEFAULT_SMALL_GRAPH_PATH)
    parser.add_argument("--big-graph", type=Path, default=DEFAULT_BIG_GRAPH_PATH)
    parser.add_argument("--big-dist", type=Path, default=DEFAULT_BIG_DIST_PATH)
    parser.add_argument("--patrol-to-big", type=Path, default=DEFAULT_PATROL_TO_BIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--n-patrols", type=int, default=4)
    args = parser.parse_args()

    small_graph = load_json(args.small_graph)
    big_graph = load_json(args.big_graph)
    big_dist_data = load_json(args.big_dist)
    patrol_to_big_data = load_json(args.patrol_to_big)

    draw_routes(
        small_graph=small_graph,
        big_graph=big_graph,
        big_dist_data=big_dist_data,
        patrol_to_big_data=patrol_to_big_data,
        out_png=args.out,
        n_patrols=args.n_patrols,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
