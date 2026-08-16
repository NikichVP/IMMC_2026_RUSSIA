from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from typing import Any, Dict, List, Tuple

from simulation import edge_travel_time_hours, precompute_sector_assignment


DEFAULT_GRAPH_PATH = "etosha_grid_graph.json"
DEFAULT_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_OUT_JSON = "single_patrol_demo_output.json"
DEFAULT_OUT_PNG = "single_patrol_demo_route.png"
DEFAULT_BASE_CELL_ID = "r20_c168"
DEFAULT_HOURS_BUDGET = 12.0
DEFAULT_SOFTMAX_TAU = 0.8
DEFAULT_RANDOM_SEED = 7
DEFAULT_MAX_STEPS = 1000


def _load_priority_map(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "node_priority" in loaded and isinstance(loaded["node_priority"], dict):
        return {
            str(k): float(v.get("priority_P_i", 0.0)) if isinstance(v, dict) else float(v)
            for k, v in loaded["node_priority"].items()
        }
    if isinstance(loaded, dict):
        return {str(k): float(v) for k, v in loaded.items()}
    raise ValueError("Unsupported priority file format")


def _build_weighted_adjacency(
    edges: Dict[str, Dict[str, Dict[str, Any]]],
    allowed_nodes: set[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for src, neis in edges.items():
        if allowed_nodes is not None and src not in allowed_nodes:
            continue
        out[src] = {
            dst: edge_travel_time_hours(ef)
            for dst, ef in neis.items()
            if allowed_nodes is None or dst in allowed_nodes
        }
    return out


def _dijkstra_all_times(
    weighted_adj: Dict[str, Dict[str, float]],
    src: str,
) -> Dict[str, float]:
    best: Dict[str, float] = {src: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, src)]
    while pq:
        t_cur, cur = heapq.heappop(pq)
        if t_cur > best.get(cur, math.inf):
            continue
        for nxt, t_edge in weighted_adj.get(cur, {}).items():
            cand = t_cur + t_edge
            if cand < best.get(nxt, math.inf):
                best[nxt] = cand
                heapq.heappush(pq, (cand, nxt))
    return best


def _shortest_path_and_time(
    weighted_adj: Dict[str, Dict[str, float]],
    src: str,
    dst: str,
) -> Tuple[List[str], float]:
    if src == dst:
        return [src], 0.0

    best: Dict[str, float] = {src: 0.0}
    prev: Dict[str, str] = {}
    pq: List[Tuple[float, str]] = [(0.0, src)]

    while pq:
        t_cur, cur = heapq.heappop(pq)
        if t_cur > best.get(cur, math.inf):
            continue
        if cur == dst:
            break
        for nxt, t_edge in weighted_adj.get(cur, {}).items():
            cand = t_cur + t_edge
            if cand < best.get(nxt, math.inf):
                best[nxt] = cand
                prev[nxt] = cur
                heapq.heappush(pq, (cand, nxt))

    if dst not in best:
        return [], math.inf

    path_rev = [dst]
    cur = dst
    while cur != src:
        cur = prev[cur]
        path_rev.append(cur)
    path = list(reversed(path_rev))
    return path, best[dst]


def _softmax_probs(
    candidates: List[str],
    priority_by_node: Dict[str, float],
    tau: float,
) -> Dict[str, float]:
    if not candidates:
        return {}
    logits = [float(priority_by_node.get(cid, 0.0)) / tau for cid in candidates]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    if z <= 0.0:
        uniform = 1.0 / len(candidates)
        return {cid: uniform for cid in candidates}
    return {cid: e / z for cid, e in zip(candidates, exps)}


def _sample_by_probs(probs: Dict[str, float], rng: random.Random) -> str:
    items = sorted(probs.items(), key=lambda x: x[0])
    r = rng.random()
    acc = 0.0
    for cid, p in items:
        acc += p
        if acc >= r:
            return cid
    return items[-1][0]


def simulate_single_sortie_softmax(
    graph: Dict[str, Any],
    base_cell_id: str,
    *,
    priority_by_node: Dict[str, float],
    max_patrol_hours: float = 8.0,
    tau: float = 1.0,
    random_seed: int = 42,
    max_steps: int = 200,
    avoid_immediate_backtrack: bool = True,
) -> Dict[str, Any]:
    """
    Один выход патруля:
      - старт в base_cell_id
      - на каждом шаге выбираем y из допустимого множества:
        S = {y in N(x) ∩ C_i : spent + T(x,y) + T(y,base) <= H}
      - выбор y по softmax(priority(y)/tau)
      - если S пусто, возврат на базу и завершение выхода
    """
    if max_patrol_hours <= 0:
        raise ValueError("max_patrol_hours must be > 0")
    if tau <= 0:
        raise ValueError("tau must be > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    nodes: Dict[str, Any] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]
    if base_cell_id not in nodes:
        raise ValueError(f"Unknown base cell: {base_cell_id}")

    valid_cells = {cid for cid, nf in nodes.items() if nf.get("median_elevation_m") is not None}
    if priority_by_node:
        allowed_cells = valid_cells & set(priority_by_node.keys())
    else:
        allowed_cells = set(valid_cells)
    if base_cell_id not in allowed_cells:
        raise ValueError(
            f"Base cell {base_cell_id} is outside allowed valid+priority cells"
        )

    weighted_adj = _build_weighted_adjacency(edges, allowed_nodes=allowed_cells)
    t_to_base = _dijkstra_all_times(weighted_adj, base_cell_id)

    # При одном активном посте его сектор — фактически вся связная компонента.
    sector_assignment, _ = precompute_sector_assignment(graph, [base_cell_id])
    sector_cells = {
        cid
        for cid, owner in sector_assignment.items()
        if owner == base_cell_id and cid in allowed_cells
    }
    sector_cells.add(base_cell_id)

    rng = random.Random(random_seed)
    current = base_cell_id
    spent = 0.0
    steps = 0
    path = [base_cell_id]
    step_log: List[Dict[str, Any]] = []
    prev_cell: str | None = None

    while steps < max_steps:
        admissible: List[str] = []
        admissible_meta: Dict[str, Dict[str, float]] = {}
        for nxt, t_xy in weighted_adj.get(current, {}).items():
            if nxt not in sector_cells:
                continue
            t_y_base = t_to_base.get(nxt, math.inf)
            feasible = spent + t_xy + t_y_base <= max_patrol_hours
            if feasible:
                admissible.append(nxt)
                admissible_meta[nxt] = {
                    "t_xy_h": float(t_xy),
                    "t_y_base_h": float(t_y_base),
                    "priority": float(priority_by_node.get(nxt, 0.0)),
                }

        if avoid_immediate_backtrack and prev_cell is not None and len(admissible) > 1:
            admissible = [cid for cid in admissible if cid != prev_cell]

        if not admissible:
            if current != base_cell_id:
                ret_path, t_back = _shortest_path_and_time(weighted_adj, current, base_cell_id)
                if ret_path and math.isfinite(t_back):
                    spent += t_back
                    path.extend(ret_path[1:])
            break

        probs = _softmax_probs(admissible, priority_by_node, tau)
        chosen = _sample_by_probs(probs, rng)
        t_step = weighted_adj[current][chosen]

        step_log.append(
            {
                "step": steps + 1,
                "from": current,
                "candidates_count": len(admissible),
                "candidates": [
                    {
                        "cell_id": cid,
                        "priority": admissible_meta[cid]["priority"],
                        "probability": probs[cid],
                        "t_xy_h": admissible_meta[cid]["t_xy_h"],
                        "t_y_base_h": admissible_meta[cid]["t_y_base_h"],
                    }
                    for cid in sorted(admissible)
                ],
                "chosen": chosen,
                "t_xy_h": float(t_step),
                "spent_before_h": float(spent),
                "spent_after_h": float(spent + t_step),
            }
        )

        prev_cell = current
        current = chosen
        spent += t_step
        steps += 1
        path.append(current)

    # Гарантируем завершение выхода на базе.
    if current != base_cell_id:
        ret_path, t_back = _shortest_path_and_time(weighted_adj, current, base_cell_id)
        if ret_path and math.isfinite(t_back):
            spent += t_back
            path.extend(ret_path[1:])
            current = base_cell_id

    return {
        "base_cell_id": base_cell_id,
        "max_patrol_hours": max_patrol_hours,
        "tau": tau,
        "random_seed": random_seed,
        "max_steps": max_steps,
        "steps_done": steps,
        "total_time_h": spent,
        "ended_at_base": current == base_cell_id,
        "path": path,
        "step_log": step_log,
    }


def run_single_sortie_demo(
    graph_path: str = DEFAULT_GRAPH_PATH,
    priority_path: str = DEFAULT_PRIORITY_PATH,
    base_cell_id: str = "r20_c168",
    *,
    max_patrol_hours: float = DEFAULT_HOURS_BUDGET,
    tau: float = DEFAULT_SOFTMAX_TAU,
    random_seed: int = DEFAULT_RANDOM_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Dict[str, Any]:
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    priority_by_node = _load_priority_map(priority_path)
    return simulate_single_sortie_softmax(
        graph,
        base_cell_id,
        priority_by_node=priority_by_node,
        max_patrol_hours=max_patrol_hours,
        tau=tau,
        random_seed=random_seed,
        max_steps=max_steps,
        avoid_immediate_backtrack=True,
    )


def _build_inside_mask(meta: Dict[str, Any], nodes: Dict[str, Dict[str, Any]]) -> np.ndarray:
    import numpy as np

    n_rows, n_cols = meta["grid_shape_rows_cols"]
    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for nf in nodes.values():
        rr = int(nf["row"])
        cc = int(nf["col"])
        inside_mask[rr, cc] = nf.get("median_elevation_m") is not None
    return inside_mask


def _path_to_xy(path: List[str], nodes: Dict[str, Dict[str, Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for cid in path:
        nf = nodes.get(cid)
        if nf is None:
            continue
        x, y = nf["centroid_m"]
        out.append((float(x), float(y)))
    return out


def plot_single_sortie(
    graph: Dict[str, Any],
    sortie: Dict[str, Any],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    meta = graph["meta"]
    nodes = graph["node_features"]
    inside_mask = _build_inside_mask(meta, nodes)

    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size
    extent = [min_x, max_x, min_y, max_y]

    path_xy = _path_to_xy(sortie.get("path", []), nodes)
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    ax_full, ax_zoom = axes

    bg = np.where(inside_mask, 1.0, 0.0)
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.imshow(
            bg,
            extent=extent,
            origin="lower",
            cmap=ListedColormap(["#f0f0f0", "#c7c7c7"]),
            interpolation="nearest",
            alpha=0.95,
            zorder=1,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (meters, projected CRS)")
        ax.set_ylabel("Y (meters, projected CRS)")

    ax_full.set_xlim(min_x, max_x)
    ax_full.set_ylim(min_y, max_y)
    ax_full.set_title("Full map")

    if path_xy:
        # Одна понятная линия маршрута.
        for ax in axes:
            ax.plot(xs, ys, color="#0b84f3", linewidth=2.8, alpha=0.95, zorder=4)
            ax.scatter(xs[0], ys[0], s=150, marker="*", color="#1a9d49", edgecolors="black", linewidths=0.8, zorder=6)
            ax.scatter(xs[-1], ys[-1], s=62, marker="o", color="#d62728", edgecolors="white", linewidths=0.9, zorder=6)

        # Стрелки направления только на zoom.
        arrow_stride = max(1, len(path_xy) // 24)
        for i in range(0, len(path_xy) - 1, arrow_stride):
            ax_zoom.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops={"arrowstyle": "->", "color": "#0a4f8b", "lw": 1.0, "alpha": 0.9},
                zorder=7,
            )

        # Zoom вокруг маршрута.
        min_px, max_px = min(xs), max(xs)
        min_py, max_py = min(ys), max(ys)
        span_x = max(max_px - min_px, 1000.0)
        span_y = max(max_py - min_py, 1000.0)
        pad_x = max(5000.0, 0.35 * span_x)
        pad_y = max(5000.0, 0.35 * span_y)
        ax_zoom.set_xlim(min_px - pad_x, max_px + pad_x)
        ax_zoom.set_ylim(min_py - pad_y, max_py + pad_y)
        ax_zoom.set_title("Route zoom")

        ax_zoom.text(
            0.02,
            0.02,
            "Legend: light bg=outside map, dark bg=valid map, green*=start, red●=end, blue line=route",
            transform=ax_zoom.transAxes,
            fontsize=9,
            color="#202020",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#7a7a7a", "alpha": 0.85},
            zorder=8,
        )
    else:
        ax_zoom.set_xlim(min_x, max_x)
        ax_zoom.set_ylim(min_y, max_y)
        ax_zoom.set_title("Route zoom (empty)")

    fig.suptitle(
        (
            f"Single patrol sortie | base={sortie['base_cell_id']} | "
            f"steps={sortie['steps_done']} | total_time_h={sortie['total_time_h']:.3f} | "
            f"ended_at_base={sortie['ended_at_base']}"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one patrol sortie with detailed softmax step log")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_PATH, help="Path to graph JSON")
    parser.add_argument("--priority", default=DEFAULT_PRIORITY_PATH, help="Path to priority JSON")
    parser.add_argument("--base", default=DEFAULT_BASE_CELL_ID, help="Base cell_id")
    parser.add_argument("--H", type=float, default=DEFAULT_HOURS_BUDGET, help="Sortie time budget in hours")
    parser.add_argument("--tau", type=float, default=DEFAULT_SOFTMAX_TAU, help="Softmax temperature")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Max forward steps before forced return")
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_JSON,
        help="Output JSON with detailed route log",
    )
    parser.add_argument(
        "--png",
        default=DEFAULT_OUT_PNG,
        help="Output PNG with route visualization",
    )
    args = parser.parse_args()

    with open(args.graph, "r", encoding="utf-8") as f:
        graph = json.load(f)
    priority_by_node = _load_priority_map(args.priority)
    result = simulate_single_sortie_softmax(
        graph,
        args.base,
        priority_by_node=priority_by_node,
        max_patrol_hours=args.H,
        tau=args.tau,
        random_seed=args.seed,
        max_steps=args.max_steps,
        avoid_immediate_backtrack=True,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    plot_single_sortie(
        graph=graph,
        sortie=result,
        out_path=args.png,
    )

    print(f"Saved JSON: {args.out}")
    print(f"Saved PNG: {args.png}")
    print(f"Base: {result['base_cell_id']}")
    print(f"Steps: {result['steps_done']}")
    print(f"Total time (h): {result['total_time_h']:.4f}")
    print(f"Ended at base: {result['ended_at_base']}")
    print(f"Path length (nodes): {len(result['path'])}")
    if result["step_log"]:
        first = result["step_log"][0]
        print(
            "Step#1 sample: "
            f"from={first['from']} candidates={first['candidates_count']} chosen={first['chosen']}"
        )


if __name__ == "__main__":
    main()
