from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SMALL_GRAPH_PATH = BASE_DIR / "etosha_grid_graph_with_big_squares.json"
DEFAULT_BIG_GRAPH_PATH = BASE_DIR / "etosha_big_square_graph_14x14.json"
DEFAULT_PRIORITY_PATH = BASE_DIR / "etosha_node_priority_compact_clamped.json"
DEFAULT_OUT_PNG = BASE_DIR / "show" / "priority_big_blocks_map.png"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def build_priority_arrays(
    small_graph: Dict[str, Any],
    priority_map: Dict[str, float],
) -> Dict[str, Any]:
    meta = small_graph["meta"]
    nodes = small_graph["node_features"]

    n_rows, n_cols = meta["grid_shape_rows_cols"]
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    cell_size = float(meta["cell_size_m"])
    min_x, min_y = meta["grid_origin_m"]
    min_x = float(min_x)
    min_y = float(min_y)
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    priority_arr = np.zeros((n_rows, n_cols), dtype="float32")
    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outside_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for cell_id, nf in nodes.items():
        rr = int(nf["row"])
        cc = int(nf["col"])
        if nf.get("median_elevation_m") is None:
            outside_mask[rr, cc] = True
            priority_arr[rr, cc] = 0.0
        else:
            inside_mask[rr, cc] = True
            priority_arr[rr, cc] = float(priority_map.get(cell_id, 0.0))

    return {
        "priority_arr": priority_arr,
        "inside_mask": inside_mask,
        "outside_mask": outside_mask,
        "extent": [min_x, max_x, min_y, max_y],
    }


def plot_priority_with_big_blocks(
    small_graph: Dict[str, Any],
    big_graph: Dict[str, Any],
    priority_map: Dict[str, float],
    out_png: Path,
) -> None:
    arr = build_priority_arrays(small_graph, priority_map)
    priority_arr = arr["priority_arr"]
    inside_mask = arr["inside_mask"]
    outside_mask = arr["outside_mask"]
    extent = arr["extent"]

    fig, ax = plt.subplots(figsize=(13, 10), dpi=220)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters, projected CRS)")
    ax.set_ylabel("Y (meters, projected CRS)")

    outside_img = np.where(outside_mask, 1.0, np.nan)
    inside_img = np.where(inside_mask, 1.0, np.nan)
    ax.imshow(
        outside_img,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#efefef"]),
        interpolation="nearest",
        alpha=1.0,
        zorder=1,
    )
    ax.imshow(
        inside_img,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#d6d6d6"]),
        interpolation="nearest",
        alpha=1.0,
        zorder=2,
    )

    inside_priority = np.where(inside_mask, priority_arr, np.nan)
    vmax = float(np.nanpercentile(inside_priority, 99.0)) if np.any(np.isfinite(inside_priority)) else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        inside_priority,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
        alpha=0.95,
        vmin=0.0,
        vmax=vmax,
        zorder=3,
    )

    big_nodes: Dict[str, Dict[str, Any]] = big_graph["node_features"]
    for bid, bnf in big_nodes.items():
        bbox = bnf.get("bbox_m")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        width = x1 - x0
        height = y1 - y0
        if width <= 0.0 or height <= 0.0:
            continue

        is_border = bool(bnf.get("is_border", False))
        lw = 1.35 if is_border else 0.9
        alpha = 0.72 if is_border else 0.58
        ax.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor="#111111",
                linewidth=lw,
                alpha=alpha,
                zorder=5,
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Node priority (clamped)")

    big_meta = big_graph.get("meta", {})
    big_shape = big_meta.get("big_grid_shape_rows_cols")
    title_suffix = ""
    if isinstance(big_shape, list) and len(big_shape) == 2:
        title_suffix = f" | big grid {big_shape[0]}x{big_shape[1]}"
    ax.set_title(f"Priority map with big-square boundaries{title_suffix}")
    ax.text(
        0.01,
        0.01,
        "Gray = outside/inside base, heat = priority, black lines = big squares",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#777777", "alpha": 0.9},
        zorder=10,
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clamped priority map with big-square boundaries from big graph."
    )
    parser.add_argument("--small-graph", type=Path, default=DEFAULT_SMALL_GRAPH_PATH)
    parser.add_argument("--big-graph", type=Path, default=DEFAULT_BIG_GRAPH_PATH)
    parser.add_argument("--priority", type=Path, default=DEFAULT_PRIORITY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PNG)
    args = parser.parse_args()

    small_graph = load_json(args.small_graph)
    big_graph = load_json(args.big_graph)
    raw_priority = load_json(args.priority)
    priority_map = {str(k): float(v) for k, v in raw_priority.items()}

    plot_priority_with_big_blocks(
        small_graph=small_graph,
        big_graph=big_graph,
        priority_map=priority_map,
        out_png=args.out,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
