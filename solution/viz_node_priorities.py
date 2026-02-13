from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


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


DEFAULT_GRAPH_PATH = "etosha_grid_graph.json"
DEFAULT_PRIORITY_PATH = "etosha_node_priority_compact_clamped.json"
DEFAULT_OUT_PNG = "etosha_node_priorities_map.png"
DEFAULT_OUT_JSON = "etosha_node_priority_compact_clamped.json"


def _load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    raise ValueError("Unsupported priority JSON format")


def build_clamped_priority(
    graph: Dict[str, Any],
    raw_priority: Dict[str, float],
) -> Dict[str, float]:
    nodes: Dict[str, Dict[str, Any]] = graph["node_features"]
    out: Dict[str, float] = {}
    for cid, nf in nodes.items():
        if nf.get("median_elevation_m") is None:
            # Жесткое правило: вне валидной карты приоритет всегда 0.
            out[cid] = 0.0
        else:
            out[cid] = float(raw_priority.get(cid, 0.0))
    return out


def _build_arrays(
    graph: Dict[str, Any],
    clamped_priority: Dict[str, float],
) -> Dict[str, Any]:
    meta = graph["meta"]
    nodes = graph["node_features"]

    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    min_x, min_y = meta["grid_origin_m"]
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    priority_arr = np.zeros((n_rows, n_cols), dtype="float32")
    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    outside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    animal_score_arr = np.zeros((n_rows, n_cols), dtype="float32")
    water_mask = np.zeros((n_rows, n_cols), dtype=bool)
    plant_mask = np.zeros((n_rows, n_cols), dtype=bool)
    coverage_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for cid, nf in nodes.items():
        rr = int(nf["row"])
        cc = int(nf["col"])
        p = float(clamped_priority.get(cid, 0.0))
        priority_arr[rr, cc] = p
        if nf.get("median_elevation_m") is None:
            outside_mask[rr, cc] = True
        else:
            inside_mask[rr, cc] = True
            poi = nf.get("poi_type_counts") or {}
            animals = nf.get("animals_present") or []
            an_i = 0.0
            for animal in animals:
                an_i += float(ANIMAL_WEIGHTS.get(str(animal), 0.0))
            animal_score_arr[rr, cc] = float(an_i)
            water_mask[rr, cc] = bool(int(poi.get("waterhole", 0)) > 0 or int(poi.get("waterhole_dry", 0)) > 0)
            plant_mask[rr, cc] = bool(nf.get("has_plant", False))
            coverage_mask[rr, cc] = bool(
                int(poi.get("patrol_house", 0)) > 0 or int(poi.get("photo_trap", 0)) > 0
            )

    return {
        "priority_arr": priority_arr,
        "inside_mask": inside_mask,
        "outside_mask": outside_mask,
        "animal_score_arr": animal_score_arr,
        "water_mask": water_mask,
        "plant_mask": plant_mask,
        "coverage_mask": coverage_mask,
        "extent": [min_x, max_x, min_y, max_y],
    }


def plot_priority_map(
    graph: Dict[str, Any],
    clamped_priority: Dict[str, float],
    out_png: str,
) -> None:
    arr = _build_arrays(graph, clamped_priority)
    priority_arr = arr["priority_arr"]
    inside_mask = arr["inside_mask"]
    outside_mask = arr["outside_mask"]
    animal_score_arr = arr["animal_score_arr"]
    water_mask = arr["water_mask"]
    plant_mask = arr["plant_mask"]
    coverage_mask = arr["coverage_mask"]
    extent = arr["extent"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 14), dpi=200)
    ax_top, ax_bottom = axes

    def draw_base(ax):
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

    draw_base(ax_top)
    draw_base(ax_bottom)

    inside_prior = np.where(inside_mask, priority_arr, np.nan)
    vmax = float(np.nanpercentile(inside_prior, 99.0)) if np.any(np.isfinite(inside_prior)) else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    im = ax_top.imshow(
        inside_prior,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
        alpha=0.96,
        vmin=0.0,
        vmax=vmax,
        zorder=3,
    )

    cbar = fig.colorbar(im, ax=ax_top, fraction=0.035, pad=0.02)
    cbar.set_label("Node priority")
    ax_top.set_title("Node priorities map (outside valid map is always 0)")
    ax_top.text(
        0.01,
        0.01,
        "Light gray = outside map (priority=0), color = priority inside map",
        transform=ax_top.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#777777", "alpha": 0.8},
    )

    # Bottom panel: transparent overlay of all priority factors from compute_node_priority.
    animal_inside = np.where(inside_mask, animal_score_arr, np.nan)
    an_vmax = float(np.nanpercentile(animal_inside, 99.0)) if np.any(np.isfinite(animal_inside)) else 1.0
    if an_vmax <= 0.0:
        an_vmax = 1.0
    ax_bottom.imshow(
        animal_inside,
        extent=extent,
        origin="lower",
        cmap="Blues",
        interpolation="nearest",
        alpha=0.55,
        vmin=0.0,
        vmax=an_vmax,
        zorder=3,
    )
    ax_bottom.imshow(
        np.where(water_mask, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#00cfe8"]),
        interpolation="nearest",
        alpha=0.55,
        zorder=4,
    )
    ax_bottom.imshow(
        np.where(plant_mask, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#2ca02c"]),
        interpolation="nearest",
        alpha=0.45,
        zorder=5,
    )
    ax_bottom.imshow(
        np.where(coverage_mask, 1.0, np.nan),
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#b565d9"]),
        interpolation="nearest",
        alpha=0.55,
        zorder=6,
    )
    ax_bottom.set_title("Priority factors overlay (semi-transparent)")

    legend_handles = [
        Patch(facecolor="#4a90e2", edgecolor="none", alpha=0.55, label="AN_i: animals weighted score"),
        Patch(facecolor="#00cfe8", edgecolor="none", alpha=0.55, label="WA_i: waterhole / dry waterhole"),
        Patch(facecolor="#2ca02c", edgecolor="none", alpha=0.45, label="PL_i: plant cells"),
        Patch(facecolor="#b565d9", edgecolor="none", alpha=0.55, label="COV_i: patrol/photo coverage"),
        Patch(facecolor="#efefef", edgecolor="#bbbbbb", alpha=1.0, label="Outside map (priority forced to 0)"),
    ]
    ax_bottom.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize node priorities for each graph vertex")
    parser.add_argument("--graph", default=DEFAULT_GRAPH_PATH, help="Path to graph JSON")
    parser.add_argument("--priority", default=DEFAULT_PRIORITY_PATH, help="Input priority JSON")
    parser.add_argument("--out-png", default=DEFAULT_OUT_PNG, help="Output PNG path")
    parser.add_argument(
        "--out-json",
        default=DEFAULT_OUT_JSON,
        help="Output compact JSON with clamped priorities (outside=0)",
    )
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    raw_priority = _load_priority_map(args.priority)
    clamped = build_clamped_priority(graph, raw_priority)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(clamped, f, ensure_ascii=False, indent=2)

    plot_priority_map(graph, clamped, args.out_png)

    nodes = graph["node_features"]
    outside_nodes = [cid for cid, nf in nodes.items() if nf.get("median_elevation_m") is None]
    outside_nonzero = sum(1 for cid in outside_nodes if float(clamped.get(cid, 0.0)) != 0.0)
    print(f"Saved PNG: {args.out_png}")
    print(f"Saved JSON: {args.out_json}")
    print(f"Node count: {len(nodes)}")
    print(f"Outside nodes: {len(outside_nodes)}")
    print(f"Outside nodes with non-zero priority: {outside_nonzero}")


if __name__ == "__main__":
    main()
