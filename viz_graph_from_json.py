import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


GRAPH_JSON = "etosha_grid_graph.json"
OUT_ANIMALS_PNG = "etosha_graph_animals_vertical.png"
OUT_INFRA_PNG = "etosha_graph_infra_vertical.png"
OUT_FIRE_PNG = "etosha_graph_fire_no_fire.png"
OUT_SECURITY_PNG = "etosha_graph_patrol_photo_overlay.png"
OUT_PLANTS_PNG = "etosha_graph_plants_overlay.png"
ANIMAL_COLOR_CYCLE = [
    "#e41a1c",  # red
    "#1f78b4",  # blue
    "#33a02c",  # green
    "#ff7f00",  # orange
    "#6a3d9a",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
    "#66a61e",  # olive
    "#17becf",  # cyan
    "#d81b60",  # magenta
]


def load_graph(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["meta"], data["node_features"], data["edge_features"]


def collect_graph_layers(meta: dict, node_features: dict):
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    min_x, min_y = meta["grid_origin_m"]

    inside_mask = np.zeros((n_rows, n_cols), dtype=bool)
    no_fire_mask = np.zeros((n_rows, n_cols), dtype=bool)
    has_plant_mask = np.zeros((n_rows, n_cols), dtype=bool)
    elevation_arr = np.full((n_rows, n_cols), np.nan, dtype="float32")
    road_length_arr = np.zeros((n_rows, n_cols), dtype="float32")

    animal_cells = defaultdict(set)
    animal_counts = Counter()
    poi_cells_by_type = defaultdict(list)
    poi_type_counts = Counter()
    poi_cells_set = set()

    for feats in node_features.values():
        rr = int(feats["row"])
        cc = int(feats["col"])
        x = min_x + (cc + 0.5) * cell_size
        y = min_y + (rr + 0.5) * cell_size

        elev = feats.get("median_elevation_m")
        is_inside = elev is not None
        if elev is not None:
            inside_mask[rr, cc] = True
            elevation_arr[rr, cc] = float(elev)

        road_length_arr[rr, cc] = float(feats.get("road_total_length_m", 0.0))
        no_fire_mask[rr, cc] = bool(feats.get("no_fire_zone", False))
        has_plant_mask[rr, cc] = bool(is_inside and feats.get("has_plant", False))

        poi_dict = feats.get("poi_type_counts", {})
        if is_inside and poi_dict:
            poi_cells_set.add((rr, cc))
        for poi_type, count in poi_dict.items():
            if not is_inside:
                continue
            if int(count) <= 0:
                continue
            poi_cells_by_type[str(poi_type)].append((x, y))
            poi_type_counts[str(poi_type)] += int(count)

        for animal_type in feats.get("animals_present", []):
            animal_cells[str(animal_type)].add((rr, cc))
            animal_counts[str(animal_type)] += 1

    return {
        "inside_mask": inside_mask,
        "no_fire_mask": no_fire_mask,
        "has_plant_mask": has_plant_mask,
        "elevation_arr": elevation_arr,
        "road_length_arr": road_length_arr,
        "animal_cells": animal_cells,
        "animal_counts": animal_counts,
        "poi_cells_by_type": poi_cells_by_type,
        "poi_type_counts": poi_type_counts,
        "poi_cells_count": len(poi_cells_set),
    }


def style_axis(ax, min_x, max_x, min_y, max_y):
    ax.add_patch(
        Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            facecolor="#f2f2f2",
            edgecolor="#8a8a8a",
            linewidth=1.2,
            zorder=0,
        )
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.set_xlabel("X (meters, projected CRS)")
    ax.set_ylabel("Y (meters, projected CRS)")


def overlay_plants(ax, has_plant_mask: np.ndarray, extent: list[float], alpha: float, zorder: int):
    plants_img = np.where(has_plant_mask, 1.0, np.nan)
    ax.imshow(
        plants_img,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#2f855a"]),
        alpha=alpha,
        interpolation="nearest",
        zorder=zorder,
    )


def draw_animals_figure(
    meta: dict,
    inside_arr: np.ndarray,
    extent: list[float],
    animal_cells: dict[str, set[tuple[int, int]]],
    animal_groups: list[dict],
):
    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=180)
    for i in range(3):
        ax = axes[i]
        style_axis(ax, min_x, max_x, min_y, max_y)
        ax.imshow(
            inside_arr,
            extent=extent,
            origin="lower",
            cmap=ListedColormap(["#d9d9d9"]),
            alpha=0.9,
            interpolation="nearest",
            zorder=1,
        )

        if i < len(animal_groups):
            group = animal_groups[i]
            group_types = group["types"]
            legend_count = 0
            for j, animal_type in enumerate(group_types):
                cells = animal_cells.get(animal_type, set())
                if not cells:
                    continue

                type_mask = np.full((n_rows, n_cols), np.nan, dtype="float32")
                for rr, cc in cells:
                    type_mask[rr, cc] = 1.0

                color = ANIMAL_COLOR_CYCLE[j % len(ANIMAL_COLOR_CYCLE)]
                ax.imshow(
                    type_mask,
                    extent=extent,
                    origin="lower",
                    cmap=ListedColormap([color]),
                    interpolation="nearest",
                    alpha=0.34,
                    zorder=2,
                )
                # Контур для читаемости границ области поверх полупрозрачной заливки.
                ax.contour(
                    np.nan_to_num(type_mask, nan=0.0),
                    levels=[0.5],
                    colors=[color],
                    linewidths=1.05,
                    origin="lower",
                    extent=extent,
                    zorder=3,
                )
                ax.plot(
                    [],
                    [],
                    color=color,
                    linewidth=6,
                    alpha=0.55,
                    label=f"{animal_type} ({len(cells)})",
                )
                legend_count += 1

            source_name = group["used"] if group["used"] else group["requested"]
            ax.set_title(f"{group['requested']} (source: {source_name})")
            if legend_count > 0:
                ax.legend(
                    loc="upper right",
                    ncol=2,
                    fontsize=7,
                    frameon=True,
                    framealpha=0.52,
                    facecolor="white",
                    edgecolor="#7a7a7a",
                    title="Areas / types",
                    title_fontsize=8,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No animal polygons found for this file",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#555555",
                    zorder=3,
                )
        else:
            ax.set_title("Animal destinations: none")

    group_names = [g.get("requested", "unknown") for g in animal_groups]
    fig.suptitle(
        (
            f"Animals from graph JSON by groups ({', '.join(group_names)}) "
            f"| crs={meta.get('metric_crs', 'unknown')}"
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_ANIMALS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return animal_groups


def draw_infra_figure(
    meta: dict,
    inside_arr: np.ndarray,
    extent: list[float],
    elevation_arr: np.ndarray,
    road_length_arr: np.ndarray,
    poi_cells_by_type: dict[str, list[tuple[float, float]]],
    poi_type_counts: Counter,
):
    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    fig, axes = plt.subplots(2, 1, figsize=(12, 14), dpi=180)

    ax1 = axes[0]
    style_axis(ax1, min_x, max_x, min_y, max_y)
    ax1.imshow(
        elevation_arr,
        extent=extent,
        origin="lower",
        cmap="terrain",
        interpolation="nearest",
        alpha=0.95,
        zorder=1,
    )
    road_arr = np.where(road_length_arr > 0.0, road_length_arr, np.nan)
    im1 = ax1.imshow(
        road_arr,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        interpolation="nearest",
        alpha=0.9,
        zorder=2,
    )
    ax1.set_title("Road length by cell (m) on elevation background")
    plt.colorbar(
        im1,
        ax=ax1,
        fraction=0.035,
        pad=0.02,
        shrink=0.68,
        label="road_total_length_m",
    )

    ax2 = axes[1]
    style_axis(ax2, min_x, max_x, min_y, max_y)
    ax2.imshow(
        inside_arr,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#d9d9d9"]),
        alpha=0.9,
        interpolation="nearest",
        zorder=1,
    )
    poi_palette = plt.get_cmap("tab20")
    poi_types_sorted = [t for t, _ in poi_type_counts.most_common()]
    for i, poi_type in enumerate(poi_types_sorted):
        pts = poi_cells_by_type.get(poi_type, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.scatter(
            xs,
            ys,
            s=34,
            marker="o",
            color=poi_palette(i % 20),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            label=f"{poi_type} ({len(pts)})",
            zorder=2,
        )
    if poi_types_sorted:
        ax2.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=4,
            fontsize=7,
            frameon=True,
            title="POI types",
            title_fontsize=8,
        )
    ax2.set_title("POI by type (cells)", pad=22)

    fig.tight_layout()
    fig.savefig(OUT_INFRA_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return poi_types_sorted


def draw_fire_figure(
    meta: dict,
    inside_arr: np.ndarray,
    extent: list[float],
    elevation_arr: np.ndarray,
    no_fire_mask: np.ndarray,
):
    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=180)
    style_axis(ax, min_x, max_x, min_y, max_y)
    # Бинарная карта: белый = огня не было, тёмно-красный = огонь был замечен.
    valid_mask = ~np.isnan(elevation_arr)
    no_fire_binary = no_fire_mask.astype(bool)
    fire_binary = valid_mask & (~no_fire_binary)
    no_fire_binary = valid_mask & no_fire_binary

    fire_img = np.full((no_fire_binary.shape[0], no_fire_binary.shape[1]), np.nan, dtype="float32")
    fire_img[no_fire_binary] = 0.0
    fire_img[fire_binary] = 1.0

    cmap = ListedColormap(["#ffffff", "#7f0000"])
    ax.imshow(
        fire_img,
        extent=extent,
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        zorder=2,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title("Fire map: white=no fire, dark red=fire observed")

    fig.tight_layout()
    fig.savefig(OUT_FIRE_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_security_figure(
    meta: dict,
    inside_arr: np.ndarray,
    extent: list[float],
    road_length_arr: np.ndarray,
    poi_cells_by_type: dict[str, list[tuple[float, float]]],
):
    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=180)
    style_axis(ax, min_x, max_x, min_y, max_y)
    ax.imshow(
        inside_arr,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#e2e2e2"]),
        alpha=0.95,
        interpolation="nearest",
        zorder=1,
    )

    roads_img = np.where(road_length_arr > 0.0, np.log1p(road_length_arr), np.nan)
    ax.imshow(
        roads_img,
        extent=extent,
        origin="lower",
        cmap="copper",
        interpolation="nearest",
        alpha=0.82,
        zorder=2,
    )

    type_style = {
        "patrol_house": {
            "color": "#1f77b4",
            "label": "Patrol houses",
            "marker": "^",
            "zorder": 5,
        },
        "photo_trap": {
            "color": "#d62728",
            "label": "Photo traps",
            "marker": "o",
            "zorder": 4,
        },
    }
    overlap_radius = 0.14 * cell_size

    # Если точки разных типов попали в один и тот же центр клетки, разводим их
    # небольшим смещением, чтобы обе были видны на карте.
    coord_to_items = defaultdict(list)
    for poi_type in type_style:
        for x, y in poi_cells_by_type.get(poi_type, []):
            key = (round(float(x), 3), round(float(y), 3))
            coord_to_items[key].append((poi_type, float(x), float(y)))

    shifted_points = {poi_type: [] for poi_type in type_style}
    shifted_lines = []
    for items in coord_to_items.values():
        if len(items) == 1:
            poi_type, x0, y0 = items[0]
            shifted_points[poi_type].append((x0, y0))
            continue

        n = len(items)
        items_sorted = sorted(items, key=lambda t: t[0])
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        for i, (poi_type, x0, y0) in enumerate(items_sorted):
            dx = overlap_radius * np.cos(angles[i])
            dy = overlap_radius * np.sin(angles[i])
            x1 = x0 + dx
            y1 = y0 + dy
            shifted_points[poi_type].append((x1, y1))
            shifted_lines.append((x0, y0, x1, y1))

    for x0, y0, x1, y1 in shifted_lines:
        ax.plot([x0, x1], [y0, y1], color="#5a5a5a", linewidth=0.6, alpha=0.65, zorder=3)

    legend_added = False
    draw_order = ["photo_trap", "patrol_house"]
    for poi_type in draw_order:
        style = type_style[poi_type]
        pts = poi_cells_by_type.get(poi_type, [])
        pts_shifted = shifted_points.get(poi_type, [])
        if not pts:
            continue
        xs = [p[0] for p in pts_shifted]
        ys = [p[1] for p in pts_shifted]
        ax.scatter(
            xs,
            ys,
            s=70,
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.96,
            label=f"{style['label']} ({len(pts)})",
            zorder=style["zorder"],
        )
        legend_added = True

    if legend_added:
        ax.legend(
            loc="upper right",
            fontsize=8,
            frameon=True,
            framealpha=0.7,
            facecolor="white",
            edgecolor="#6a6a6a",
            title="Security layers",
            title_fontsize=9,
        )

    ax.set_title("Security map from graph: roads + patrol houses + photo traps")
    fig.tight_layout()
    fig.savefig(OUT_SECURITY_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_plants_figure(
    meta: dict,
    inside_arr: np.ndarray,
    extent: list[float],
    has_plant_mask: np.ndarray,
):
    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=180)
    style_axis(ax, min_x, max_x, min_y, max_y)
    ax.imshow(
        inside_arr,
        extent=extent,
        origin="lower",
        cmap=ListedColormap(["#e0e0e0"]),
        alpha=0.95,
        interpolation="nearest",
        zorder=1,
    )

    # Полигональные перекрытия уже объединены при билде, здесь рисуем итоговое покрытие.
    overlay_plants(ax=ax, has_plant_mask=has_plant_mask, extent=extent, alpha=0.48, zorder=2)
    ax.set_title("Important plants coverage (semi-transparent overlay)")
    fig.tight_layout()
    fig.savefig(OUT_PLANTS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render graph layers from a fine-grid JSON file")
    parser.add_argument("--graph", default=GRAPH_JSON, help="Input graph JSON")
    parser.add_argument("--out-dir", default=".", help="Directory for the five output PNG files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global OUT_ANIMALS_PNG, OUT_INFRA_PNG, OUT_FIRE_PNG, OUT_SECURITY_PNG, OUT_PLANTS_PNG
    OUT_ANIMALS_PNG = str(out_dir / Path(OUT_ANIMALS_PNG).name)
    OUT_INFRA_PNG = str(out_dir / Path(OUT_INFRA_PNG).name)
    OUT_FIRE_PNG = str(out_dir / Path(OUT_FIRE_PNG).name)
    OUT_SECURITY_PNG = str(out_dir / Path(OUT_SECURITY_PNG).name)
    OUT_PLANTS_PNG = str(out_dir / Path(OUT_PLANTS_PNG).name)

    meta, node_features, edge_features = load_graph(args.graph)
    layers = collect_graph_layers(meta, node_features)
    animal_groups = meta.get("animal_groups", [])

    min_x, min_y = meta["grid_origin_m"]
    n_rows, n_cols = meta["grid_shape_rows_cols"]
    cell_size = float(meta["cell_size_m"])
    max_x = min_x + n_cols * cell_size
    max_y = min_y + n_rows * cell_size
    extent = [min_x, max_x, min_y, max_y]
    inside_arr = np.where(layers["inside_mask"], 1.0, np.nan)

    used_animal_groups = draw_animals_figure(
        meta=meta,
        inside_arr=inside_arr,
        extent=extent,
        animal_cells=layers["animal_cells"],
        animal_groups=animal_groups,
    )
    poi_types_sorted = draw_infra_figure(
        meta=meta,
        inside_arr=inside_arr,
        extent=extent,
        elevation_arr=layers["elevation_arr"],
        road_length_arr=layers["road_length_arr"],
        poi_cells_by_type=layers["poi_cells_by_type"],
        poi_type_counts=layers["poi_type_counts"],
    )
    draw_fire_figure(
        meta=meta,
        inside_arr=inside_arr,
        extent=extent,
        elevation_arr=layers["elevation_arr"],
        no_fire_mask=layers["no_fire_mask"],
    )
    draw_security_figure(
        meta=meta,
        inside_arr=inside_arr,
        extent=extent,
        road_length_arr=layers["road_length_arr"],
        poi_cells_by_type=layers["poi_cells_by_type"],
    )
    draw_plants_figure(
        meta=meta,
        inside_arr=inside_arr,
        extent=extent,
        has_plant_mask=layers["has_plant_mask"],
    )

    degree_dist = dict(sorted(Counter(len(v) for v in edge_features.values()).items()))
    print(f"Saved: {OUT_ANIMALS_PNG}")
    print(f"Saved: {OUT_INFRA_PNG}")
    print(f"Saved: {OUT_FIRE_PNG}")
    print(f"Saved: {OUT_SECURITY_PNG}")
    print(f"Saved: {OUT_PLANTS_PNG}")
    print(f"Nodes: {len(node_features)}")
    print(f"Inside cells: {int(np.sum(layers['inside_mask']))}")
    print(
        "Animal groups: "
        + ", ".join(
            [
                f"{g['requested']} -> {g['used'] if g['used'] else 'missing'} ({len(g['types'])} types)"
                for g in used_animal_groups
            ]
        )
    )
    print(f"Road cells: {int(np.sum(layers['road_length_arr'] > 0.0))}")
    print(f"POI cells: {layers['poi_cells_count']}")
    print(f"No-fire cells: {int(np.sum(layers['no_fire_mask']))}")
    print(f"Plant cells: {int(np.sum(layers['has_plant_mask']))}")
    print(f"POI types shown: {len(poi_types_sorted)}")
    print(f"Degree distribution: {degree_dist}")


if __name__ == "__main__":
    main()
