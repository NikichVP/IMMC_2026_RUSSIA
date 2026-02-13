import json
import os
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import LineString, box
from shapely.ops import unary_union


DEM_PATH = "map-hieght.tif"
POI_GPKG = "etosha_poi.gpkg"
ROADS_GPKG = "etosha_roads.gpkg"
FIRE_SPOTS_GPKG = "fire-spots.gpkg"
PLANTS_GPKG = "plants.gpkg"
TYPE_COL = "type"
EXTRA_POI_LAYERS = [
    ("patrol-houses.gpkg", "patrol_house"),
    ("photo-trap.gpkg", "photo_trap"),
]

CELL_SIZE_M = 1000.0
DIAGONAL_DISTANCE_M = CELL_SIZE_M * (2.0 ** 0.5)
OUT_JSON = "etosha_grid_graph.json"
FIRE_NO_FIRE_DILATE_STEPS = 2
FIRE_NO_FIRE_SMOOTH_ITERS = 6
ANIMAL_DEST_FILES = [
    "animal-destinations1.gpkg",
    "animal-destinations2.gpkg",
    "animal-destinations3.gpkg",
]
ANIMAL_DEST_ALIASES = {
    # В проекте первый файл сейчас называется во множественном числе.
    "animal-destinations1.gpkg": "animals-destinations.gpkg",
}


def cell_id(row: int, col: int) -> str:
    return f"r{row}_c{col}"


def linear_to_row_col(linear_idx: int, n_cols: int) -> tuple[int, int]:
    return linear_idx // n_cols, linear_idx % n_cols


def pick_metric_crs(src_crs, bounds) -> str:
    if src_crs is not None and not src_crs.is_geographic:
        return src_crs.to_string()

    lon = 0.5 * (bounds.left + bounds.right)
    lat = 0.5 * (bounds.bottom + bounds.top)
    zone = int((lon + 180.0) // 6.0) + 1
    epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
    return f"EPSG:{epsg}"


def reproject_dem_to_metric(dem_path: str, dst_crs: str) -> tuple[np.ndarray, object]:
    with rasterio.open(dem_path) as src:
        src_arr = src.read(1).astype("float32")
        src_nodata = src.nodata

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype="float32")

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return dst_arr, dst_transform


def aligned_grid_bounds(transform, height: int, width: int, cell_size_m: float) -> tuple[float, float, int, int]:
    left, bottom, right, top = array_bounds(height, width, transform)
    min_x = np.floor(left / cell_size_m) * cell_size_m
    min_y = np.floor(bottom / cell_size_m) * cell_size_m
    max_x = np.ceil(right / cell_size_m) * cell_size_m
    max_y = np.ceil(top / cell_size_m) * cell_size_m

    n_cols = int(round((max_x - min_x) / cell_size_m))
    n_rows = int(round((max_y - min_y) / cell_size_m))
    return min_x, min_y, n_rows, n_cols


def compute_median_elevation_by_cell(
    dem_arr: np.ndarray,
    transform,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> dict[int, float]:
    height, width = dem_arr.shape

    x_centers = transform.c + (np.arange(width, dtype="float64") + 0.5) * transform.a
    linear_chunks = []
    value_chunks = []

    for r in range(height):
        row_vals = dem_arr[r]
        valid = np.isfinite(row_vals)
        if not np.any(valid):
            continue

        y = transform.f + (r + 0.5) * transform.e
        cell_row = int(np.floor((y - min_y) / cell_size_m))
        if cell_row < 0 or cell_row >= n_rows:
            continue

        valid_cols = np.flatnonzero(valid)
        cell_cols = np.floor((x_centers[valid_cols] - min_x) / cell_size_m).astype(np.int32)
        in_bounds = (cell_cols >= 0) & (cell_cols < n_cols)
        if not np.any(in_bounds):
            continue

        cell_cols = cell_cols[in_bounds]
        vals = row_vals[valid_cols][in_bounds].astype("float32")
        linear_idx = cell_row * n_cols + cell_cols

        linear_chunks.append(linear_idx.astype(np.int64))
        value_chunks.append(vals)

    if not linear_chunks:
        return {}

    all_linear = np.concatenate(linear_chunks)
    all_vals = np.concatenate(value_chunks)

    order = np.argsort(all_linear)
    all_linear = all_linear[order]
    all_vals = all_vals[order]

    unique_linear, starts, counts = np.unique(all_linear, return_index=True, return_counts=True)

    medians = {}
    for linear_idx, start, count in zip(unique_linear, starts, counts):
        chunk = all_vals[start:start + count]
        medians[int(linear_idx)] = float(np.median(chunk))

    return medians


def assign_poi_to_cells(
    poi_gdf: gpd.GeoDataFrame,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> dict[int, Counter]:
    poi_counts = defaultdict(Counter)

    for _, row in poi_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type != "Point":
            geom = geom.representative_point()

        col = int(np.floor((geom.x - min_x) / cell_size_m))
        rr = int(np.floor((geom.y - min_y) / cell_size_m))
        if col < 0 or col >= n_cols or rr < 0 or rr >= n_rows:
            continue

        linear_idx = rr * n_cols + col
        poi_type = str(row.get(TYPE_COL, "unknown"))
        poi_counts[linear_idx][poi_type] += 1

    return dict(poi_counts)


def assign_roads_to_cells(
    roads_gdf: gpd.GeoDataFrame,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> tuple[dict[int, float], dict[int, int]]:
    road_length_m = defaultdict(float)
    road_count = defaultdict(int)

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        min_gx, min_gy, max_gx, max_gy = geom.bounds
        c0 = max(0, int(np.floor((min_gx - min_x) / cell_size_m)))
        c1 = min(n_cols - 1, int(np.floor((max_gx - min_x) / cell_size_m)))
        r0 = max(0, int(np.floor((min_gy - min_y) / cell_size_m)))
        r1 = min(n_rows - 1, int(np.floor((max_gy - min_y) / cell_size_m)))

        if c0 > c1 or r0 > r1:
            continue

        for rr in range(r0, r1 + 1):
            y0 = min_y + rr * cell_size_m
            y1 = y0 + cell_size_m
            for cc in range(c0, c1 + 1):
                x0 = min_x + cc * cell_size_m
                x1 = x0 + cell_size_m
                inter = geom.intersection(box(x0, y0, x1, y1))
                if inter.is_empty:
                    continue

                length_m = float(inter.length)
                if length_m <= 0.0:
                    continue

                linear_idx = rr * n_cols + cc
                road_length_m[linear_idx] += length_m
                road_count[linear_idx] += 1

    return dict(road_length_m), dict(road_count)


def load_animal_polygons(dem_crs, metric_crs: str) -> tuple[gpd.GeoDataFrame, list[dict]]:
    frames = []
    animal_groups = []

    for requested in ANIMAL_DEST_FILES:
        path = requested if os.path.exists(requested) else ANIMAL_DEST_ALIASES.get(requested)
        if path is None or not os.path.exists(path):
            print(f"Skip missing animal file: {requested}")
            animal_groups.append({"requested": requested, "used": None, "types": []})
            continue

        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(dem_crs)
        if gdf.crs is not None and gdf.crs.to_string() != metric_crs:
            gdf = gdf.to_crs(metric_crs)

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
        if TYPE_COL not in gdf.columns:
            gdf[TYPE_COL] = "unknown_animal"

        types = sorted([str(t) for t in gdf[TYPE_COL].dropna().unique()], key=str.lower)
        animal_groups.append({"requested": requested, "used": path, "types": types})

        frames.append(gdf[[TYPE_COL, "geometry"]].copy())
        if requested == path:
            print(f"Animal polygons loaded: {requested}")
        else:
            print(f"Animal polygons loaded: {requested} -> {path}")

    if not frames:
        return (
            gpd.GeoDataFrame({TYPE_COL: [], "geometry": []}, geometry="geometry", crs=metric_crs),
            animal_groups,
        )

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    return merged, animal_groups


def load_fire_spots(dem_crs, metric_crs: str) -> tuple[gpd.GeoDataFrame, dict]:
    status = {
        "requested": FIRE_SPOTS_GPKG,
        "used": FIRE_SPOTS_GPKG if os.path.exists(FIRE_SPOTS_GPKG) else None,
        "polygon_count": 0,
    }
    if not os.path.exists(FIRE_SPOTS_GPKG):
        print(f"Skip missing fire file: {FIRE_SPOTS_GPKG}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs), status

    gdf = gpd.read_file(FIRE_SPOTS_GPKG)
    if gdf.crs is None:
        gdf = gdf.set_crs(dem_crs)
    if gdf.crs is not None and gdf.crs.to_string() != metric_crs:
        gdf = gdf.to_crs(metric_crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    status["polygon_count"] = int(len(gdf))
    print(f"Fire spots loaded: {FIRE_SPOTS_GPKG} polygons={len(gdf)}")
    return gdf[["geometry"]].copy(), status


def load_plants(dem_crs, metric_crs: str) -> tuple[gpd.GeoDataFrame, dict]:
    status = {
        "requested": PLANTS_GPKG,
        "used": PLANTS_GPKG if os.path.exists(PLANTS_GPKG) else None,
        "polygon_count": 0,
        "merged_polygon_count": 0,
    }
    if not os.path.exists(PLANTS_GPKG):
        print(f"Skip missing plants file: {PLANTS_GPKG}")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs), status

    gdf = gpd.read_file(PLANTS_GPKG)
    if gdf.crs is None:
        gdf = gdf.set_crs(dem_crs)
    if gdf.crs is not None and gdf.crs.to_string() != metric_crs:
        gdf = gdf.to_crs(metric_crs)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    status["polygon_count"] = int(len(gdf))
    if len(gdf) == 0:
        print(f"Plants loaded: {PLANTS_GPKG} polygons=0")
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=metric_crs), status

    # Перекрытия растений объединяем в единое покрытие.
    merged_geom = unary_union(list(gdf.geometry.values))
    merged_polygons = []
    if merged_geom is not None and not merged_geom.is_empty:
        if merged_geom.geom_type == "Polygon":
            merged_polygons = [merged_geom]
        elif merged_geom.geom_type == "MultiPolygon":
            merged_polygons = list(merged_geom.geoms)
        elif merged_geom.geom_type == "GeometryCollection":
            merged_polygons = [geom for geom in merged_geom.geoms if geom.geom_type == "Polygon"]

    status["merged_polygon_count"] = int(len(merged_polygons))
    print(
        f"Plants loaded: {PLANTS_GPKG} polygons={status['polygon_count']} merged={status['merged_polygon_count']}"
    )
    return gpd.GeoDataFrame({"geometry": merged_polygons}, geometry="geometry", crs=metric_crs), status


def load_extra_poi_layers(dem_crs, metric_crs: str) -> tuple[gpd.GeoDataFrame, list[dict]]:
    frames = []
    status = []

    for path, poi_type in EXTRA_POI_LAYERS:
        layer_status = {
            "requested": path,
            "used": path if os.path.exists(path) else None,
            "poi_type": poi_type,
            "feature_count": 0,
        }
        if not os.path.exists(path):
            print(f"Skip missing POI layer: {path}")
            status.append(layer_status)
            continue

        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(dem_crs)
        if gdf.crs is not None and gdf.crs.to_string() != metric_crs:
            gdf = gdf.to_crs(metric_crs)

        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        layer_status["feature_count"] = int(len(gdf))
        status.append(layer_status)
        if len(gdf) == 0:
            continue

        typed = gdf[["geometry"]].copy()
        typed[TYPE_COL] = poi_type
        frames.append(typed[[TYPE_COL, "geometry"]])
        print(f"Extra POI loaded: {path} -> {poi_type} features={len(typed)}")

    if not frames:
        return (
            gpd.GeoDataFrame({TYPE_COL: [], "geometry": []}, geometry="geometry", crs=metric_crs),
            status,
        )

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    return merged, status


def assign_animals_to_cells(
    animals_gdf: gpd.GeoDataFrame,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> dict[int, set]:
    animals_by_cell = defaultdict(set)
    if len(animals_gdf) == 0:
        return {}

    for _, row in animals_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        min_gx, min_gy, max_gx, max_gy = geom.bounds
        c0 = max(0, int(np.floor((min_gx - min_x) / cell_size_m)))
        c1 = min(n_cols - 1, int(np.floor((max_gx - min_x) / cell_size_m)))
        r0 = max(0, int(np.floor((min_gy - min_y) / cell_size_m)))
        r1 = min(n_rows - 1, int(np.floor((max_gy - min_y) / cell_size_m)))

        if c0 > c1 or r0 > r1:
            continue

        animal_type = str(row.get(TYPE_COL, "unknown_animal"))
        for rr in range(r0, r1 + 1):
            y0 = min_y + rr * cell_size_m
            y1 = y0 + cell_size_m
            for cc in range(c0, c1 + 1):
                x0 = min_x + cc * cell_size_m
                x1 = x0 + cell_size_m
                inter = geom.intersection(box(x0, y0, x1, y1))
                if inter.is_empty:
                    continue
                if inter.area <= 0.0:
                    continue

                linear_idx = rr * n_cols + cc
                animals_by_cell[linear_idx].add(animal_type)

    return dict(animals_by_cell)


def assign_no_fire_to_cells(
    fire_spots_gdf: gpd.GeoDataFrame,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> set[int]:
    no_fire_cells = set()
    if len(fire_spots_gdf) == 0:
        return no_fire_cells

    for _, row in fire_spots_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        min_gx, min_gy, max_gx, max_gy = geom.bounds
        c0 = max(0, int(np.floor((min_gx - min_x) / cell_size_m)))
        c1 = min(n_cols - 1, int(np.floor((max_gx - min_x) / cell_size_m)))
        r0 = max(0, int(np.floor((min_gy - min_y) / cell_size_m)))
        r1 = min(n_rows - 1, int(np.floor((max_gy - min_y) / cell_size_m)))
        if c0 > c1 or r0 > r1:
            continue

        for rr in range(r0, r1 + 1):
            y0 = min_y + rr * cell_size_m
            y1 = y0 + cell_size_m
            for cc in range(c0, c1 + 1):
                x0 = min_x + cc * cell_size_m
                x1 = x0 + cell_size_m
                inter = geom.intersection(box(x0, y0, x1, y1))
                if inter.is_empty or inter.area <= 0.0:
                    continue
                no_fire_cells.add(rr * n_cols + cc)

    return no_fire_cells


def assign_plants_to_cells(
    plants_gdf: gpd.GeoDataFrame,
    min_x: float,
    min_y: float,
    n_rows: int,
    n_cols: int,
    cell_size_m: float,
) -> set[int]:
    plant_cells = set()
    if len(plants_gdf) == 0:
        return plant_cells

    for _, row in plants_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        min_gx, min_gy, max_gx, max_gy = geom.bounds
        c0 = max(0, int(np.floor((min_gx - min_x) / cell_size_m)))
        c1 = min(n_cols - 1, int(np.floor((max_gx - min_x) / cell_size_m)))
        r0 = max(0, int(np.floor((min_gy - min_y) / cell_size_m)))
        r1 = min(n_rows - 1, int(np.floor((max_gy - min_y) / cell_size_m)))
        if c0 > c1 or r0 > r1:
            continue

        for rr in range(r0, r1 + 1):
            y0 = min_y + rr * cell_size_m
            y1 = y0 + cell_size_m
            for cc in range(c0, c1 + 1):
                x0 = min_x + cc * cell_size_m
                x1 = x0 + cell_size_m
                inter = geom.intersection(box(x0, y0, x1, y1))
                if inter.is_empty or inter.area <= 0.0:
                    continue
                plant_cells.add(rr * n_cols + cc)

    return plant_cells


def dilate_mask(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(max(0, int(steps))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def smooth_binary_map(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    arr = mask.astype("float32")
    for _ in range(max(0, int(iterations))):
        p = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
        arr = (
            1 * p[:-2, :-2] + 2 * p[:-2, 1:-1] + 1 * p[:-2, 2:]
            + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:]
            + 1 * p[2:, :-2] + 2 * p[2:, 1:-1] + 1 * p[2:, 2:]
        ) / 16.0
    return np.clip(arr, 0.0, 1.0)


def road_between_cells_stats(
    roads_sindex,
    road_geometries,
    min_x: float,
    min_y: float,
    cell_size_m: float,
    n_cols: int,
    road_len_by_cell: dict[int, float],
    r1: int,
    c1: int,
    r2: int,
    c2: int,
) -> dict:
    if r1 == r2 and abs(c1 - c2) == 1:
        x = min_x + max(c1, c2) * cell_size_m
        y0 = min_y + r1 * cell_size_m
        y1 = y0 + cell_size_m
        shared_border = LineString([(x, y0), (x, y1)])
    elif c1 == c2 and abs(r1 - r2) == 1:
        y = min_y + max(r1, r2) * cell_size_m
        x0 = min_x + c1 * cell_size_m
        x1 = x0 + cell_size_m
        shared_border = LineString([(x0, y), (x1, y)])
    else:
        return {
            "road_between_cells": False,
            "roads_between_cells_count": 0,
            "roads_between_cells_length_m": 0.0,
        }

    a_x0 = min_x + c1 * cell_size_m
    a_y0 = min_y + r1 * cell_size_m
    b_x0 = min_x + c2 * cell_size_m
    b_y0 = min_y + r2 * cell_size_m

    cell_a = box(a_x0, a_y0, a_x0 + cell_size_m, a_y0 + cell_size_m)
    cell_b = box(b_x0, b_y0, b_x0 + cell_size_m, b_y0 + cell_size_m)

    road_count = 0
    for idx in roads_sindex.intersection(shared_border.bounds):
        geom = road_geometries[idx]
        if geom is None or geom.is_empty:
            continue
        if not geom.intersects(shared_border):
            continue

        len_a = float(geom.intersection(cell_a).length)
        len_b = float(geom.intersection(cell_b).length)
        if len_a <= 0.0 or len_b <= 0.0:
            continue

        road_count += 1

    if road_count <= 0:
        return {
            "road_between_cells": False,
            "roads_between_cells_count": 0,
            "roads_between_cells_length_m": 0.0,
        }

    linear_a = r1 * n_cols + c1
    linear_b = r2 * n_cols + c2
    total_inside_roads = float(
        road_len_by_cell.get(linear_a, 0.0) + road_len_by_cell.get(linear_b, 0.0)
    )

    return {
        "road_between_cells": True,
        "roads_between_cells_count": int(road_count),
        "roads_between_cells_length_m": round(total_inside_roads, 3),
    }


def build_graph():
    with rasterio.open(DEM_PATH) as src:
        dem_crs = src.crs
        dem_bounds = src.bounds

    metric_crs = pick_metric_crs(dem_crs, dem_bounds)

    poi = gpd.read_file(POI_GPKG)
    roads = gpd.read_file(ROADS_GPKG)

    if TYPE_COL not in poi.columns:
        poi[TYPE_COL] = "unknown"
    if poi.crs is None:
        poi = poi.set_crs(dem_crs)
    if roads.crs is None:
        roads = roads.set_crs(dem_crs)

    poi_metric = poi.to_crs(metric_crs)
    roads_metric = roads.to_crs(metric_crs)
    extra_poi_metric, extra_poi_status = load_extra_poi_layers(
        dem_crs=dem_crs, metric_crs=metric_crs
    )
    if len(extra_poi_metric) > 0:
        poi_metric = gpd.GeoDataFrame(
            pd.concat(
                [
                    poi_metric[[TYPE_COL, "geometry"]].copy(),
                    extra_poi_metric[[TYPE_COL, "geometry"]].copy(),
                ],
                ignore_index=True,
            ),
            geometry="geometry",
            crs=metric_crs,
        )
    else:
        poi_metric = poi_metric[[TYPE_COL, "geometry"]].copy()

    animals_metric, animal_groups = load_animal_polygons(dem_crs=dem_crs, metric_crs=metric_crs)
    fire_spots_metric, fire_status = load_fire_spots(dem_crs=dem_crs, metric_crs=metric_crs)
    plants_metric, plants_status = load_plants(dem_crs=dem_crs, metric_crs=metric_crs)

    dem_metric, dem_transform = reproject_dem_to_metric(DEM_PATH, metric_crs)
    min_x, min_y, n_rows, n_cols = aligned_grid_bounds(
        dem_transform, dem_metric.shape[0], dem_metric.shape[1], CELL_SIZE_M
    )

    elev_median = compute_median_elevation_by_cell(
        dem_metric, dem_transform, min_x, min_y, n_rows, n_cols, CELL_SIZE_M
    )
    valid_map_cells = set(elev_median.keys())
    poi_counts = assign_poi_to_cells(poi_metric, min_x, min_y, n_rows, n_cols, CELL_SIZE_M)
    poi_counts = {idx: cnt for idx, cnt in poi_counts.items() if idx in valid_map_cells}
    road_len, road_cnt = assign_roads_to_cells(
        roads_metric, min_x, min_y, n_rows, n_cols, CELL_SIZE_M
    )
    animals_by_cell = assign_animals_to_cells(
        animals_metric, min_x, min_y, n_rows, n_cols, CELL_SIZE_M
    )
    no_fire_cells = assign_no_fire_to_cells(
        fire_spots_metric, min_x, min_y, n_rows, n_cols, CELL_SIZE_M
    )
    plant_cells = assign_plants_to_cells(
        plants_metric, min_x, min_y, n_rows, n_cols, CELL_SIZE_M
    )
    plant_cells = {idx for idx in plant_cells if idx in valid_map_cells}
    no_fire_raw_mask = np.zeros((n_rows, n_cols), dtype=bool)
    for linear_idx in no_fire_cells:
        rr, cc = linear_to_row_col(linear_idx, n_cols)
        no_fire_raw_mask[rr, cc] = True
    no_fire_expanded_mask = dilate_mask(no_fire_raw_mask, steps=FIRE_NO_FIRE_DILATE_STEPS)
    no_fire_smoothed = smooth_binary_map(no_fire_expanded_mask, iterations=FIRE_NO_FIRE_SMOOTH_ITERS)
    no_fire_final_mask = no_fire_smoothed >= 0.5

    node_features = {}
    total_cells = n_rows * n_cols
    for linear_idx in range(total_cells):
        rr, cc = linear_to_row_col(linear_idx, n_cols)
        x0 = min_x + cc * CELL_SIZE_M
        y0 = min_y + rr * CELL_SIZE_M
        x1 = x0 + CELL_SIZE_M
        y1 = y0 + CELL_SIZE_M

        cid = cell_id(rr, cc)
        poi_counter = poi_counts.get(linear_idx, Counter())
        road_length = float(road_len.get(linear_idx, 0.0))
        animals_here = sorted(list(animals_by_cell.get(linear_idx, set())))
        no_fire_here = bool(no_fire_final_mask[rr, cc])
        has_plant_here = bool(linear_idx in plant_cells)

        node_features[cid] = {
            "row": int(rr),
            "col": int(cc),
            "is_boarder": bool(rr == 0 or cc == 0 or rr == n_rows - 1 or cc == n_cols - 1),
            "bbox_m": [float(x0), float(y0), float(x1), float(y1)],
            "centroid_m": [float(x0 + 0.5 * CELL_SIZE_M), float(y0 + 0.5 * CELL_SIZE_M)],
            "median_elevation_m": (
                float(elev_median[linear_idx]) if linear_idx in elev_median else None
            ),
            "poi_type_counts": {k: int(v) for k, v in sorted(poi_counter.items())},
            "road_total_length_m": round(road_length, 3),
            "road_segment_count": int(road_cnt.get(linear_idx, 0)),
            "animals_present": animals_here,
            "no_fire_zone": bool(no_fire_here),
            "has_plant": bool(has_plant_here),
        }

    edge_features = {cid: {} for cid in node_features}
    active_lookup = {
        (node_features[cid]["row"], node_features[cid]["col"]): cid for cid in node_features
    }
    roads_sindex = roads_metric.sindex
    road_geometries = roads_metric.geometry.values
    edge_road_cache = {}

    for cid, feats in node_features.items():
        rr = feats["row"]
        cc = feats["col"]
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
            nr = rr + dr
            nc = cc + dc
            neighbor_id = active_lookup.get((nr, nc))
            if neighbor_id is None:
                continue

            z1 = feats["median_elevation_m"]
            z2 = node_features[neighbor_id]["median_elevation_m"]
            zdiff = None if z1 is None or z2 is None else abs(float(z1) - float(z2))
            is_diagonal = dr != 0 and dc != 0

            if is_diagonal:
                edge_road_stats = {
                    "road_between_cells": False,
                    "roads_between_cells_count": 0,
                    "roads_between_cells_length_m": 0.0,
                }
                distance_m = DIAGONAL_DISTANCE_M
                shared_border_m = 0.0
            else:
                edge_key = ((rr, cc), (nr, nc)) if (rr, cc) < (nr, nc) else ((nr, nc), (rr, cc))
                if edge_key not in edge_road_cache:
                    edge_road_cache[edge_key] = road_between_cells_stats(
                        roads_sindex=roads_sindex,
                        road_geometries=road_geometries,
                        min_x=min_x,
                        min_y=min_y,
                        cell_size_m=CELL_SIZE_M,
                        n_cols=n_cols,
                        road_len_by_cell=road_len,
                        r1=rr,
                        c1=cc,
                        r2=nr,
                        c2=nc,
                    )
                edge_road_stats = edge_road_cache[edge_key]
                distance_m = CELL_SIZE_M
                shared_border_m = CELL_SIZE_M

            edge_features[cid][neighbor_id] = {
                "distance_m": distance_m,
                "shared_border_m": shared_border_m,
                "elevation_median_diff_m": zdiff,
                **edge_road_stats,
            }

    return node_features, edge_features, {
        "cell_size_m": CELL_SIZE_M,
        "metric_crs": metric_crs,
        "grid_origin_m": [float(min_x), float(min_y)],
        "grid_shape_rows_cols": [int(n_rows), int(n_cols)],
        "node_count": len(node_features),
        "animal_groups": animal_groups,
        "additional_poi_layers": extra_poi_status,
        "fire_spots": {
            **fire_status,
            "no_fire_cell_count": int(np.sum(no_fire_final_mask)),
            "no_fire_expanded_cell_count": int(np.sum(no_fire_expanded_mask)),
            "no_fire_dilate_steps": FIRE_NO_FIRE_DILATE_STEPS,
            "no_fire_smooth_iters": FIRE_NO_FIRE_SMOOTH_ITERS,
        },
        "plants": {
            **plants_status,
            "plant_cell_count": int(len(plant_cells)),
        },
    }


def main():
    node_features, edge_features, meta = build_graph()
    payload = {
        "meta": meta,
        "node_features": node_features,
        "edge_features": edge_features,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved graph data to {OUT_JSON}")
    print(f"Nodes: {meta['node_count']}")


if __name__ == "__main__":
    main()
