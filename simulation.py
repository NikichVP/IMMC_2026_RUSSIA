# simulation.py
from __future__ import annotations

import heapq
import json
import itertools
import math
from collections import Counter
from typing import Dict, Any, Iterable, List, Tuple, FrozenSet


ROAD_SPEED_KMH = 60.0
OFFROAD_SPEED_KMH = 30.0
METERS_PER_KM = 1000.0
DEFAULT_EDGE_DISTANCE_M = 1000.0


def loss_stub(sim_result: Dict[str, Any]) -> float:
    # TODO: сюда потом твой реальный loss
    return 0.0


def edge_travel_time_hours(edge_feat: Dict[str, Any]) -> float:
    """
    Время перехода между соседними клетками (центр -> центр).
    Если на ребре есть дорога, едем ROAD_SPEED_KMH, иначе OFFROAD_SPEED_KMH.
    """
    distance_m = float(edge_feat.get("distance_m", DEFAULT_EDGE_DISTANCE_M))
    speed_kmh = ROAD_SPEED_KMH if bool(edge_feat.get("road_between_cells", False)) else OFFROAD_SPEED_KMH
    return distance_m / (speed_kmh * METERS_PER_KM)


def shortest_travel_time_hours(
    graph: Dict[str, Any],
    src_cell_id: str,
    dst_cell_id: str,
) -> float:
    """
    Кратчайшее время пути между двумя клетками по графу (Dijkstra).
    """
    nodes: Dict[str, Any] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]

    if src_cell_id not in nodes:
        raise ValueError(f"Unknown source cell_id: {src_cell_id}")
    if dst_cell_id not in nodes:
        raise ValueError(f"Unknown destination cell_id: {dst_cell_id}")
    if src_cell_id == dst_cell_id:
        return 0.0

    best: Dict[str, float] = {src_cell_id: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, src_cell_id)]

    while pq:
        t_cur, cur = heapq.heappop(pq)
        if t_cur > best.get(cur, math.inf):
            continue
        if cur == dst_cell_id:
            return t_cur

        for nxt, e_feat in edges.get(cur, {}).items():
            cand = t_cur + edge_travel_time_hours(e_feat)
            if cand < best.get(nxt, math.inf):
                best[nxt] = cand
                heapq.heappush(pq, (cand, nxt))

    return math.inf


def precompute_sector_assignment(
    graph: Dict[str, Any],
    patrol_cells: Iterable[str],
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Разметка "секторов ответственности":
      sector(v) = argmin_{a in patrol_cells} d(a, v)
    где d(a, v) — минимальное время пути по графу.

    Returns:
      - nearest_post_by_cell: cell_id -> ближайший активный пост
      - travel_time_hours_by_cell: cell_id -> время до ближайшего поста
    """
    nodes: Dict[str, Any] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]

    posts = sorted(set(patrol_cells))
    if not posts:
        raise ValueError("patrol_cells is empty; at least one active post is required")
    unknown = [p for p in posts if p not in nodes]
    if unknown:
        raise ValueError(f"Unknown patrol cells: {unknown[:5]}")

    nearest_post_by_cell: Dict[str, str] = {}
    travel_time_hours_by_cell: Dict[str, float] = {}
    pq: List[Tuple[float, str, str]] = []

    # Multi-source Dijkstra: каждая "волна" помнит, от какого поста пришла.
    for p in posts:
        nearest_post_by_cell[p] = p
        travel_time_hours_by_cell[p] = 0.0
        heapq.heappush(pq, (0.0, p, p))

    while pq:
        t_cur, cur, src_post = heapq.heappop(pq)

        best_t = travel_time_hours_by_cell.get(cur, math.inf)
        best_src = nearest_post_by_cell.get(cur)
        if t_cur > best_t:
            continue
        if best_src is not None and abs(t_cur - best_t) <= 1e-12 and src_post > best_src:
            continue

        for nxt, e_feat in edges.get(cur, {}).items():
            cand_t = t_cur + edge_travel_time_hours(e_feat)
            old_t = travel_time_hours_by_cell.get(nxt, math.inf)
            old_src = nearest_post_by_cell.get(nxt)

            better_time = cand_t < old_t - 1e-12
            tie_break = abs(cand_t - old_t) <= 1e-12 and (old_src is None or src_post < old_src)

            if better_time or tie_break:
                travel_time_hours_by_cell[nxt] = cand_t
                nearest_post_by_cell[nxt] = src_post
                heapq.heappush(pq, (cand_t, nxt, src_post))

    return nearest_post_by_cell, travel_time_hours_by_cell


def precompute_sector_assignments_for_active_post_subsets(
    graph: Dict[str, Any],
    patrol_posts: Iterable[str],
) -> Dict[FrozenSet[str], Tuple[Dict[str, str], Dict[str, float]]]:
    """
    Предподсчёт разметки для всех непустых подмножеств активных блок-постов.
    Ключ кэша: frozenset активных cell_id постов.
    """
    unique_posts = sorted(set(patrol_posts))
    cache: Dict[FrozenSet[str], Tuple[Dict[str, str], Dict[str, float]]] = {}
    for k in range(1, len(unique_posts) + 1):
        for subset in itertools.combinations(unique_posts, k):
            key = frozenset(subset)
            cache[key] = precompute_sector_assignment(graph, subset)
    return cache


def simulate_patrol(graph: Dict[str, Any], starts: List[str], **kwargs) -> Dict[str, Any]:
    # TODO: ты заменишь это своей симуляцией движения патрулей
    # starts: список cell_id, откуда выходят K патрулей
    return {
        "starts": starts,
        "patrol_post_counts": kwargs.get("patrol_post_counts", {}),
        "active_posts_unique": kwargs.get("active_posts_unique", []),
        "sector_assignment": kwargs.get("sector_assignment", {}),
        "sector_travel_time_hours": kwargs.get("sector_travel_time_hours", {}),
    }


def sweep_patrols(
    graph_path: str,
    ks: List[int],
    *,
    candidate_poi_prefixes: Tuple[str, ...] = ("patrol_house",),
    allow_multiple_patrols_per_post: bool = True,
    max_combos: int | None = None,   # если хочешь ограничить перебор (иначе может взорваться)
    **simulate_kwargs
) -> List[Tuple[int, Tuple[str, ...], float]]:
    """
    Перебирает:
      - K = количество патрулей
      - все наборы стартовых постов starts (|starts|=K) из кандидатов
        * если allow_multiple_patrols_per_post=True, допускает несколько патрулей на одном посту
    Для каждого запускает simulate_patrol() и считает loss_stub().

    Возвращает список (K, starts_tuple, loss).
    """
    with open(graph_path, "r", encoding="utf-8") as f:
        g = json.load(f)

    nodes: Dict[str, Any] = g["node_features"]

    # Кандидаты стартовых постов из poi_type_counts.
    # Каждый block post идентифицируется своим cell_id (уникальная сущность).
    candidates_raw: List[str] = []
    for cid, nf in nodes.items():
        d = nf.get("poi_type_counts") or {}
        ok = False
        for k in d.keys():
            for pref in candidate_poi_prefixes:
                if k.startswith(pref):
                    ok = True
                    break
            if ok:
                break
        if ok:
            candidates_raw.append(cid)

    candidates = sorted(set(candidates_raw))

    if not candidates:
        raise ValueError("No candidate posts found in poi_type_counts for given prefixes.")

    # Предподсчёт секторной разметки для всех непустых подмножеств постов.
    # Для 8 patrol_house это 2^8 - 1 = 255 вариантов.
    sector_cache = precompute_sector_assignments_for_active_post_subsets(g, candidates)

    out: List[Tuple[int, Tuple[str, ...], float]] = []
    used = 0

    for K in ks:
        if K <= 0:
            continue
        if not allow_multiple_patrols_per_post and K > len(candidates):
            continue

        starts_iter = (
            itertools.combinations_with_replacement(candidates, K)
            if allow_multiple_patrols_per_post
            else itertools.combinations(candidates, K)
        )
        for starts in starts_iter:
            patrol_post_counts = dict(Counter(starts))
            active_posts_unique = sorted(patrol_post_counts.keys())
            cache_key = frozenset(active_posts_unique)
            sector_assignment, sector_travel_time_hours = sector_cache[cache_key]
            sim_res = simulate_patrol(
                g,
                list(starts),
                patrol_post_counts=patrol_post_counts,
                active_posts_unique=active_posts_unique,
                sector_assignment=sector_assignment,
                sector_travel_time_hours=sector_travel_time_hours,
                **simulate_kwargs,
            )
            loss = loss_stub(sim_res)
            out.append((K, starts, loss))

            used += 1
            if max_combos is not None and used >= max_combos:
                return out

    return out
