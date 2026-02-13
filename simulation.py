# simulation.py
from __future__ import annotations

import heapq
import json
import itertools
import math
import os
import random
from collections import Counter
from typing import Dict, Any, Iterable, List, Tuple, FrozenSet


ROAD_SPEED_KMH = 60.0
OFFROAD_SPEED_KMH = 30.0
METERS_PER_KM = 1000.0
DEFAULT_EDGE_DISTANCE_M = 1000.0
DEFAULT_PRIORITY_PATH = "solution/etosha_node_priority_compact_clamped.json"
DEFAULT_SIM_DAYS = 365
DEFAULT_PATROLS_PER_DAY = 2
DEFAULT_MAX_PATROL_HOURS = 8.0
DEFAULT_SOFTMAX_TAU = 1.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_STEPS_PER_SORTIE = 10000


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


def _build_weighted_adjacency(
    edges: Dict[str, Dict[str, Dict[str, Any]]],
    allowed_nodes: set[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    weighted: Dict[str, Dict[str, float]] = {}
    for src, neis in edges.items():
        if allowed_nodes is not None and src not in allowed_nodes:
            continue
        weighted[src] = {
            dst: edge_travel_time_hours(ef)
            for dst, ef in neis.items()
            if allowed_nodes is None or dst in allowed_nodes
        }
    return weighted


def _dijkstra_all_times(
    weighted_adj: Dict[str, Dict[str, float]],
    src_cell_id: str,
) -> Dict[str, float]:
    best: Dict[str, float] = {src_cell_id: 0.0}
    pq: List[Tuple[float, str]] = [(0.0, src_cell_id)]
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


def _load_priority_map(
    node_priority: Dict[str, Any] | None,
    priority_path: str | None,
) -> Tuple[Dict[str, float], str]:
    if node_priority is not None:
        out = {str(k): float(v) for k, v in node_priority.items()}
        return out, "inline"

    if priority_path and os.path.exists(priority_path):
        with open(priority_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            if "node_priority" in loaded and isinstance(loaded["node_priority"], dict):
                # Совместимость со старым полным форматом.
                full = loaded["node_priority"]
                out = {
                    str(k): float(v.get("priority_P_i", 0.0)) if isinstance(v, dict) else float(v)
                    for k, v in full.items()
                }
                return out, priority_path
            out = {str(k): float(v) for k, v in loaded.items()}
            return out, priority_path

    return {}, "missing"


def _pick_softmax_neighbor(
    candidates: List[str],
    priority_by_node: Dict[str, float],
    tau: float,
    rng: random.Random,
) -> str:
    if len(candidates) == 1:
        return candidates[0]

    logits = [float(priority_by_node.get(cid, 0.0)) / tau for cid in candidates]
    max_logit = max(logits)
    weights = [math.exp(lg - max_logit) for lg in logits]
    total = sum(weights)
    if total <= 0.0:
        return candidates[rng.randrange(len(candidates))]

    r = rng.random() * total
    acc = 0.0
    for cid, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return cid
    return candidates[-1]


def simulate_patrol(graph: Dict[str, Any], starts: List[str], **kwargs) -> Dict[str, Any]:
    # starts: список cell_id, откуда выходят K патрулей
    nodes: Dict[str, Any] = graph["node_features"]
    edges: Dict[str, Dict[str, Dict[str, Any]]] = graph["edge_features"]
    if not starts:
        raise ValueError("starts must contain at least one patrol base")
    unknown = [s for s in starts if s not in nodes]
    if unknown:
        raise ValueError(f"Unknown start cells: {unknown[:5]}")

    days = int(kwargs.get("days", DEFAULT_SIM_DAYS))
    patrols_per_day = int(kwargs.get("patrols_per_day", DEFAULT_PATROLS_PER_DAY))
    sorties_per_patrol = days * patrols_per_day
    max_patrol_hours = float(kwargs.get("max_patrol_hours", DEFAULT_MAX_PATROL_HOURS))
    softmax_tau = float(kwargs.get("softmax_tau", DEFAULT_SOFTMAX_TAU))
    random_seed = int(kwargs.get("random_seed", DEFAULT_RANDOM_SEED))
    max_steps_per_sortie = int(kwargs.get("max_steps_per_sortie", DEFAULT_MAX_STEPS_PER_SORTIE))
    include_paths = bool(kwargs.get("include_paths", False))
    max_saved_paths_per_patrol = int(kwargs.get("max_saved_paths_per_patrol", 5))
    if days <= 0 or patrols_per_day <= 0:
        raise ValueError("days and patrols_per_day must be > 0")
    if max_patrol_hours <= 0:
        raise ValueError("max_patrol_hours must be > 0")
    if softmax_tau <= 0:
        raise ValueError("softmax_tau must be > 0")
    if max_steps_per_sortie <= 0:
        raise ValueError("max_steps_per_sortie must be > 0")

    patrol_post_counts = kwargs.get("patrol_post_counts", dict(Counter(starts)))
    active_posts_unique = kwargs.get("active_posts_unique", sorted(set(starts)))
    sector_assignment = kwargs.get("sector_assignment", {}) or {}
    sector_travel_time_hours = kwargs.get("sector_travel_time_hours", {}) or {}

    # Если снаружи секторная разметка не передана, строим на лету.
    if not sector_assignment:
        sector_assignment, sector_travel_time_hours = precompute_sector_assignment(graph, active_posts_unique)

    priority_by_node, priority_source = _load_priority_map(
        node_priority=kwargs.get("node_priority"),
        priority_path=kwargs.get("priority_path", DEFAULT_PRIORITY_PATH),
    )

    valid_cells = {cid for cid, nf in nodes.items() if nf.get("median_elevation_m") is not None}
    if priority_by_node:
        allowed_cells = valid_cells & set(priority_by_node.keys())
    else:
        allowed_cells = set(valid_cells)

    blocked_starts = [s for s in starts if s not in allowed_cells]
    if blocked_starts:
        raise ValueError(
            f"Start cells are not allowed (outside valid+priority space): {blocked_starts[:5]}"
        )

    weighted_adj = _build_weighted_adjacency(edges, allowed_nodes=allowed_cells)
    base_to_return_times: Dict[str, Dict[str, float]] = {
        base: _dijkstra_all_times(weighted_adj, base) for base in sorted(set(starts))
    }

    all_cells = set(allowed_cells)
    sector_cells_by_base: Dict[str, set[str]] = {}
    for cid, owner_base in sector_assignment.items():
        if cid not in allowed_cells:
            continue
        sector_cells_by_base.setdefault(owner_base, set()).add(cid)
    for base in starts:
        sector_cells_by_base.setdefault(base, set()).add(base)

    rng = random.Random(random_seed)
    patrol_units = []
    global_total_steps = 0
    global_total_travel_hours = 0.0
    global_total_returns = 0
    global_total_dead_end_returns = 0

    for patrol_idx, base in enumerate(starts, start=1):
        sector_cells = sector_cells_by_base.get(base, all_cells)
        if not sector_cells:
            sector_cells = {base}
        t_to_base = base_to_return_times[base]

        visit_counter: Counter[str] = Counter()
        total_steps = 0
        total_travel_hours = 0.0
        total_returns = 0
        total_dead_end_returns = 0
        sample_paths: List[List[str]] = []
        sortie_hours_samples: List[float] = []
        sortie_steps_samples: List[int] = []

        for _ in range(sorties_per_patrol):
            current = base
            spent = 0.0
            steps = 0
            if include_paths and len(sample_paths) < max_saved_paths_per_patrol:
                path = [base]
            else:
                path = None

            while steps < max_steps_per_sortie:
                admissible: List[str] = []
                for nxt, t_xy in weighted_adj.get(current, {}).items():
                    if nxt not in sector_cells:
                        continue
                    t_y_to_base = t_to_base.get(nxt, math.inf)
                    if spent + t_xy + t_y_to_base <= max_patrol_hours:
                        admissible.append(nxt)

                if not admissible:
                    # Если дальше идти нельзя, возвращаемся на базу кратчайшим путем.
                    if current != base:
                        t_back = t_to_base.get(current, math.inf)
                        if math.isfinite(t_back):
                            spent += t_back
                            total_travel_hours += t_back
                            total_dead_end_returns += 1
                            current = base
                            if path is not None:
                                path.append(base)
                    break

                nxt = _pick_softmax_neighbor(
                    candidates=admissible,
                    priority_by_node=priority_by_node,
                    tau=softmax_tau,
                    rng=rng,
                )
                t_step = weighted_adj[current][nxt]
                spent += t_step
                total_travel_hours += t_step
                total_steps += 1
                steps += 1
                current = nxt
                visit_counter[current] += 1
                if path is not None:
                    path.append(current)

            # Страховка: если цикл остановился не на базе (например, по шаговому лимиту),
            # принудительно возвращаем патруль на базу.
            if current != base:
                t_back = t_to_base.get(current, math.inf)
                if math.isfinite(t_back):
                    spent += t_back
                    total_travel_hours += t_back
                    total_dead_end_returns += 1
                    current = base
                    if path is not None:
                        path.append(base)

            if current == base:
                total_returns += 1

            if path is not None and len(sample_paths) < max_saved_paths_per_patrol:
                sample_paths.append(path)
            sortie_hours_samples.append(spent)
            sortie_steps_samples.append(steps)

        global_total_steps += total_steps
        global_total_travel_hours += total_travel_hours
        global_total_returns += total_returns
        global_total_dead_end_returns += total_dead_end_returns

        patrol_units.append(
            {
                "patrol_id": patrol_idx,
                "base_cell_id": base,
                "sorties_count": sorties_per_patrol,
                "total_steps": total_steps,
                "total_travel_hours": total_travel_hours,
                "avg_steps_per_sortie": (total_steps / sorties_per_patrol),
                "avg_hours_per_sortie": (total_travel_hours / sorties_per_patrol),
                "returns_to_base_count": total_returns,
                "dead_end_returns_count": total_dead_end_returns,
                "unique_visited_cells_count": int(len(visit_counter)),
                "top_visited_cells": [
                    {"cell_id": cid, "visits": int(cnt)} for cid, cnt in visit_counter.most_common(20)
                ],
                "sample_sortie_paths": sample_paths,
                "sortie_hours_sample_first10": sortie_hours_samples[:10],
                "sortie_steps_sample_first10": sortie_steps_samples[:10],
            }
        )

    return {
        "starts": starts,
        "patrol_post_counts": patrol_post_counts,
        "active_posts_unique": active_posts_unique,
        "sector_assignment": sector_assignment,
        "sector_travel_time_hours": sector_travel_time_hours,
        "simulation_params": {
            "days": days,
            "patrols_per_day": patrols_per_day,
            "sorties_per_patrol": sorties_per_patrol,
            "max_patrol_hours": max_patrol_hours,
            "softmax_tau": softmax_tau,
            "random_seed": random_seed,
            "priority_source": priority_source,
            "max_steps_per_sortie": max_steps_per_sortie,
        },
        "patrol_units": patrol_units,
        "summary": {
            "patrol_units_count": len(starts),
            "total_sorties": len(starts) * sorties_per_patrol,
            "total_steps": global_total_steps,
            "total_travel_hours": global_total_travel_hours,
            "returns_to_base_count": global_total_returns,
            "dead_end_returns_count": global_total_dead_end_returns,
        },
    }


def simulate_patrol_year_softmax(
    graph: Dict[str, Any],
    starts: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """
    Явный entrypoint годовой симуляции:
      - 365 * 2 выходов на патруль (по умолчанию),
      - переходы по softmax(priority/tau),
      - допустимые соседи только из N(x) ∩ C_i и с ограничением возврата в пределах H.
    """
    return simulate_patrol(graph, starts, **kwargs)


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
    simulate_kwargs = dict(simulate_kwargs)
    if "node_priority" not in simulate_kwargs:
        priority_by_node, _ = _load_priority_map(
            node_priority=None,
            priority_path=simulate_kwargs.get("priority_path", DEFAULT_PRIORITY_PATH),
        )
        simulate_kwargs["node_priority"] = priority_by_node

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
