from __future__ import annotations

from typing import List, Dict, Any


def select_optimal_sound_tracker_coverage(
    *,
    allocated_budget: float,
    sound_tracker_cost_per_km: float,
    total_border_km: float,
    border_segments: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Выбирает оптимальные километры границы для закрытия sound trackers.
    Модель: одинаковая стоимость на 1 км, поэтому закрываем сначала сегменты
    с максимальным приоритетом (priority_per_km), затем при необходимости частично.
    """
    if allocated_budget < 0:
        raise ValueError("allocated_budget must be non-negative")
    if sound_tracker_cost_per_km <= 0:
        raise ValueError("sound_tracker_cost_per_km must be > 0")
    if total_border_km < 0:
        raise ValueError("total_border_km must be non-negative")

    normalized_segments: List[Dict[str, Any]] = []
    for idx, seg in enumerate(border_segments or []):
        seg_id = str(seg.get("id", f"segment_{idx + 1}"))
        length_km = float(seg.get("length_km", 0.0))
        priority_per_km = float(seg.get("priority_per_km", 0.0))
        if length_km <= 0:
            continue
        normalized_segments.append(
            {
                "id": seg_id,
                "length_km": length_km,
                "priority_per_km": priority_per_km,
            }
        )

    coverage_budget_km = min(allocated_budget / sound_tracker_cost_per_km, total_border_km)
    selected_segments: List[Dict[str, Any]] = []

    covered_km = 0.0
    priority_score = 0.0
    remaining_km = coverage_budget_km

    if normalized_segments:
        for seg in sorted(normalized_segments, key=lambda x: x["priority_per_km"], reverse=True):
            if remaining_km <= 0:
                break
            cover_here_km = min(seg["length_km"], remaining_km)
            if cover_here_km <= 0:
                continue
            selected_segments.append(
                {
                    "id": seg["id"],
                    "covered_km": cover_here_km,
                    "segment_length_km": seg["length_km"],
                    "priority_per_km": seg["priority_per_km"],
                    "is_full_segment": abs(cover_here_km - seg["length_km"]) <= 1e-9,
                }
            )
            covered_km += cover_here_km
            priority_score += cover_here_km * seg["priority_per_km"]
            remaining_km -= cover_here_km
    else:
        covered_km = coverage_budget_km

    spent_budget = covered_km * sound_tracker_cost_per_km
    unspent_budget = allocated_budget - spent_budget

    return {
        "allocated_budget": allocated_budget,
        "spent_budget": spent_budget,
        "unspent_budget": unspent_budget,
        "covered_km": covered_km,
        "selected_segments": selected_segments,
        "priority_score": priority_score,
    }


def _build_base_assignment(total_ranger_groups: int, base_candidates: List[str]) -> List[Dict[str, Any]]:
    if total_ranger_groups <= 0:
        return []
    if not base_candidates:
        base_candidates = [f"TBD_BASE_{i + 1}" for i in range(total_ranger_groups)]

    assignment: List[Dict[str, Any]] = []
    for i in range(total_ranger_groups):
        base_cell_id = base_candidates[i % len(base_candidates)]
        assignment.append(
            {
                "group_id": i + 1,
                "base_cell_id": base_cell_id,
            }
        )
    return assignment


def recommend_allocation_stub(
    *,
    remaining_budget: float,
    current_ranger_groups: int,
    ranger_group_cost: float,
    sound_tracker_cost_per_km: float,
    total_border_km: float,
    base_candidates: List[str],
    border_segments: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be non-negative")
    if current_ranger_groups < 0:
        raise ValueError("current_ranger_groups must be non-negative")
    if ranger_group_cost <= 0:
        raise ValueError("ranger_group_cost must be > 0")
    if sound_tracker_cost_per_km <= 0:
        raise ValueError("sound_tracker_cost_per_km must be > 0")
    if total_border_km < 0:
        raise ValueError("total_border_km must be non-negative")

    # Stub policy:
    # 1) all remaining money into additional ranger groups
    # 2) all leftover goes to sound trackers until full border is covered
    extra_ranger_groups = int(remaining_budget // ranger_group_cost)
    budget_after_rangers = remaining_budget - extra_ranger_groups * ranger_group_cost

    max_sound_budget = total_border_km * sound_tracker_cost_per_km
    sound_tracker_budget = min(budget_after_rangers, max_sound_budget)
    sound_coverage_plan = select_optimal_sound_tracker_coverage(
        allocated_budget=sound_tracker_budget,
        sound_tracker_cost_per_km=sound_tracker_cost_per_km,
        total_border_km=total_border_km,
        border_segments=border_segments,
    )
    unallocated_budget = budget_after_rangers - sound_coverage_plan["spent_budget"]

    total_ranger_groups = current_ranger_groups + extra_ranger_groups
    recommended_bases = _build_base_assignment(total_ranger_groups, base_candidates)

    return {
        "extra_ranger_groups": extra_ranger_groups,
        "sound_tracker_budget": sound_tracker_budget,
        "sound_covered_km": sound_coverage_plan["covered_km"],
        "sound_coverage_plan": sound_coverage_plan,
        "total_ranger_groups": total_ranger_groups,
        "recommended_bases": recommended_bases,
        "unallocated_budget": unallocated_budget,
        "policy_note": "STUB: all remaining budget -> rangers, leftover -> sound trackers (up to full border)",
    }
