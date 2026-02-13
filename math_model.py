from __future__ import annotations

import argparse
from dataclasses import dataclass

import simulations2


# =========================
# Fill Constants Here
# =========================
MIN_RANGER_GROUPS = 2
RANGER_GROUP_COST = 30000.0
GPS_FULL_PACKAGE_COST = 60000.0
TOTAL_BORDER_KM = 820.0
SOUND_TRACKER_COST_PER_KM = 750.0
DEFAULT_BASE_CANDIDATES = (
    "r20_c168",
    "r21_c15",
    "r27_c16",
    "r36_c165",
    "r51_c224",
    "r76_c273",
    "r76_c283",
    "r79_c135",
)


@dataclass
class BudgetPlan:
    total_budget: float
    spent_on_mandatory_rangers: float
    mandatory_ranger_groups: int
    gps_full_bought: bool
    spent_on_gps: float
    remaining_for_optimization: float
    minimum_ranger_target_met: bool


def allocate_pre_optimization_budget(total_budget: float) -> BudgetPlan:
    if total_budget < 0:
        raise ValueError("Budget must be non-negative")
    if RANGER_GROUP_COST <= 0:
        raise ValueError("RANGER_GROUP_COST must be > 0")
    if GPS_FULL_PACKAGE_COST < 0:
        raise ValueError("GPS_FULL_PACKAGE_COST must be >= 0")

    remaining = total_budget
    mandatory_ranger_groups = int(remaining // RANGER_GROUP_COST)
    mandatory_ranger_groups = min(mandatory_ranger_groups, MIN_RANGER_GROUPS)
    spent_on_mandatory_rangers = mandatory_ranger_groups * RANGER_GROUP_COST
    remaining -= spent_on_mandatory_rangers

    minimum_ranger_target_met = mandatory_ranger_groups >= MIN_RANGER_GROUPS

    gps_full_bought = False
    spent_on_gps = 0.0
    if minimum_ranger_target_met and remaining >= GPS_FULL_PACKAGE_COST:
        gps_full_bought = True
        spent_on_gps = GPS_FULL_PACKAGE_COST
        remaining -= spent_on_gps

    return BudgetPlan(
        total_budget=total_budget,
        spent_on_mandatory_rangers=spent_on_mandatory_rangers,
        mandatory_ranger_groups=mandatory_ranger_groups,
        gps_full_bought=gps_full_bought,
        spent_on_gps=spent_on_gps,
        remaining_for_optimization=remaining,
        minimum_ranger_target_met=minimum_ranger_target_met,
    )


def print_pre_optimization_report(plan: BudgetPlan) -> None:
    print("=== PRE-OPTIMIZATION BUDGET DISTRIBUTION ===")
    print(f"Input budget: {plan.total_budget:,.2f}")
    print(
        f"Step 1: Rangers first -> {plan.mandatory_ranger_groups} groups "
        f"for {plan.spent_on_mandatory_rangers:,.2f}"
    )
    if not plan.minimum_ranger_target_met:
        missing = MIN_RANGER_GROUPS - plan.mandatory_ranger_groups
        print(
            f"WARNING: Minimum target {MIN_RANGER_GROUPS} groups not reached. "
            f"Missing groups: {missing}"
        )

    if plan.gps_full_bought:
        print(f"Step 2: Full GPS package purchased for {plan.spent_on_gps:,.2f}")
    else:
        print("Step 2: Full GPS package NOT purchased (insufficient budget after Step 1)")

    print(f"Step 3: Remaining budget sent to optimization: {plan.remaining_for_optimization:,.2f}")
    print(
        f"Sound tracker economics: {SOUND_TRACKER_COST_PER_KM:,.2f} per km, "
        f"total border = {TOTAL_BORDER_KM:.1f} km"
    )
    print()


def print_final_report(plan: BudgetPlan, recommendation: dict) -> None:
    print("=== FINAL VERDICT (STUB via simulations2.py) ===")
    print(f"Total input budget: {plan.total_budget:,.2f}")
    print(
        f"Mandatory rangers: {plan.mandatory_ranger_groups} groups "
        f"(spent {plan.spent_on_mandatory_rangers:,.2f})"
    )
    print(f"GPS full package: {'YES' if plan.gps_full_bought else 'NO'}")
    print(f"Extra ranger groups (optimization): {recommendation['extra_ranger_groups']}")
    print(f"Sound tracker budget (optimization): {recommendation['sound_tracker_budget']:,.2f}")
    print(f"Sound-covered border (optimization): {recommendation['sound_covered_km']:.2f} km")
    print(f"Total ranger groups: {recommendation['total_ranger_groups']}")
    print(f"Unallocated remainder: {recommendation['unallocated_budget']:,.2f}")
    print("Recommended ranger bases:")
    for base in recommendation["recommended_bases"]:
        print(f" - group #{base['group_id']}: {base['base_cell_id']}")
    print()
    print(f"Stub policy note: {recommendation['policy_note']}")


def run_math_model(total_budget: float) -> dict:
    plan = allocate_pre_optimization_budget(total_budget)
    print_pre_optimization_report(plan)

    recommendation = simulations2.recommend_allocation_stub(
        remaining_budget=plan.remaining_for_optimization,
        current_ranger_groups=plan.mandatory_ranger_groups,
        ranger_group_cost=RANGER_GROUP_COST,
        sound_tracker_cost_per_km=SOUND_TRACKER_COST_PER_KM,
        total_border_km=TOTAL_BORDER_KM,
        base_candidates=list(DEFAULT_BASE_CANDIDATES),
    )

    print_final_report(plan, recommendation)
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-to-protection math model runner")
    parser.add_argument("budget", type=float, help="Total budget (single number)")
    args = parser.parse_args()
    run_math_model(args.budget)


if __name__ == "__main__":
    main()
