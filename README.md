# Etosha Wildlife Protection — IM²C 2026

Code and simulation artifacts for our **International Mathematical Modeling Challenge 2026** solution on allocating limited conservation resources in **Etosha National Park, Namibia**.

Our team received an **IM²C Meritorious Award** at the international level.

## Problem

The model treats wildlife protection as a constrained resource-allocation problem: limited ranger teams, tracking infrastructure, detection coverage, travel time, and multiple threat classes must be balanced across a large park.

The implementation combines a spatial graph representation of Etosha with threat priorities, patrol allocation, annual-risk simulation, and budget search.

## Technical pipeline

1. **Spatial graph construction** — convert park infrastructure and relevant locations into a graph/grid representation.
2. **Threat prioritization** — assign spatial priority values used by allocation and simulation components.
3. **Patrol allocation** — choose patrol-house assignments and candidate coverage areas under travel-time and resource constraints.
4. **Infrastructure allocation** — model GPS and acoustic/sound-tracker coverage under a fixed budget.
5. **Two-stage plan search** — screen candidate allocations on a shorter simulation horizon, then evaluate the strongest plans over a full simulated year.
6. **Loss minimization** — compare feasible plans by simulated annual loss, with coverage and travel time used as tie-breakers.

## Key code

- `build_graph.py` — graph construction
- `math_solution/math_model.py` — budget allocation and top-level plan search
- `math_solution/patrol_alloc_greedy_unique.py` — patrol allocation
- `math_solution/risk_year_simulation.py` — annual threat/risk simulation
- `math_solution/select_sound_border_cells.py` — sound-tracker border coverage
- `math_solution/viz_patrol_alloc_k2.py` — allocation visualization
- `single_patrol_demo.py` — single-patrol route demonstration

## Example model configuration

The working model includes configurable values for patrol cost, GPS deployment, sound-tracker cost per kilometer, minimum ranger groups, total border length, simulation horizon, and candidate-screening depth. Different budget scenarios can be evaluated without changing the overall pipeline.

## Outputs

The repository contains generated maps and reports for multiple budget scenarios, including patrol allocations, threat overlays, and best-plan visualizations.

## My contribution

I worked on the mathematical model with a teammate, developed the optimization/search approach used to choose resource allocations, and implemented the computational pipeline and simulations used to evaluate candidate plans.

> This is the original working repository. Before any public release, large generated files, intermediate artifacts, and any third-party data with redistribution restrictions should be separated from the curated code release.
