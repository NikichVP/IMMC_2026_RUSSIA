# Etosha Wildlife Protection — IM²C 2026

Code, selected visualizations, and the final report from our 2026 International Mathematical Modeling Challenge solution to **Protecting Wildlife at Scale**, using Etosha National Park in Namibia as the case study.

**[Read the final report](./IMMC_2026_Report.pdf)**

Our team received an **IM²C Meritorious Award** at the international level. In 2026, 68 finalist teams from 37 countries and regions reached the international round; 11 teams received Meritorious recognition. [Official results](https://immchallenge.org/2026-results/).

![Aggregated priority map](figures/priority_blocks.png)

## Model

We treated wildlife protection as a constrained resource-allocation problem. The computational pipeline combines a spatial graph of the park with threat priorities, patrol routing, monitoring coverage, seasonal risk simulation, and budget search.

1. **Spatial representation** — convert geospatial layers into a navigable fine-grid graph, then aggregate it into larger blocks for tractable optimization.
2. **Priority scoring** — assign local and neighborhood-aware priority values from wildlife, water, vegetation, infrastructure, and existing coverage features.
3. **Patrol allocation** — distribute ranger groups across patrol houses and candidate areas under travel-time and overlap constraints.
4. **Monitoring allocation** — place GPS and acoustic/sound-tracker coverage within a fixed budget.
5. **Risk simulation** — simulate seasonal threat events and interception under the assumptions defined in the model.
6. **Plan search** — screen feasible allocations on a shorter horizon, then evaluate the strongest candidates over a full simulated year.

## Code map

| File | Role |
|---|---|
| `build_graph.py` | Geospatial ingestion and fine-grid construction |
| `solution/compute_node_priority.py` | Spatial priority scoring |
| `solution/build_big_square_graphs.py` | Block-level graph construction |
| `solution/build_big_dist_with_portals.py` | Inter-block route and travel-cost preprocessing |
| `math_solution/patrol_alloc_greedy_unique.py` | Patrol allocation search |
| `math_solution/select_sound_border_cells.py` | Border monitoring placement |
| `math_solution/risk_year_simulation.py` | Seasonal annual-risk simulation |
| `math_solution/math_model.py` | Budget logic and top-level plan search |
| `math_solution/viz_patrol_alloc_k2.py` | Route/allocation visualization |
| `single_patrol_demo.py` | Single-patrol routing demonstration |

## Selected outputs

### Spatial priority

![Priority blocks](figures/priority_blocks.png)

### Patrol routing

![Patrol routes](figures/patrol_routes.png)

### Example optimized allocation

![Best-plan visualization](figures/best_plan.png)

## Reproduction note

This repository is a curated version of the original working code. Large generated graph matrices, raw geospatial source files, and intermediate artifacts are intentionally excluded from the current tree. The expected data products and filenames are documented in [`data/README.md`](./data/README.md).

The simulation parameters in `math_solution/risk_year_simulation.py` are competition-model assumptions, not field-calibrated conservation forecasts. The final report documents the model, assumptions, limitations, and decision framework used in the submission.

## My contribution

I co-developed the mathematical model, developed the optimization/search approach used to compare resource allocations, and implemented the computational pipeline and simulations used to evaluate candidate plans.
