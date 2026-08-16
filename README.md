# Etosha Wildlife Protection — IM²C 2026

Research code, selected visualizations, and the final submission from our 2026 International Mathematical Modeling Challenge solution to **Protecting Wildlife at Scale**, using Etosha National Park in Namibia as the case study.

**[Read the final report](./IMMC_2026_Report.pdf)**

Our team received an **IM²C Meritorious Award** at the international level. In 2026, 68 finalist teams from 37 countries and regions reached the international round; 11 teams received Meritorious recognition. [Official results](https://immchallenge.org/2026-results/).

## Model

We treated wildlife protection as a constrained resource-allocation problem. The computational pipeline combines a spatial graph of the park with threat priorities, patrol routing, monitoring coverage, seasonal risk simulation, and budget search.

1. **Spatial representation** — convert geospatial layers into a navigable fine-grid graph, then aggregate it into larger blocks for tractable optimization.
2. **Priority scoring** — assign local and neighborhood-aware priority values from wildlife, water, vegetation, infrastructure, and existing coverage features.
3. **Patrol allocation** — distribute ranger groups across patrol houses and candidate areas under travel-time and overlap constraints.
4. **Monitoring allocation** — place GPS and acoustic/sound-tracker coverage within a fixed budget.
5. **Risk simulation** — simulate seasonal threat events and interception under the assumptions defined in the model.
6. **Plan search** — screen feasible allocations on a shorter horizon, then evaluate the strongest candidates over a full simulated year.

## Repository map

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

## Environment

The geospatial preprocessing and visualization code uses Python with the dependencies in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A complete end-to-end rerun also requires the source geospatial layers and large derived graph/matrix files described in [`data/README.md`](./data/README.md). Those data products are intentionally not redistributed in the current tree.

## Selected outputs

### Spatial priority

![Priority blocks](figures/priority_blocks.png)

### Patrol routing

![Representative patrol plan for three teams](figures/best_plan.png)

## Reproducibility and scope

This is a curated release of the original competition repository. It preserves the modeling, optimization, simulation, preprocessing, and visualization code together with representative outputs and the final paper, while excluding large generated data products and raw geospatial inputs.

The simulation parameters in `math_solution/risk_year_simulation.py` are **competition-model assumptions**, not field-calibrated conservation forecasts. The repository should therefore be read as a mathematical modeling and optimization project rather than an operational wildlife-management system. The final report documents the assumptions, limitations, and decision framework used in the submission.

GitHub Actions performs a repository-wide Python syntax check on pushes and pull requests. This verifies source-code integrity, not end-to-end numerical reproducibility without the excluded data dependencies.

## My contribution

I co-developed the mathematical model, developed the optimization/search approach used to compare resource allocations, and implemented the computational pipeline and simulations used to evaluate candidate plans.
