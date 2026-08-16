# Etosha Wildlife Protection — IM²C 2026

Research code, selected visualizations, and the final submission from our 2026 International Mathematical Modeling Challenge solution to **Protecting Wildlife at Scale**, using Etosha National Park in Namibia as the case study.

**[Read the final report](./IMMC_2026_Report.pdf)**

Our team received an **IM²C Meritorious Award** at the international level. In 2026, 68 finalist teams from 37 countries and regions reached the international round; 11 teams received Meritorious recognition. [Official results](https://immchallenge.org/2026-results/).

![Example optimized allocation](figures/best_plan.png)

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
| `simulation.py` | Fine-grid patrol-sector and softmax route simulation |
| `single_patrol_demo.py` | Single-patrol routing demonstration |

## Environment

The geospatial preprocessing and visualization code uses Python with the dependencies in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository includes the generated block graph, travel matrices, priorities, patrol-house mapping, and original result artifacts used by the retained optimizer. Rebuilding those artifacts from scratch still requires the source geospatial layers and the fine-grid graph described in [`data/README.md`](./data/README.md).

## Verified runs

Run the automated runtime suite:

```bash
python -m unittest discover -s tests -v
```

Run the patrol allocator against the restored Etosha artifacts:

```bash
python math_solution/patrol_alloc_greedy_unique.py \
  --Kmin 2 --Kmax 2 --topL 20 --Tlim 12 \
  --out /tmp/patrol-smoke.json
```

Run the top-level annual-risk model on the small, self-contained fixture:

```bash
python math_solution/math_model.py \
  --total-budget 0 \
  --dist examples/synthetic/dist.json \
  --sound-graph examples/synthetic/small_graph.json \
  --sound-priority examples/synthetic/priority.json \
  --out /tmp/math-model-smoke.json \
  --report-out /tmp/math-model-smoke.md \
  --skip-plot --no-progress
```

The fixture exercises the orchestration and full 365-day risk-simulation path. The first command above exercises patrol allocation on the restored competition-scale block data.

## Selected outputs

### Spatial priority

![Priority blocks](figures/priority_blocks.png)

### Patrol routing

![Representative patrol plan for three teams](figures/patrol_routes.png)

## Reproducibility and scope

This release preserves the original modeling, optimization, simulation, preprocessing, and visualization code together with the generated block-level planning artifacts, representative outputs, and final paper. Raw GIS inputs and the generated 33,264-node fine-grid graph are not included because they were never tracked in the repository.

The simulation parameters in `math_solution/risk_year_simulation.py` are **competition-model assumptions**, not field-calibrated conservation forecasts. The repository should therefore be read as a mathematical modeling and optimization project rather than an operational wildlife-management system. The final report documents the assumptions, limitations, and decision framework used in the submission.

GitHub Actions performs a repository-wide syntax check plus unit and integration tests. The suite covers fine-grid patrol simulation, border monitoring selection, a full synthetic annual-risk run, and an optimizer run against the restored Etosha block-level data. Rebuilding the geospatial inputs themselves remains outside the repository because the raw layers are not included.

## My contribution

I co-developed the mathematical model, developed the optimization/search approach used to compare resource allocations, and implemented the computational pipeline and simulations used to evaluate candidate plans.
