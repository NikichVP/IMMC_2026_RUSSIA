# Data dependencies

The original working repository generated several JSON artifacts from geospatial inputs. The block-level artifacts needed by the retained patrol optimizer have been restored to this repository. The raw source layers and generated fine-grid graph were never tracked and are still required to rebuild the spatial pipeline from scratch.

## Expected generated products

The optimization and simulation scripts expect the following local files when reproducing the full pipeline:

- `solution/etosha_grid_graph_with_big_squares.json` — fine-grid graph annotated with block membership (**not tracked; must be regenerated or supplied locally**)
- `solution/etosha_big_square_graph_14x14.json` — aggregated block graph (**included**)
- `solution/etosha_node_priority_compact_clamped.json` — compact priority values (**included**)
- `solution/big_dist_with_portals_time_priority.json` — precomputed inter-block route and travel information (**included**)
- `solution/patrol_house_to_big_cell.json` — mapping from patrol houses to aggregated cells (**included**)

The raw graph builder (`build_graph.py`) also expects the geospatial source layers used during the competition analysis, including elevation, road, point-of-interest, fire, vegetation, wildlife-destination, patrol-house, and photo-trap layers.

## Pipeline relationship

At a high level, the retained code follows this dependency chain:

1. `build_graph.py` constructs the fine spatial graph from source GIS layers.
2. `solution/build_big_square_graphs.py` aggregates the fine graph into planning blocks.
3. `solution/compute_node_priority.py` computes compact spatial priority scores.
4. `solution/build_big_dist_with_portals.py` precomputes travel relationships between aggregated blocks.
5. The scripts in `math_solution/` consume those derived products for patrol allocation, monitoring placement, simulation, and budget search.

Some local mapping artifacts used by the competition workflow, such as the patrol-house-to-block mapping, are data dependencies rather than source code and must also be present for a full rerun.

## Public-release scope

The repository preserves the modeling and optimization implementation, the final report, aggregate visual outputs, and the tracked competition-era derived artifacts. It does **not** claim to redistribute the original GIS dataset. A self-contained synthetic fixture under `examples/synthetic/` supports runtime verification without pretending to reproduce the Etosha geospatial preprocessing.
