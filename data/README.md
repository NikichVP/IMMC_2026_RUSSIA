# Data dependencies

The original working repository generated several large JSON artifacts from geospatial inputs. They are intentionally excluded from the curated public tree to keep the repository readable and to avoid republishing raw or intermediate geospatial data.

## Expected generated products

The optimization and simulation scripts expect the following local files when reproducing the full pipeline:

- `solution/etosha_grid_graph_with_big_squares.json` — fine-grid graph annotated with block membership
- `solution/etosha_big_square_graph_14x14.json` — aggregated block graph
- `solution/etosha_node_priority_compact_clamped.json` — compact priority values
- `solution/big_dist_with_portals_time_priority.json` — precomputed inter-block route and travel information
- `solution/patrol_house_to_big_cell.json` — mapping from patrol houses to aggregated cells

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

The repository preserves the modeling and optimization implementation, the final report, and selected aggregate visual outputs. It does **not** claim to be a self-contained geospatial dataset or a one-command reproduction package. This distinction is intentional: the excluded files are large derived artifacts or source data that are not needed to inspect the mathematical and computational approach.
