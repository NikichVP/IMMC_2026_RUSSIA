# Data products

The original working repository generated several large JSON artifacts from geospatial inputs. They are intentionally excluded from the curated tree to keep the repository readable and to avoid republishing raw/intermediate geospatial data.

The core scripts expect the following generated files when reproducing the full pipeline locally:

- `solution/etosha_grid_graph_with_big_squares.json` — fine-grid graph annotated with block membership
- `solution/etosha_big_square_graph_14x14.json` — aggregated block graph
- `solution/etosha_node_priority_compact_clamped.json` — compact priority values
- `solution/big_dist_with_portals_time_priority.json` — precomputed inter-block route/travel information
- `solution/patrol_house_to_big_cell.json` — mapping from patrol houses to aggregated cells

The raw graph builder (`build_graph.py`) also expects the geospatial source layers used during the competition analysis, including elevation, road, point-of-interest, fire, vegetation, wildlife-destination, patrol-house, and photo-trap layers.

These files are data dependencies rather than source code. The public-facing repository preserves the modeling and optimization implementation plus selected aggregate visual outputs.
