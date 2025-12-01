# Persistence Forests and Generalized Landscapes

Reference implementation accompanying our submission on persistence forests and generalized persistence landscapes. The code is meant to be usable by anyone interested in experimenting with the algorithm and reproducing figures/benchmarks from the manuscript (still under review).

## What this repo provides
- `PersistenceForest` (primary entry point) builds the forest of optimal cycles for an alpha complex, together with barcodes and cycle representatives over the filtration.
- Generalized persistence landscapes via `forest_landscapes.py` and `cycle_rep_vectorisations.py`, including utilities to compare functionals on cycles.
- Plotting helpers (`forest_plotting.py`) for barcodes, dendrogram-like views, snapshots along the filtration, and optional animations; `color_scheme.py` keeps plots consistent.
- End-to-end example in `pers_forest_example.py` showing forest construction, plotting, landscape computation, and simple ML vectorisation.
- Benchmark tooling in `benchmark.py` to reproduce runtime plots reported in the paper.
- Paper figure notebook `paper-examples.ipy` was used to generate submission graphics.

Legacy note: `LoopForest.py` is an older version kept for reference and is no longer part of the workflow.

## Installation
Tested with Python 3.13.3. Install dependencies with pip:
```bash
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install numpy matplotlib seaborn pandas scikit-learn gudhi
# optional: pillow/ffmpeg for animations, jupyter for notebooks
```

## Quickstart
```python
import numpy as np
from PersistenceForest import PersistenceForest
from cycle_rep_vectorisations import signed_chain_edge_length

# 1) Create a point cloud
rng = np.random.default_rng(0)
pts = rng.standard_normal((300, 2))

# 2) Build the persistence forest (alpha complex)
forest = PersistenceForest(pts, print_info=True)

# 3) Visualize
forest.plot_barcode(min_bar_length=0.01, coloring="forest")
forest.plot_at_filtration(0.5)

# 4) Generalized landscapes
forest.compute_generalized_landscape_family(
    cycle_func=signed_chain_edge_length,
    max_k=5,
    num_grid_points=512,
    label="edge-length",
)
forest.plot_landscape_family(label="edge-length")
```
Run the richer demo (plots + simple ML vectorisation) with:
```bash
python pers_forest_example.py
```

## Generalized landscapes and vectorisation
- Define cycle functionals in `cycle_rep_vectorisations.py` (examples: edge length, area, connected components, signed/unsigned variants).
- `forest.compute_generalized_landscape_family(...)` builds families for one functional; `plot_landscape_comparison_between_functionals` contrasts multiple labels.
- `MultiLandscapeVectorizer` turns multiple forests into fixed-length feature vectors (optionally with basic stats) for downstream ML models.

## Benchmarks
`benchmark.py` measures construction time across sampling schemes (`point_cloud_sampling.py`). Toggle the `if False/True` blocks in `__main__` to:
- generate CSVs into `benchmarks/` for chosen samplers/size schedules, or
- render the runtime plots from existing CSVs.
Artifacts used in the manuscript are in `benchmarks/` (e.g., `persistence_forest_benchmark2.csv`).

## Repository guide
- `PersistenceForest.py` – forest construction, barcodes, plotting wrappers, generalized landscapes.
- `forest_plotting.py` – shared plotting/animation utilities.
- `forest_landscapes.py` – landscape computation and visualisation.
- `cycle_rep_vectorisations.py` – cycle functionals and vectoriser.
- `color_scheme.py` – consistent color palettes across plots.
- `pers_forest_example.py` – main usage example.
- `benchmark.py` – runtime benchmarks.
- `paper-examples.ipy`, `generalized_landscape_plots/`, `paper_figures/` – scripts/notebooks for paper figures.
- `point_cloud_sampling.py`, `point_cloud_generator.py` – synthetic data utilities.
- `LoopForest.py` – deprecated predecessor, kept only for historical reference.

## Notes
- The repository accompanies a submission that has not yet been accepted; interfaces may evolve slightly.
- Animations require a working Matplotlib animation backend (Pillow or ffmpeg).
