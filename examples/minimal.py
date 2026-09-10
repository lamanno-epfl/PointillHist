"""Minimal PointillHist run: graphs -> network -> train -> predict -> save.

Edit the paths below (relative to the working directory) and run it with a Python that has
the packages of docs/environment.md.
`python examples/minimal.py --demo` runs the same steps on the synthetic dataset
of tests/_dataset.py instead (about a minute on CPU).
"""
import os
import sys

import anndata as ad
import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import pointillhist as ph

# ---- inputs ---------------------------------------------------------------
SPATIAL_PATHS = ["data/sec_E9_1.h5ad", "data/sec_E9_2.h5ad", "data/sec_E10_1.h5ad"]
REFERENCE = "data/reference.csv"                 # cell types x genes
TIMEPOINTS = ["E9", "E9", "E10"]                 # one label per section, or None
REGION_KEY = "region"                            # obs column with the region, or None
TYPE_PRIORS = "data/proportions_by_timepoint.csv"  # cell types x timepoint labels, or "uniform"
TYPE_REGIONS = "data/type_regions.csv"           # cell types x regions (0/1), or None
OUT_DIR = "results"
GRAPH_KWARGS = {}   # tile_side, grid_spacing, cell_cell_maxdist, fraction_overlap default to "auto"
NET_KWARGS = dict(hidden_size=128, n_heads=4)
TRAIN_KWARGS = dict(cell_loss_type="zip", num_epochs=100)

if "--demo" in sys.argv:
    import tempfile
    sys.path.insert(0, os.path.join(HERE, "..", "tests"))
    import _dataset

    OUT_DIR = tempfile.mkdtemp(prefix="pointillhist_demo_")
    REFERENCE, SPATIAL_PATHS = _dataset.build(OUT_DIR)
    TIMEPOINTS, REGION_KEY = list(_dataset.TIMEPOINTS), "region"
    reference = pd.read_csv(REFERENCE, index_col=0)
    # true composition of each section as a cell types x timepoint-labels count table
    TYPE_PRIORS = pd.DataFrame({tp: ad.read_h5ad(p).obs["true_label"].value_counts()
                                for p, tp in zip(SPATIAL_PATHS, TIMEPOINTS)}).fillna(0)
    TYPE_PRIORS.index = [f"type_{k}" for k in TYPE_PRIORS.index]
    TYPE_REGIONS = pd.DataFrame(1.0, index=reference.index, columns=list(_dataset.REGIONS))
    GRAPH_KWARGS = dict(tile_side=200, min_cells=20, min_dots=10, grid_spacing=40,
                        cell_cell_k_neighbors=8, cell_cell_maxdist=120, fraction_overlap=0.0,
                        counts_min_cell=-1, coarse_grid_side=4)
    NET_KWARGS = dict(hidden_size=32)
    TRAIN_KWARGS = dict(cell_loss_type="zip", num_epochs=3)

# ---- the four calls -------------------------------------------------------
graphs = ph.pp.generate_graphs(SPATIAL_PATHS, REFERENCE, timepoints=TIMEPOINTS,
                               region_key=REGION_KEY, **GRAPH_KWARGS)
net = ph.tr.networks(graphs, **NET_KWARGS)
net, history = ph.tr.train(net, graphs, REFERENCE, type_priors=TYPE_PRIORS,
                           type_regions=TYPE_REGIONS, **TRAIN_KWARGS)
result = ph.eval.predict(net, graphs)

# ---- outputs --------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
predictions = pd.DataFrame({
    "unique_cell_id": result["unique_cell_ids"],
    "x": result["all_positions"][:, 0],
    "y": result["all_positions"][:, 1],
    "section": result["all_sections"],
    "timepoint": result["all_timepoints"],
    "predicted_cell_type": result["all_cell_types"],
    # confidence = the largest softmax probability; all_probs holds only the renormalised top 5 (see predict)
    "max_prob": torch.softmax(torch.as_tensor(result["all_logits"]) / float(net.temperature), dim=1).max(dim=1).values.numpy(),
})
try:
    predictions.to_parquet(os.path.join(OUT_DIR, "predictions.parquet"), index=False)
except ImportError:
    predictions.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
ph.pl.history(history)
plt.savefig(os.path.join(OUT_DIR, "history.png"), dpi=150, bbox_inches="tight")
print(f"{len(predictions)} cells predicted; outputs in {OUT_DIR}")
