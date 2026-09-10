"""A demonstration of PointillHist's optional inputs in one run.

Start with ``examples/minimal.ipynb`` for a realistic mapping. This script shows how the
optional features fit together: several sections trained jointly with a timepoint label each,
a region label per cell with a cell types x regions table for the anatomical prior, and a
cell types x timepoints table of expected proportions as the type prior. It is a template, not
a benchmark: edit the paths below (relative to the working directory) and run it in an
environment where PointillHist is installed (see the README).

``python examples/minimal.py --demo`` runs the same steps on a small synthetic dataset that the
script generates itself (two sections of 320 cells x 40 genes, six types, three regions), in
well under a minute on the CPU.
"""
import os
import sys

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


# ---- synthetic dataset for --demo -----------------------------------------
DEMO_TIMEPOINTS = ("E10", "E9")                  # one label per synthetic section
DEMO_REGIONS = ("region_a", "region_b", "region_c")


def synthetic_dataset(out_dir, n_genes=40, n_types=6, n_cells=320):
    """Write two synthetic sections and their reference to ``out_dir``.

    Each section holds ``n_cells`` cells of ``n_types`` types on a jittered lattice, with Poisson
    counts over ``n_genes`` genes drawn from the reference profiles, a region label in three
    horizontal bands (about 10 % of the cells unannotated) and a timepoint label; the true type
    is kept in ``obs["true_label"]``. Fixed seeds make the files identical on every machine.
    Returns ``(reference_csv, section_paths)``.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(rng.gamma(shape=1.5, scale=4.0, size=(n_types, n_genes)),
                             index=[f"type_{i}" for i in range(n_types)],
                             columns=[f"gene_{i:03d}" for i in range(n_genes)])
    reference_csv = os.path.join(out_dir, "reference.csv")
    reference.to_csv(reference_csv)

    section_paths = []
    for seed, timepoint in enumerate(DEMO_TIMEPOINTS):
        local = np.random.default_rng(seed)
        labels = local.integers(0, n_types, size=n_cells)
        counts = local.poisson(reference.values[labels] * 0.6).astype(np.float32)
        side = int(np.ceil(np.sqrt(n_cells)))
        gx, gy = np.meshgrid(np.arange(side), np.arange(side))
        coords = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells].astype(np.float64)
        coords = coords * 30.0 + local.uniform(-6.0, 6.0, size=coords.shape)
        band = np.clip(coords[:, 1] // (side * 10.0), 0, len(DEMO_REGIONS) - 1).astype(int)
        region = np.array(DEMO_REGIONS, dtype=object)[band]
        region[local.random(n_cells) < 0.1] = None

        adata = ad.AnnData(X=counts)
        adata.var_names = list(reference.columns)
        adata.obs_names = [f"s{seed}_cell_{i:04d}" for i in range(n_cells)]
        adata.obs["true_label"] = labels
        adata.obs["region"] = pd.Categorical(region)
        adata.obsm["spatial"] = coords
        adata.uns["timepoint"] = timepoint
        path = os.path.join(out_dir, f"section_{'AB'[seed]}.h5ad")
        adata.write_h5ad(path)
        section_paths.append(path)
    return reference_csv, section_paths


if "--demo" in sys.argv:
    import tempfile

    OUT_DIR = tempfile.mkdtemp(prefix="pointillhist_demo_")
    REFERENCE, SPATIAL_PATHS = synthetic_dataset(OUT_DIR)
    TIMEPOINTS, REGION_KEY = list(DEMO_TIMEPOINTS), "region"
    reference = pd.read_csv(REFERENCE, index_col=0)
    # true composition of each section as a cell types x timepoint-labels count table
    TYPE_PRIORS = pd.DataFrame({tp: ad.read_h5ad(p).obs["true_label"].value_counts()
                                for p, tp in zip(SPATIAL_PATHS, TIMEPOINTS)}).fillna(0)
    TYPE_PRIORS.index = [f"type_{k}" for k in TYPE_PRIORS.index]
    TYPE_REGIONS = pd.DataFrame(1.0, index=reference.index, columns=list(DEMO_REGIONS))
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
