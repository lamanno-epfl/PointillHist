"""A template for datasets larger than memory: graphs and predictions on disk, one or several GPUs.

Two steps. Build the graphs once, in one process (no GPU needed), then train and predict, with
python on one GPU or with torchrun on several GPUs of a node:

    python examples/train_large.py build [work_folder]
    python examples/train_large.py train [work_folder]                        # one GPU
    torchrun --nproc_per_node=4 examples/train_large.py train [work_folder]   # four GPUs

Host memory stays at a few GB whatever the size of the dataset: generate_graphs writes the graphs
section by section, training reads one batch of graphs at a time, and predict writes each graph's
results to Parquet (``pip install "pointillhist[parquet]"``). Keep the work folder on a local
disk, or in /dev/shm: training reads the graph files at every step. The predictions folder must
not exist yet when ``train`` starts.

``--demo`` uses the skin tables of examples/data/skin, cut into 12 small tiles, and trains for 30
epochs: ``python examples/train_large.py --demo`` runs both steps in a new temporary folder (about
half a minute on a GPU), whose path is printed; ``--demo build <folder>`` and
``--demo train <folder>`` run them one at a time.
"""
import glob
import os
import sys
import tempfile

import pandas as pd
import torch
import torch.distributed as dist

import pointillhist as ph

# ---- inputs ---------------------------------------------------------------
SECTIONS = sorted(glob.glob("data/sections/*.h5ad"))   # one .h5ad per section (or (dots.csv, cells.csv) pairs)
REFERENCE = "data/reference.csv"                       # cell types x genes
GRAPH_KWARGS = {}                                      # tiling parameters, "auto" by default
TRAIN_KWARGS = dict(batch_size=2, num_epochs=100)      # batch_size is per GPU with torchrun
WORK = "work"                                          # graphs, checkpoints, network and predictions

args = [a for a in sys.argv[1:] if a != "--demo"]
if "--demo" in sys.argv:
    skin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "skin")
    SECTIONS = [(os.path.join(skin, "dots.csv"), os.path.join(skin, "cells.csv"))]
    REFERENCE = pd.read_csv(os.path.join(skin, "cell_type_expression.csv"), index_col=0).T
    GRAPH_KWARGS = dict(tile_side=300, fraction_overlap=0.1, grid_spacing=40, cell_cell_maxdist=60)
    TRAIN_KWARGS = dict(batch_size=2, num_epochs=30)
    if not args:
        args = ["build+train", tempfile.mkdtemp(prefix="pointillhist_large_demo_")]
if not args or args[0] not in ("build", "train", "build+train"):
    sys.exit(__doc__)
step = args[0]
WORK = args[1] if len(args) > 1 else WORK
GRAPHS = os.path.join(WORK, "graphs")
PREDICTIONS = os.path.join(WORK, "predictions")

# ---- build: once, in one process ------------------------------------------
if step in ("build", "build+train"):
    graphs = ph.pp.generate_graphs(SECTIONS, REFERENCE, save_dir=GRAPHS, **GRAPH_KWARGS)
    print(f"{len(graphs)} graphs of {graphs.index['n_core_cells'].sum():,} cells written to {GRAPHS}")

# ---- train and predict: python or torchrun --------------------------------
if step in ("train", "build+train"):
    graphs = ph.pp.load_graphs(GRAPHS)
    net = ph.tr.networks(graphs)
    # train_distributed is train in a single process; under torchrun every process trains on its
    # share of the graphs (it creates the process group: nccl on GPUs, gloo on the CPU)
    net, history = ph.tr.train_distributed(net, graphs, REFERENCE, checkpoint_dir=os.path.join(WORK, "checkpoints"),
                                           **TRAIN_KWARGS)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        torch.save(net.state_dict(), os.path.join(WORK, "net.pth"))
    # every process predicts its share of the graphs and writes it
    ph.eval.predict(net, graphs, out=PREDICTIONS)
    if rank == 0:
        cells = pd.read_parquet(os.path.join(PREDICTIONS, "cells"), columns=["section", "cell_type"])
        print(f"{len(cells):,} cells predicted; results in {PREDICTIONS} (network in {WORK}/net.pth)")
        print(cells.groupby("section")["cell_type"].value_counts().groupby(level=0).head(3))
    if dist.is_initialized():
        dist.destroy_process_group()
