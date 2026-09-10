"""
Zhuang ABCA-2 (MERFISH, 66 coronal sections) mapped to the ~1200 Yao 2023 supertypes with pointillhist.
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch

import pointillhist as ph

# Where the downloaded data live and where the outputs go; edit the three paths to your layout.
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the directory holding the clone
DATA_DIR = os.path.join(DATA_ROOT, "notebooks_Merfish", "ABCA2")
REFERENCE = os.path.join(DATA_ROOT, "zhuang", "PH_Yao2023_avg", "reference_matrix_Yao2023_supertype.csv")
OUT = os.path.join(DATA_ROOT, "notebooks_Merfish", "ABCA2_supertype", "pointillhist")
os.makedirs(OUT, exist_ok=True)

# reference: ~1200 supertypes x 1122 panel genes
reference = pd.read_csv(REFERENCE, index_col=0)
reference = reference.loc[reference.index.notna()]
sections = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5ad")))
print(f"{len(sections)} sections, {reference.shape[0]} supertypes x {reference.shape[1]} genes")

# graphs: one per section, every tiling parameter automatic (cached, delete graphs.pt to rebuild)
GRAPHS = os.path.join(OUT, "graphs.pt")
if os.path.exists(GRAPHS):
    graphs = torch.load(GRAPHS, weights_only=False)
else:
    graphs = ph.pp.generate_graphs(sections, reference)
    torch.save(graphs, GRAPHS)
    print(f"{len(graphs)} graphs built and saved to {GRAPHS}.")

# network and training at the package defaults
net = ph.tr.networks(graphs)

net, history = ph.tr.train(net, graphs, reference, batch_size=4, lambda_density_min=50,   # density weight 100 -> 50 (default floor 10)
                           checkpoint_dir=os.path.join(OUT, "checkpoints"))

torch.save(net.state_dict(), os.path.join(OUT, "net.pth"))

# predictions: one row per cell; the supertype name without its leading number matches obs["supertype_label"]
result = ph.eval.predict(net, graphs)
predictions = pd.DataFrame({
    "cell_id": result["unique_cell_ids"],
    "section": result["all_sections"],
    "x": result["all_positions"][:, 0],
    "y": result["all_positions"][:, 1],
    "supertype": result["all_cell_types"],
    "supertype_label": [name.split(" ", 1)[1] for name in result["all_cell_types"]],
})

# aggregate the supetypes back to subclass and class using the Yao et al. 2023 taxonomy (supertype -> subclass -> class)
HIERARCHY = os.path.join(DATA_ROOT, "zhuang", "celltypes_info.xlsx")      # Yao 2023 taxonomy: supertype -> subclass -> class
hierarchy = pd.read_excel(HIERARCHY, sheet_name="supertype_annotation").set_index("supertype_label")
predictions["subclass_label"] = predictions["supertype_label"].map(hierarchy["subclass_label"]).values
predictions["class_label"] = predictions["supertype_label"].map(hierarchy["class_label"]).values

predictions.to_csv(os.path.join(OUT, "predictions.csv"), index=False)
with open(os.path.join(OUT, "result.pkl"), "wb") as f:       # probabilities, embeddings, positions; not the counts
    pickle.dump({k: v for k, v in result.items() if k != "expression_matrices"}, f)
print(f"{len(predictions):,} cells predicted; outputs in {OUT}")