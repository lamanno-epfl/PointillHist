<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_dark.svg">
    <img src="docs/logo.svg" alt="PointillHist" width="300">
  </picture>
</p>

<h3 align="center">Context-aware cell identity assignment for spatial transcriptomics</h3>

<p align="center">
  <a href="PAPER_URL_PLACEHOLDER"><img src="https://img.shields.io/badge/paper-bioRxiv%202026-b31b1b.svg" alt="paper"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab.svg" alt="python">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A5%202.4-ee4c2c.svg" alt="pytorch">
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-%E2%89%A5%202.6-3c2179.svg" alt="pyg">
</p>

---

PointillHist assigns a cell type to every cell of a spatial transcriptomics dataset from a
single-cell reference alone: a table of mean expression per cell type is the only
supervision. Instead of classifying each cell on its own, it trains a graph transformer on
tiles of the tissue in which every cell is embedded together with its neighbours and with a
coarse grid that summarises the composition of its surroundings, so each assignment is
consistent with its spatial context. It scales to millions of cells and trains several
sections, timepoints and conditions together.

The method is described in
[**Context-aware cell identity assignment maps the 3D cellular architecture of the human embryonic brain**](PAPER_URL_PLACEHOLDER).

**Contents:** [Installation](#installation) · [Quick start](#quick-start) · [Inputs and outputs](#inputs-and-outputs) · [Examples](#examples) · [Hyperparameters](#hyperparameters) · [GPU usage](#gpu-usage) · [Citation](#citation) · [License](#license)

## Installation

PointillHist is a pure-Python package for Python 3.10 or later. Its dependencies are
`torch`, `torch_geometric`, `anndata`, `numpy`, `scipy`, `pandas`, `tqdm`, `matplotlib`,
`seaborn` and `scikit-learn`; the compiled extensions of PyTorch Geometric (`pyg_lib`,
`torch_scatter`, ...) are not needed. Two extras exist: `umap` for the optional embedding of
the predictions (`ph.eval.umap`) and `examples` for the optional dependencies of the example
scripts (`openpyxl` for the ABCA-2 taxonomy sheet, `pyarrow` for the parquet output of
`minimal.py`).

**pip**, into an existing environment that has `git` on its PATH (on Linux the `torch` wheel
from PyPI includes CUDA; see the note on NVIDIA drivers below):

```bash
pip install "pointillhist[umap,examples] @ git+https://github.com/lamanno-epfl/PointillHist.git"
```

**uv**, from a clone, with the versions pinned in [`uv.lock`](uv.lock), resolved separately
for Python 3.10, 3.11 and 3.12 or newer; we run the demo and our internal tests on all three:

```bash
git clone https://github.com/lamanno-epfl/PointillHist.git
cd PointillHist
uv sync --all-extras                          # creates .venv with the locked versions
uv run python examples/minimal.py --demo      # or: source .venv/bin/activate
```

**conda**, letting conda provide Python and pip the packages (PointillHist itself is not on
conda, and the PyPI builds of `torch` and `torch_geometric` are the ones we test):

```bash
conda create -n pointillhist python=3.11 git      # git: pip installs from the git URL below
conda activate pointillhist
pip install "pointillhist[umap,examples] @ git+https://github.com/lamanno-epfl/PointillHist.git"
```

The `torch` that PyPI serves by default is a CUDA 13 build (2.14.0+cu130 in September 2026)
and needs an NVIDIA driver of version 580 or newer (`nvidia-smi` prints it); with an older
driver the installation succeeds but `torch.cuda.is_available()` is `False`. In that case, on
a CPU-only machine, or for a specific CUDA version, install `torch` first following
[pytorch.org](https://pytorch.org/get-started/locally/), for example
`pip install torch --index-url https://download.pytorch.org/whl/cu126`, and then PointillHist
with pip as above. The uv equivalent is in [`docs/environment.md`](docs/environment.md), which
also lists the supported version ranges, the platforms we test on and the versions the
manuscript results were produced with.

To check the installation, `python -c "import pointillhist as ph; print(ph.__version__)"`
must print the version, and `python examples/minimal.py --demo` (from a clone) runs the
whole pipeline on a small synthetic dataset in well under a minute, GPU or not.

## Quick start

Four calls take a dataset from files to a cell type per cell.

```python
import pointillhist as ph

# 1. data: spatial sections (raw counts + coordinates) and a reference (cell types x genes)
sections = ["data/section_1.h5ad", "data/section_2.h5ad"]
reference = "data/reference.csv"

# 2. graphs: every section is tiled into graphs of cells, neighbours and a coarse grid
graphs = ph.pp.generate_graphs(sections, reference)

# 3. training: the network is sized from the graphs
net = ph.tr.networks(graphs)
net, history = ph.tr.train(net, graphs, reference)

# 4. prediction: one cell type per cell, with probabilities, embeddings and positions
result = ph.eval.predict(net, graphs)
```

`result["all_cell_types"]` holds the assignment of every cell, in the order of
`result["unique_cell_ids"]`. Nothing is written to disk unless you ask for it:
`torch.save(net.state_dict(), "net.pth")` keeps the trained network
(`train(..., checkpoint_dir="checkpoints")` also saves it every 20 epochs), and
`ph.tr.load_model("net.pth", graphs)` rebuilds it later for prediction on the same graphs.

## Inputs and outputs

**Spatial sections.** One `.h5ad` file per section with raw counts in `X` and the
coordinates in `obsm["spatial"]` (or `obs["x"]`, `obs["y"]`); any coordinate unit works,
since every distance is derived from the data. A section can also be given as a
`(dots.csv, cells.csv)` pair of molecule and cell tables, whose coordinates must be in
micrometres. Optional lists `timepoints=` and `conditions=` give one label per section.

**Reference.** A csv or `DataFrame` of mean expression, rows = cell types, columns =
genes, or an `AnnData` of single cells with the type in `obs` (`reference_key=`). The
genes used are the reference genes present in the first section, in reference order; every
other section must contain them, and genes outside the reference are ignored.

**Predictions.** `predict` returns a dict with, per cell, the assigned type
(`all_cell_types`) and its index (`all_labels`), the logits (`all_logits`), the five largest
probabilities renormalised to sum 1 as a `scipy.sparse` matrix (`all_probs`; `top_k=None`
keeps every probability in a dense array), positions, section and timepoint labels, and the
cell embeddings; plus the embeddings and positions of the grid nodes. Apart from `all_probs`
every per-cell and per-grid-node entry is a NumPy array, ready for `pandas` or `pickle`;
`cell_types`, `section_ids` and `label_list` are plain lists.

## Examples

| example | what it shows |
|---|---|
| [`examples/minimal.py`](examples/minimal.py) | The four calls above with the predictions and the loss curves written to a `results/` folder. `python examples/minimal.py --demo` runs it on a small synthetic dataset generated on the fly by `ph.datasets.synthetic`, so nothing needs downloading; the demo writes to a temporary directory whose path is printed at the end. |
| [`examples/train_abca2_supertype.py`](examples/train_abca2_supertype.py) | The Zhuang MERFISH atlas of the adult mouse brain (ABCA-2, 66 sections, 1.2 M cells) mapped to the 1 195 supertypes of the Yao 2023 taxonomy, then backtracked to subclasses and classes. |
| [`examples/p1pup_mapping.ipynb`](examples/p1pup_mapping.ipynb) | A whole Xenium Prime section of a newborn mouse (1.3 M cells, 5 010 genes) mapped to 181 cell types of a whole-body reference, with the figures of the manuscript. |

The two dataset examples expect the public data to be downloaded; their headers say where
the files are read from. The notebook additionally reads three project-specific files from a
`figures/` folder next to it (the cell-type palette, a UMAP of the reference cells and a
cached segmentation for the region insets); they serve only the manuscript figures and are
not part of the repository.

## Hyperparameters

The defaults are meant to work out of the box, and the graph tiling adapts itself to the
cell density and the transcript counts of the data. The few settings worth knowing are
grouped in three families, graph generation, model and training, and explained with
practical guidance in the **[hyperparameter guide](guide/hyperparameters.md)**.

## GPU usage

PointillHist runs on CUDA; training on the CPU works but is slow. During the first epoch
the graphs stay in host memory and a copy of each one is sent to the GPU for its step, while
the memory a step needs is measured. From the second epoch on, all graphs are kept on the
GPU whenever they fit next to that working set, which removes the copies; otherwise the
copying mode continues, at about 1.5× the runtime. A line printed after the first epoch says
which mode was chosen.

If training runs out of memory in the first epoch, the tiles themselves are too large: lower
`tile_side` in `generate_graphs` so that every graph and its working set become smaller. If
it happens later, after the graphs were moved to the GPU, force the copying mode with
`keep_on_device=False` in `train`.

## Citation

If PointillHist is useful in your work, please cite the manuscript:

> Gargoori Motlagh A, Borm L, et al. (2026). *Context-aware cell identity assignment maps
> the 3D cellular architecture of the human embryonic brain.* bioRxiv.
> [PAPER_URL_PLACEHOLDER](PAPER_URL_PLACEHOLDER)

```bibtex
@article{gargoori2026pointillhist,
  title   = {Context-aware cell identity assignment maps the 3D cellular architecture of the human embryonic brain},
  author  = {Gargoori Motlagh, Alireza and Borm, Lars and others},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {DOI_PLACEHOLDER}
}
```

## License

PointillHist is released under the LICENSE_NAME_PLACEHOLDER license, see [`LICENSE`](LICENSE).
