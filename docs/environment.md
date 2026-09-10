# Environment

## Supported versions

[`pyproject.toml`](../pyproject.toml) declares the dependency ranges and `pip install`
resolves them for your Python. The lower bounds are at or below the versions the manuscript
results were produced with (the same minor series in every case except `numpy`, 1.26 against
2.1) and the package was verified at exactly these floors; it is also run on current
releases, and the two columns below are the two environments we use.

| package | required | manuscript (Python 3.10) | also tested (Python 3.11) |
|---|---|---|---|
| torch | >= 2.4 | 2.4.0 (CUDA 12.1) | 2.7.0 (CUDA 12.6) |
| torch_geometric | >= 2.6 | 2.6.1 | 2.6.1 |
| anndata | >= 0.10.9 | 0.10.9 | 0.11.4 |
| numpy | >= 1.26 | 2.1.0 | 2.2.6 |
| scipy | >= 1.14 | 1.14.1 | 1.16.3 |
| pandas | >= 2.2, < 3 | 2.2.2 | 2.3.3 |
| tqdm | >= 4.66 | 4.66.5 | 4.67.1 |
| matplotlib | >= 3.9 | 3.9.2 | 3.10.3 |
| seaborn | >= 0.13 | 0.13.2 | 0.13.2 |
| scikit-learn | >= 1.5 | 1.5.2 | 1.8.0 |
| umap-learn (extra `umap`, for `ph.eval.umap`) | >= 0.5 | 0.5.7 | 0.5.9 |
| openpyxl (extra `examples`, for the ABCA-2 example) | >= 3.1 | 3.1.5 | 3.1.5 |
| pyarrow (extra `examples`, for the parquet output of `minimal.py`) | >= 17 | 17.0.0 | 20.0.0 |

`pandas` stays below 3.0 for now: pandas 3 changes the default string dtype, and the
molecule-table (`dots.csv`, `cells.csv`) path of `generate_graphs` fails on it; the AnnData
path works.

[`uv.lock`](../uv.lock) is a locked resolution of these ranges for `uv sync`, computed
separately for Python 3.10, 3.11 and 3.12 or newer, so the versions it installs depend on the
Python minor (in September 2026: torch 2.14, PyTorch Geometric 2.8, pandas 2.3 and anndata
0.11 to 0.13). We ran the demo and our internal tests on all three resolutions, with Python
3.10 to 3.14, on the GPU for Python 3.12. `uv lock --upgrade` refreshes it.

PyTorch Geometric's compiled extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`,
`torch_cluster`) are not required: PointillHist uses only the pure-Python parts of the
library, and `examples/minimal.py --demo` and our internal tests run without them. They are
optional accelerators (`pyg_lib` speeds up the `HeteroLinear` layers PointillHist uses). The
wheels on [data.pyg.org](https://data.pyg.org/whl/) are built per `torch` and CUDA version and
indexed by `torch.__version__`, so the URL must match the `torch` you have; for the `torch`
2.14.0 with CUDA 13.0 that `uv.lock` installs:

```bash
pip install pyg_lib -f https://data.pyg.org/whl/torch-2.14.0+cu130.html
```

For `torch` 2.13 and newer data.pyg.org ships only `pyg_lib`; the pages for older versions,
such as `torch-2.7.0+cu126.html`, also carry `torch_scatter`, `torch_sparse` and `torch_cluster`.

## Platforms, CUDA and CPU builds of torch

We develop and test on Linux (x86_64, NVIDIA GPUs). On Linux the `torch` wheel on PyPI
includes the CUDA runtime, so the default installation runs on the GPU, provided the NVIDIA
driver is new enough for that CUDA version: the `torch` 2.14.0 pinned in `uv.lock` is a
CUDA 13 build and needs driver 580 or newer; with an older driver `torch.cuda.is_available()`
is `False` and you need a `cu126` or `cu128` build, installed as shown below. On Windows the PyPI
wheel is CPU-only, and macOS has no CUDA; both are untested by us but the package has no
platform-specific code. Intel Macs are not supported, because `torch` 2.4 and later ship no
wheels for them; on Apple Silicon the `torch` pinned in `uv.lock` needs macOS 14 or newer,
older systems can install an older `torch` (2.4 runs on macOS 11) before PointillHist. To
choose a build explicitly, install `torch` before PointillHist:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu126    # a specific CUDA version
uv pip install torch --torch-backend=auto                               # uv: pick from the installed driver
```

With uv, from the root of a clone (`git clone` and `cd PointillHist` as in the
[installation guide](../guide/installation.md)), there
are two ways to get a different `torch` than the locked CUDA 13 build. Either skip the lock and
install into a plain virtual environment (`--torch-backend=auto` picks the CUDA build matching
the installed driver, `cpu` the CPU build):

```bash
uv venv && uv pip install --torch-backend=cpu -e ".[umap,examples]"
```

or, after `uv sync`, swap the locked `torch` inside the project with
`uv pip install --reinstall-package torch --torch-backend=auto torch` (or `cpu`); without
`--reinstall-package` uv sees `torch` as already satisfied and leaves the locked build in
place. In that second case run things with `uv run --no-sync` or from the activated `.venv`:
a plain `uv run` or `uv sync` restores the locked build.

## Reproducing the manuscript environment

The exact versions of the manuscript column above, in a fresh virtual environment created
from the root of a clone (the `-e .` installs that checkout):

```bash
uv venv --python 3.10 .venv-manuscript && source .venv-manuscript/bin/activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e . torch_geometric==2.6.1 anndata==0.10.9 numpy==2.1.0 scipy==1.14.1 \
    pandas==2.2.2 tqdm==4.66.5 matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.2
```

(`python -m venv` and `pip` work the same way.) `anndata` 0.10 needs `scipy` < 1.15 to read
backed `.h5ad` files, which is why that combination pins `scipy==1.14.1`.
