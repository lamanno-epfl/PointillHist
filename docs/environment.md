# Environment

## Supported versions

[`pyproject.toml`](../pyproject.toml) declares the dependency ranges and `pip install`
resolves them for your Python. The lower bounds are the versions the manuscript results were
produced with; the package is also run on current releases, and the two columns below are
the two environments we use.

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

[`uv.lock`](../uv.lock) pins one complete, tested set of these for `uv sync`; it is what
"the versions we test with" means in the README. `uv lock --upgrade` refreshes it.

PyTorch Geometric's compiled extensions (`pyg_lib`, `torch_scatter`, `torch_sparse`,
`torch_cluster`) are not required: PointillHist uses only the pure-Python parts of the
library, and `examples/minimal.py --demo` and the test suite run without them. They are
optional accelerators; if you want them, install the wheels matching your `torch` and CUDA
versions from [data.pyg.org](https://data.pyg.org/whl/):

```bash
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

## CUDA and CPU builds of torch

On Linux the `torch` wheel on PyPI includes the CUDA runtime, so the default installation
runs on the GPU. On Windows the PyPI wheel is CPU-only, and macOS has no CUDA. To choose a
build explicitly, install `torch` before PointillHist:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cu126    # a specific CUDA version
uv pip install torch --torch-backend=auto                               # uv: pick from the installed driver
```

After `uv sync`, the same `uv pip install torch ...` command inside the project replaces the
locked `torch` in `.venv` with the build you want.

## Reproducing the manuscript environment

The exact versions of the manuscript column above, in a fresh virtual environment:

```bash
uv venv --python 3.10 .venv-manuscript && source .venv-manuscript/bin/activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -e . torch_geometric==2.6.1 anndata==0.10.9 numpy==2.1.0 scipy==1.14.1 \
    pandas==2.2.2 tqdm==4.66.5 matplotlib==3.9.2 seaborn==0.13.2 scikit-learn==1.5.2
```

(`python -m venv` and `pip` work the same way.) `anndata` 0.10 needs `scipy` < 1.15 to read
backed `.h5ad` files, which is why that combination pins `scipy==1.14.1`.
