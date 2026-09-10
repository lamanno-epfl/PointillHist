# Installation guide

PointillHist is a pure-Python package for Python 3.10 or later. Its dependencies (`torch`,
`torch_geometric`, `anndata`, `numpy`, `scipy`, `pandas`, `tqdm`, `matplotlib`, `seaborn`,
`scikit-learn`) are installed automatically; the compiled extensions of PyTorch Geometric
(`pyg_lib`, `torch_scatter`, ...) are not needed. We develop and test on Linux with NVIDIA
GPUs; training on the CPU works but is slow. The supported version ranges, the tested
environments and the platform notes are in [`docs/environment.md`](../docs/environment.md).

Two extras exist: `umap` adds `umap-learn` for `ph.eval.umap`, and `examples` adds `openpyxl`
(the ABCA-2 example reads its taxonomy from an Excel sheet) and `pyarrow` (parquet output of
`examples/minimal.py`). Add them as `pointillhist[umap,examples]` in the commands below, or
leave them out.

## With pip

Into an existing environment that has `git` on its PATH:

```bash
pip install "pointillhist[umap,examples] @ git+https://github.com/lamanno-epfl/PointillHist.git"
```

On Linux the `torch` wheel from PyPI includes CUDA; see [Choosing the torch build](#choosing-the-torch-build)
before installing on a machine with an older NVIDIA driver, without a GPU, or on Windows.

## With uv

From a clone, with the versions pinned in [`uv.lock`](../uv.lock):

```bash
git clone https://github.com/lamanno-epfl/PointillHist.git
cd PointillHist
uv sync --all-extras                          # creates .venv with the locked versions
uv run python examples/minimal.py --demo      # or: source .venv/bin/activate
```

The lock is resolved separately for Python 3.10, 3.11 and 3.12 or newer, and we run the demo
and our internal tests on all three. It pins the CUDA 13 build of `torch`; for another build
either skip the lock, in a plain virtual environment from the same clone,

```bash
uv venv && uv pip install --torch-backend=auto -e ".[umap,examples]"   # auto: match the NVIDIA driver; or cpu
```

or, after `uv sync`, swap `torch` with `uv pip install --reinstall-package torch --torch-backend=auto torch`
and then run things with `uv run --no-sync` or from the activated `.venv`, because a plain
`uv run` or `uv sync` restores the locked build.

## With conda

conda provides Python (and `git`, which pip needs for the URL), pip installs the package.
PointillHist itself is not on conda, and the PyPI builds of `torch` and `torch_geometric` are
the ones we test.

```bash
conda create -n pointillhist python=3.11 git
conda activate pointillhist
pip install "pointillhist[umap,examples] @ git+https://github.com/lamanno-epfl/PointillHist.git"
```

## Choosing the torch build

The `torch` that PyPI serves by default on Linux is a CUDA 13 build (2.14.0+cu130 in
September 2026) and needs an NVIDIA driver of version 580 or newer (`nvidia-smi` prints it).
With an older driver the installation succeeds but `torch.cuda.is_available()` is `False`. In
that case, on a CPU-only machine, or for a specific CUDA version, install `torch` first and
PointillHist afterwards:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126    # CUDA 12.6; cu128 also exists
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU-only
uv pip install torch --torch-backend=auto                               # uv: pick from the installed driver
```

The Windows wheel on PyPI is CPU-only and macOS has no CUDA; Intel Macs are not supported
because `torch` 2.4 and later ship no wheels for them. Details in
[`docs/environment.md`](../docs/environment.md#platforms-cuda-and-cpu-builds-of-torch).

## Checking the installation

```bash
python -c "import pointillhist as ph; print(ph.__version__)"
python examples/minimal.py --demo      # from a clone
```

The demo trains on a small synthetic dataset, GPU or not, and ends with a line like
`640 cells predicted; outputs in /tmp/pointillhist_demo_...` in well under a minute.

## Optional accelerators and the manuscript environment

`pyg_lib` and the other compiled PyTorch Geometric extensions are optional accelerators whose
wheels must match your `torch` and CUDA versions; the commands, and a recipe that reproduces
the exact versions the manuscript results were produced with, are in
[`docs/environment.md`](../docs/environment.md).
