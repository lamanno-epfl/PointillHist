# Environment

PointillHist is imported from the directory that contains the clone (or with that directory on
`PYTHONPATH`) and has no build step. The versions
below are the ones the manuscript results were produced with; newer releases of the same
major versions are expected to work.

| package | version |
|---|---|
| Python | 3.10 |
| torch | 2.4.0 (CUDA 12.1) |
| torch_geometric | 2.6.1 |
| anndata | 0.10.9 |
| numpy | 2.2 |
| scipy | 1.14 |
| pandas | 2.2 |
| tqdm | 4.66 |
| matplotlib | 3.9 |
| seaborn | 0.13 |
| scikit-learn | 1.5 |
| umap-learn (optional, `ph.eval.umap`) | 0.5 |
| openpyxl (optional, ABCA-2 example) | 3.1 |

A minimal installation with pip, after installing `torch` for your CUDA version from
[pytorch.org](https://pytorch.org):

```bash
pip install torch_geometric anndata numpy scipy pandas tqdm matplotlib seaborn scikit-learn
pip install umap-learn openpyxl   # optional
```

`anndata` 0.10 requires `scipy` < 1.15 for reading backed `.h5ad` files; pin
`scipy==1.14.1` if you use that combination.
