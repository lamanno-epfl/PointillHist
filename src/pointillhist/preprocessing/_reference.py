import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

__all__ = ["reference_from_anndata", "load_reference"]


def reference_from_anndata(adata, cell_type_key, save_path=None):
    """Mean expression of ``adata.X`` per cell type, as a (cell types x genes) DataFrame.

    ``adata`` is an AnnData or a path to an .h5ad. Rows are the sorted unique
    values of ``adata.obs[cell_type_key]``, columns are ``adata.var_names``.
    If ``save_path`` is given the frame is also written to csv.
    """
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    # unique on the raw labels so integer types sort numerically (as pandas groupby does)
    types, codes = np.unique(np.asarray(adata.obs[cell_type_key]), return_inverse=True)
    types = pd.Index(types).astype(str)
    n_cells = len(codes)
    indicator = csr_matrix(
        (np.ones(n_cells), (codes, np.arange(n_cells))), shape=(len(types), n_cells)
    )
    summed = indicator @ adata.X
    summed = summed.toarray() if issparse(summed) else np.asarray(summed)
    counts = np.bincount(codes, minlength=len(types))
    reference = pd.DataFrame(summed / counts[:, None], index=types, columns=adata.var_names)
    if save_path is not None:
        reference.to_csv(save_path, index=True)
    return reference


def load_reference(reference, reference_key=None):
    """Return the (cell types x genes) reference as a DataFrame with str index/columns.

    Accepts a DataFrame, a .csv path (``index_col=0``), an AnnData or a .h5ad
    path (the two latter need ``reference_key``, the obs column with the cell type).
    """
    if isinstance(reference, os.PathLike):
        reference = os.fspath(reference)
    if isinstance(reference, pd.DataFrame):
        reference = reference.copy()
    elif isinstance(reference, str) and reference.endswith(".csv"):
        reference = pd.read_csv(reference, index_col=0)
    elif isinstance(reference, ad.AnnData) or (
        isinstance(reference, str) and reference.endswith(".h5ad")
    ):
        if reference_key is None:
            raise ValueError("reference_key is required when the reference is an AnnData/.h5ad")
        reference = reference_from_anndata(reference, reference_key)
    else:
        raise TypeError(
            "reference must be a DataFrame, a .csv path, an AnnData or a .h5ad path"
        )
    reference.index = reference.index.astype(str)
    reference.columns = reference.columns.astype(str)
    return reference
