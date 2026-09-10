"""A small synthetic dataset for trying PointillHist without downloading anything.

``synthetic`` writes two AnnData sections and the matching cell-type reference to a
directory and returns their paths: 320 cells x 40 genes per section, six cell types,
three anatomical regions in ``obs["region"]`` with about 10 % of the cells unannotated,
and one timepoint label per section. Everything is drawn from fixed seeds, so the files
are identical on every machine; ``examples/minimal.py --demo`` and the internal test
suite both run on them.
"""
import os

import numpy as np
import pandas as pd

N_GENES = 40
N_TYPES = 6
N_CELLS = 320
VERSION = "v2"
SECTIONS = (f"section_A_{VERSION}.h5ad", f"section_B_{VERSION}.h5ad")
TIMEPOINTS = ("E10", "E9")
REGIONS = ("region_a", "region_b", "region_c")
REFERENCE_CSV = "reference.csv"

__all__ = ["synthetic", "N_GENES", "N_TYPES", "N_CELLS", "SECTIONS", "TIMEPOINTS", "REGIONS"]


def _reference(rng):
    profile = rng.gamma(shape=1.5, scale=4.0, size=(N_TYPES, N_GENES))
    genes = [f"gene_{i:03d}" for i in range(N_GENES)]
    types = [f"type_{i}" for i in range(N_TYPES)]
    return pd.DataFrame(profile, index=types, columns=genes)


def _section(rng, reference, section_seed):
    import anndata as ad

    local = np.random.default_rng(section_seed)
    labels = local.integers(0, N_TYPES, size=N_CELLS)
    rates = reference.values[labels] * 0.6
    counts = local.poisson(rates).astype(np.float32)

    # Cells laid out on a jittered lattice so the tiling produces several tiles.
    side = int(np.ceil(np.sqrt(N_CELLS)))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([gx.ravel(), gy.ravel()])[:N_CELLS].astype(np.float64)
    coords = coords * 30.0 + local.uniform(-6.0, 6.0, size=coords.shape)

    # Three horizontal bands of regions, with ~10% of the cells unannotated.
    # Drawn after everything above so the earlier draws are unchanged.
    band = np.clip(coords[:, 1] // (side * 10.0), 0, len(REGIONS) - 1).astype(int)
    region = np.array(REGIONS, dtype=object)[band]
    region[local.random(N_CELLS) < 0.1] = None

    adata = ad.AnnData(X=counts)
    adata.var_names = list(reference.columns)
    adata.obs_names = [f"s{section_seed}_cell_{i:04d}" for i in range(N_CELLS)]
    adata.obs["true_label"] = labels
    adata.obs["region"] = pd.Categorical(region)
    adata.obsm["spatial"] = coords
    adata.uns["timepoint"] = TIMEPOINTS[section_seed]
    return adata


def synthetic(out_dir):
    """Write the synthetic dataset to ``out_dir`` if absent and return its paths.

    Returns ``(reference_csv, section_paths)``: the path of the cell types x genes reference
    table and the list of the two section ``.h5ad`` files, ready for
    ``ph.pp.generate_graphs(section_paths, reference_csv, timepoints=list(TIMEPOINTS))``.
    Existing files are reused, so the dataset is only generated once per directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    ref_path = os.path.join(out_dir, REFERENCE_CSV)
    section_paths = [os.path.join(out_dir, name) for name in SECTIONS]

    if os.path.exists(ref_path) and all(os.path.exists(p) for p in section_paths):
        return ref_path, section_paths

    rng = np.random.default_rng(0)
    reference = _reference(rng)
    reference.to_csv(ref_path)
    for seed, path in enumerate(section_paths):
        _section(rng, reference, seed).write_h5ad(path)
    return ref_path, section_paths
