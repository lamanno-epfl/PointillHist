import math
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.sparse import issparse
from sklearn.neighbors import KDTree
from torch_geometric.nn import SimpleConv

from ._reference import load_reference
from ._topology import (
    hex_grid_by_spacing,
    build_tile_graph,
    assignment_edges,
)

__all__ = ["generate_graphs", "auto_graph_parameters"]


def split_into_tiles(
    df_cells,
    df_gridpoints,
    df_dots=None,
    df_cells_features=None,
    tile_side=1000,
    fraction_overlap=0.2,
    pad_fraction=0.05,
):
    """Divide the full field of view into square tiles of side ``tile_side``.

    Each tile has a unique core and a frame of ``fraction_overlap * tile_side``
    shared with its neighbours. ``df_cells``, ``df_gridpoints`` and the optional
    ``df_dots`` / ``df_cells_features`` need 'X' and 'Y' columns; each is cut
    per tile and gets an 'is_core' column. On the dots path only tiles with at
    least one core dot are kept, and no cell features are returned.

    Returns (vertexes, core_vertexes, cells_dfs, dots_dfs, cells_features_dfs,
    gridpoints_dfs), one entry per tile (``dots_dfs`` / ``cells_features_dfs``
    is None when the corresponding input is None).
    """
    (
        fovs_cells_df,
        fovs_dots_df,
        fovs_cells_features_df,
        fovs_vertexes,
        fovs_core_vertexes,
        fovs_gridpoints_df,
    ) = ([], [], [], [], [], [])

    if df_dots is not None:
        # Calculate the range for X and Y coordinates
        x_min, x_max = min(df_cells["X"].min(), df_dots["X"].min()), max(
            df_cells["X"].max(), df_dots["X"].max()
        )
        y_min, y_max = min(df_cells["Y"].min(), df_dots["Y"].min()), max(
            df_cells["Y"].max(), df_dots["Y"].max()
        )
    else:
        # If no dots are provided, use only cells to determine the range
        x_min, x_max = df_cells["X"].min(), df_cells["X"].max()
        y_min, y_max = df_cells["Y"].min(), df_cells["Y"].max()

    # Add a bit of a frame
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_left = x_min - pad_fraction * x_range
    x_right = x_max + pad_fraction * x_range
    y_bottom = y_min - pad_fraction * y_range
    y_top = y_max + pad_fraction * y_range

    # Determine the number of grids in X and Y directions
    num_grids_x = int(np.ceil((x_right - x_left) / tile_side))
    num_grids_y = int(np.ceil((y_top - y_bottom) / tile_side))

    for i in range(num_grids_x):
        for j in range(num_grids_y):
            # Calculate core tile boundaries
            x_start_core = x_left + i * tile_side
            x_end_core = x_start_core + tile_side
            y_start_core = y_bottom + j * tile_side
            y_end_core = y_start_core + tile_side

            # Calculate extended tile boundaries (with overlap)
            x_start_extended = x_start_core - fraction_overlap * tile_side
            x_end_extended = x_end_core + fraction_overlap * tile_side
            y_start_extended = y_start_core - fraction_overlap * tile_side
            y_end_extended = y_end_core + fraction_overlap * tile_side

            tile_df_cells = df_cells[
                (df_cells["X"] >= x_start_extended)
                & (df_cells["X"] < x_end_extended)
                & (df_cells["Y"] >= y_start_extended)
                & (df_cells["Y"] < y_end_extended)
            ].copy()

            tile_df_cells["is_core"] = (
                (tile_df_cells["X"] >= x_start_core)
                & (tile_df_cells["X"] < x_end_core)
                & (tile_df_cells["Y"] >= y_start_core)
                & (tile_df_cells["Y"] < y_end_core)
            )

            tile_df_gridpoints = df_gridpoints[
                (df_gridpoints["X"] >= x_start_extended)
                & (df_gridpoints["X"] < x_end_extended)
                & (df_gridpoints["Y"] >= y_start_extended)
                & (df_gridpoints["Y"] < y_end_extended)
            ].copy()

            tile_df_gridpoints["is_core"] = (
                (tile_df_gridpoints["X"] >= x_start_core)
                & (tile_df_gridpoints["X"] < x_end_core)
                & (tile_df_gridpoints["Y"] >= y_start_core)
                & (tile_df_gridpoints["Y"] < y_end_core)
            )
            if df_cells_features is not None:
                df_cells_features["X"] = df_cells["X"]
                df_cells_features["Y"] = df_cells["Y"]
                tile_df_cells_features = df_cells_features[
                    (df_cells_features["X"] >= x_start_extended)
                    & (df_cells_features["X"] < x_end_extended)
                    & (df_cells_features["Y"] >= y_start_extended)
                    & (df_cells_features["Y"] < y_end_extended)
                ].copy()
                tile_df_cells_features["is_core"] = (
                    (tile_df_cells_features["X"] >= x_start_core)
                    & (tile_df_cells_features["X"] < x_end_core)
                    & (tile_df_cells_features["Y"] >= y_start_core)
                    & (tile_df_cells_features["Y"] < y_end_core)
                )

            if df_dots is not None:
                tile_df_dots = df_dots[
                    (df_dots["X"] >= x_start_extended)
                    & (df_dots["X"] < x_end_extended)
                    & (df_dots["Y"] >= y_start_extended)
                    & (df_dots["Y"] < y_end_extended)
                ].copy()

                tile_df_dots["is_core"] = (
                    (tile_df_dots["X"] >= x_start_core)
                    & (tile_df_dots["X"] < x_end_core)
                    & (tile_df_dots["Y"] >= y_start_core)
                    & (tile_df_dots["Y"] < y_end_core)
                )
                if tile_df_dots["is_core"].sum() > 0:
                    fovs_core_vertexes.append(
                        (x_start_core, x_end_core, y_start_core, y_end_core)
                    )
                    fovs_vertexes.append(
                        (x_start_extended, x_end_extended, y_start_extended, y_end_extended)
                    )

                    fovs_cells_df.append(tile_df_cells)
                    fovs_dots_df.append(tile_df_dots)
                    fovs_gridpoints_df.append(tile_df_gridpoints)
                fovs_cells_features_df = None
            else:
                fovs_dots_df = None
                fovs_cells_df.append(tile_df_cells)
                fovs_cells_features_df.append(tile_df_cells_features)
                fovs_core_vertexes.append(
                    (x_start_core, x_end_core, y_start_core, y_end_core)
                )
                fovs_vertexes.append(
                    (x_start_extended, x_end_extended, y_start_extended, y_end_extended)
                )
                fovs_gridpoints_df.append(tile_df_gridpoints)

    return (
        fovs_vertexes,
        fovs_core_vertexes,
        fovs_cells_df,
        fovs_dots_df,
        fovs_cells_features_df,
        fovs_gridpoints_df,
    )


def _read_dots(path_dots):
    """Dots csv with columns X, Y, gene (accepts x/y and target)."""
    df_dots = pd.read_csv(path_dots)
    if "X" not in df_dots.columns:
        df_dots.rename(columns={"x": "X", "y": "Y"}, inplace=True)
    if "gene" not in df_dots.columns:
        df_dots.rename(columns={"target": "gene"}, inplace=True)
    return df_dots


def _graphs_from_tables(
    path_dots,
    path_cells,
    genes,
    region_key,
    tile_side,
    min_cells,
    min_dots,
    cell_cell_k_neighbors,
    grid_spacing,
    fraction_overlap,
    counts_min_cell,
    device,
    coarse_grid_side,
):
    """Tile graphs of one section given as (dots csv, cells csv).

    Returns the graphs (with ``unique_cell_ids`` set) and, when ``region_key``
    is given, the per-graph array of raw region labels (else an empty list).
    """
    df_dots = _read_dots(path_dots)
    df_dots = df_dots[df_dots["gene"].isin(genes)]
    df_dots.reset_index(drop=True, inplace=True)

    df_cell = pd.read_csv(path_cells)
    if "X" not in df_cell.columns:
        df_cell.rename(columns={"x": "X", "y": "Y"}, inplace=True)
    if "unique_cell_id" not in df_cell.columns:
        df_cell["unique_cell_id"] = df_cell.index.astype(str)
    columns = ["X", "Y", "unique_cell_id"] + ([region_key] if region_key is not None else [])
    df_cell = df_cell[columns]

    # Limits of the full canvas. The `.min()` inside the max() is deliberate
    # (historical quirk): changing it changes every tile, see CLEANUP_PLAN.md §11.
    x_min, x_max = min((df_cell["X"].min(), df_dots["X"].min())), max(
        (df_cell["X"].max(), df_dots["X"].min())
    )
    y_min, y_max = min((df_cell["Y"].min(), df_dots["Y"].min())), max(
        (df_cell["Y"].max(), df_dots["Y"].min())
    )

    # filtering cells
    if counts_min_cell >= 0:
        dc_edge_index = assignment_edges(
            df_dots,
            df_cell,
            strategy="knn-maxdist",
            k_neighbors=1,
            radius=70)
        counts = SimpleConv(aggr="sum", flow="source_to_target")(
                (torch.ones(df_dots.shape[0])[:, None], torch.ones(df_cell.shape[0])[:, None]),  # note second entry is not used
                torch.tensor(dc_edge_index, dtype=torch.long),
            ).squeeze()
        bool_cells = counts > counts_min_cell
        df_cell = df_cell.loc[bool_cells.cpu().numpy(), :]

    full_region_vxs = x_min, x_max, y_min, y_max
    # create gridpoints
    all_gridpoints, diameter = hex_grid_by_spacing(
        full_region_vxs, grid_spacing, pad_fraction=0.05
    )
    df_gridpoints = pd.DataFrame(all_gridpoints, columns=["X", "Y"])
    (
        fovs_vertexes,
        fovs_core_vertexes,
        fovs_cells_df,
        fovs_dots_df,
        _,
        fovs_gridpoints_df,
    ) = split_into_tiles(
        df_cell,
        df_gridpoints,
        df_dots=df_dots,
        df_cells_features=None,
        tile_side=tile_side,
        fraction_overlap=fraction_overlap,
    )

    # Filter out FOVs with less than min_cells / min_dots
    indices_1 = [
        i
        for i, cells_df in enumerate(fovs_cells_df)
        if cells_df["is_core"].sum() >= max([min_cells, cell_cell_k_neighbors + 1])
    ]
    indices_2 = [
        i
        for i, dots_df in enumerate(fovs_dots_df)
        if dots_df["is_core"].sum() >= min_dots
    ]
    # Deliberate quirk: `or` (not `&`) makes min_dots inert whenever any tile
    # passes min_cells; changing it changes every tile, see CLEANUP_PLAN.md §11.
    indices = list(set(indices_1) or set(indices_2))

    graphs, region_labels = [], []
    for i in indices:
        graph = build_tile_graph(
            fovs_vertexes[i],
            fovs_core_vertexes[i],
            fovs_cells_df[i],
            fovs_gridpoints_df[i],
            genes,
            df_dots_fov=fovs_dots_df[i],
            grid_triangle_side=diameter / 2,
            cell_cell_strategy="knn-maxdist",
            cell_cell_k_neighbors=cell_cell_k_neighbors,
            cell_cell_maxdist=90,  # deliberate: fixed on the dots path, see CLEANUP_PLAN.md §11
            dot_cell_strategy="knn-maxdist",
            dot_cell_k_neighbors=1,
            dot_cell_maxdist=70,
            pos_is_separate=True,
            use_stored_cell_assign=False,
            coarse_grid_side=coarse_grid_side
        ).to(device)
        # Skip empty tiles
        if (
            graph["dots", "could_come_from", "cells"].edge_index.shape[1] == 0
            or graph["cells", "is_watched_by", "gridpoints"].edge_index.shape[1] == 0
        ):
            continue
        graph.unique_cell_ids = fovs_cells_df[i]["unique_cell_id"].values
        graphs.append(graph)
        if region_key is not None:
            region_labels.append(fovs_cells_df[i][region_key].values)

    return graphs, region_labels


def _graphs_from_anndata(
    path_adata,
    genes,
    region_key,
    tile_side,
    min_cells,
    cell_cell_k_neighbors,
    cell_cell_maxdist,
    grid_spacing,
    fraction_overlap,
    counts_min_cell,
    device,
    coarse_grid_side,
):
    """Tile graphs of one section given as an .h5ad (cells x genes counts).

    Returns the graphs (with ``unique_cell_ids`` set) and, when ``region_key``
    is given, the per-graph array of raw region labels (else an empty list).
    """
    adata = ad.read_h5ad(path_adata)

    try:
        adata.obs['X'] = adata.obsm['spatial'][:, 0]
        adata.obs['Y'] = adata.obsm['spatial'][:, 1]
    except KeyError:
        adata.obs['X'] = adata.obs['x'].values
        adata.obs['Y'] = adata.obs['y'].values
    adata.obs['unique_cell_id'] = adata.obs_names

    # Limits of the full canvas. The `.min()` inside the max() is deliberate
    # (historical quirk): changing it changes every tile, see CLEANUP_PLAN.md §11.
    x_min, x_max = min((adata.obs["X"].min(), adata.obs["X"].min())), max(
        (adata.obs["X"].max(), adata.obs["X"].min())
    )
    y_min, y_max = min((adata.obs["Y"].min(), adata.obs["Y"].min())), max(
        (adata.obs["Y"].max(), adata.obs["Y"].min())
    )

    # filter cells with less than counts_min_cell counts
    if counts_min_cell >= 0:
        counts = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)
        bool_cells = counts > counts_min_cell
        adata = adata[bool_cells, :].copy()
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise ValueError(f"{path_adata} lacks reference genes {missing[:10]}")
    adata = adata[:, genes]

    columns = ["X", "Y", "unique_cell_id"] + ([region_key] if region_key is not None else [])
    df_cell = adata.obs[columns].copy()
    full_region_vxs = x_min, x_max, y_min, y_max
    # create gridpoints
    all_gridpoints, diameter = hex_grid_by_spacing(
        full_region_vxs, grid_spacing, pad_fraction=0.05
    )
    df_gridpoints = pd.DataFrame(all_gridpoints, columns=["X", "Y"])

    X = adata.X
    if issparse(X):
        X = X.toarray()
    df_cells_features = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)

    (
        fovs_vertexes,
        fovs_core_vertexes,
        fovs_cells_df,
        _,
        fovs_cells_features_df,
        fovs_gridpoints_df,
    ) = split_into_tiles(
        df_cells=df_cell,
        df_dots=None,
        df_cells_features=df_cells_features,
        df_gridpoints=df_gridpoints,
        tile_side=tile_side,
        fraction_overlap=fraction_overlap,
    )

    # Filter out FOVs with less than min_cells (list(set(...)) fixes the tile order)
    indices_1 = [
        i
        for i, cells_df in enumerate(fovs_cells_df)
        if cells_df["is_core"].sum() >= max([min_cells, cell_cell_k_neighbors + 1])
    ]
    indices = list(set(indices_1))

    graphs, region_labels = [], []
    for i in indices:
        graph = build_tile_graph(
            fovs_vertexes[i],
            fovs_core_vertexes[i],
            fovs_cells_df[i],
            fovs_gridpoints_df[i],
            genes,
            df_cells_features_fov=fovs_cells_features_df[i],
            grid_triangle_side=diameter / 2,
            cell_cell_strategy="knn-maxdist",
            cell_cell_k_neighbors=cell_cell_k_neighbors,
            cell_cell_maxdist=cell_cell_maxdist,
            dot_cell_strategy="knn-maxdist",
            dot_cell_k_neighbors=1,
            dot_cell_maxdist=70,
            pos_is_separate=True,
            use_stored_cell_assign=False,
            coarse_grid_side=coarse_grid_side
        ).to(device)
        # Skip empty tiles
        if graph["cells", "is_watched_by", "gridpoints"].edge_index.shape[1] == 0:
            continue
        graph.unique_cell_ids = fovs_cells_df[i]["unique_cell_id"].values
        graphs.append(graph)
        if region_key is not None:
            region_labels.append(fovs_cells_df[i][region_key].values)

    return graphs, region_labels


def _dense_codes(labels, n_files, name):
    """Per-file labels (default all 0) and their dense integer codes."""
    labels = [0] * n_files if labels is None else list(labels)
    if len(labels) != n_files:
        raise ValueError(f"{name} has {len(labels)} entries, expected one per spatial path ({n_files})")
    _, codes = np.unique(labels, return_inverse=True)
    return labels, codes


def _panel_genes(reference, spatial_path):
    """Reference genes present in the first section, in reference column order."""
    if isinstance(spatial_path, (tuple, list)):
        cols = pd.read_csv(spatial_path[0], usecols=lambda c: c in ("gene", "target"))
        panel = set(cols["gene"] if "gene" in cols.columns else cols["target"])  # as _read_dots
    else:
        adata = ad.read_h5ad(spatial_path, backed="r")
        panel = set(adata.var_names)
        adata.file.close()
    return [g for g in reference.columns if g in panel]


def _region_strings(labels):
    """Region labels as str plus a mask of the annotated cells (NaN/None/""/"nan" are not)."""
    labels = pd.Series(np.asarray(labels, dtype=object))
    as_str = labels.astype(str).values
    annotated = (~labels.isna()).values & ~np.isin(as_str, ["", "nan", "None"])
    return as_str, annotated


def _region_one_hot(as_str, annotated, regions):
    """(n_cells, n_regions) float32 one-hot; unannotated cells get an all-zero row."""
    column = pd.Index(regions).get_indexer(as_str)
    one_hot = np.zeros((len(as_str), len(regions)), dtype=np.float32)
    one_hot[np.flatnonzero(annotated), column[annotated]] = 1.0
    return torch.tensor(one_hot)


def _section_summary(spatial_path, genes, max_cells=20_000):
    """Cell coordinates of one section, and its transcripts on ``genes`` (all genes if None).

    Returns ``(xy, n_transcripts, n_counted_cells)``; the transcripts are counted on
    the first ``max_cells`` cells only, which is enough for a mean per cell.
    """
    if isinstance(spatial_path, (tuple, list)):
        dots, cells = _read_dots(spatial_path[0]), pd.read_csv(spatial_path[1])
        xy = cells[["X", "Y"] if "X" in cells.columns else ["x", "y"]].to_numpy(dtype=float)
        n_transcripts = len(dots) if genes is None else int(dots["gene"].isin(genes).sum())
        return xy, n_transcripts, len(cells)
    adata = ad.read_h5ad(spatial_path, backed="r")
    xy = np.asarray(adata.obsm["spatial"])[:, :2] if "spatial" in adata.obsm else adata.obs[["x", "y"]].to_numpy()
    X = adata.X[:max_cells]
    if genes is not None:
        X = X[:, adata.var_names.get_indexer([g for g in genes if g in adata.var_names])]
    adata.file.close()
    return np.asarray(xy, dtype=float), float(X.sum()), X.shape[0]


def auto_graph_parameters(spatial_paths, genes=None, max_cells_per_graph=100_000):
    """Tiling parameters read off the cell coordinates and counts of the sections.

    Per section (returned as lists, one entry per path): ``tile_side`` covers
    the section, so it is one graph, unless it has more than
    ``max_cells_per_graph`` cells, in which case it is cut into square tiles
    that stay below ``max_cells_per_graph`` cells (frames included) even in
    the densest tenth of the section, with ``fraction_overlap`` 0.1 (0
    otherwise). Shared by all sections:
    distances scale with ``d_nn``, the median distance from a cell to its
    closest neighbour (median over the sections); cell-cell edges reach
    ``cell_cell_maxdist = 20 d_nn`` and fine gridpoints sit ``grid_spacing =
    10 d_nn`` apart, which puts a few dozen cells within a gridpoint's reach at
    uniform density. Sparse data needs more cells per gridpoint for a stable
    composition, so below 50 transcripts per cell (counted on ``genes``, the
    reference panel) the spacing grows as sqrt(50 / transcripts per cell).
    """
    if isinstance(spatial_paths, (str, tuple)):
        spatial_paths = [spatial_paths]
    d_nn, tile_side, fraction_overlap, n_transcripts, n_counted = [], [], [], 0.0, 0
    for path in spatial_paths:
        xy, transcripts, counted = _section_summary(path, genes)
        d_nn.append(np.median(KDTree(xy).query(xy, k=2)[0][:, 1]))
        width, height = float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))
        if len(xy) <= max_cells_per_graph:
            # one tile: the canvas is padded to 1.1 x extent, a hair more so ceil() adds no empty tile
            tile_side.append(1.12 * max(width, height))
            fraction_overlap.append(0.0)
        else:
            # square tiles that stay below max_cells_per_graph cells (10 % frames included)
            # even in the densest part of the section, the busiest of 10 x 10 bins
            bins, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=10)
            density = bins.max() / (width * height / 100)
            tile_side.append(math.sqrt(max_cells_per_graph / density) / 1.2)
            fraction_overlap.append(0.1)
        n_transcripts += transcripts
        n_counted += counted
    d_nn = float(np.median(d_nn))
    transcripts_per_cell = n_transcripts / n_counted
    return dict(
        tile_side=tile_side,
        grid_spacing=10 * d_nn * math.sqrt(max(1.0, 50.0 / transcripts_per_cell)),
        cell_cell_maxdist=20 * d_nn,
        fraction_overlap=fraction_overlap,
    )


def _per_section(value, n_files, name):
    """A number is used for every section; a list must have one entry per section."""
    values = [value] * n_files if np.isscalar(value) else list(value)
    if len(values) != n_files:
        raise ValueError(f"{name} has {len(values)} entries, expected one per spatial path ({n_files})")
    return values


def generate_graphs(
    spatial_paths,
    reference,
    reference_key=None,
    timepoints=None,
    conditions=None,
    region_key=None,
    tile_side="auto",
    grid_spacing="auto",
    cell_cell_maxdist="auto",
    fraction_overlap="auto",
    min_cells=10,
    min_dots=10,
    cell_cell_k_neighbors=30,
    counts_min_cell=-1,
    coarse_grid_side=7,
    device="cpu",
):
    """Tile every spatial section into HeteroData graphs.

    Parameters
    ----------
    spatial_paths : list
        .h5ad paths (cells x genes counts, coordinates in ``obsm["spatial"]`` or
        ``obs["x"/"y"]``), or (dots csv, cells csv) tuples.
    reference : DataFrame | csv path | AnnData | .h5ad path
        Cell types x genes reference (AnnData needs ``reference_key``).
    reference_key : str or None
        obs column with the cell type when ``reference`` is an AnnData/.h5ad.
    timepoints, conditions : list or None
        One label per spatial path (default: all the same).
    region_key : str or None
        obs column (AnnData) / csv column (cells csv) with the region label of
        each cell; sets ``graph.cell_regions`` and ``graph.regions``. Labels
        are matched as ``str(label)``, so the columns of a ``type_regions``
        table must carry the same names.
    tile_side, grid_spacing, cell_cell_maxdist, fraction_overlap : number or "auto"
        Tiling parameters in the units of the coordinates. Any of them left at
        "auto" is set by :func:`auto_graph_parameters` from all the sections.
        ``tile_side`` and ``fraction_overlap`` may also be lists with one
        entry per spatial path (which is what "auto" produces: one graph per
        section, only sections above 100 000 cells are cut into tiles).
    min_cells, min_dots, cell_cell_k_neighbors, counts_min_cell, coarse_grid_side, device
        Remaining tiling and topology parameters. Dots path: ``min_dots``
        currently has no effect and ``cell_cell_maxdist`` is fixed at 90
        (deliberate, see CLEANUP_PLAN.md §11); on the AnnData path ``min_dots``
        is unused.

    Returns
    -------
    list of HeteroData, each carrying section_label, section, timepoint_label,
    timepoint, condition_label, condition, cell_types, genes, unique_cell_ids
    (and cell_regions, regions when ``region_key`` is given).
    """
    if len(spatial_paths) == 0:
        raise ValueError("spatial_paths is empty")
    reference = load_reference(reference, reference_key)
    cell_types = list(reference.index)
    n_files = len(spatial_paths)
    timepoint_labels, timepoint_codes = _dense_codes(timepoints, n_files, "timepoints")
    condition_labels, condition_codes = _dense_codes(conditions, n_files, "conditions")
    genes = _panel_genes(reference, spatial_paths[0])
    if not genes:
        raise ValueError(
            f"none of the {reference.shape[1]} reference genes is in {spatial_paths[0]} "
            f"(e.g. {list(reference.columns[:3])})"
        )

    given = dict(tile_side=tile_side, grid_spacing=grid_spacing,
                 cell_cell_maxdist=cell_cell_maxdist, fraction_overlap=fraction_overlap)
    if any(isinstance(v, str) and v == "auto" for v in given.values()):
        auto = auto_graph_parameters(spatial_paths, genes)
        given = {k: auto[k] if isinstance(v, str) else v for k, v in given.items()}
    grid_spacing, cell_cell_maxdist = given["grid_spacing"], given["cell_cell_maxdist"]
    tile_sides = _per_section(given["tile_side"], n_files, "tile_side")
    overlaps = _per_section(given["fraction_overlap"], n_files, "fraction_overlap")
    if grid_spacing >= min(tile_sides):
        raise ValueError(f"grid_spacing ({grid_spacing:.3g}) must be smaller than tile_side ({min(tile_sides):.3g})")

    graphs, region_labels = [], []
    for i, path in enumerate(tqdm.tqdm(spatial_paths, desc="Generating graphs")):
        if isinstance(path, (tuple, list)):
            tiles, labels = _graphs_from_tables(
                path[0], path[1], genes, region_key,
                tile_side=tile_sides[i], min_cells=min_cells, min_dots=min_dots,
                cell_cell_k_neighbors=cell_cell_k_neighbors, grid_spacing=grid_spacing,
                fraction_overlap=overlaps[i], counts_min_cell=counts_min_cell,
                device=device, coarse_grid_side=coarse_grid_side)
            section_label = os.path.basename(path[1])
        else:
            tiles, labels = _graphs_from_anndata(
                path, genes, region_key,
                tile_side=tile_sides[i], min_cells=min_cells,
                cell_cell_k_neighbors=cell_cell_k_neighbors, cell_cell_maxdist=cell_cell_maxdist,
                grid_spacing=grid_spacing, fraction_overlap=overlaps[i],
                counts_min_cell=counts_min_cell, device=device, coarse_grid_side=coarse_grid_side)
            section_label = os.path.basename(path)
        if not tiles:
            warnings.warn(f"{section_label}: no tile passed the filters (min_cells={min_cells})")
        for graph in tiles:
            graph.section_label = section_label
            graph.section = i
            graph.timepoint_label = timepoint_labels[i]
            graph.timepoint = int(timepoint_codes[i])
            graph.condition_label = condition_labels[i]
            graph.condition = int(condition_codes[i])
            graph.cell_types = cell_types
            graph.genes = genes
        graphs += tiles
        region_labels += labels

    if region_key is not None:
        region_labels = [_region_strings(labels) for labels in region_labels]
        regions = sorted({r for as_str, annotated in region_labels for r in as_str[annotated]})
        if not regions:
            raise ValueError(f"no annotated cells found in column {region_key!r}")
        for graph, (as_str, annotated) in zip(graphs, region_labels):
            graph.cell_regions = _region_one_hot(as_str, annotated, regions).to(device)
            graph.regions = regions

    return graphs
