import itertools
import math
import os
import pickle
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import torch
import tqdm
from scipy.sparse import issparse
from sklearn.neighbors import KDTree
from torch_geometric.nn import SimpleConv

from . import _graph_worker
from ._reference import load_reference
from ._store import _GraphWriter
from ._topology import (
    hex_grid_by_spacing,
    build_tile_graph,
    assignment_edges,
)

__all__ = ["generate_graphs", "auto_graph_parameters"]


def _inside(df, box):
    """Mask of the rows of ``df`` inside ``box`` ({axis: (start, end)}, end excluded)."""
    mask = True
    for axis, (start, end) in box.items():
        mask = mask & (df[axis] >= start) & (df[axis] < end)
    return mask


def _cut(df, core, extended):
    """Rows of ``df`` inside the ``extended`` box, with 'is_core' for those inside ``core``."""
    tile = df[_inside(df, extended)].copy()
    tile["is_core"] = _inside(tile, core)
    return tile


def split_into_tiles(
    df_cells,
    df_gridpoints,
    df_dots=None,
    df_cells_features=None,
    tile_side=1000,
    fraction_overlap=0.2,
    pad_fraction=0.05,
):
    """Divide the full field of view into square (cubic in 3D) tiles of side ``tile_side``.

    Each tile has a unique core and a frame of ``fraction_overlap * tile_side``
    shared with its neighbours. ``df_cells``, ``df_gridpoints`` and the optional
    ``df_dots`` / ``df_cells_features`` need 'X' and 'Y' columns (and 'Z' if
    ``df_cells`` has one); each is cut per tile and gets an 'is_core' column.
    On the dots path only tiles with at least one core dot are kept, and no
    cell features are returned.

    Returns (vertexes, core_vertexes, cells_dfs, dots_dfs, cells_features_dfs,
    gridpoints_dfs), one entry per tile (``dots_dfs`` / ``cells_features_dfs``
    is None when the corresponding input is None). Vertexes are
    (x_min, x_max, y_min, y_max[, z_min, z_max]).
    """
    (
        fovs_cells_df,
        fovs_dots_df,
        fovs_cells_features_df,
        fovs_vertexes,
        fovs_core_vertexes,
        fovs_gridpoints_df,
    ) = ([], [], [], [], [], [])

    axes = [axis for axis in ("X", "Y", "Z") if axis in df_cells.columns]
    starts, num_grids = [], []
    for axis in axes:
        if df_dots is not None:
            # Calculate the range of the cells and dots
            low = min(df_cells[axis].min(), df_dots[axis].min())
            high = max(df_cells[axis].max(), df_dots[axis].max())
        else:
            # If no dots are provided, use only cells to determine the range
            low, high = df_cells[axis].min(), df_cells[axis].max()

        # Add a bit of a frame
        extent = high - low
        start = low - pad_fraction * extent
        end = high + pad_fraction * extent
        starts.append(start)
        num_grids.append(int(np.ceil((end - start) / tile_side)))

    if df_cells_features is not None:
        for axis in axes:
            df_cells_features[axis] = df_cells[axis]

    for index in itertools.product(*[range(n) for n in num_grids]):
        # Core tile boundaries, and the extended ones (with overlap)
        core, extended = {}, {}
        for axis, start, i in zip(axes, starts, index):
            start_core = start + i * tile_side
            end_core = start_core + tile_side
            core[axis] = (start_core, end_core)
            extended[axis] = (start_core - fraction_overlap * tile_side,
                              end_core + fraction_overlap * tile_side)
        core_vertexes = tuple(v for bounds in core.values() for v in bounds)
        vertexes = tuple(v for bounds in extended.values() for v in bounds)

        tile_df_cells = _cut(df_cells, core, extended)
        tile_df_gridpoints = _cut(df_gridpoints, core, extended)

        if df_dots is not None:
            tile_df_dots = _cut(df_dots, core, extended)
            if tile_df_dots["is_core"].sum() > 0:
                fovs_core_vertexes.append(core_vertexes)
                fovs_vertexes.append(vertexes)

                fovs_cells_df.append(tile_df_cells)
                fovs_dots_df.append(tile_df_dots)
                fovs_gridpoints_df.append(tile_df_gridpoints)
            fovs_cells_features_df = None
        else:
            fovs_dots_df = None
            fovs_cells_df.append(tile_df_cells)
            fovs_cells_features_df.append(_cut(df_cells_features, core, extended))
            fovs_core_vertexes.append(core_vertexes)
            fovs_vertexes.append(vertexes)
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
    """Dots csv with columns X, Y[, Z], gene (accepts x/y/z and target)."""
    df_dots = pd.read_csv(path_dots)
    if "X" not in df_dots.columns:
        df_dots.rename(columns={"x": "X", "y": "Y"}, inplace=True)
    if "Z" not in df_dots.columns:
        df_dots.rename(columns={"z": "Z"}, inplace=True)
    if "gene" not in df_dots.columns:
        df_dots.rename(columns={"target": "gene"}, inplace=True)
    return df_dots


def _axes(cells):
    """Coordinate columns of a section: X, Y, plus Z if present, numeric and not constant."""
    z = cells.get("Z")
    is_3d = z is not None and pd.api.types.is_numeric_dtype(z) and z.nunique() > 1
    return ["X", "Y", "Z"] if is_3d else ["X", "Y"]


def _anndata_coordinates(adata):
    """X, Y[, Z] of the cells, from ``obsm["spatial"]`` (2 or 3 columns) or ``obs["x"/"y"/"z"]``."""
    if "spatial" in adata.obsm:
        cells = pd.DataFrame(np.asarray(adata.obsm["spatial"])[:, :3], index=adata.obs_names)
    else:
        cells = adata.obs[[c for c in ("x", "y", "z") if c in adata.obs.columns]]
    cells = cells.set_axis(["X", "Y", "Z"][:cells.shape[1]], axis=1)
    return cells[_axes(cells)]


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
    if region_key is not None:
        df_cell["region"] = df_cell[region_key]  # kept apart from a region column named z/Z
    if "X" not in df_cell.columns:
        df_cell.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)
    if "unique_cell_id" not in df_cell.columns:
        df_cell["unique_cell_id"] = df_cell.index.astype(str)
    axes = _axes(df_cell)
    if "Z" in axes and "Z" not in df_dots.columns:
        raise ValueError(f"{path_cells} has z coordinates but {path_dots} has none")
    columns = axes + ["unique_cell_id"] + (["region"] if region_key is not None else [])
    df_cell = df_cell[columns]
    df_dots = df_dots[axes + ["gene"]]

    # Limits of the full canvas. The `.min()` inside the max() is deliberate
    # (historical quirk): changing it changes every tile, see CLEANUP_PLAN.md §11.
    full_region_vxs = ()
    for axis in axes:
        full_region_vxs += (min((df_cell[axis].min(), df_dots[axis].min())),
                            max((df_cell[axis].max(), df_dots[axis].min())))

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

    # create gridpoints
    all_gridpoints, diameter = hex_grid_by_spacing(
        full_region_vxs, grid_spacing, pad_fraction=0.05
    )
    df_gridpoints = pd.DataFrame(all_gridpoints, columns=axes)
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
        if fovs_gridpoints_df[i].empty:  # a thin border tile without fine gridpoints
            continue
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
            region_labels.append(fovs_cells_df[i]["region"].values)

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
    if region_key is not None:
        adata.obs["region"] = adata.obs[region_key].values  # before X, Y, Z are written

    coordinates = _anndata_coordinates(adata)
    axes = list(coordinates.columns)
    for axis in axes:
        adata.obs[axis] = coordinates[axis].values
    adata.obs['unique_cell_id'] = adata.obs_names

    # Limits of the full canvas. The `.min()` inside the max() is deliberate
    # (historical quirk): changing it changes every tile, see CLEANUP_PLAN.md §11.
    full_region_vxs = ()
    for axis in axes:
        full_region_vxs += (min((adata.obs[axis].min(), adata.obs[axis].min())),
                            max((adata.obs[axis].max(), adata.obs[axis].min())))

    # filter cells with less than counts_min_cell counts
    if counts_min_cell >= 0:
        counts = adata.X.sum(axis=1).A1 if issparse(adata.X) else adata.X.sum(axis=1)
        bool_cells = counts > counts_min_cell
        adata = adata[bool_cells, :].copy()
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise ValueError(f"{path_adata} lacks reference genes {missing[:10]}")
    adata = adata[:, genes]

    columns = axes + ["unique_cell_id"] + (["region"] if region_key is not None else [])
    df_cell = adata.obs[columns].copy()
    # create gridpoints
    all_gridpoints, diameter = hex_grid_by_spacing(
        full_region_vxs, grid_spacing, pad_fraction=0.05
    )
    df_gridpoints = pd.DataFrame(all_gridpoints, columns=axes)

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
        if fovs_gridpoints_df[i].empty:  # a thin border tile without fine gridpoints
            continue
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
            region_labels.append(fovs_cells_df[i]["region"].values)

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

    Returns ``(xyz, n_transcripts, n_counted_cells)`` (``xyz`` has 2 or 3 columns, see
    ``_axes``); the transcripts are counted on the first ``max_cells`` cells only,
    which is enough for a mean per cell.
    """
    if isinstance(spatial_path, (tuple, list)):
        dots, cells = _read_dots(spatial_path[0]), pd.read_csv(spatial_path[1])
        if "X" not in cells.columns:
            cells = cells.rename(columns={"x": "X", "y": "Y", "z": "Z"})
        xyz = cells[_axes(cells)].to_numpy(dtype=float)
        n_transcripts = len(dots) if genes is None else int(dots["gene"].isin(genes).sum())
        return xyz, n_transcripts, len(cells)
    adata = ad.read_h5ad(spatial_path, backed="r")
    xyz = _anndata_coordinates(adata).to_numpy(dtype=float)
    X = adata.X[:max_cells]
    if genes is not None:
        X = X[:, adata.var_names.get_indexer([g for g in genes if g in adata.var_names])]
    adata.file.close()
    return xyz, float(X.sum()), X.shape[0]


def auto_graph_parameters(spatial_paths, genes=None, max_cells_per_graph=100_000):
    """Tiling parameters read off the cell coordinates and counts of the sections.

    Per section (returned as lists, one entry per path): ``tile_side`` covers
    the section, so it is one graph, unless it has more than
    ``max_cells_per_graph`` cells, in which case it is cut into square tiles
    that stay below ``max_cells_per_graph`` cells (frames included) even in
    the densest tenth of the section, with ``fraction_overlap`` 0.1 (0
    otherwise). In 3D, tiles are cubes and a single tile also covers the z
    extent. Shared by all sections:
    distances scale with ``d_nn``, the median distance from a cell to its
    closest neighbour (median over the sections); cell-cell edges reach
    ``cell_cell_maxdist = 20 d_nn`` and fine gridpoints sit ``grid_spacing =
    10 d_nn`` apart, which puts a few dozen cells within a gridpoint's reach at
    uniform density (``5 d_nn`` in 3D, where the reach is a sphere: about a
    hundred cells). Sparse data needs more cells per gridpoint for a stable
    composition, so below 50 transcripts per cell (counted on ``genes``, the
    reference panel) the spacing grows as sqrt(50 / transcripts per cell).
    """
    if isinstance(spatial_paths, (str, tuple)):
        spatial_paths = [spatial_paths]
    d_nn, tile_side, fraction_overlap, n_transcripts, n_counted = [], [], [], 0.0, 0
    is_3d = False
    for path in spatial_paths:
        xyz, transcripts, counted = _section_summary(path, genes)
        is_3d = is_3d or xyz.shape[1] == 3
        d_nn.append(np.median(KDTree(xyz).query(xyz, k=2)[0][:, 1]))
        extent = np.ptp(xyz, axis=0)
        if len(xyz) <= max_cells_per_graph:
            # one tile: the canvas is padded to 1.1 x extent, a hair more so ceil() adds no empty tile
            tile_side.append(1.12 * float(extent.max()))
            fraction_overlap.append(0.0)
        else:
            # square tiles that stay below max_cells_per_graph cells (10 % frames included)
            # even in the densest part of the section, the busiest of 10 x 10 bins
            width, height = float(extent[0]), float(extent[1])
            bins, _, _ = np.histogram2d(xyz[:, 0], xyz[:, 1], bins=10)
            density = bins.max() / (width * height / 100)
            tile_side.append(math.sqrt(max_cells_per_graph / density) / 1.2)
            fraction_overlap.append(0.1)
        n_transcripts += transcripts
        n_counted += counted
    d_nn = float(np.median(d_nn))
    transcripts_per_cell = n_transcripts / n_counted
    # 10 d_nn would put about 800 cells within the spherical reach of a 3D gridpoint
    grid_factor = 5 if is_3d else 10
    return dict(
        tile_side=tile_side,
        grid_spacing=grid_factor * d_nn * math.sqrt(max(1.0, 50.0 / transcripts_per_cell)),
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
    save_dir=None,
    n_workers=1,
):
    """Tile every spatial section into HeteroData graphs.

    Parameters
    ----------
    spatial_paths : list
        .h5ad paths (cells x genes counts, 2 or 3 coordinates in ``obsm["spatial"]``
        or ``obs["x"/"y"/"z"]``), or (dots csv, cells csv) tuples (x, y[, z]).
        A constant z is ignored (2D section); 2D and 3D sections cannot be mixed.
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
        Tiles are cubes for 3D sections.
    min_cells, min_dots, cell_cell_k_neighbors, counts_min_cell, coarse_grid_side, device
        Remaining tiling and topology parameters. Dots path: ``min_dots``
        currently has no effect and ``cell_cell_maxdist`` is fixed at 90
        (deliberate, see CLEANUP_PLAN.md §11); on the AnnData path ``min_dots``
        is unused.
    save_dir : str or None
        None (default): the graphs are returned as a list held in memory. A
        folder that does not exist or is empty: the graphs are written there
        section by section as they are built, so memory holds one section (its
        counts and all its tiles) at a time, and are returned as a
        :class:`DiskGraphs`, which loads each graph (on the CPU) when it is
        accessed. The graphs are the same in both cases. If building fails,
        what was written is removed again.
    n_workers : int
        With ``save_dir``, the number of sections built at the same time, by
        as many Python worker processes (on the CPU); the graphs, warnings and
        errors are the same as with 1 (default). Each worker first imports
        pointillhist (several seconds, about 1 GB of memory), then holds one
        section at a time, so it pays off when sections take longer than
        that to build. The threads of this process
        (``torch.get_num_threads()``) are shared out among the workers.

    Returns
    -------
    list of HeteroData (a DiskGraphs with ``save_dir``), each carrying
    section_label, section, timepoint_label, timepoint, condition_label,
    condition, cell_types, genes, unique_cell_ids (and cell_regions, regions
    when ``region_key`` is given).
    """
    if len(spatial_paths) == 0:
        raise ValueError("spatial_paths is empty")
    if isinstance(n_workers, (bool, np.bool_)) or not isinstance(n_workers, (int, np.integer)) or n_workers < 1:
        raise ValueError(f"n_workers must be a positive integer, got {n_workers!r}")
    if n_workers > 1 and save_dir is None:
        raise ValueError("n_workers > 1 needs save_dir: sections are built in parallel only when the graphs "
                         "are written to disk")
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

    settings = dict(genes=genes, region_key=region_key, tile_sides=tile_sides, overlaps=overlaps,
                    min_cells=min_cells, min_dots=min_dots, cell_cell_k_neighbors=cell_cell_k_neighbors,
                    cell_cell_maxdist=cell_cell_maxdist, grid_spacing=grid_spacing,
                    counts_min_cell=counts_min_cell, device=device, coarse_grid_side=coarse_grid_side,
                    timepoint_labels=timepoint_labels, timepoint_codes=timepoint_codes,
                    condition_labels=condition_labels, condition_codes=condition_codes, cell_types=cell_types)
    if save_dir is not None:
        with _GraphWriter(save_dir) as writer:   # an error removes what was written
            if n_workers > 1:
                _build_graphs_parallel(spatial_paths, settings, writer, n_workers)
            else:
                _build_graphs(spatial_paths, settings, writer)
            if region_key is not None and not writer.region_vocabulary:
                raise ValueError(f"no annotated cells found in column {region_key!r}")
            return writer.close()
    graphs, region_labels = _build_graphs(spatial_paths, settings)

    if region_key is not None:
        region_labels = [_region_strings(labels) for labels in region_labels]
        regions = sorted({r for as_str, annotated in region_labels for r in as_str[annotated]})
        if not regions:
            raise ValueError(f"no annotated cells found in column {region_key!r}")
        for graph, (as_str, annotated) in zip(graphs, region_labels):
            graph.cell_regions = _region_one_hot(as_str, annotated, regions).to(device)
            graph.regions = regions

    return graphs


def _section_settings(settings, i):
    """The arguments of _section_tiles for section ``i``."""
    return dict(
        genes=settings["genes"], region_key=settings["region_key"], tile_side=settings["tile_sides"][i],
        overlap=settings["overlaps"][i], min_cells=settings["min_cells"], min_dots=settings["min_dots"],
        cell_cell_k_neighbors=settings["cell_cell_k_neighbors"], cell_cell_maxdist=settings["cell_cell_maxdist"],
        grid_spacing=settings["grid_spacing"], counts_min_cell=settings["counts_min_cell"],
        device=settings["device"], coarse_grid_side=settings["coarse_grid_side"],
        timepoint_label=settings["timepoint_labels"][i], timepoint_code=int(settings["timepoint_codes"][i]),
        condition_label=settings["condition_labels"][i], condition_code=int(settings["condition_codes"][i]),
        cell_types=settings["cell_types"],
    )


def _section_tiles(i, path, genes, region_key, tile_side, overlap, min_cells, min_dots, cell_cell_k_neighbors,
                   cell_cell_maxdist, grid_spacing, counts_min_cell, device, coarse_grid_side, timepoint_label,
                   timepoint_code, condition_label, condition_code, cell_types):
    """The graphs of section ``i`` with their section, timepoint and condition attributes, the raw
    region labels of each graph, and the section label."""
    if isinstance(path, (tuple, list)):
        tiles, labels = _graphs_from_tables(
            path[0], path[1], genes, region_key,
            tile_side=tile_side, min_cells=min_cells, min_dots=min_dots,
            cell_cell_k_neighbors=cell_cell_k_neighbors, grid_spacing=grid_spacing,
            fraction_overlap=overlap, counts_min_cell=counts_min_cell,
            device=device, coarse_grid_side=coarse_grid_side)
        section_label = os.path.basename(path[1])
    else:
        tiles, labels = _graphs_from_anndata(
            path, genes, region_key,
            tile_side=tile_side, min_cells=min_cells,
            cell_cell_k_neighbors=cell_cell_k_neighbors, cell_cell_maxdist=cell_cell_maxdist,
            grid_spacing=grid_spacing, fraction_overlap=overlap,
            counts_min_cell=counts_min_cell, device=device, coarse_grid_side=coarse_grid_side)
        section_label = os.path.basename(path)
    for graph in tiles:
        graph.section_label = section_label
        graph.section = i
        graph.timepoint_label = timepoint_label
        graph.timepoint = timepoint_code
        graph.condition_label = condition_label
        graph.condition = condition_code
        graph.cell_types = cell_types
        graph.genes = genes
    return tiles, labels, section_label


def _check_dimensions(n_vertexes, section_label, first, min_cells):
    """2D/3D bookkeeping for a section whose graphs have ``n_vertexes`` vertex coordinates (None: no graph).
    ``first`` is (n_vertexes, section_label) of the first section with graphs; returns it."""
    if n_vertexes is None:
        warnings.warn(f"{section_label}: no tile passed the filters (min_cells={min_cells})")
    elif first is None:
        first = (n_vertexes, section_label)
    elif n_vertexes != first[0]:
        raise ValueError(f"{section_label} is {n_vertexes // 2}D but "
                         f"{first[1]} is {first[0] // 2}D: "
                         "2D and 3D sections cannot be mixed")
    return first


def _build_graphs(spatial_paths, settings, writer=None):
    """The section loop of generate_graphs: the graphs and raw region labels of every section, or,
    with a writer, each section's graphs written as soon as they are built (nothing returned)."""
    graphs, region_labels = [], []
    first = None   # (len(vertexes_fov), section_label) of the first graph, for the 2D/3D check
    for i, path in enumerate(tqdm.tqdm(spatial_paths, desc="Generating graphs")):
        tiles, labels, section_label = _section_tiles(i, path, **_section_settings(settings, i))
        first = _check_dimensions(len(tiles[0].vertexes_fov) if tiles else None, section_label, first,
                                  settings["min_cells"])
        if writer is None:
            graphs += tiles
            region_labels += labels
        else:
            for j, graph in enumerate(tiles):
                writer.add(graph, _region_strings(labels[j]) if settings["region_key"] is not None else None)
            tiles = labels = graph = None   # only one section in memory
    return graphs, region_labels


def _write_section(folder, token, shared, i, path, threads, section):
    """Build section ``i`` and write its graphs into ``folder`` (run by _graph_worker in its own process)."""
    torch.set_num_threads(threads)
    tiles, labels, section_label = _section_tiles(i, path, **section)
    writer = _GraphWriter.attached(folder, token, shared)
    try:
        for j, graph in enumerate(tiles):
            writer.add(graph, _region_strings(labels[j]) if section["region_key"] is not None else None)
    except BaseException:
        writer.abort(remove_folders=False)
        raise
    return dict(section_label=section_label, n_vertexes=len(tiles[0].vertexes_fov) if tiles else None,
                rows=writer.rows, region_vocabulary=writer.region_vocabulary, streamed=writer.streamed_regions)


class _RemoteTraceback(Exception):
    """The traceback of an error raised in a worker process (set as the ``__cause__`` of the error)."""

    def __str__(self):
        return "\n\n" + self.args[0]


def _read_replies(stream, worker, replies):
    """Put every reply a worker process writes, then (worker, None) when it ends, into ``replies``."""
    try:
        while True:
            message = _graph_worker.read_message(stream)
            replies.put((worker, message))
            if message is None:
                return
    except BaseException:
        replies.put((worker, None))


def _reemit(recorded):
    """Emit the warnings a worker recorded while building a section as if they were raised here (the
    filters of this process apply). Each section is shown afresh, as when building in this process,
    where the filter changes made by pandas and scikit-learn reset "once per place" between sections."""
    registry = {}
    for category, category_name, message, filename, lineno, module in recorded:
        try:
            category = pickle.loads(category)
        except Exception:
            category = None
        if not (isinstance(category, type) and issubclass(category, Warning)):
            category, message = UserWarning, f"{category_name}: {message}"
        warnings.warn_explicit(message, category, filename, lineno, module=module, registry=registry)


def _signal_name(number):
    try:
        return signal.Signals(number).name
    except ValueError:
        return "unknown signal"


def _worker_error(reply, path, process, log):
    """The exception to raise for a section whose worker failed (``reply``) or ended without a reply (None)."""
    if reply is None:
        code = process.wait()
        how = (f"was killed by signal {-code} ({_signal_name(-code)}; SIGKILL often means out of memory)"
               if code < 0 else f"ended with exit code {code}")
        log.seek(0)
        output = log.read().decode(errors="replace").strip()
        return RuntimeError(f"the process building {path} {how}"
                            + (f"; its last output:\n{output[-4000:]}" if output else ""))
    error = None
    if reply["error"] is not None:
        try:
            error = pickle.loads(reply["error"])
        except Exception:
            pass
    if not isinstance(error, BaseException):
        error = RuntimeError(f"building {path} failed: {reply['error_text']}")
    error.__cause__ = _RemoteTraceback(reply["traceback"])
    return error


def _build_graphs_parallel(spatial_paths, settings, writer, n_workers):
    """_build_graphs with a writer, run by ``n_workers`` worker processes (_graph_worker) that build one
    section at a time each. Replies are taken in section order, so warnings, errors and the 2D/3D check
    come out as with _build_graphs."""
    if not sys.executable:
        raise RuntimeError("n_workers > 1 starts Python processes, but sys.executable is not set")
    spatial_paths = list(spatial_paths)   # by position, as the other per-section settings
    n_sections = len(spatial_paths)
    n_workers = min(n_workers, n_sections)
    shared = {"cell_types": settings["cell_types"], "genes": settings["genes"]}
    threads = max(1, torch.get_num_threads() // n_workers)   # the CPUs this process uses, shared out
    parts = __name__.split(".")
    root = os.path.abspath(__file__)
    for _ in parts:   # the folder holding the package, which the workers import
        root = os.path.dirname(root)
    search = [root] + [os.path.abspath(p) for p in sys.path if isinstance(p, str)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(dict.fromkeys(search)), OMP_NUM_THREADS=str(threads),
               PYTHONWARNINGS="ignore")   # warnings raised while building are recorded and emitted here
    command = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_graph_worker.py"),
               root, ".".join(parts[:-2])]
    writer.owns_all = True   # an error removes every graph file, whichever process wrote it
    replies = queue.Queue()
    processes, logs, readers = [], [], []
    busy, results = {}, {}   # worker -> section it builds; section -> (worker, reply or None if it ended)
    handed, done, first, stop = 0, 0, None, False

    def hand_out(worker):
        nonlocal handed
        section = dict(_section_settings(settings, handed), device="cpu")
        task = pickle.dumps(dict(folder=writer.path, token=writer.token, shared=shared, i=handed,
                                 path=spatial_paths[handed], threads=threads, section=section))
        busy[worker] = handed
        handed += 1
        try:
            _graph_worker.write_message(processes[worker].stdin, task)
        except OSError:   # the worker has ended, which its reader reports
            pass

    bar = tqdm.tqdm(total=n_sections, desc="Generating graphs")
    try:
        for worker in range(n_workers):
            logs.append(tempfile.TemporaryFile())   # its output, shown if it ends unexpectedly
            processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=logs[-1], env=env))
            readers.append(threading.Thread(target=_read_replies, daemon=True,
                                            args=(processes[-1].stdout, worker, replies)))
            readers[-1].start()
        for worker in range(n_workers):
            hand_out(worker)
        while done < n_sections:
            while done not in results:
                try:
                    worker, message = replies.get(timeout=1.0)
                except queue.Empty:
                    continue
                if worker not in busy:   # an idle worker ended
                    continue
                reply = None
                if message is not None:
                    try:
                        reply = pickle.loads(message)
                    except Exception as error:
                        reply = dict(error=None, error_text=f"its reply cannot be read ({error!r})",
                                     traceback="", warnings=[])
                results[busy.pop(worker)] = (worker, reply)
                stop = stop or reply is None or "result" not in reply
                if not stop and handed < n_sections:   # after a failure, later sections are not needed
                    hand_out(worker)
            worker, reply = results.pop(done)
            if reply is not None:
                _reemit(reply["warnings"])
            if reply is None or "result" not in reply:
                raise _worker_error(reply, spatial_paths[done], processes[worker], logs[worker])
            outcome = reply["result"]
            first = _check_dimensions(outcome["n_vertexes"], outcome["section_label"], first,
                                      settings["min_cells"])
            writer.merge(outcome["rows"], outcome["region_vocabulary"], outcome["streamed"], shared)
            done += 1
            bar.update(1)
    finally:
        bar.close()
        for process in processes:
            try:
                process.stdin.close()   # a worker ends at the end of its input
            except OSError:
                pass
            if done < n_sections:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for reader in readers:
            reader.join(timeout=10)
        for process in processes:
            process.stdout.close()
        for log in logs:
            log.close()
