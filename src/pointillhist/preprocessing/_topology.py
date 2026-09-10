import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import HeteroData

__all__ = [
    "one_hot_encode_genes",
    "knn_edges",
    "assignment_edges",
    "cell_to_grid_edges",
    "hex_grid_by_count",
    "hex_grid_by_spacing",
    "build_tile_graph",
]


def one_hot_encode_genes(dots_df, genes):
    """(n_dots, n_genes) int one-hot of ``dots_df["gene"]`` over the ``genes`` list."""
    one_hots = dots_df["gene"].values[:, None] == np.asarray(genes)
    return one_hots.astype(int)

def knn_edges(
    cell_centroid_df,
    strategy="knn",
    self_edges=True,
    k_neighbors=16,
    radius=40,
    extra_features=None,
):
    """
    Construct the edges among the points of one dataframe.

    Parameters
    ----------
    cell_centroid_df : pd.DataFrame
        DataFrame with cell coordinates (x/y or X/Y).
    strategy : str
        One of ['knn', 'ball', 'knn-maxdist', 'ball-kmax'].
    self_edges : bool
        Whether to include self-loops.
    k_neighbors : int
        Number of neighbors for knn/ball-kmax.
    radius : float
        Radius for ball or knn-maxdist.
    extra_features : np.array or None
        Optional additional features to include in node features.

    Returns
    -------
    node_features : np.ndarray
        Array of shape (n_nodes, feature_dim).
    edge_index : np.ndarray
        Array of shape (2, n_edges).
    """
    if "x" in cell_centroid_df.columns:
        coords = cell_centroid_df[["x", "y"]].values
    elif "X" in cell_centroid_df.columns:
        coords = cell_centroid_df[["X", "Y"]].values
    else:
        raise ValueError("Coordinates not found in dataframe.")

    N = coords.shape[0]

    # Clamp k_neighbors so kdt.query never asks for more neighbors than exist.
    # Triggered for tiny grid FOVs where N can be < k_neighbors + 1.
    max_k = N - int(not self_edges)
    if k_neighbors > max_k:
        k_neighbors = max(max_k, 0)

    # ----- Build local edges based on strategy -----
    if strategy == "knn":
        kdt = KDTree(coords)
        ind = kdt.query(coords, k=k_neighbors + int(not self_edges), return_distance=False)
        ind = ind[:, int(not self_edges):]
        edge_index = np.stack([
            np.repeat(np.arange(N), k_neighbors),
            ind.flatten()
        ])
    elif strategy == "ball":
        kdt = KDTree(coords)
        indices, _ = kdt.query_radius(coords, r=radius, sort_results=True, return_distance=True)
        edge_index = np.concatenate([
            np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]),
            np.concatenate(indices)
        ]).reshape(2, -1)
    elif strategy == "knn-maxdist":
        kdt = KDTree(coords)
        tmp_dist, tmp_ind = kdt.query(coords, k=k_neighbors + int(not self_edges), return_distance=True)
        tmp_dist = tmp_dist[:, int(not self_edges):]
        tmp_ind = tmp_ind[:, int(not self_edges):]
        indices = [tmp_ind[i][tmp_dist[i] < radius] for i in range(N)]
        edge_index = np.concatenate([
            np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]),
            np.concatenate(indices)
        ]).reshape(2, -1)
    elif strategy == "ball-kmax":
        kdt = KDTree(coords)
        indices, _ = kdt.query_radius(coords, r=radius, sort_results=True, return_distance=True)
        indices = [idx[int(not self_edges):k_neighbors] for idx in indices]
        edge_index = np.concatenate([
            np.concatenate([np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]),
            np.concatenate(indices)
        ]).reshape(2, -1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # ----- Node features -----
    if extra_features is not None:
        node_features = np.column_stack([coords, extra_features]).astype(np.float32)
    else:
        node_features = coords.astype(np.float32)

    return node_features, edge_index

def assignment_edges(
    dots_df,
    cell_centroid_df=None,
    strategy="knn",
    k_neighbors=2,
    radius=12,
    weight_edges=False,
):
    """
    Constructs the dot-cell edges.

    Parameters
    ----------
    dots_df : pd.DataFrame
        Dot coordinates.
    cell_centroid_df : pd.DataFrame
        Cell coordinates.
    strategy : str
        One of ['knn', 'ball', 'knn-maxdist', 'ball-kmax', 'precomputed'].
    k_neighbors : int
        Number of neighbors (for knn/ball-kmax).
    radius : float
        Radius threshold.
    weight_edges : bool
        Return distances if True.

    Returns
    -------
    edge_index : np.ndarray of shape (2, num_edges)
    edge_weight : np.ndarray or None (if enabled)
    """
    if "x" in dots_df.columns:
        dot_coords = dots_df[["x", "y"]].values
    elif "X" in dots_df.columns:
        dot_coords = dots_df[["X", "Y"]].values

    if cell_centroid_df is not None:
        if "x" in cell_centroid_df.columns:
            cell_coords = cell_centroid_df[["x", "y"]].values
        elif "X" in cell_centroid_df.columns:
            cell_coords = cell_centroid_df[["X", "Y"]].values
    else:
        raise ValueError("`cell_centroid_df` must be provided.")

    edge_weight = None

    if strategy == "precomputed":
        if "cell" in dots_df.columns:
            try:
                edge_index = np.column_stack(
                    [np.arange(len(dots_df.index)), dots_df["cell"].values]
                ).T
            except KeyError:
                raise ValueError("Column `cell` must exist in dots_df.")
            if weight_edges:
                raise NotImplementedError("Edge weights not supported with precomputed strategy.")
        else:
            raise ValueError("`cell` column is required for precomputed strategy.")

    elif strategy == "knn":
        kdt = KDTree(cell_coords)
        if weight_edges:
            dist, indices = kdt.query(dot_coords, k=k_neighbors, return_distance=True)
            edge_weight = dist
        else:
            indices = kdt.query(dot_coords, k=k_neighbors, return_distance=False)
        edge_index = np.stack([
            np.repeat(np.arange(len(dot_coords)), k_neighbors),
            indices.flatten()
        ])

    elif strategy == "ball":
        kdt = KDTree(cell_coords)
        indices, _ = kdt.query_radius(dot_coords, r=radius, sort_results=True, return_distance=True)
        first_node = [np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]
        edge_index = np.concatenate(
            [np.concatenate(first_node), np.concatenate(indices)]
        ).reshape(2, -1)

    elif strategy == "knn-maxdist":
        kdt = KDTree(cell_coords)
        tmp_dist, tmp_ind = kdt.query(dot_coords, k=k_neighbors, return_distance=True)
        indices = []
        edge_weight = [] if weight_edges else None
        for i in range(len(dot_coords)):
            valid = tmp_dist[i] < radius
            indices.append(tmp_ind[i][valid])
            if weight_edges:
                edge_weight.append(tmp_dist[i][valid])
        first_node = [np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]
        edge_index = np.concatenate(
            [np.concatenate(first_node), np.concatenate(indices)]
        ).reshape(2, -1)
        if weight_edges:
            edge_weight = np.concatenate(edge_weight)

    elif strategy == "ball-kmax":
        kdt = KDTree(cell_coords)
        indices, dist = kdt.query_radius(dot_coords, r=radius, sort_results=True, return_distance=True)
        indices = [idx[:k_neighbors] for idx in indices]
        first_node = [np.full(len(idx), i, dtype=int) for i, idx in enumerate(indices)]
        edge_index = np.concatenate(
            [np.concatenate(first_node), np.concatenate(indices)]
        ).reshape(2, -1)
        if weight_edges:
            edge_weight = np.concatenate([d[:k_neighbors] for d in dist])

    else:
        raise ValueError(f"Strategy '{strategy}' not implemented.")

    if weight_edges:
        return edge_index, edge_weight
    else:
        return edge_index

def cell_to_grid_edges(grid_df, cell_centroid_df, radius=100):
    """Cell -> gridpoint edges: every cell within ``radius`` of a gridpoint."""
    edge_indexes = assignment_edges(
        grid_df,
        cell_centroid_df,
        "ball",
        None,
        radius,
        False,
    )
    return edge_indexes[::-1, :].copy()

def hex_grid_by_count(vertexes, N=25):
    """Note this is a triangular/hexagonal grid
    where each point is equidistand from its 6 neighbors

    Parameters
    ----------
    vertexes : tuple
        The tuple containing the vertexes of the field of view.
    N : int, default=25
        The number of gridpoints to generate.
        Note that this is an approximate number because the grid is hexagonal
        and some gridpoints may be outside the field of view.

    Returns
    -------
    points : np.array
        The array containing the gridpoints.
    """
    n = int(np.sqrt(N))
    x_min, x_max, y_min, y_max = vertexes
    spacing = (x_max - x_min) / n
    x = np.arange(x_min, x_max, spacing)
    y = np.arange(y_min, y_max, np.sin(np.pi / 3) * spacing)
    y = y[:-1] + (y[-1] - y[-2])
    X, Y = np.meshgrid(x, y)
    X[::2] += spacing / 2.0
    points = np.stack((X.flatten(), Y.flatten()), axis=1)
    points = points[points[:, 0] < x_max, :]
    points = points[points[:, 1] < y_max, :]
    diameter = np.sin(np.pi / 3) * 2 * spacing
    return points, diameter

def hex_grid_by_spacing(
    vertexes, spacing=150, pad_fraction=0.05, offset_fraction=0.2
):
    """Note this is a triangular/hexagonal grid
    where each point is equidistand from its 6 neighbors

    Parameters
    ----------
    vertexes : tuple
        The tuple containing the vertexes of the field of view.
    spacing : int, default=150
        The typical spacing between the gridpoints in pixels.

    Returns
    -------
    points : np.array
        The array containing the gridpoints.
    """

    x_min, x_max, y_min, y_max = vertexes
    full_range_x = x_max - x_min
    full_range_y = y_max - y_min
    x_min -= pad_fraction * full_range_x
    x_max += pad_fraction * full_range_x
    y_min -= pad_fraction * full_range_y
    y_max += pad_fraction * full_range_y

    x = np.arange(
        float(x_min), float(x_max) + float(spacing), float(spacing)
    )
    y = np.arange(
        float(y_min),
        float(y_max) + float(spacing),
        np.sin(np.pi / 3) * spacing,
    )
    # offset the grid to avoid the border
    x = x + offset_fraction * (x[-1] - x[-2])
    y = y + offset_fraction * (y[-1] - y[-2])

    X, Y = np.meshgrid(x, y)
    X[::2] += spacing / 2.0
    points = np.stack((X.flatten(), Y.flatten()), axis=1)
    points = points[points[:, 0] <= x_max, :]
    points = points[points[:, 1] <= y_max, :]
    diameter = np.sin(np.pi / 3) * 2 * spacing
    return points, diameter

def build_tile_graph(
    vertexes_fov,
    core_vertexes_fov,
    df_cells_fov,
    df_gridpoints_fov,
    genes,
    df_dots_fov=None,
    df_cells_features_fov=None,
    grid_triangle_side=150,
    cell_cell_strategy="knn-maxdist",
    cell_cell_k_neighbors=25,
    cell_cell_maxdist=90,  # in pixels
    dot_cell_strategy="knn-maxdist",
    dot_cell_k_neighbors=1,
    dot_cell_maxdist=70,  # in pixels
    pos_is_separate=True,
    use_stored_cell_assign=False,
    coarse_grid_side=7,
):
    """Generates the PyG HeteroData object of one tile.

    Parameters
    ----------
    vertexes_fov, core_vertexes_fov : tuple
        (x_min, x_max, y_min, y_max) of the tile and of its core.
    df_cells_fov : pd.DataFrame
        Cells of the tile (X, Y, is_core).
    df_gridpoints_fov : pd.DataFrame
        Gridpoints of the tile (X, Y, is_core).
    genes : list
        Gene order of the dot one-hot (dots path).
    df_dots_fov : pd.DataFrame or None
        Dots of the tile (X, Y, gene, is_core); dots path only.
    df_cells_features_fov : pd.DataFrame or None
        Per-cell counts (genes..., X, Y, is_core); AnnData path only.
    grid_triangle_side : float
        Side of the hexagonal grid triangles, sets the cell -> gridpoint radius.
    cell_cell_strategy, cell_cell_k_neighbors, cell_cell_maxdist
        Cell-cell edge construction (see ``knn_edges``).
    dot_cell_strategy, dot_cell_k_neighbors, dot_cell_maxdist
        Dot-cell edge construction (see ``assignment_edges``).
    pos_is_separate : bool
        If True, positions go in ``.pos`` and not in ``.x``.
    use_stored_cell_assign : bool
        Use the ``cell`` column of the dots as the dot-cell assignment.
    coarse_grid_side : int
        Side of the coarse (long-range) grid; 0 disables it.

    Returns
    -------
    data : HeteroData
    """
    # Identify the cell-cell edges
    cells_node_features, cells_edge_index = knn_edges(
        df_cells_fov,
        strategy=cell_cell_strategy,
        k_neighbors=cell_cell_k_neighbors,
        radius=cell_cell_maxdist,
    )
    if df_dots_fov is not None:
        one_hot = one_hot_encode_genes(df_dots_fov, genes)

        dots_pos = df_dots_fov[["X", "Y"]].values
        if pos_is_separate:
            dots_node_features = one_hot.astype(np.float32)
        else:
            dots_node_features = np.column_stack([dots_pos, one_hot]).astype(np.float32)

    # k=6 because the grid is hexagonal
    grid_node_features, grid_edge_index = knn_edges(
        df_gridpoints_fov,
        strategy="knn",
        k_neighbors=6,
        self_edges=False,
    )

    if df_dots_fov is not None:
        if use_stored_cell_assign:
            dc_edge_index = assignment_edges(
                df_dots_fov, df_cells_fov, strategy="precomputed"
            )
        else:
            dc_edge_index = assignment_edges(
                df_dots_fov,
                df_cells_fov,
                strategy=dot_cell_strategy,
                k_neighbors=dot_cell_k_neighbors,
                radius=dot_cell_maxdist,
            )

    cg_edge_index = cell_to_grid_edges(
        df_gridpoints_fov, df_cells_fov, radius=grid_triangle_side * 1.05
    )

    lr_grid_node_features = None
    lr_grid_edge_index = None
    c_lr_edge_index = None

    if coarse_grid_side > 0:
        # 1. Generate the grid points for this tile using the vertexes
        # We use N = M^2 to get an MxM grid
        lr_points, lr_diameter = hex_grid_by_count(vertexes_fov, N=coarse_grid_side**2)
        df_lr_grid = pd.DataFrame(lr_points, columns=["X", "Y"])

        # 2. Make Grid-Grid edges (Hexagonal approach, same as regular grid)
        # Using simple KNN with k=6 creates the hexagonal lattice topology
        lr_grid_node_features, lr_grid_edge_index = knn_edges(
            df_lr_grid,
            strategy="knn",
            k_neighbors=6, 
            self_edges=False
        )

        # 3. Make Cell-Grid edges ("is_watched_by")
        # We scale the radius based on the new diameter (spacing) of the coarse grid
        # lr_diameter/2 is effectively the 'triangle side' for this new grid
        c_lr_edge_index = cell_to_grid_edges(
            df_lr_grid, df_cells_fov, radius=(lr_diameter / 2) * 1.05
        )

    data = HeteroData()

    if pos_is_separate:
        
        if df_cells_features_fov is not None:
            gene_expression = df_cells_features_fov.values[:, :-3].astype(np.float32)
            data["cells"].x = torch.tensor(gene_expression, dtype=torch.float)
            # @TODO .x as the gene expression? 
            data["cells"].pos = torch.tensor(cells_node_features, dtype=torch.float)
        elif df_dots_fov is not None:
            data["dots"].x = torch.tensor(dots_node_features, dtype=torch.float)
            data["dots"].pos = torch.tensor(dots_pos, dtype=torch.float)
            
            data["cells"].pos = torch.tensor(cells_node_features, dtype=torch.float)            
        data["gridpoints"].pos = torch.tensor(grid_node_features, dtype=torch.float)
        if lr_grid_node_features is not None:
            data["longrange_grid"].pos = torch.tensor(lr_grid_node_features, dtype=torch.float)
    else:
        data["dots"].x = torch.tensor(
            dots_node_features, dtype=torch.float
        )  # it inclues the position

        data["cells"].x = torch.tensor(cells_node_features, dtype=torch.float)

        data["gridpoints"].x = torch.tensor(grid_node_features, dtype=torch.float)
        if lr_grid_node_features is not None:
             data["longrange_grid"].x = torch.tensor(lr_grid_node_features, dtype=torch.float)

    if df_dots_fov is not None:
        data["dots"].is_core = torch.tensor(df_dots_fov["is_core"].values, dtype=torch.bool)
        # data["dots", "is_close_to", "dots"].edge_index = torch.tensor(
        # dots_edge_index, dtype=torch.long
        data["dots", "could_come_from", "cells"].edge_index = torch.tensor(
            dc_edge_index, dtype=torch.long
        )
    data["cells"].is_core = torch.tensor(
        df_cells_fov["is_core"].values, dtype=torch.bool
    )
    data["gridpoints"].is_core = torch.tensor(
        df_gridpoints_fov["is_core"].values, dtype=torch.bool
    )

    data["cells", "is_close_to", "cells"].edge_index = torch.tensor(
        cells_edge_index, dtype=torch.long
    )
    data["cells", "is_watched_by", "gridpoints"].edge_index = torch.tensor(
        cg_edge_index, dtype=torch.long
    )
    data["gridpoints", "is_close_to", "gridpoints"].edge_index = torch.tensor(
        grid_edge_index, dtype=torch.long
    )

    data["cells"].num_nodes = cells_node_features.shape[0]

    if lr_grid_edge_index is not None:
        data["longrange_grid", "is_close_to", "longrange_grid"].edge_index = torch.tensor(
            lr_grid_edge_index, dtype=torch.long
        )
        data["cells", "is_watched_by", "longrange_grid"].edge_index = torch.tensor(
            c_lr_edge_index, dtype=torch.long
        )

    # Add graph levels attributes
    data.vertexes_fov = vertexes_fov
    data.core_vertexes_fov = core_vertexes_fov

    return data
