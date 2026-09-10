from collections import defaultdict

import numpy as np
import torch
import tqdm
from scipy import sparse

__all__ = ["predict", "cell_cell_interactions", "umap"]


def _top_k(probs, k):
    """CSR matrix (cells x types) keeping the k largest probabilities of every cell, renormalised to sum 1."""
    k = min(int(k), probs.shape[1])
    values, index = torch.topk(probs, k, dim=1)
    # ties at the maximum may leave the argmax (first maximal index) out of the selection: keep it in slot 0
    argmax = probs.argmax(dim=1)
    missing = ~(index == argmax[:, None]).any(dim=1)
    index[missing, 0] = argmax[missing]
    values = (values / values.sum(dim=1, keepdim=True)).cpu().numpy()
    indptr = np.arange(0, probs.shape[0] * k + 1, k)
    matrix = sparse.csr_matrix((values.ravel(), index.cpu().numpy().ravel(), indptr), shape=tuple(probs.shape))
    matrix.sort_indices()
    return matrix


def predict(net, graphs, top_k=5):
    """
    Run the network in evaluation mode on every graph and stack the outputs.

    Parameters:
        net: The trained model (its device and ``temperature`` buffer are used).
        graphs: List of HeteroData graphs as returned by ``generate_graphs``.
        top_k: Keep only the ``top_k`` largest probabilities of every cell, renormalised
            to sum 1, and return ``all_probs`` as a ``scipy.sparse`` CSR matrix
            (cells x types) instead of a dense array; ``None`` keeps every probability
            in a dense array. ``all_labels`` is the argmax either way.

    Returns:
        A dictionary with, per core cell: all_labels, all_cell_types, all_probs,
        all_logits, all_positions, all_scales, expression_matrices,
        embeddings_cell, unique_cell_ids, all_sections, all_timepoints,
        all_conditions; per grid node: embeddings_grid, grid_positions,
        grid_sections, grid_timepoints; per graph: label_list; plus the
        cell_types names and the section_ids.
    """
    if len(graphs) == 0:
        raise ValueError("predict needs at least one graph")
    if top_k is not None and int(top_k) < 1:
        raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
    device = next(net.parameters()).device
    temperature = float(net.temperature)
    cell_types = list(graphs[0].cell_types)

    logit_list = []
    probs_list = []
    label_list = []
    expression_matrices = []
    all_positions = []
    all_grid_positions = []
    embeddings_cell_list = []
    embeddings_grid_list = []
    all_scales = []
    all_timepoints = []
    all_sections = []
    all_conditions = []
    all_unique_cell_ids = []
    all_grid_sections = []
    all_grid_timepoints = []

    net.eval()
    with torch.no_grad():
        for graph in tqdm.tqdm(graphs, desc="Predicting"):
            graph.to(device)
            counts, logits, out_cell, out_grid, scale, _ = net(graph)
            is_core = graph["cells"].is_core
            pos = graph["cells"].pos[is_core].detach().cpu().numpy()
            pos_grid = graph["longrange_grid"].pos.detach().cpu().numpy()
            all_positions.append(pos)
            all_grid_positions.append(pos_grid)
            embeddings_cell_list.append(out_cell[is_core].detach().cpu().numpy())
            embeddings_grid_list.append(out_grid.detach().cpu().numpy())
            logit_list.append(logits[is_core].detach().cpu().numpy())
            probs = torch.softmax(logits[is_core] / temperature, dim=1).detach()
            probs_list.append(probs.cpu().numpy() if top_k is None else _top_k(probs, top_k))
            expression_matrices.append(counts[is_core].detach().cpu().numpy())
            all_scales.append(scale[is_core].detach().cpu().numpy())
            all_unique_cell_ids.append(np.asarray(graph.unique_cell_ids)[is_core.detach().cpu().numpy()])
            label_list.append(torch.argmax(logits[is_core], dim=1).cpu().numpy())
            graph.to("cpu")

            all_sections.append(np.array([graph.section_label] * len(pos)))
            all_timepoints.append(np.array([graph.timepoint_label] * len(pos)))
            all_conditions.append(np.array([graph.condition_label] * len(pos)))
            all_grid_sections.append(np.array([graph.section_label] * len(pos_grid)))
            all_grid_timepoints.append(np.array([graph.timepoint_label] * len(pos_grid)))

    all_labels = np.hstack(label_list)
    all_sections = np.hstack(all_sections)

    return {
        "all_labels": all_labels,
        "all_cell_types": np.asarray(cell_types)[all_labels],
        "all_probs": np.vstack(probs_list) if top_k is None else sparse.vstack(probs_list, format="csr"),
        "all_logits": np.vstack(logit_list),
        "all_positions": np.vstack(all_positions),
        "all_scales": np.vstack(all_scales),
        "expression_matrices": np.vstack(expression_matrices),
        "embeddings_cell": np.vstack(embeddings_cell_list),
        "unique_cell_ids": np.hstack(all_unique_cell_ids),
        "all_sections": all_sections,
        "all_timepoints": np.hstack(all_timepoints),
        "all_conditions": np.hstack(all_conditions),
        "embeddings_grid": np.vstack(embeddings_grid_list),
        "grid_positions": np.vstack(all_grid_positions),
        "grid_sections": np.hstack(all_grid_sections),
        "grid_timepoints": np.hstack(all_grid_timepoints),
        "label_list": label_list,
        "cell_types": cell_types,
        "section_ids": list(np.unique(all_sections)),
    }


def cell_cell_interactions(predictions, graphs, return_normalized=False):
    """
    Count predicted cell-type pairs across the cell-cell edges of each section.

    Only edges between two core cells are counted, so cells in the overlapping
    tile frames are not double-counted. Matrices are averaged over the tiles of
    a section.

    Parameters:
        predictions: Dictionary returned by predict().
        graphs: The graphs that were passed to predict().
        return_normalized: Divide each tile's matrix by its edge count first.

    Returns:
        {section_label: (n_types, n_types) matrix}
    """
    n_types = len(predictions["cell_types"])
    matrices_by_section = defaultdict(list)

    for graph, labels in zip(graphs, predictions["label_list"]):
        is_core = graph["cells"].is_core.detach().cpu().numpy().astype(bool)
        core_index = np.cumsum(is_core) - 1  # original index -> core index
        src, tgt = graph["cells", "is_close_to", "cells"].edge_index.detach().cpu().numpy()
        both_core = is_core[src] & is_core[tgt]
        src, tgt = src[both_core], tgt[both_core]

        matrix = np.zeros((n_types, n_types), dtype=int)
        np.add.at(matrix, (labels[core_index[src]], labels[core_index[tgt]]), 1)
        if return_normalized and len(src) > 0:
            matrix = matrix / len(src)
        matrices_by_section[graph.section_label].append(matrix)

    return {sid: np.mean(ms, axis=0) for sid, ms in matrices_by_section.items()}


def umap(
    results,
    key,
    subsample_rate=None,
    random_state=42,
    n_neighbors=30,
    min_dist=0.1,
    metric='cosine',
    low_memory=True
):
    """
    Compute 2D UMAP embeddings once, with optional subsampling.

    Parameters
    ----------
    results : dict
        Output of ``predict``; ``results["embeddings_cell"]`` is embedded.
    key : array-like or list of array-like
        Per-cell annotation(s) (e.g. ``results["all_cell_types"]``) returned
        alongside the embedding, subsampled the same way.
    subsample_rate : int, optional
        Keep every ``subsample_rate``-th cell (None keeps all).
    random_state : int
        Random seed for reproducibility.
    n_neighbors, min_dist, metric, low_memory : passed to UMAP.

    Returns
    -------
    X_umap : np.ndarray, shape (n_output, 2)
        2D UMAP coordinates.
    idx : np.ndarray or None
        Indices of sampled points, or None if no subsampling.
    """

    from umap import UMAP

    embeddings = results['embeddings_cell']
    X = np.asarray(embeddings)
    n = X.shape[0]
    idx = None
    if subsample_rate is not None:
        if isinstance(subsample_rate, float):
            k = int(subsample_rate)
        else:
            k = subsample_rate
        idx = np.arange(0, n, k)
        X = X[idx]

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=low_memory
    )
    X_umap = reducer.fit_transform(X)

    # Adjust keys if subsampled
    if isinstance(key, (list, tuple)):
        keys = [np.asarray(k)[idx] if idx is not None else np.asarray(k)
                for k in key]
    else:
        keys = np.asarray(key)
        if idx is not None:
            keys = keys[idx]

    return X_umap, keys
