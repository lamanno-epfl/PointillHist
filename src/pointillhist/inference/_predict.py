import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from scipy import sparse

from ..preprocessing._store import _graph_values

__all__ = ["predict", "read_predictions", "cell_cell_interactions", "umap"]


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


#: per-cell and per-grid-node entries of the dict ``predict`` returns, in its order
CELL_FIELDS = ("all_cell_types", "all_probs", "all_logits", "all_positions", "all_scales",
               "expression_matrices", "embeddings_cell", "unique_cell_ids", "all_sections",
               "all_timepoints", "all_conditions")
GRID_FIELDS = ("embeddings_grid", "grid_positions", "grid_sections", "grid_timepoints")
#: entries that are always returned
ALWAYS = ("all_labels", "label_list", "cell_types", "section_ids")
#: what ``predict(..., out=...)`` writes when ``fields`` is not given (about 0.1 kB per cell)
COMPACT_FIELDS = ("all_cell_types", "all_probs", "all_positions", "unique_cell_ids",
                  "all_sections", "all_timepoints", "all_conditions")
FORMAT = 1
META_FILE = "predictions.pt"


def _fields(fields, default):
    """The optional entries asked for, in the order of the returned dict."""
    if fields is None:
        fields = default
    elif isinstance(fields, str):
        fields = [fields]
    fields = list(fields)
    unknown = [f for f in fields if f not in CELL_FIELDS + GRID_FIELDS + ALWAYS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; the fields are {list(CELL_FIELDS + GRID_FIELDS)} "
                         f"(and {list(ALWAYS)}, which are always returned)")
    return [f for f in CELL_FIELDS + GRID_FIELDS if f in fields]


def _labels_array(label, n):
    """The per-cell label array ``predict`` returns for a graph (np.array of ``n`` copies)."""
    return np.array([label] * n)


def _section_ids(labels_and_counts):
    """``list(np.unique(all_sections))`` without building the per-cell array (same dtype promotion)."""
    return list(np.unique(np.hstack([_labels_array(label, min(n, 1)) for label, n in labels_and_counts])))


def _predict_graph(net, graph, device, temperature, top_k, wanted):
    """The outputs of one graph for its core cells (numpy / scipy), restricted to ``wanted`` plus the labels."""
    graph.to(device)
    counts, logits, out_cell, out_grid, scale, _ = net(graph)
    is_core = graph["cells"].is_core
    pos = graph["cells"].pos[is_core].detach().cpu().numpy()
    pos_grid = graph["longrange_grid"].pos.detach().cpu().numpy()
    piece = {"n_cells": len(pos), "n_grid": len(pos_grid)}
    if "all_positions" in wanted:
        piece["all_positions"] = pos
    if "grid_positions" in wanted:
        piece["grid_positions"] = pos_grid
    if "embeddings_cell" in wanted:
        piece["embeddings_cell"] = out_cell[is_core].detach().cpu().numpy()
    if "embeddings_grid" in wanted:
        piece["embeddings_grid"] = out_grid.detach().cpu().numpy()
    if "all_logits" in wanted:
        piece["all_logits"] = logits[is_core].detach().cpu().numpy()
    if "all_probs" in wanted:
        probs = torch.softmax(logits[is_core] / temperature, dim=1).detach()
        piece["all_probs"] = probs.cpu().numpy() if top_k is None else _top_k(probs, top_k)
    if "expression_matrices" in wanted:
        piece["expression_matrices"] = counts[is_core].detach().cpu().numpy()
    if "all_scales" in wanted:
        piece["all_scales"] = scale[is_core].detach().cpu().numpy()
    if "unique_cell_ids" in wanted:
        piece["unique_cell_ids"] = np.asarray(graph.unique_cell_ids)[is_core.detach().cpu().numpy()]
    piece["all_labels"] = torch.argmax(logits[is_core], dim=1).cpu().numpy()
    graph.to("cpu")
    piece["labels"] = (graph.section_label, graph.timepoint_label, graph.condition_label)
    return piece


def _assemble(pieces, cell_types, top_k, wanted):
    """The dict of ``predict`` from per-graph pieces (each with the labels and the ``wanted`` entries)."""
    label_list = [p["all_labels"] for p in pieces]
    all_labels = np.hstack(label_list)
    per_cell = {
        "all_sections": lambda p: _labels_array(p["labels"][0], p["n_cells"]),
        "all_timepoints": lambda p: _labels_array(p["labels"][1], p["n_cells"]),
        "all_conditions": lambda p: _labels_array(p["labels"][2], p["n_cells"]),
        "grid_sections": lambda p: _labels_array(p["labels"][0], p["n_grid"]),
        "grid_timepoints": lambda p: _labels_array(p["labels"][1], p["n_grid"]),
    }
    result = {"all_labels": all_labels}
    for field in CELL_FIELDS + GRID_FIELDS:
        if field not in wanted:
            continue
        if field == "all_cell_types":
            result[field] = np.asarray(cell_types)[all_labels]
        elif field in per_cell:
            result[field] = np.hstack([per_cell[field](p) for p in pieces])
        elif field == "all_probs" and top_k is not None:
            result[field] = sparse.vstack([p[field] for p in pieces], format="csr")
        elif field == "unique_cell_ids":
            result[field] = np.hstack([p[field] for p in pieces])
        else:
            result[field] = np.vstack([p[field] for p in pieces])
    result["label_list"] = label_list
    result["cell_types"] = cell_types
    if "all_sections" in result:
        result["section_ids"] = list(np.unique(result["all_sections"]))
    else:
        result["section_ids"] = _section_ids((p["labels"][0], p["n_cells"]) for p in pieces)
    return result


def predict(net, graphs, top_k=5, out=None, fields=None):
    """
    Run the network in evaluation mode on every graph and stack the outputs.

    Parameters:
        net: The trained model (its device and ``temperature`` buffer are used).
        graphs: List of HeteroData graphs as returned by ``generate_graphs``, or a
            ``DiskGraphs``, whose graphs are loaded one at a time.
        top_k: Keep only the ``top_k`` largest probabilities of every cell, renormalised
            to sum 1, and return ``all_probs`` as a ``scipy.sparse`` CSR matrix
            (cells x types) instead of a dense array; ``None`` keeps every probability
            in a dense array. ``all_labels`` is the argmax either way.
        out: None (default) returns the outputs in memory. A folder (absent or empty)
            writes them there graph by graph, so memory holds one graph's outputs at a
            time (needs ``pyarrow``, the ``parquet`` extra); read them back with
            ``read_predictions``. ``<out>/cells`` is a Parquet dataset with one row per
            core cell (graph, label, cell_type, cell_id, section, timepoint, condition,
            x, y[, z], top_types / top_probs or probs, and logits, scale, expression,
            embedding when asked for), readable directly with
            ``pandas.read_parquet``; ``<out>/grid`` holds the grid-node fields, if any
            were asked for; ``predictions.pt`` the rest. Under torchrun
            (a process group initialised, e.g. by ``train_distributed``) every process
            must call ``predict`` with the same ``out``: each one predicts its share of
            the graphs on its own device.
        fields: The optional entries to return or write (see below). None: all of them
            in memory; with ``out``, only ``all_cell_types``, ``all_probs``,
            ``all_positions``, ``unique_cell_ids``, ``all_sections``,
            ``all_timepoints`` and ``all_conditions`` (about 0.1 kB per cell; the
            logits, scales, expression and embeddings take a few kB per cell).

    Returns:
        Without ``out``, a dictionary with, per core cell: all_labels, all_cell_types,
        all_probs, all_logits, all_positions, all_scales, expression_matrices,
        embeddings_cell, unique_cell_ids, all_sections, all_timepoints,
        all_conditions; per grid node: embeddings_grid, grid_positions,
        grid_sections, grid_timepoints; per graph: label_list; plus the cell_types
        names and the section_ids. ``all_labels``, ``label_list``, ``cell_types`` and
        ``section_ids`` are always there; ``fields`` selects the others.
        With ``out``, the folder path.
    """
    if len(graphs) == 0:
        raise ValueError("predict needs at least one graph")
    if top_k is not None and int(top_k) < 1:
        raise ValueError(f"top_k must be a positive integer or None, got {top_k!r}")
    wanted = _fields(fields, CELL_FIELDS + GRID_FIELDS if out is None else COMPACT_FIELDS)
    device = next(net.parameters()).device
    temperature = float(net.temperature)
    first = graphs[0]
    cell_types = list(first.cell_types)
    positions = first["cells"].pos   # its width and dtype set the position columns of the Parquet output
    positions = (positions.shape[1], positions.dtype)
    del first
    if out is not None:
        return _predict_to_parquet(net, graphs, positions, top_k, os.fspath(out), wanted, device, temperature,
                                   cell_types)

    pieces = []
    net.eval()
    with torch.no_grad():
        for graph in tqdm.tqdm(graphs, desc="Predicting"):
            pieces.append(_predict_graph(net, graph, device, temperature, top_k, wanted))
            del graph   # a DiskGraphs graph is released before the next one is loaded
    return _assemble(pieces, cell_types, top_k, wanted)


# ---------------------------------------------------------------- Parquet output
def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("predictions written to disk need pyarrow: "
                          'pip install "pointillhist[parquet]" (or pip install pyarrow)') from exc
    return pa, pq


def _shard_path(out, kind, rank, world_size):
    """``<out>/cells`` and ``<out>/grid`` are Parquet datasets with one part per process."""
    return os.path.join(out, kind, f"part-{rank:05d}-of-{world_size:05d}.parquet")


def _temporary(path):
    # a leading underscore: Parquet readers skip the part while it is being written
    folder, name = os.path.split(path)
    return os.path.join(folder, f"_{name}.tmp{os.getpid()}")


def _finished_output(out):
    """Why ``out`` cannot receive predictions (an empty string if it can)."""
    if not os.path.exists(out):
        return ""
    if not os.path.isdir(out):
        return f"{out} exists and is not a folder"
    names = os.listdir(out)
    for kind in ("cells", "grid"):
        folder = os.path.join(out, kind)
        if os.path.isdir(folder):
            names += os.listdir(folder)
    if META_FILE in names or any(n.endswith(".parquet") for n in names):
        return f"{out} already holds predictions; remove it or choose another folder"
    return ""


def _schema(pa, wanted, top_k, n_types, positions, net):
    """Arrow schemas of the per-cell and per-grid-node tables."""
    pos_dim, pos_dtype = positions
    pos_type = pa.from_numpy_dtype(torch.empty(0, dtype=pos_dtype).numpy().dtype)
    axes = ["x", "y", "z"][:pos_dim] if pos_dim <= 3 else [f"pos{i}" for i in range(pos_dim)]
    cells = [("graph", pa.int32()), ("label", pa.int64())]
    if "all_cell_types" in wanted:
        cells.append(("cell_type", pa.string()))
    if "unique_cell_ids" in wanted:
        cells.append(("cell_id", pa.string()))
    for field, column in (("all_sections", "section"), ("all_timepoints", "timepoint"),
                          ("all_conditions", "condition")):
        if field in wanted:
            cells.append((column, pa.string()))
    if "all_positions" in wanted:
        cells += [(axis, pos_type) for axis in axes]
    if "all_probs" in wanted:
        if top_k is None:
            cells.append(("probs", pa.list_(pa.float32(), n_types)))
        else:
            k = min(int(top_k), n_types)
            cells += [("top_types", pa.list_(pa.int32(), k)), ("top_probs", pa.list_(pa.float32(), k))]
    if "all_logits" in wanted:
        cells.append(("logits", pa.list_(pa.float32(), n_types)))
    if "all_scales" in wanted:
        cells.append(("scale", pa.float32()))
    if "expression_matrices" in wanted:
        cells.append(("expression", pa.list_(pa.float32(), net.n_genes)))
    if "embeddings_cell" in wanted:
        cells.append(("embedding", pa.list_(pa.float32(), net.hidden_size)))
    grid = [("graph", pa.int32())]
    for field, column in (("grid_sections", "section"), ("grid_timepoints", "timepoint")):
        if field in wanted:
            grid.append((column, pa.string()))
    if "grid_positions" in wanted:
        grid += [(axis, pos_type) for axis in axes]
    if "embeddings_grid" in wanted:
        grid.append(("embedding", pa.list_(pa.float32(), net.hidden_size)))
    return pa.schema(cells), (pa.schema(grid) if len(grid) > 1 else None), axes


def _fixed_list(pa, array, width, value_type):
    return pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(array).reshape(-1), type=value_type), width)


def _tables(pa, piece, position, schema, grid_schema, axes):
    """The per-cell and per-grid-node Arrow tables of one graph."""
    n, n_grid = piece["n_cells"], piece["n_grid"]
    section, timepoint, condition = (str(label) for label in piece["labels"])
    columns = {"graph": pa.array(np.full(n, position, dtype=np.int32)),
               "label": pa.array(piece["all_labels"], type=pa.int64())}
    names = schema.names
    if "cell_type" in names:
        columns["cell_type"] = pa.array(piece["cell_types"], type=pa.string())
    if "cell_id" in names:
        ids = piece["unique_cell_ids"]
        columns["cell_id"] = pa.array([str(i) for i in ids] if ids.dtype == object else ids.astype(str),
                                      type=pa.string())
    for column, value in (("section", section), ("timepoint", timepoint), ("condition", condition)):
        if column in names:
            columns[column] = pa.array([value] * n, type=pa.string())
    if axes[0] in names:
        pos = piece["all_positions"]
        for i, axis in enumerate(axes):
            columns[axis] = pa.array(np.ascontiguousarray(pos[:, i]), type=schema.field(axis).type)
    if "probs" in names:
        columns["probs"] = _fixed_list(pa, piece["all_probs"], schema.field("probs").type.list_size, pa.float32())
    if "top_types" in names:
        matrix = piece["all_probs"]
        k = schema.field("top_types").type.list_size
        columns["top_types"] = _fixed_list(pa, matrix.indices.astype(np.int32), k, pa.int32())
        columns["top_probs"] = _fixed_list(pa, matrix.data, k, pa.float32())
    for field, column in (("all_logits", "logits"), ("expression_matrices", "expression"),
                          ("embeddings_cell", "embedding")):
        if column in names:
            columns[column] = _fixed_list(pa, piece[field], schema.field(column).type.list_size, pa.float32())
    if "scale" in names:
        columns["scale"] = pa.array(piece["all_scales"].reshape(-1), type=pa.float32())
    cells = pa.table([columns[name] for name in names], schema=schema)
    grid = None
    if grid_schema is not None:
        columns = {"graph": pa.array(np.full(n_grid, position, dtype=np.int32))}
        for column, value in (("section", section), ("timepoint", timepoint)):
            if column in grid_schema.names:
                columns[column] = pa.array([value] * n_grid, type=pa.string())
        if axes[0] in grid_schema.names:
            for i, axis in enumerate(axes):
                columns[axis] = pa.array(np.ascontiguousarray(piece["grid_positions"][:, i]),
                                         type=grid_schema.field(axis).type)
        if "embedding" in grid_schema.names:
            columns["embedding"] = _fixed_list(pa, piece["embeddings_grid"],
                                               grid_schema.field("embedding").type.list_size, pa.float32())
        grid = pa.table([columns[name] for name in grid_schema.names], schema=grid_schema)
    return cells, grid


def _predict_to_parquet(net, graphs, positions, top_k, out, wanted, device, temperature, cell_types):
    pa, pq = _pyarrow()
    distributed = dist.is_available() and dist.is_initialized()
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    problem = [_finished_output(out) if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(problem, src=0)
    if problem[0]:
        raise FileExistsError(problem[0])
    os.makedirs(out, exist_ok=True)

    written = list(wanted)
    schema, grid_schema, axes = _schema(pa, written, top_k, len(cell_types), positions, net)
    info = json.dumps({"cell_types": cell_types, "fields": written, "top_k": top_k,
                       "rank": rank, "world_size": world_size})
    schema = schema.with_metadata({"pointillhist": info})
    grid_schema = None if grid_schema is None else grid_schema.with_metadata({"pointillhist": info})
    paths = {"cells": _shard_path(out, "cells", rank, world_size)}
    if grid_schema is not None:
        paths["grid"] = _shard_path(out, "grid", rank, world_size)
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    writers = {kind: pq.ParquetWriter(_temporary(path), schema if kind == "cells" else grid_schema)
               for kind, path in paths.items()}
    done = []
    names = np.asarray(cell_types)
    net.eval()
    try:
        with torch.no_grad():
            positions = range(rank, len(graphs), world_size)
            for position in tqdm.tqdm(positions, desc="Predicting", disable=rank != 0):
                piece = _predict_graph(net, graphs[position], device, temperature, top_k, set(written))
                if "all_cell_types" in written:
                    piece["cell_types"] = names[piece["all_labels"]]
                cells, grid = _tables(pa, piece, position, schema, grid_schema, axes)
                if len(cells):
                    writers["cells"].write_table(cells, row_group_size=len(cells))
                if grid is not None and len(grid):
                    writers["grid"].write_table(grid, row_group_size=len(grid))
                ids = piece.get("unique_cell_ids")
                done.append((position, piece["n_cells"], piece["n_grid"],
                             None if ids is None else ids.dtype.str if ids.dtype != object else "object"))
                del piece, cells, grid
    finally:
        for writer in writers.values():
            writer.close()
    for kind, path in paths.items():
        os.replace(_temporary(path), path)

    shares = [done]
    if distributed:
        shares = [None] * world_size
        dist.all_gather_object(shares, done)
    if rank == 0:
        rows = sorted((row for share in shares for row in share), key=lambda row: row[0])
        if [row[0] for row in rows] != list(range(len(graphs))):
            raise RuntimeError("the processes did not predict every graph once; were the graphs the same everywhere?")
        labels = zip(*(_graph_values(graphs, name) for name in ("section_label", "timepoint_label",
                                                                 "condition_label")))
        meta = {
            "format": FORMAT, "n_graphs": len(graphs), "world_size": world_size, "fields": written,
            "top_k": top_k, "cell_types": cell_types, "axes": axes, "grid": grid_schema is not None,
            "pos_dtype": schema.field(axes[0]).type.to_pandas_dtype() if axes[0] in schema.names else
            (grid_schema.field(axes[0]).type.to_pandas_dtype() if grid_schema is not None
             and axes[0] in grid_schema.names else None),
            "n_genes": net.n_genes, "hidden_size": net.hidden_size,
            "graphs": [dict(section_label=s, timepoint_label=t, condition_label=c, n_cells=n, n_grid=m,
                            ids_dtype=d)
                       for (s, t, c), (_, n, m, d) in zip(labels, rows)],
        }
        tmp = os.path.join(out, f"{META_FILE}.tmp{os.getpid()}")
        torch.save(meta, tmp)
        os.replace(tmp, os.path.join(out, META_FILE))
    if distributed:
        dist.barrier()
    return out


def read_predictions(out, fields=None, sections=None):
    """
    Read predictions written by ``predict(..., out=out)`` into the dict ``predict`` returns.

    Parameters:
        out: The folder given to ``predict``.
        fields: The optional entries to read, among those written (None: all written ones).
            ``all_labels``, ``label_list``, ``cell_types`` and ``section_ids`` are always read.
        sections: None, or section labels (``all_sections`` values) to read; the other
            graphs are skipped, so only the requested cells are held in memory.

    Returns:
        The dict of ``predict`` for the graphs read, in graph order; for all sections and
        fields it equals what ``predict`` returns in memory.
    """
    pa, pq = _pyarrow()
    out = os.fspath(out)
    meta_path = os.path.join(out, META_FILE)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found: no predictions there, or their writing did not finish")
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{out}: predictions format {meta.get('format')!r}, this version reads format {FORMAT}")
    written = meta["fields"]
    wanted = _fields(fields, written)
    missing = [f for f in wanted if f not in written]
    if missing:
        raise ValueError(f"field(s) {missing} were not written to {out}; written: {written}")
    kinds = ["cells"] + (["grid"] if meta["grid"] else [])
    shards = {kind: [_shard_path(out, kind, r, meta["world_size"]) for r in range(meta["world_size"])]
              for kind in kinds}
    absent = [p for paths in shards.values() for p in paths if not os.path.exists(p)]
    if absent:
        raise FileNotFoundError(f"missing prediction file(s) {absent}")

    graph_rows = meta["graphs"]
    selected = [g for g, row in enumerate(graph_rows)
                if sections is None or any(_same_label(row["section_label"], s) for s in sections)]
    wanted_set = set(wanted)
    columns = {"cells": ["graph", "label"], "grid": ["graph"]}
    top_k = meta["top_k"]
    if "all_positions" in wanted_set:
        columns["cells"] += meta["axes"]
    if "all_probs" in wanted_set:
        columns["cells"] += ["probs"] if top_k is None else ["top_types", "top_probs"]
    for field, column in (("all_logits", "logits"), ("all_scales", "scale"),
                          ("expression_matrices", "expression"), ("embeddings_cell", "embedding"),
                          ("unique_cell_ids", "cell_id")):
        if field in wanted_set:
            columns["cells"].append(column)
    if "grid_positions" in wanted_set:
        columns["grid"] += meta["axes"]
    if "embeddings_grid" in wanted_set:
        columns["grid"].append("embedding")
    need_grid = meta["grid"] and len(columns["grid"]) > 1

    tables = {kind: _row_groups(pq, shards[kind], set(selected), columns[kind])
              for kind in (["cells", "grid"] if need_grid else ["cells"])}
    n_types = len(meta["cell_types"])
    pieces = []
    for g in selected:
        row = graph_rows[g]
        piece = {"n_cells": row["n_cells"], "n_grid": row["n_grid"],
                 "labels": (row["section_label"], row["timepoint_label"], row["condition_label"])}
        cells = tables["cells"].get(g)
        n = row["n_cells"]
        piece["all_labels"] = np.array(_column(cells, "label", np.int64))
        if "all_positions" in wanted_set:
            piece["all_positions"] = _positions(cells, meta["axes"], meta["pos_dtype"])
        if "all_probs" in wanted_set:
            if top_k is None:
                piece["all_probs"] = _matrix(cells, "probs", n, n_types)
            else:
                k = min(int(top_k), n_types)
                indices = _matrix(cells, "top_types", n, k, np.int32).astype(np.int64).ravel()
                values = _matrix(cells, "top_probs", n, k).ravel()
                matrix = sparse.csr_matrix((values, indices, np.arange(0, n * k + 1, k)), shape=(n, n_types))
                matrix.sort_indices()
                piece["all_probs"] = matrix
        if "all_logits" in wanted_set:
            piece["all_logits"] = _matrix(cells, "logits", n, n_types)
        if "all_scales" in wanted_set:
            piece["all_scales"] = _column(cells, "scale", np.float32).reshape(-1, 1)
        if "expression_matrices" in wanted_set:
            piece["expression_matrices"] = _matrix(cells, "expression", n, meta["n_genes"])
        if "embeddings_cell" in wanted_set:
            piece["embeddings_cell"] = _matrix(cells, "embedding", n, meta["hidden_size"])
        if "unique_cell_ids" in wanted_set:
            piece["unique_cell_ids"] = _ids(cells, row["ids_dtype"])
        if need_grid:
            grid = tables["grid"].get(g)
            if "grid_positions" in wanted_set:
                piece["grid_positions"] = _positions(grid, meta["axes"], meta["pos_dtype"])
            if "embeddings_grid" in wanted_set:
                piece["embeddings_grid"] = _matrix(grid, "embedding", row["n_grid"], meta["hidden_size"])
        pieces.append(piece)
    if not pieces:
        raise ValueError(f"no graph of {out} belongs to the sections {list(sections)}")
    return _assemble(pieces, meta["cell_types"], top_k, wanted)


def _same_label(label, wanted):
    try:
        return bool(label == wanted)
    except Exception:
        return False


def _row_groups(pq, paths, selected, columns):
    """{graph position: Arrow table} for the selected graphs, read row group by row group."""
    tables = {}
    for path in paths:
        handle = pq.ParquetFile(path)
        graph_column = handle.schema_arrow.get_field_index("graph")
        for i in range(handle.num_row_groups):
            statistics = handle.metadata.row_group(i).column(graph_column).statistics
            if statistics is not None and statistics.has_min_max:
                position = int(statistics.min)
                if position not in selected:
                    continue
                table = handle.read_row_group(i, columns=columns)
            else:
                table = handle.read_row_group(i, columns=columns)
                position = int(table.column("graph")[0].as_py())
                if position not in selected:
                    continue
            tables[position] = table if position not in tables else _concat_tables([tables[position], table])
    return tables


def _concat_tables(tables):
    import pyarrow as pa
    return pa.concat_tables(tables)


def _column(table, name, dtype):
    """A 1-D column as numpy (empty of ``dtype`` for a graph without rows)."""
    if table is None:
        return np.zeros(0, dtype=dtype)
    return table.column(name).to_numpy()


def _positions(table, axes, dtype):
    if table is None:
        return np.zeros((0, len(axes)), dtype=dtype)
    return np.column_stack([table.column(axis).to_numpy() for axis in axes])


def _matrix(table, name, n, width, dtype=np.float32):
    """A fixed-size list column as an (n, width) array."""
    if table is None:
        return np.zeros((0, width), dtype=dtype)
    column = table.column(name).combine_chunks()
    return column.flatten().to_numpy().reshape(n, column.type.list_size)


def _ids(table, dtype):
    """Cell ids, stored as strings, back in their numpy dtype."""
    if table is None:
        values = np.zeros(0, dtype=object)
    else:
        values = table.column("cell_id").to_numpy(zero_copy_only=False)
    if dtype == "object" or dtype is None:
        return values.astype(object)
    if np.dtype(dtype).kind == "b":
        return values == "True"
    return values.astype(np.dtype(dtype))

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
