import bisect
import json
import math
import os
import time
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from scipy import sparse

from ..preprocessing._store import DiskGraphs, _graph_values

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
#: a row group of the Parquet parts is written once it holds this many bytes or rows
ROW_GROUP_BYTES = 64 * 2**20
ROW_GROUP_ROWS = 1_000_000


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


def _predict_graph(net, graph, device, temperature, top_k, wanted, restore=True):
    """The outputs of one graph for its core cells (numpy / scipy), restricted to ``wanted`` plus the labels.
    The graph is moved to ``device`` and, with ``restore``, back to the CPU afterwards."""
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
    if restore:
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
        out: None (default) returns the outputs in memory. A folder that does not exist
            or is empty: the outputs are written there graph by graph, so memory holds one
            graph's outputs at a time (needs ``pyarrow``, the ``parquet`` extra); read
            them back with ``read_predictions``. ``<out>/cells`` is a Parquet dataset with
            one row per core cell, readable directly with ``pandas.read_parquet``: graph
            (position in ``graphs``), label (index into ``cell_types``), cell_type,
            cell_id (as text), section, timepoint, condition (as text), x, y[, z];
            top_types and top_probs, the ``top_k`` most probable type indices and their
            renormalised probabilities, most probable first (probs, every probability,
            with ``top_k=None``); and logits, scale, expression, embedding when asked
            for. ``<out>/grid`` holds the grid-node fields, if any were asked for, and
            ``predictions.pt`` the rest (the cell types among them). If writing fails in
            a single process, what was written is removed. Under torchrun (a process
            group initialised, e.g. by ``train_distributed``) every process must call
            ``predict`` with the same ``out``, on a file system all of them read, and
            the same graphs: each one predicts its share of the graphs on the device of
            its ``net`` and writes one part, then returns; process 0 returns once every
            part is written and the output is complete, or raises if another process
            failed (the folder then keeps what was written and must be removed before
            a new run).
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
        return _predict_to_parquet(net, graphs, positions, top_k, _output_folder(out), wanted, device,
                                   temperature, cell_types)

    pieces = []
    net.eval()
    # a list's graphs are moved back to the CPU as before; a DiskGraphs graph is dropped once predicted
    restore = not isinstance(graphs, DiskGraphs)
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(graphs)), desc="Predicting"):
            pieces.append(_predict_graph(net, graphs[i], device, temperature, top_k, wanted, restore))
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


def _is_temporary(name):
    return name.startswith("_part-") and ".parquet.tmp" in name


def _output_problem(out):
    """Why ``out`` cannot receive predictions (an empty string if it can): it must be absent, empty,
    or hold nothing but the temporary parts an interrupted ``predict`` may have left."""
    if not os.path.exists(out):
        return ""
    if not os.path.isdir(out):
        return f"{out} exists and is not a folder"
    names = os.listdir(out)
    inner = []
    for kind in ("cells", "grid"):
        if kind in names and os.path.isdir(os.path.join(out, kind)):
            inner += os.listdir(os.path.join(out, kind))
    if META_FILE in names:
        return f"{out} already holds predictions; remove it or choose another folder"
    if any(n.endswith(".parquet") for n in inner) or any(n.startswith(("_rows-", "_failed-", "_writing-"))
                                                          for n in names):
        return (f"{out} holds an unfinished prediction output (no {META_FILE}: a run was interrupted or "
                "one of its processes failed); remove it or choose another folder")
    if set(names) - {"cells", "grid"} or not all(_is_temporary(n) for n in inner):
        return f"{out} is not empty; predictions are written to a new or empty folder"
    return ""


def _output_folder(out):
    out = os.fspath(out)
    if not out:
        raise ValueError("the output folder path is empty")
    return out


def _schema(pa, wanted, top_k, n_types, positions, net, wide):
    """Arrow schemas of the per-cell and per-grid-node tables (network outputs as float32, or float64
    when ``wide``; the dtypes of the outputs are restored when reading)."""
    pos_dim, pos_dtype = positions
    value = pa.float64() if wide else pa.float32()
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
            cells.append(("probs", pa.list_(value, n_types)))
        else:
            k = min(int(top_k), n_types)
            cells += [("top_types", pa.list_(pa.int32(), k)), ("top_probs", pa.list_(value, k))]
    if "all_logits" in wanted:
        cells.append(("logits", pa.list_(value, n_types)))
    if "all_scales" in wanted:
        cells.append(("scale", value))
    if "expression_matrices" in wanted:
        cells.append(("expression", pa.list_(value, net.n_genes)))
    if "embeddings_cell" in wanted:
        cells.append(("embedding", pa.list_(value, net.hidden_size)))
    grid = [("graph", pa.int32())]
    for field, column in (("grid_sections", "section"), ("grid_timepoints", "timepoint")):
        if field in wanted:
            grid.append((column, pa.string()))
    if "grid_positions" in wanted:
        grid += [(axis, pos_type) for axis in axes]
    if "embeddings_grid" in wanted:
        grid.append(("embedding", pa.list_(value, net.hidden_size)))
    return pa.schema(cells), (pa.schema(grid) if len(grid) > 1 else None), axes


def _fixed_list(pa, array, width, value_type):
    values = np.ascontiguousarray(array, dtype=value_type.to_pandas_dtype()).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(values, type=value_type), width)


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
        if ids.dtype.kind == "S":   # bytes, kept byte for byte as latin-1 text
            columns["cell_id"] = pa.array([i.decode("latin-1") for i in ids.tolist()], type=pa.string())
        elif ids.dtype != object:
            columns["cell_id"] = pa.array(ids.astype(str), type=pa.string())
        else:
            try:   # str, with None / NaN for missing ids (written as nulls)
                columns["cell_id"] = pa.array(ids, type=pa.string(), from_pandas=True)
            except (TypeError, ValueError, pa.ArrowException):   # objects other than str
                warnings.warn("cell ids that are not strings are written, and read back, as text")
                piece["ids_as_text"] = True
                columns["cell_id"] = pa.array([None if _missing(i) else str(i) for i in ids], type=pa.string())
    for column, value in (("section", section), ("timepoint", timepoint), ("condition", condition)):
        if column in names:
            columns[column] = pa.array([value] * n, type=pa.string())
    if axes[0] in names:
        pos = piece["all_positions"]
        for i, axis in enumerate(axes):
            columns[axis] = pa.array(np.ascontiguousarray(pos[:, i]), type=schema.field(axis).type)
    if "probs" in names:
        field = schema.field("probs").type
        columns["probs"] = _fixed_list(pa, piece["all_probs"], field.list_size, field.value_type)
    if "top_types" in names:
        matrix = piece["all_probs"]   # k entries per row, in type order
        k = schema.field("top_types").type.list_size
        probs, types = matrix.data.reshape(-1, k), matrix.indices.reshape(-1, k)
        order = np.argsort(-probs, axis=1, kind="stable")   # most probable first
        columns["top_types"] = _fixed_list(pa, np.take_along_axis(types, order, axis=1).astype(np.int32), k,
                                           pa.int32())
        columns["top_probs"] = _fixed_list(pa, np.take_along_axis(probs, order, axis=1), k,
                                           schema.field("top_probs").type.value_type)
    for field, column in (("all_logits", "logits"), ("expression_matrices", "expression"),
                          ("embeddings_cell", "embedding")):
        if column in names:
            kind = schema.field(column).type
            columns[column] = _fixed_list(pa, piece[field], kind.list_size, kind.value_type)
    if "scale" in names:
        kind = schema.field("scale").type
        columns["scale"] = pa.array(np.ascontiguousarray(piece["all_scales"], dtype=kind.to_pandas_dtype()).reshape(-1),
                                    type=kind)
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
            kind = grid_schema.field("embedding").type
            columns["embedding"] = _fixed_list(pa, piece["embeddings_grid"], kind.list_size, kind.value_type)
        grid = pa.table([columns[name] for name in grid_schema.names], schema=grid_schema)
    return cells, grid


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def _rows_path(out, rank, world_size):
    return os.path.join(out, f"_rows-{rank:05d}-of-{world_size:05d}.pt")


def _predict_to_parquet(net, graphs, positions, top_k, out, wanted, device, temperature, cell_types):
    pa, pq = _pyarrow()
    top_k = None if top_k is None else int(top_k)
    distributed = dist.is_available() and dist.is_initialized()
    if not distributed and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        raise RuntimeError("predict(out=) runs under torchrun without a process group: every process would "
                           "predict everything into the same files. Create the group first "
                           "(torch.distributed.init_process_group(), or train_distributed, which creates it).")
    rank, world_size = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    if distributed and device.type == "cuda" and dist.get_backend() == "nccl":
        torch.cuda.set_device(device)   # the object collectives below run on the current device

    # start: process 0 checks and prepares the folder (a failure is passed on to the other processes);
    # every process must then see its marker
    created, failure = [], None
    plan = [None, None, None]   # kind of problem, its message, token
    if rank == 0:
        try:
            problem = _output_problem(out)
            if problem:
                plan = ["exists", problem, None]
            else:
                created = [folder for folder in (out, os.path.join(out, "cells"), os.path.join(out, "grid"))
                           if not os.path.exists(folder)]
                os.makedirs(out, exist_ok=True)
                for kind in ("cells", "grid"):   # temporary parts left by a killed run
                    folder = os.path.join(out, kind)
                    for name in os.listdir(folder) if os.path.isdir(folder) else []:
                        if _is_temporary(name):
                            _remove(os.path.join(folder, name))
                token = uuid.uuid4().hex
                open(os.path.join(out, f"_writing-{token}"), "w").close()
                plan = [None, None, token]
        except Exception as error:
            failure = error
            for folder in reversed(created):
                try:
                    os.rmdir(folder)
                except OSError:
                    pass
            plan = ["failed", f"process 0 could not prepare {out}: {type(error).__name__}: {error}", None]
    if distributed:   # the only collectives: every process starts predict at the same point
        dist.broadcast_object_list(plan, src=0)
    kind, problem, token = plan
    if failure is not None:
        raise failure
    if kind == "exists":
        raise FileExistsError(problem)
    if kind is not None:
        raise RuntimeError(problem)
    marker = os.path.join(out, f"_writing-{token}")
    seen = [os.path.exists(marker)]
    if distributed:
        seen = [None] * world_size
        dist.all_gather_object(seen, os.path.exists(marker))
    if not all(seen):
        if rank == 0:
            _cleanup(out, marker, created)
        blind = [r for r, ok in enumerate(seen) if not ok]
        raise RuntimeError(f"process(es) {blind} cannot see {out}: under torchrun the output folder must be on a "
                           "file system that every process reads")

    try:
        details = _write_parts(pa, pq, net, graphs, positions, top_k, out, wanted, device, temperature,
                               cell_types, rank, world_size)
    except BaseException as error:
        for kind in ("cells", "grid"):
            _remove(_temporary(_shard_path(out, kind, rank, world_size)))
        if distributed:   # process 0 notices it while waiting
            with open(os.path.join(out, f"_failed-{rank:05d}-of-{world_size:05d}"), "w") as handle:
                handle.write(f"{type(error).__name__}: {error}")
        else:
            _cleanup(out, marker, created)
        raise
    if rank == 0:
        try:
            _write_meta(out, graphs, world_size, details, marker)
        except BaseException:
            if not distributed and not os.path.exists(os.path.join(out, META_FILE)):
                _cleanup(out, marker, created)
            raise
    return out


def _cleanup(out, marker, created):
    """Single process: remove what this call wrote (parts, the list of its graphs, temporary files), the
    marker and the folders it created (when they are empty)."""
    for kind in ("cells", "grid"):
        _remove(_shard_path(out, kind, 0, 1))
        _remove(_temporary(_shard_path(out, kind, 0, 1)))
    rows = _rows_path(out, 0, 1)
    for path in (rows, f"{rows}.tmp{os.getpid()}", os.path.join(out, f"{META_FILE}.tmp{os.getpid()}")):
        _remove(path)
    _remove(marker)
    for folder in reversed(created):
        try:
            os.rmdir(folder)
        except OSError:
            pass


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _write_parts(pa, pq, net, graphs, positions, top_k, out, wanted, device, temperature, cell_types,
                 rank, world_size):
    """Predict this process's share of the graphs, write its Parquet parts, then the list of its graphs
    (``_rows-<rank>``, which process 0 waits for)."""
    written = list(wanted)
    share = range(rank, len(graphs), world_size)
    restore = not isinstance(graphs, DiskGraphs)
    net.eval()

    def predicted(position):
        with torch.no_grad():
            return _predict_graph(net, graphs[position], device, temperature, top_k, set(written), restore)

    # the first graph tells whether the network outputs float64 (a process without graphs: its parameters)
    first = predicted(share[0]) if len(share) else None
    if first is None:
        wide = next(net.parameters()).dtype == torch.float64
    else:
        wide = np.dtype(np.float64).str in _output_dtypes(first).values()
    schema, grid_schema, axes = _schema(pa, written, top_k, len(cell_types), positions, net, wide)
    info = json.dumps({"cell_types": cell_types, "fields": written, "top_k": top_k,
                       "rank": rank, "world_size": world_size})
    schema = schema.with_metadata({"pointillhist": info})
    grid_schema = None if grid_schema is None else grid_schema.with_metadata({"pointillhist": info})
    paths = {"cells": _shard_path(out, "cells", rank, world_size)}
    if grid_schema is not None:
        paths["grid"] = _shard_path(out, "grid", rank, world_size)
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    writers, buffers = {}, {kind: [] for kind in paths}
    buffered = {kind: [0, 0] for kind in paths}   # bytes, rows
    done, dtypes, measured = [], {}, False   # measured: on a graph with core cells
    if first is not None:
        dtypes = _output_dtypes(first)
    names = np.asarray(cell_types)

    def flush(kind):
        if buffers[kind]:
            table = pa.concat_tables(buffers[kind])
            writers[kind].write_table(table, row_group_size=len(table))
            buffers[kind], buffered[kind] = [], [0, 0]

    try:
        for kind, path in paths.items():
            writers[kind] = pq.ParquetWriter(_temporary(path), schema if kind == "cells" else grid_schema)
        for position in tqdm.tqdm(share, desc="Predicting", disable=rank != 0):
            piece, first = (predicted(position) if first is None else first), None
            if piece["n_cells"] and not measured:
                dtypes, measured = _output_dtypes(piece), True
            if "all_cell_types" in written:
                piece["cell_types"] = names[piece["all_labels"]]
            tables = dict(zip(("cells", "grid"), _tables(pa, piece, position, schema, grid_schema, axes)))
            for kind in paths:
                if len(tables[kind]):
                    buffers[kind].append(tables[kind])
                    buffered[kind][0] += tables[kind].nbytes
                    buffered[kind][1] += len(tables[kind])
                # several graphs per row group, so that the footer stays small
                if buffered[kind][0] >= ROW_GROUP_BYTES or buffered[kind][1] >= ROW_GROUP_ROWS:
                    flush(kind)
            ids = piece.get("unique_cell_ids")
            done.append((position, piece["n_cells"], piece["n_grid"],
                         None if ids is None else ids.dtype.str if ids.dtype != object else
                         "object-as-text" if piece.get("ids_as_text") else "object"))
            del piece, tables
        for kind in paths:
            flush(kind)
    finally:
        for writer in writers.values():
            writer.close()
    for path in paths.values():
        os.replace(_temporary(path), path)
    rows_path = _rows_path(out, rank, world_size)
    torch.save({"rows": done, "dtypes": dtypes, "measured": measured}, f"{rows_path}.tmp{os.getpid()}")
    os.replace(f"{rows_path}.tmp{os.getpid()}", rows_path)
    return dict(fields=written, top_k=top_k, cell_types=cell_types, axes=axes, grid=grid_schema is not None,
                pos_dtype=_position_dtype(schema, grid_schema, axes), n_genes=net.n_genes,
                hidden_size=net.hidden_size)


def _output_dtypes(piece):
    """numpy dtypes of the network outputs of a graph, restored when reading."""
    dtypes = {}
    for field in ("all_probs", "all_logits", "all_scales", "expression_matrices", "embeddings_cell",
                  "embeddings_grid"):
        if field in piece:
            value = piece[field]
            dtypes[field] = (value.data if sparse.issparse(value) else value).dtype.str
    return dtypes


def _position_dtype(schema, grid_schema, axes):
    for table_schema in (schema, grid_schema):
        if table_schema is not None and axes[0] in table_schema.names:
            return table_schema.field(axes[0]).type.to_pandas_dtype()
    return None


def _write_meta(out, graphs, world_size, details, marker):
    """``predictions.pt``, written by process 0 once every process has written its part: the output is
    complete when it exists."""
    paths = [_rows_path(out, r, world_size) for r in range(world_size)]
    last_note = time.monotonic()
    while True:
        failed = sorted(name for name in os.listdir(out) if name.startswith("_failed-"))
        if failed:
            reasons = []
            for name in failed:
                with open(os.path.join(out, name)) as handle:
                    reasons.append(f"process {int(name.split('-')[1])}: {handle.read()}")
            raise RuntimeError(f"prediction failed in {'; '.join(reasons)}. {out} keeps what was written: remove it "
                               "before running again")
        missing = [path for path in paths if not os.path.exists(path)]
        if not missing:
            break
        if time.monotonic() - last_note > 600:
            print(f"predict: waiting for {len(missing)} of {world_size} processes to finish writing", flush=True)
            last_note = time.monotonic()
        time.sleep(1.0)
    shares = [torch.load(path, weights_only=False) for path in paths]
    rows = sorted((row for share in shares for row in share["rows"]), key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(graphs))):
        raise RuntimeError("the processes did not predict every graph once; were the graphs the same everywhere?")
    labels = zip(*(_graph_values(graphs, name) for name in ("section_label", "timepoint_label",
                                                             "condition_label")))
    dtypes = next((share["dtypes"] for share in shares if share.get("measured")),
                  next((share["dtypes"] for share in shares if share["dtypes"]), {}))
    meta = dict(format=FORMAT, n_graphs=len(graphs), world_size=world_size, dtypes=dtypes, **details,
                graphs=[dict(section_label=s, timepoint_label=t, condition_label=c, n_cells=n, n_grid=m,
                             ids_dtype=d)
                        for (s, t, c), (_, n, m, d) in zip(labels, rows)])
    tmp = os.path.join(out, f"{META_FILE}.tmp{os.getpid()}")
    torch.save(meta, tmp)
    os.replace(tmp, os.path.join(out, META_FILE))
    for path in paths:
        _remove(path)
    _remove(marker)


def read_predictions(out, fields=None, sections=None):
    """
    Read predictions written by ``predict(..., out=out)`` into the dict ``predict`` returns.

    Parameters:
        out: The folder given to ``predict``.
        fields: The optional entries to read, among those written (None: all written ones).
            ``all_labels``, ``label_list``, ``cell_types`` and ``section_ids`` are always read.
        sections: None, or a section label or a list of them (``all_sections`` values);
            only the graphs of these sections are read, so only their cells are held in
            memory. For ``cell_cell_interactions`` pass the same graphs, e.g.
            ``graphs.subset(graphs.index.section_label.isin(sections))`` for a DiskGraphs.

    Returns:
        The dict of ``predict`` for the graphs read, in graph order; for all sections and
        fields it equals what ``predict`` returns in memory, except that cell ids that were
        Python objects other than str come back as str, and missing ids (None) as NaN.
    """
    pa, pq = _pyarrow()
    out = _output_folder(out)
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
    world_size = meta["world_size"]
    kinds = ["cells"] + (["grid"] if meta["grid"] else [])
    parts = {kind: [_shard_path(out, kind, r, world_size) for r in range(world_size)] for kind in kinds}
    absent = [path for paths in parts.values() for path in paths if not os.path.exists(path)]
    if absent:
        raise FileNotFoundError(f"missing prediction file(s) {absent}")

    graph_rows = meta["graphs"]
    if sections is not None:
        sections = [sections] if isinstance(sections, (str, bytes)) or not isinstance(sections, Iterable) \
            else list(sections)
        try:
            lookup = set(sections)
        except TypeError:   # unhashable labels
            lookup = None
    selected = [g for g, row in enumerate(graph_rows)
                if sections is None or (_in_lookup(row["section_label"], lookup) if lookup is not None
                                        else any(_same_label(row["section_label"], s) for s in sections))]
    if not selected:
        raise ValueError(f"no graph of {out} belongs to the sections {sections}")
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

    readers = {kind: _PartReader(pq, parts[kind], selected, world_size, columns[kind])
               for kind in (["cells", "grid"] if need_grid else ["cells"])}
    n_types = len(meta["cell_types"])
    dtypes = {field: np.dtype(meta.get("dtypes", {}).get(field, np.float32)) for field in
              ("all_probs", "all_logits", "all_scales", "expression_matrices", "embeddings_cell", "embeddings_grid")}
    pieces = []
    for g in selected:
        row = graph_rows[g]
        piece = {"n_cells": row["n_cells"], "n_grid": row["n_grid"],
                 "labels": (row["section_label"], row["timepoint_label"], row["condition_label"])}
        cells = readers["cells"].graph(g)
        n = row["n_cells"]
        piece["all_labels"] = np.array(_column(cells, "label", np.int64))
        if "all_positions" in wanted_set:
            piece["all_positions"] = _positions(cells, meta["axes"], meta["pos_dtype"])
        if "all_probs" in wanted_set:
            if top_k is None:
                piece["all_probs"] = _matrix(cells, "probs", n, n_types, dtypes["all_probs"])
            else:
                k = min(int(top_k), n_types)
                indices = _matrix(cells, "top_types", n, k, np.int32).astype(np.int64).ravel()
                values = _matrix(cells, "top_probs", n, k, dtypes["all_probs"]).ravel().copy()   # sorted below
                matrix = sparse.csr_matrix((values, indices, np.arange(0, n * k + 1, k)), shape=(n, n_types))
                matrix.sort_indices()
                piece["all_probs"] = matrix
        if "all_logits" in wanted_set:
            piece["all_logits"] = _matrix(cells, "logits", n, n_types, dtypes["all_logits"])
        if "all_scales" in wanted_set:
            piece["all_scales"] = _column(cells, "scale", dtypes["all_scales"]).reshape(-1, 1)
        if "expression_matrices" in wanted_set:
            piece["expression_matrices"] = _matrix(cells, "expression", n, meta["n_genes"],
                                                   dtypes["expression_matrices"])
        if "embeddings_cell" in wanted_set:
            piece["embeddings_cell"] = _matrix(cells, "embedding", n, meta["hidden_size"],
                                               dtypes["embeddings_cell"])
        if "unique_cell_ids" in wanted_set:
            piece["unique_cell_ids"] = _ids(cells, row["ids_dtype"])
        if need_grid:
            grid = readers["grid"].graph(g)
            if "grid_positions" in wanted_set:
                piece["grid_positions"] = _positions(grid, meta["axes"], meta["pos_dtype"])
            if "embeddings_grid" in wanted_set:
                piece["embeddings_grid"] = _matrix(grid, "embedding", row["n_grid"], meta["hidden_size"],
                                                   dtypes["embeddings_grid"])
        pieces.append(piece)
        del cells
    return _assemble(pieces, meta["cell_types"], top_k, wanted)


class _PartReader:
    """Reads the rows of one graph from the Parquet parts, one row group in memory per part.

    Process ``r`` of ``w`` wrote the graphs ``r, r + w, ...`` in order, several per row group, so the
    ``graph`` statistics of a row group give the range of positions it holds.
    """

    def __init__(self, pq, paths, selected, world_size, columns):
        self.columns = columns
        self.groups = {}   # position -> [(part, row group), ...]
        self.handles = []
        self.cache = {}    # part -> (row group, table, graph positions)
        wanted = sorted(selected)
        for rank, path in enumerate(paths):
            handle = pq.ParquetFile(path)
            self.handles.append(handle)
            column = handle.schema_arrow.get_field_index("graph")
            for i in range(handle.num_row_groups):
                statistics = handle.metadata.row_group(i).column(column).statistics
                if statistics is not None and statistics.has_min_max:
                    low, high = int(statistics.min), int(statistics.max)
                else:
                    values = handle.read_row_group(i, columns=["graph"]).column("graph").to_numpy()
                    low, high = int(values.min()), int(values.max())
                for position in wanted[bisect.bisect_left(wanted, low):bisect.bisect_right(wanted, high)]:
                    if position % world_size == rank:
                        self.groups.setdefault(position, []).append((rank, i))

    def graph(self, position):
        """The rows of graph ``position`` (None when it has none)."""
        slices = []
        for rank, i in self.groups.get(position, []):
            cached = self.cache.get(rank)
            if cached is None or cached[0] != i:
                table = self.handles[rank].read_row_group(i, columns=self.columns)
                cached = (i, table, table.column("graph").to_numpy())
                self.cache[rank] = cached
            _, table, positions = cached
            first, last = np.searchsorted(positions, position, "left"), np.searchsorted(positions, position, "right")
            if last > first:
                slices.append(table.slice(first, last - first))
        if not slices:
            return None
        if len(slices) == 1:
            return slices[0]
        import pyarrow as pa
        return pa.concat_tables(slices)


def _same_label(label, wanted):
    try:
        return bool(label == wanted)
    except Exception:
        return False


def _in_lookup(label, lookup):
    try:
        return label in lookup
    except TypeError:
        return False


def _column(table, name, dtype):
    """A 1-D column as numpy of ``dtype`` (empty for a graph without rows)."""
    if table is None:
        return np.zeros(0, dtype=dtype)
    return table.column(name).to_numpy().astype(dtype, copy=False)


def _positions(table, axes, dtype):
    if table is None:
        return np.zeros((0, len(axes)), dtype=dtype)
    return np.column_stack([table.column(axis).to_numpy() for axis in axes])


def _matrix(table, name, n, width, dtype=np.float32):
    """A fixed-size list column as an (n, width) array of ``dtype``."""
    if table is None:
        return np.zeros((0, width), dtype=dtype)
    column = table.column(name).combine_chunks()
    return column.flatten().to_numpy().reshape(n, column.type.list_size).astype(dtype, copy=False)


def _ids(table, dtype):
    """Cell ids, stored as strings (nulls for missing ids), back in their numpy dtype."""
    if table is None:
        return np.zeros(0, dtype=object if dtype in ("object", "object-as-text", None) else np.dtype(dtype))
    column = table.column("cell_id")
    values = column.to_numpy(zero_copy_only=False)
    if dtype in ("object", "object-as-text") or dtype is None:
        values = values.astype(object)
        values[column.is_null().to_numpy(zero_copy_only=False)] = np.nan
        return values
    if np.dtype(dtype).kind == "b":
        return values == "True"
    if np.dtype(dtype).kind == "S":
        return np.array([v.encode("latin-1") for v in values.tolist()], dtype=np.dtype(dtype))
    return values.astype(np.dtype(dtype))


def cell_cell_interactions(predictions, graphs, return_normalized=False):
    """
    Count predicted cell-type pairs across the cell-cell edges of each section.

    Only edges between two core cells are counted, so cells in the overlapping
    tile frames are not double-counted. Matrices are averaged over the tiles of
    a section.

    Parameters:
        predictions: Dictionary returned by predict() or read_predictions().
        graphs: The graphs the predictions were made on, in the same order (a ValueError is
            raised when their number or their core cells do not match the predictions).
        return_normalized: Divide each tile's matrix by its edge count first.

    Returns:
        {section_label: (n_types, n_types) matrix}
    """
    n_types = len(predictions["cell_types"])
    matrices_by_section = defaultdict(list)
    label_list = predictions["label_list"]
    mismatch = ("the predictions cover {} graphs but {} graphs were given; pass the graphs the predictions were "
                "made on (after read_predictions(sections=...): graphs.subset(graphs.index.section_label.isin("
                "sections)) for a DiskGraphs)")
    if hasattr(graphs, "__len__") and len(graphs) != len(label_list):
        raise ValueError(mismatch.format(len(label_list), len(graphs)))

    n_graphs = 0
    for graph in graphs:
        if n_graphs == len(label_list):   # an iterable without a length that goes on
            raise ValueError(mismatch.format(len(label_list), "more"))
        labels = label_list[n_graphs]
        n_graphs += 1
        is_core = graph["cells"].is_core.detach().cpu().numpy().astype(bool)
        if int(is_core.sum()) != len(labels):
            raise ValueError(f"a graph of {graph.section_label} has {int(is_core.sum())} core cells but its "
                             f"predictions have {len(labels)}: the graphs do not match the predictions")
        core_index = np.cumsum(is_core) - 1  # original index -> core index
        src, tgt = graph["cells", "is_close_to", "cells"].edge_index.detach().cpu().numpy()
        both_core = is_core[src] & is_core[tgt]
        src, tgt = src[both_core], tgt[both_core]

        matrix = np.zeros((n_types, n_types), dtype=int)
        np.add.at(matrix, (labels[core_index[src]], labels[core_index[tgt]]), 1)
        if return_normalized and len(src) > 0:
            matrix = matrix / len(src)
        matrices_by_section[graph.section_label].append(matrix)
    if n_graphs != len(label_list):
        raise ValueError(mismatch.format(len(label_list), n_graphs))

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
