"""Graphs kept in a folder on disk and loaded one at a time."""
import copy
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

__all__ = ["DiskGraphs", "save_graphs", "load_graphs"]

FORMAT = 1
INDEX_FILE = "index.pt"
GRAPH_DIR = "graphs"
#: graph attributes stored once in the index when every graph carries the same value
SHARED = ("cell_types", "genes", "regions")
#: graph attributes every graph must carry (set by generate_graphs); copied to the index
REQUIRED = ("section", "section_label", "timepoint", "timepoint_label", "condition", "condition_label",
            "cell_types", "genes")
_MARK = "__pointillhist_packed__"


def _tensor_bytes(graph):
    """Bytes of every tensor of a HeteroData (the in-memory size ``train`` uses for placement)."""
    return sum(v.numel() * v.element_size() for store in graph.stores for v in store.values() if torch.is_tensor(v))


def _equal(a, b):
    """Exact equality of two attribute values (lists of names, arrays, scalars)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and bool(np.all(a == b))
    if torch.is_tensor(a):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    return bool(a == b)


def _graph_values(graphs, name):
    """Per-graph value of a graph attribute; a DiskGraphs reads it from its index."""
    if isinstance(graphs, DiskGraphs):
        return graphs._column(name)
    return [getattr(g, name) for g in graphs]


# ---------------------------------------------------------------- tensors
def _pack_tensor(t):
    """Lossless compact form of a tensor: int32 for int64 values that fit, CSR for sparse 2-D floats."""
    t = t.detach().cpu()
    if t.dtype == torch.int64 and t.numel() and t.min() >= -2**31 and t.max() < 2**31:
        return {_MARK: "int32", "values": t.to(torch.int32)}
    if t.is_floating_point() and t.dim() == 2 and t.numel():
        rows, cols = ((t != 0) | torch.signbit(t)).nonzero(as_tuple=True)   # -0.0 is kept explicitly
        if 2 * rows.numel() <= t.numel() and t.shape[1] < 2**31:
            values = t[rows, cols]
            small = values.to(torch.int16)
            # int16 only when every value comes back bit for bit (no -0.0, NaN, fractions or overflow)
            if torch.equal(small.to(t.dtype), values) and not (values == 0).any():
                values = small
            return {_MARK: "csr", "row_counts": torch.bincount(rows, minlength=t.shape[0]).to(torch.int32),
                    "cols": cols.to(torch.int32), "values": values, "shape": tuple(t.shape), "dtype": t.dtype}
    if t.untyped_storage().nbytes() > t.numel() * t.element_size():
        t = t.clone()   # a view would save its whole storage
    return t


def _unpack_tensor(packed):
    kind = packed[_MARK]
    if kind == "int32":
        return packed["values"].to(torch.int64)
    if kind == "csr":
        shape, dtype = packed["shape"], packed["dtype"]
        dense = torch.zeros(shape, dtype=dtype)
        rows = torch.repeat_interleave(torch.arange(shape[0]), packed["row_counts"].to(torch.int64))
        dense[rows, packed["cols"].to(torch.int64)] = packed["values"].to(dtype)
        return dense
    raise ValueError(f"unknown packed tensor kind {kind!r}")


def _is_packed(value):
    return isinstance(value, dict) and _MARK in value


# ---------------------------------------------------------------- graphs
def _region_codes(as_str, annotated):
    """Per-cell region labels -> (sorted labels of this graph, int32 code per cell, -1 if unannotated)."""
    vocabulary = sorted(set(as_str[annotated]))
    codes = np.full(len(as_str), -1, dtype=np.int32)
    codes[annotated] = pd.Index(vocabulary).get_indexer(as_str[annotated])
    return vocabulary, torch.from_numpy(codes)


def _region_one_hot_from_codes(vocabulary, codes, regions):
    from ._graphs import _region_one_hot     # the function generate_graphs uses for in-memory graphs

    annotated = codes.numpy() >= 0
    as_str = np.full(len(annotated), "", dtype=object)
    as_str[annotated] = np.asarray(vocabulary, dtype=object)[codes.numpy()[annotated]]
    return _region_one_hot(as_str, annotated, regions)


def _pack_graph(graph, shared, regions=None):
    """Store order and attribute order of ``graph`` with packed tensors and markers for shared values.

    ``regions`` = (as_str, annotated) adds ``cell_regions`` / ``regions`` as generate_graphs does
    at the end, resolved against the folder's region list when the graph is loaded.
    """
    stores = []
    for store in graph.stores:
        key = getattr(store, "_key", None)
        kind = "global" if key is None else ("edge" if isinstance(key, tuple) else "node")
        items = []
        for name, value in store.items():
            if kind == "global" and name in shared and _equal(value, shared[name]):
                value = {_MARK: "shared"}
            elif torch.is_tensor(value):
                value = _pack_tensor(value)
            items.append((name, value))
        if kind == "global" and regions is not None:
            vocabulary, codes = _region_codes(*regions)
            items.append(("cell_regions", {_MARK: "region_codes", "vocabulary": vocabulary, "codes": codes}))
            items.append(("regions", {_MARK: "shared"}))
        stores.append((kind, key, items))
    return stores


def _unpack_graph(stores, shared):
    graph = HeteroData()
    for kind, key, items in stores:
        for name, value in items:
            if _is_packed(value):
                if value[_MARK] == "shared":
                    value = copy.copy(shared[name])
                elif value[_MARK] == "region_codes":
                    value = _region_one_hot_from_codes(value["vocabulary"], value["codes"], shared["regions"])
                else:
                    value = _unpack_tensor(value)
            if kind == "global":
                setattr(graph, name, value)
            else:
                graph[key][name] = value
    return graph


def _atomic_save(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class _GraphWriter:
    """Writes graphs into an empty folder one at a time; ``close`` writes the index last."""

    def __init__(self, path):
        path = os.fspath(path)
        if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
            raise FileExistsError(f"{path} already exists and is not an empty folder")
        os.makedirs(os.path.join(path, GRAPH_DIR), exist_ok=True)
        self.path = path
        self.rows = []
        self.shared = {}
        self.tiles = {}
        self.region_vocabulary = set()
        self.streamed_regions = False

    def add(self, graph, regions=None):
        """Write one graph; ``regions`` = (as_str, annotated) per cell for graphs built with a region_key."""
        missing = [name for name in REQUIRED if not hasattr(graph, name)]
        if "cells" not in graph.node_types or "is_core" not in graph["cells"]:
            missing.append("cells.is_core")
        if missing:
            raise ValueError(f"graph {len(self.rows)} lacks the attribute(s) {missing}; "
                             "build the graphs with ph.pp.generate_graphs")
        if regions is not None and any(name in graph._global_store for name in ("cell_regions", "regions")):
            raise ValueError("the graph already carries cell_regions / regions")
        for name in SHARED:
            if name in graph._global_store and name not in self.shared:
                self.shared[name] = copy.copy(graph._global_store[name])
        if regions is not None:
            self.streamed_regions = True
            self.region_vocabulary.update(regions[0][regions[1]])
        section = int(graph.section)
        tile = self.tiles.get(section, 0)
        self.tiles[section] = tile + 1
        name = f"s{section:06d}_t{tile:06d}.pt"
        _atomic_save({"format": FORMAT, "stores": _pack_graph(graph, self.shared, regions)},
                     os.path.join(self.path, GRAPH_DIR, name))
        n_cells = int(graph["cells"].num_nodes)
        vertexes = getattr(graph, "vertexes_fov", None)
        row = {attr: getattr(graph, attr) for attr in REQUIRED if attr not in SHARED}
        row.update(file=name, n_cells=n_cells, n_core_cells=int(graph["cells"].is_core.sum()),
                   nbytes=_tensor_bytes(graph), ndim=None if vertexes is None else len(vertexes) // 2,
                   region_cells=n_cells if regions is not None else 0)
        self.rows.append(row)

    def close(self):
        if self.streamed_regions:
            if "regions" in self.shared:
                raise ValueError("graphs with and without streamed regions cannot be mixed")
            self.shared["regions"] = sorted(self.region_vocabulary)
            for row in self.rows:   # cell_regions is (n_cells, n_regions) float32 once loaded
                row["nbytes"] += row["region_cells"] * len(self.shared["regions"]) * 4
        for row in self.rows:
            del row["region_cells"]
        _atomic_save({"format": FORMAT, "shared": self.shared, "graphs": self.rows},
                     os.path.join(self.path, INDEX_FILE))
        return DiskGraphs(self.path)


class DiskGraphs(torch.utils.data.Dataset):
    """
    Graphs kept in a folder on disk, loaded one at a time when accessed.

    Made by ``generate_graphs(..., save_dir=...)``, :func:`save_graphs` or :func:`load_graphs`, and
    accepted wherever a list of graphs is: ``networks``, ``train``, ``train_distributed``, ``predict``,
    ``load_model``, ``cell_cell_interactions``. Only the graphs a step needs are in memory, so host
    memory stays bounded whatever the number of graphs. ``graphs[i]`` loads graph ``i`` (on the CPU)
    as a new object each time; slices and :meth:`subset` give views on the same folder; iterating
    loads the graphs one after the other.

    The folder holds ``index.pt`` (the per-graph metadata of :attr:`index` and the cell types, genes
    and regions shared by all graphs) and one file per graph. The files are pickled: open only
    folders you trust, as for ``torch.load``.
    """

    def __init__(self, path, _index=None, _positions=None):
        self.path = os.fspath(path)
        if _index is None:
            index_path = os.path.join(self.path, INDEX_FILE)
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"{index_path} not found: not a graph folder, or its writing did not finish")
            _index = torch.load(index_path, map_location="cpu", weights_only=False)
            if _index.get("format") != FORMAT:
                raise ValueError(f"{self.path}: graph folder format {_index.get('format')!r}, "
                                 f"this version of pointillhist reads format {FORMAT}")
        self._index = _index
        self._positions = list(range(len(_index["graphs"]))) if _positions is None else list(_positions)
        self._frame = None

    # -- sequence -------------------------------------------------------------------
    def __len__(self):
        return len(self._positions)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.subset(range(len(self))[i])
        if isinstance(i, bool) or not (isinstance(i, (int, np.integer)) or (torch.is_tensor(i) and i.dim() == 0)):
            raise TypeError(f"graph index must be an integer or a slice, not {type(i).__name__}")
        i = int(i)
        if not -len(self) <= i < len(self):
            raise IndexError(f"graph index {i} out of range for {len(self)} graphs")
        return self._load(self._positions[i])

    def __iter__(self):
        for position in self._positions:
            yield self._load(position)

    def subset(self, indices):
        """A view on the graphs at ``indices`` (positions in this sequence), in that order."""
        positions = []
        for i in indices:
            i = int(i)
            if not -len(self) <= i < len(self):
                raise IndexError(f"graph index {i} out of range for {len(self)} graphs")
            positions.append(self._positions[i])
        return DiskGraphs(self.path, _index=self._index, _positions=positions)

    def _load(self, position):
        row = self._index["graphs"][position]
        path = os.path.join(self.path, GRAPH_DIR, row["file"])
        packed = torch.load(path, map_location="cpu", weights_only=False)
        return _unpack_graph(packed["stores"], self._index["shared"])

    # -- metadata -------------------------------------------------------------------
    def _column(self, name):
        """Per-graph value of an index column, as stored (no graph is loaded)."""
        return [self._index["graphs"][p][name] for p in self._positions]

    @property
    def index(self):
        """One row per graph: file, section, timepoint and condition codes and labels, n_cells (frames
        included), n_core_cells, nbytes (tensor bytes once loaded) and ndim (2 or 3)."""
        if self._frame is None:
            columns = ["file", "section", "section_label", "timepoint", "timepoint_label", "condition",
                       "condition_label", "n_cells", "n_core_cells", "nbytes", "ndim"]
            self._frame = pd.DataFrame([self._index["graphs"][p] for p in self._positions], columns=columns)
        return self._frame

    @property
    def nbytes(self):
        """Total tensor bytes of the graphs once loaded."""
        return sum(self._column("nbytes"))

    @property
    def cell_types(self):
        return copy.copy(self._index["shared"].get("cell_types"))

    @property
    def genes(self):
        return copy.copy(self._index["shared"].get("genes"))

    @property
    def regions(self):
        return copy.copy(self._index["shared"].get("regions"))

    def __repr__(self):
        return (f"DiskGraphs({self.path!r}, {len(self)} graphs, "
                f"{sum(self._column('n_core_cells'))} core cells)")


def save_graphs(graphs, path):
    """
    Write ``graphs`` (a list from ``generate_graphs``, or a :class:`DiskGraphs`) to the folder ``path``,
    which must not exist or be empty, and return them as a :class:`DiskGraphs`.

    Graphs are written one at a time. Tensors are stored losslessly in a compact form (sparse
    counts, 32-bit indices) and are loaded back on the CPU.
    """
    writer = _GraphWriter(path)
    for graph in graphs:
        writer.add(graph)
    return writer.close()


def load_graphs(path):
    """Open a folder written by :func:`save_graphs` or ``generate_graphs(..., save_dir=...)``."""
    return DiskGraphs(path)
