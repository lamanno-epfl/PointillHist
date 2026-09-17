"""Graphs kept in a folder on disk and loaded one at a time."""
import copy
import math
import operator
import os
import pickle
import uuid

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


#: types whose == is exact once the types are the same
_EXACT = (str, bytes, int, type(None))


def _same_pickle(a, b):
    try:
        return pickle.dumps(a) == pickle.dumps(b)
    except Exception:
        return False


def _equal(a, b):
    """Exact equality of two attribute values: same types, dtypes and bits (-0.0 differs from 0.0).
    When in doubt the values count as different (a graph then stores its own value)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, _EXACT):
        return a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return _equal(list(a), list(b)) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, complex):
        return _equal(a.real, b.real) and _equal(a.imag, b.imag)
    if isinstance(a, np.generic):
        return a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, (pd.Index, pd.Series)):
        if isinstance(a.dtype, pd.CategoricalDtype) and not (   # == ignores the order of unordered categories
                isinstance(b.dtype, pd.CategoricalDtype) and _equal(a.dtype.categories, b.dtype.categories)):
            return False
        if isinstance(a, pd.MultiIndex):   # to_numpy gives tuples: the dtype of every level is lost
            return (isinstance(b, pd.MultiIndex) and a.nlevels == b.nlevels and _equal(list(a.names), list(b.names))
                    and all(_equal(a.get_level_values(k), b.get_level_values(k)) for k in range(a.nlevels)))
        if _frequency(a) != _frequency(b):   # a DatetimeIndex, TimedeltaIndex or PeriodIndex keeps its step
            return False
        return (a.dtype == b.dtype and _equal(list(a.names) if isinstance(a, pd.Index) else a.name,
                                              list(b.names) if isinstance(b, pd.Index) else b.name)
                and _equal(a.to_numpy(), b.to_numpy())
                and (not isinstance(a, pd.Series) or _equal(a.index, b.index)))
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype == object:
            return all(_equal(x, y) for x, y in zip(a.ravel(), b.ravel()))
        return a.tobytes() == b.tobytes()
    if torch.is_tensor(a):
        return (a.layout == b.layout == torch.strided and a.dtype == b.dtype and a.shape == b.shape
                and a.device == b.device and a.requires_grad == b.requires_grad
                and torch.equal(_bits(a), _bits(b)))
    try:
        return bool(a == b) and _same_pickle(a, b)   # e.g. time zones, set element types
    except Exception:   # e.g. an object whose == is element-wise
        return False


def _frequency(value):
    """The step of a date or period index, as (name, string) so that None and an absent freq differ."""
    freq = getattr(value, "freq", None)
    return None if freq is None else (type(freq).__name__, str(freq))


def _bits(t):
    """The raw bits of a floating or complex tensor (so that -0.0 and 0.0 differ), the tensor itself otherwise."""
    t = t.detach()
    if t.is_complex():
        t = torch.view_as_real(t.resolve_conj().resolve_neg())
    if t.is_floating_point():
        return t.contiguous().view({8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.int8}[t.element_size()])
    return t


#: floating dtypes stored as CSR when sparse (float8 types lack the comparisons it needs)
_CSR_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _graph_values(graphs, name):
    """Per-graph value of a graph attribute; a DiskGraphs reads it from its index."""
    if isinstance(graphs, DiskGraphs):
        return graphs._column(name)
    return [getattr(g, name) for g in graphs]


# ---------------------------------------------------------------- tensors
def _pack_tensor(t):
    """Lossless compact form of a tensor: int32 for int64 values that fit, CSR for sparse 2-D floats."""
    if t.requires_grad:
        return {_MARK: "requires_grad", "value": _pack_tensor(t.detach())}
    t = t.cpu()
    if t.layout != torch.strided:
        return t   # sparse tensors are saved as they are
    if t.dtype == torch.int64 and t.numel() and t.min() >= -2**31 and t.max() < 2**31:
        return {_MARK: "int32", "values": t.to(torch.int32)}
    if t.dtype in _CSR_DTYPES and t.dim() == 2 and t.numel() and t.shape[1] < 2**31:
        stored = t != 0
        stored |= torch.signbit(t)   # -0.0 is kept explicitly
        if 2 * int(stored.sum()) <= t.numel():
            rows, cols = stored.nonzero(as_tuple=True)
            del stored
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
    if kind == "requires_grad":
        value = packed["value"]
        return (_unpack_tensor(value) if _is_packed(value) else value).requires_grad_()
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
    """Store order and attribute order of ``graph`` with packed tensors and markers for shared values,
    and the names of the values replaced by a marker.

    ``regions`` = (as_str, annotated) adds ``cell_regions`` / ``regions`` as generate_graphs does
    at the end, resolved against the folder's region list when the graph is loaded.
    """
    stores, marked = [], []
    for store in graph.stores:
        key = getattr(store, "_key", None)
        kind = "global" if key is None else ("edge" if isinstance(key, tuple) else "node")
        items = []
        for name, value in store.items():
            if kind == "global" and name in shared and _equal(value, shared[name]):
                value = {_MARK: "shared"}
                marked.append(name)
            elif torch.is_tensor(value):
                value = _pack_tensor(value)
            items.append((name, value))
        if kind == "global" and regions is not None:
            vocabulary, codes = _region_codes(*regions)
            items.append(("cell_regions", {_MARK: "region_codes", "vocabulary": vocabulary, "codes": codes}))
            items.append(("regions", {_MARK: "shared"}))
            marked.append("regions")
        stores.append((kind, key, items))
    return stores, tuple(marked)


def _unpack_graph(stores, shared):
    graph = HeteroData()
    for kind, key, items in stores:
        # the global storage, or a node / edge storage (created even when it has no attribute)
        target = graph._global_store if kind == "global" else graph[key]
        for name, value in items:
            if _is_packed(value):
                if value[_MARK] == "shared":
                    value = copy.deepcopy(shared[name])
                elif value[_MARK] == "region_codes":
                    value = _region_one_hot_from_codes(value["vocabulary"], value["codes"], shared["regions"])
                else:
                    value = _unpack_tensor(value)
            target[name] = value
    return graph


def _atomic_save(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _folder(path):
    """Absolute folder path (a DiskGraphs keeps working after the working directory changes), not
    normalised, so that '..' after a symbolic link is resolved by the operating system."""
    path = os.fspath(path)
    if not path:
        raise ValueError("the folder path is empty")
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


class _GraphWriter:
    """Writes graphs into an empty folder one at a time; ``close`` writes the index last.

    Used as a context manager: when the block raises, the files written so far are removed again
    (and the folder, if it was created here), so the same call can simply be run again.
    """

    def __init__(self, path):
        path = _folder(path)
        if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
            unfinished = (os.path.isdir(path) and GRAPH_DIR in os.listdir(path)
                          and INDEX_FILE not in os.listdir(path))
            raise FileExistsError(f"{path} already exists and is not an empty folder"
                                  + (" (it holds an unfinished graph folder: remove it)" if unfinished else ""))
        self.created = not os.path.exists(path)
        os.makedirs(os.path.join(path, GRAPH_DIR), exist_ok=True)
        self.path = path
        self.token = uuid.uuid4().hex
        self._start()

    def _start(self):
        self.rows = []
        self.shared = {}
        self.tiles = {}
        self.region_vocabulary = set()
        self.streamed_regions = False
        self.closed = False
        self.owns_all = False   # abort removes every graph file (other processes wrote some)

    @classmethod
    def attached(cls, path, token, shared):
        """A writer in another process, adding graphs to a folder that a main writer created."""
        writer = cls.__new__(cls)
        writer.path, writer.token, writer.created = path, token, False
        writer._start()
        writer.shared = copy.deepcopy(shared)
        return writer

    def merge(self, rows, region_vocabulary, streamed_regions, shared):
        """Take over the graphs an attached writer wrote (in the order they must have in the index)."""
        if rows and not self.rows:
            for name, value in shared.items():
                self.shared.setdefault(name, copy.deepcopy(value))
        self.rows.extend(rows)
        self.region_vocabulary.update(region_vocabulary)
        self.streamed_regions = self.streamed_regions or streamed_regions

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None and not self.closed:
            self.abort()
        return False

    def abort(self, remove_folders=True):
        """Remove what this writer wrote (graph files, temporary files, folders it created); with
        ``owns_all``, every file of the graph folder, which was empty when the writer started."""
        graph_dir = os.path.join(self.path, GRAPH_DIR)
        for row in self.rows:
            _remove(os.path.join(graph_dir, row["file"]))
        for name in os.listdir(graph_dir) if os.path.isdir(graph_dir) else []:
            if self.owns_all or f".pt.tmp{os.getpid()}" in name:
                _remove(os.path.join(graph_dir, name))
        _remove(os.path.join(self.path, f"{INDEX_FILE}.tmp{os.getpid()}"))
        if not remove_folders:
            return
        for folder in (graph_dir, self.path) if self.created else (graph_dir,):
            try:
                os.rmdir(folder)
            except OSError:
                pass

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
                self.shared[name] = copy.deepcopy(graph._global_store[name])
        if regions is not None:
            self.streamed_regions = True
            self.region_vocabulary.update(regions[0][regions[1]])
        section = int(graph.section)
        tile = self.tiles.get(section, 0)
        name = f"s{section:06d}_t{tile:06d}.pt"
        n_cells = int(graph["cells"].num_nodes)
        vertexes = getattr(graph, "vertexes_fov", None)
        row = {attr: getattr(graph, attr) for attr in REQUIRED if attr not in SHARED}
        stores, marked = _pack_graph(graph, self.shared, regions)
        row.update(file=name, n_cells=n_cells, n_core_cells=int(graph["cells"].is_core.sum()),
                   nbytes=_tensor_bytes(graph), ndim=None if vertexes is None else len(vertexes) // 2,
                   shared_names=marked, region_cells=n_cells if regions is not None else 0)
        _atomic_save({"format": FORMAT, "token": self.token, "stores": stores},
                     os.path.join(self.path, GRAPH_DIR, name))
        self.tiles[section] = tile + 1
        self.rows.append(row)

    def close(self):
        if self.streamed_regions:
            if "regions" in self.shared:
                raise ValueError("graphs with and without streamed regions cannot be mixed")
            self.shared["regions"] = sorted(self.region_vocabulary)
        rows = [dict(row) for row in self.rows]
        regions = self.shared.get("regions")
        n_regions = 0 if regions is None else len(regions)
        for row in rows:   # cell_regions is (n_cells, n_regions) float32 once loaded
            row["nbytes"] += row.pop("region_cells") * n_regions * 4
        _atomic_save({"format": FORMAT, "token": self.token, "shared": self.shared, "graphs": rows},
                     os.path.join(self.path, INDEX_FILE))
        self.closed = True
        return DiskGraphs(self.path)


def _remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _is_bool(value):
    """A boolean, which is refused as a graph index (a list would take True as 1)."""
    return isinstance(value, (bool, np.bool_)) or (torch.is_tensor(value) and value.dtype == torch.bool)


class DiskGraphs(torch.utils.data.Dataset):
    """
    Graphs kept in a folder on disk, loaded one at a time when accessed.

    Made by ``generate_graphs(..., save_dir=...)``, :func:`save_graphs` or :func:`load_graphs`, and
    accepted wherever a list of graphs is: ``networks``, ``load_model``, ``setup_training``, ``train``,
    ``train_distributed``, ``predict``, ``cell_cell_interactions`` and ``ph.pl.fovs``. Only the graphs
    a step needs are in memory, so host memory stays bounded whatever the number of graphs.
    ``graphs[i]`` loads graph ``i`` (on the CPU) as a new object each time; slices and :meth:`subset`
    give views on the same folder; iterating loads the graphs one after the other. :attr:`index`
    describes every graph without loading it.

    The folder holds ``index.pt`` (the per-graph metadata of :attr:`index` and the cell types, genes
    and regions shared by all graphs) and one file per graph in ``graphs/``. The files are pickled: open only
    folders you trust, as for ``torch.load``. Two DiskGraphs cannot be joined with ``+``: build all
    the sections with one ``generate_graphs`` call, which numbers them consistently.
    """

    def __init__(self, path, _index=None, _positions=None):
        self.path = _folder(path)
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
        try:
            if _is_bool(i):
                raise TypeError
            i = operator.index(i)
        except TypeError:
            raise TypeError(f"graph index must be an integer or a slice, not {type(i).__name__}") from None
        if not -len(self) <= i < len(self):
            raise IndexError(f"graph index {i} out of range for {len(self)} graphs")
        return self._load(self._positions[i])

    def __iter__(self):
        for position in self._positions:
            yield self._load(position)

    def __add__(self, other):
        raise TypeError("DiskGraphs cannot be joined with +: build all the sections with one generate_graphs "
                        "call (it numbers sections, timepoints and conditions consistently)")

    __radd__ = __add__

    def subset(self, indices):
        """A view on the graphs at ``indices`` (positions in this sequence, in that order), or on the
        graphs where a boolean mask is true: an array of length ``len(self)``, or a boolean Series
        whose index labels are positions (as those of :attr:`index`, in any order)."""
        if isinstance(indices, (set, frozenset)):
            raise TypeError("graph indices must be ordered: pass a list, not a set")
        if isinstance(indices, pd.Series) and pd.api.types.is_bool_dtype(indices.dtype):
            indices = self._series_mask(indices)
        elif isinstance(indices, (pd.Series, pd.Index)):
            indices = indices.to_numpy()
        elif torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        elif not isinstance(indices, np.ndarray) and not hasattr(indices, "__array__") and not np.isscalar(indices):
            indices = list(indices)   # generators, map, dict_keys, ...
        if isinstance(indices, list):   # element by element, as numpy would promote mixed integer types
            array = np.asarray(indices, dtype=bool) if indices and all(map(_is_bool, indices)) else None
            items = indices
        else:
            array = np.asarray(indices)
            items = array.ravel().tolist() if array.ndim else [array.item()]
            if array.dtype == object and array.size and all(map(_is_bool, items)):
                array = array.astype(bool)
            elif array.dtype != bool and array.dtype.kind not in "iuO" and array.size:
                raise TypeError(f"graph indices must be integers or a boolean mask, not {array.dtype}")
        if array is not None and array.dtype == bool:
            if array.shape != (len(self),):
                raise IndexError(f"a boolean mask must have one entry per graph ({len(self)}), "
                                 f"got shape {array.shape}")
            items = np.flatnonzero(array).tolist()
        positions = []
        for i in items:
            try:
                if _is_bool(i):
                    raise TypeError
                i = operator.index(i)
            except TypeError:
                raise TypeError("graph indices must be integers or a boolean mask, "
                                f"not {type(i).__name__}") from None
            if not -len(self) <= i < len(self):
                raise IndexError(f"graph index {i} out of range for {len(self)} graphs")
            positions.append(self._positions[i])
        return DiskGraphs(self.path, _index=self._index, _positions=positions)

    def _series_mask(self, mask):
        """A boolean Series as a mask in position order, aligned by its labels (those of :attr:`index`)."""
        labels = mask.index
        if len(labels) != len(self) or labels.has_duplicates or (
                len(labels) and not (pd.api.types.is_integer_dtype(labels)
                                     and pd.RangeIndex(len(self)).isin(labels).all())):
            raise IndexError(f"a boolean Series must carry the labels of graphs.index (0 to {len(self) - 1}, "
                             "in any order), one entry per graph")
        mask = mask.reindex(pd.RangeIndex(len(self)))
        if mask.isna().any():
            raise ValueError("a boolean mask cannot contain missing values")
        return mask.to_numpy(dtype=bool)

    def _load(self, position):
        row = self._index["graphs"][position]
        path = os.path.join(self.path, GRAPH_DIR, row["file"])
        packed = torch.load(path, map_location="cpu", weights_only=False)
        if packed.get("token") != self._index.get("token"):
            raise RuntimeError(f"{path} does not belong to the folder opened as {self.path}: "
                               "the folder was written again; open it again with load_graphs")
        return _unpack_graph(packed["stores"], self._index["shared"])

    # -- metadata -------------------------------------------------------------------
    def _column(self, name):
        """Per-graph value of an index column, as stored (no graph is loaded)."""
        return [self._index["graphs"][p][name] for p in self._positions]

    @property
    def index(self):
        """One row per graph (a new DataFrame each time, indexed by position): file, section, timepoint
        and condition codes and labels, n_cells (frames included), n_core_cells, nbytes (tensor bytes
        once loaded) and ndim (2 or 3)."""
        if self._frame is None:
            columns = ["file", "section", "section_label", "timepoint", "timepoint_label", "condition",
                       "condition_label", "n_cells", "n_core_cells", "nbytes", "ndim"]
            self._frame = pd.DataFrame([self._index["graphs"][p] for p in self._positions], columns=columns)
        return self._frame.copy()

    @property
    def nbytes(self):
        """Total tensor bytes of the graphs once loaded."""
        return sum(self._column("nbytes"))

    def _first_value(self, name):
        """``name`` of the first graph of this view, from the index when that graph shares it."""
        shared = self._index["shared"]
        if self._positions:
            row = self._index["graphs"][self._positions[0]]
            if name not in row.get("shared_names", ()):   # (folders written before it was recorded)
                return getattr(self[0], name, None)
        return copy.deepcopy(shared.get(name))

    @property
    def cell_types(self):
        """The cell types of the first graph (read from the index when it shares them)."""
        return self._first_value("cell_types")

    @property
    def genes(self):
        """The genes of the first graph (read from the index when it shares them)."""
        return self._first_value("genes")

    @property
    def regions(self):
        """The regions of the first graph, or None (read from the index when it shares them)."""
        return self._first_value("regions")

    def __repr__(self):
        return (f"DiskGraphs({self.path!r}, {len(self)} graphs, "
                f"{sum(self._column('n_core_cells'))} core cells)")


def save_graphs(graphs, path):
    """
    Write ``graphs`` (a list from ``generate_graphs``, or a :class:`DiskGraphs`) to the folder ``path``,
    which must not exist or be empty, and return them as a :class:`DiskGraphs`.

    Graphs are written one at a time. Tensors are stored losslessly in a compact form (sparse
    counts, 32-bit indices) and are loaded back on the CPU. If writing fails, the files written so
    far are removed again.
    """
    with _GraphWriter(path) as writer:
        for graph in graphs:
            writer.add(graph)
        return writer.close()


def load_graphs(path):
    """Open a folder written by :func:`save_graphs` or ``generate_graphs(..., save_dir=...)``."""
    return DiskGraphs(path)
