"""Worker process of ``generate_graphs(..., save_dir=, n_workers=)``: builds and writes sections.

    python <this file> <folder holding the package> <package name>

A plain subprocess (not multiprocessing): nothing is inherited from the calling process and its script
is not imported, so scripts need no ``if __name__ == "__main__"`` guard. Tasks arrive on stdin and
results leave on stdout, each as a length-prefixed pickle. The process ends at the end of stdin, or
when the calling process disappears.
"""
import gc
import importlib
import os
import pickle
import struct
import sys
import threading
import time
import traceback
import warnings

_HEADER = struct.Struct("<Q")


def read_message(stream):
    """The next length-prefixed message of ``stream`` (bytes), or None at its end."""
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    return payload if len(payload) == size else None


def write_message(stream, payload):
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _pickled(value):
    try:
        return pickle.dumps(value)
    except Exception:
        return None


def _module_names():
    """File -> name of the modules imported here (warning filters match module names)."""
    names = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str):
            names.setdefault(os.path.abspath(path), name)
    return names


def _watch_parent(parent):
    """End this process when the one that started it is gone."""
    while True:
        time.sleep(1.0)
        if os.getppid() != parent:
            os._exit(1)


def main(root, package):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != here]   # not a package root
    sys.path.insert(0, root)
    requests, replies = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr   # what the building code prints must not mix with the replies
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    graphs = None
    while True:
        payload = read_message(requests)
        if payload is None:
            return 0
        reply, caught = {}, []
        try:
            if graphs is None:   # its import warnings are ignored (PYTHONWARNINGS): the caller showed them
                graphs = importlib.import_module(f"{package}.preprocessing._graphs")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")   # the calling process applies its filters
                task = pickle.loads(payload)
                reply["result"] = graphs._write_section(**task)
        except BaseException as error:   # handed to the calling process, which raises it
            reply["error"] = _pickled(error)
            reply["error_text"] = f"{type(error).__name__}: {error}"
            reply["traceback"] = traceback.format_exc()
        task = None
        modules = _module_names() if caught else {}
        reply["warnings"] = [(_pickled(w.category), w.category.__name__, str(w.message), w.filename, w.lineno,
                              modules.get(os.path.abspath(w.filename)) if w.filename else None)
                             for w in caught]
        encoded = _pickled(reply)
        if encoded is None:   # e.g. a label that cannot be pickled back
            encoded = pickle.dumps(dict(error=None, error_text="the result of the section cannot be pickled",
                                        traceback="", warnings=[]))
        caught = reply = None
        gc.collect()   # the section's memory, before the next one
        write_message(replies, encoded)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
