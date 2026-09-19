"""Worker process of ``generate_graphs(..., save_dir=, n_workers=)``: builds and writes sections.

A plain subprocess started as ``python -c BOOTSTRAP`` (not multiprocessing): nothing is inherited
from the calling process and its script is not imported, so scripts need no
``if __name__ == "__main__"`` guard. The bootstrap moves the task and reply pipes to private
descriptors (what the building code, or a process it starts, prints goes to the worker's log, and
they cannot read the tasks), takes the calling process's module search path, and runs ``main``.
Tasks and replies are length-prefixed pickles. The process ends at the end of its input, or when
the calling process disappears.
"""
import gc
import importlib
import os
import pickle
import struct
import sys
import traceback
import warnings

HEADER = struct.Struct("<Q")
MAX_MESSAGE = 2**36   # a longer message means the stream is corrupted

#: run by ``python -c``, before anything but the standard library can be imported
BOOTSTRAP = r"""
import sys
if sys.path and sys.path[0] == "":
    del sys.path[0]   # the current folder must not shadow the modules imported next
import importlib, os, pickle, signal, struct, threading, time
requests = os.fdopen(os.dup(0), "rb")   # os.dup: not inherited by the processes the build starts
replies = os.fdopen(os.dup(1), "wb")
null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
os.close(null)
os.dup2(2, 1)   # printed output goes to the log
header = requests.read(8)
if len(header) < 8:
    sys.exit(1)
setup = pickle.loads(requests.read(struct.unpack("<Q", header)[0]))

def watch(parent):   # the calling process is gone: stop as Ctrl-C would, then for good
    while os.getppid() == parent:
        time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(2.0)
    os._exit(1)

threading.Thread(target=watch, args=(setup["parent"],), daemon=True).start()
sys.path[:] = setup["path"]
importlib.import_module(setup["module"]).main(requests, replies, setup)
"""


def read_message(stream):
    """The next length-prefixed message of ``stream`` (bytes), or None at its end."""
    header = stream.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    (size,) = HEADER.unpack(header)
    if size > MAX_MESSAGE:
        raise ValueError(f"a message of {size} bytes: the stream is corrupted")
    payload = stream.read(size)
    return payload if len(payload) == size else None


def write_message(stream, payload):
    stream.write(HEADER.pack(len(payload)))
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


def _recorded(caught):
    """The warnings caught while building a section, as the calling process re-emits them."""
    modules = _module_names() if caught else {}
    return [dict(message=_pickled(w.message), text=str(w.message), category=_pickled(w.category),
                 category_name=w.category.__name__, filename=w.filename, lineno=w.lineno,
                 module=modules.get(os.path.abspath(w.filename)) if w.filename else None)
            for w in caught]


def main(requests, replies, setup):
    graphs = importlib.import_module(setup["module"].rsplit(".", 1)[0] + "._graphs")
    while True:
        payload = read_message(requests)
        if payload is None:
            return
        reply, caught = {}, []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")   # the calling process applies its filters
                task = pickle.loads(payload)
                reply["result"] = graphs._write_section(**task)
        except BaseException as error:   # handed to the calling process, which raises it
            reply["error"] = _pickled(error)
            reply["error_text"] = f"{type(error).__name__}: {error}"
            reply["traceback"] = traceback.format_exc()
        task = None
        reply["warnings"] = _recorded(caught)
        encoded = _pickled(reply)
        if encoded is None:   # e.g. a label that cannot be pickled back
            encoded = pickle.dumps(dict(error=None, error_text="the result of the section cannot be pickled",
                                        traceback="", warnings=[]))
        caught = reply = None
        gc.collect()   # the section's memory, before the next one
        write_message(replies, encoded)
