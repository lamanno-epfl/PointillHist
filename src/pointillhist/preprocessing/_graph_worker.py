"""Worker process of ``generate_graphs(..., save_dir=, n_workers=)``: builds and writes one section.

    python -m pointillhist.preprocessing._graph_worker <task.pkl> <result.pkl>

A plain subprocess (not multiprocessing): it inherits nothing from the calling process and does not
import the caller's script, so scripts need no ``if __name__ == "__main__"`` guard.
"""
import os
import pickle
import sys
import traceback


def main(task_path, result_path):
    with open(task_path, "rb") as handle:
        task = pickle.load(handle)
    try:
        from ._graphs import _write_section

        result = _write_section(**task)
    except BaseException as error:   # handed to the calling process, which raises it
        result = {"error": error, "traceback": traceback.format_exc()}
    tmp = f"{result_path}.tmp"
    try:
        with open(tmp, "wb") as handle:
            pickle.dump(result, handle)
    except Exception:   # an exception that cannot be pickled
        with open(tmp, "wb") as handle:
            pickle.dump({"error": RuntimeError(result.get("traceback", "")), "traceback": result.get("traceback", "")},
                        handle)
    os.replace(tmp, result_path)
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
