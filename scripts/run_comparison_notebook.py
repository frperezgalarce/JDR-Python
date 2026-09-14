"""Execute only the self-contained comparison, using this Python as its kernel."""

from pathlib import Path
import argparse
import json
import sys
from hashlib import sha256
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    adjacent = Path(__file__).resolve().parent / "jdr_vs_classical_features_1000.ipynb"
    default_notebook = (
        adjacent
        if adjacent.exists()
        else Path(__file__).resolve().parents[1]
        / "notebooks/jdr_vs_classical_features_1000.ipynb"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        nargs="?",
        default=str(default_notebook),
    )
    args = parser.parse_args()
    path = Path(args.notebook).resolve()
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    NotebookClient(
        notebook,
        km=manager,
        timeout=1800,
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)
    portable = path.parent / "inputs/cohort.csv"
    output = (
        path.parent / "outputs"
        if portable.exists()
        else path.parent.parent / "results/representation_comparison_1000"
    )
    result = dict(
        status="passed",
        fresh_kernel=True,
        kernel_python=sys.executable,
        code_cells=sum(c.cell_type == "code" for c in notebook.cells),
        notebook_sha256=sha256(path.read_bytes()).hexdigest(),
        code_cells_sha256=sha256(
            "\n\n".join(
                c.source for c in notebook.cells if c.cell_type == "code"
            ).encode()
        ).hexdigest(),
    )
    (output / "notebook_execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
