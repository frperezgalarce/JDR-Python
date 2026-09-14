"""Execute only the synthetic and 18-star validation notebooks in fresh kernels."""

import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    results = []
    for name in ("test.ipynb", "experiment4.ipynb"):
        path = root / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbformat.write(notebook, path)
        results.append(
            {
                "notebook": str(path.relative_to(root)),
                "status": "passed",
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            }
        )
    destination = root / "results" / "validation_v2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "notebook_execution.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
