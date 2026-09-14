"""Run only the explicitly requested 100-curve experiment notebook."""

from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "experiment_100_journal.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)
    (root / "results" / "benchmark_100" / "notebook_execution.json").write_text(
        json.dumps(
            {
                "notebook": str(path.relative_to(root)),
                "status": "passed",
                "fresh_kernel": True,
                "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
            },
            indent=2,
        )
        + "\n"
    )
    print("100-curve notebook completed successfully.")


if __name__ == "__main__":
    main()
