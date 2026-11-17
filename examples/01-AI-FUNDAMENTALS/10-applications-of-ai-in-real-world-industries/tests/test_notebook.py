"""Run the companion notebook end-to-end. Fail if any cell raises an error."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

FOLDER = Path(__file__).resolve().parent.parent
NOTEBOOK = FOLDER / "ai_industry_applications.ipynb"


def test_notebook_runs_without_errors() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(FOLDER)}},
    )
    client.execute()  # raises CellExecutionError if any cell fails
    # Save the executed notebook with populated outputs
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert all(out.get("output_type") != "error" for out in cell.get("outputs", []))
