"""Run every example notebook from top to bottom and fail if any cell raises.

New notebooks under examples/ are picked up automatically; no per-example test
file is needed. Notebooks are executed in memory and are never written back.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

EXAMPLES_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(
    p for p in EXAMPLES_DIR.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
)

# Keep matplotlib from trying to open a window (CI has no display).
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.mark.parametrize(
    "notebook",
    NOTEBOOKS,
    ids=[p.relative_to(EXAMPLES_DIR).as_posix() for p in NOTEBOOKS],
)
def test_notebook_runs_without_errors(notebook: Path) -> None:
    nb = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook.parent)}},
    )
    client.execute()  # raises CellExecutionError if any cell fails
