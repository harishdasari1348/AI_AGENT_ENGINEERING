"""Run the companion notebook end-to-end. Fail if any cell raises an error."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

FOLDER = Path(__file__).parent.parent
NOTEBOOK = FOLDER / "supervised_learning.ipynb"


def test_notebook_runs_without_errors():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(FOLDER)}},
    )
    client.execute()
