# Types of AI: companion notebook

[`types_of_ai.ipynb`](types_of_ai.ipynb) is a short, step-by-step Jupyter notebook for the lesson **Types of AI**. All the code is in the notebook, in small cells with a plain-language explanation before each one. It shows:

| Step | What you see |
|---|---|
| Catalogues | The two classifications of AI (by capability and by functionality) and which types exist today |
| Two thermostats | A **reactive** machine vs a **limited-memory** machine making different decisions from the same current reading |
| Classifier | Real systems (Deep Blue, spam filter, self-driving perception, an LLM chatbot) placed on both scales |

Article: [Types of AI](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/03-types-of-ai.md)

## Open the notebook

- **On GitHub:** just open the file. The saved outputs are shown.
- **In VS Code:** open the file, pick your `.venv` as the kernel and run the cells.
- **In Jupyter in the browser:**

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
pip install notebook            # Jupyter itself, if you don't already have it
jupyter notebook types_of_ai.ipynb
```

No API keys and no internet access are needed after installation. The notebook uses only the Python standard library; the pinned packages run it headlessly for the test.

## Test it

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 03-types-of-ai
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.

## Tested with

- Python 3.14.5 (Windows 11)
- ipykernel 7.4.0, nbclient 0.11.0, nbformat 5.11.1, pytest 9.2.0
- Last tested: 2026-11-17 (`1 passed`)

The Jupyter Notebook app is not pinned because it was not part of the tested environment (the test executes the notebook headlessly with nbclient). Any recent version of Jupyter Notebook or JupyterLab can open the file.

> This is teaching code. The thermostats are deliberately tiny so the difference between "no memory" and "limited memory" is easy to see.
