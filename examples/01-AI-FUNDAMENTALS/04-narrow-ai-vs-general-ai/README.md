# Narrow AI vs General AI: companion notebook

[`narrow_vs_general_ai.ipynb`](narrow_vs_general_ai.ipynb) shows what "narrow AI" means in practice, in small cells with a plain-language explanation before each one:

1. Train a small model that does **one job** (recognise 8x8 handwritten digits).
2. See that it does that job well (about 96% accuracy on unseen digits).
3. Hand it things that are **not digits** and watch it still answer, sometimes with high confidence.
4. Hand it a **sentence** and watch it fail to accept the input at all.
5. Add a simple **domain guard** so the system can notice unfamiliar inputs and hand them to a human.

All the code lives in the notebook. A few "quick check" cells use `assert` to confirm the key results as you go.

Article: [Narrow AI vs General AI](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/04-narrow-ai-vs-general-ai.md)

## Open the notebook

- **On GitHub:** just open the file. The saved outputs are shown.
- **In VS Code:** open the file, pick your `.venv` as the kernel and run the cells.
- **In Jupyter in the browser:**

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab
jupyter lab narrow_vs_general_ai.ipynb
```

No API keys or downloads are needed: the digits dataset ships with scikit-learn.

## Test it

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 04-narrow-ai-vs-general-ai
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.

## Tested with

- Python 3.14.5 (Windows 11)
- numpy 2.5.5, scikit-learn 1.9.1
- nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, pytest 9.2.0
- Last tested: 2026-11-17

> This is teaching code. The domain guard is a simple nearest-neighbour distance check, not a production out-of-distribution detector.
