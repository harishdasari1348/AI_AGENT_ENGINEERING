# Unsupervised Learning

Companion code for [Unsupervised Learning](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/07-unsupervised-learning.md).

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook unsupervised_learning.ipynb
```

## Run the tests

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 07-unsupervised-learning
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.