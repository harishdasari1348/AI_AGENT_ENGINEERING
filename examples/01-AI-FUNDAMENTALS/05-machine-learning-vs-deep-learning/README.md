# Machine Learning vs Deep Learning: companion notebook

[`ml_vs_dl.ipynb`](ml_vs_dl.ipynb) trains four models on the same small, curved dataset (scikit-learn's two "moons") and compares them, in small cells with a plain-language explanation before each one:

| Model | Kind | What it shows |
|---|---|---|
| Logistic regression on raw `x, y` | Classic ML | A straight-line model struggles on curved data |
| Logistic regression + polynomial features | Classic ML + feature engineering | A human-designed feature set fixes most of the problem |
| Decision tree | Classic ML | Easy-to-read if/else rules |
| `MLPClassifier` (2 hidden layers of 16) | Small neural network (the idea behind deep learning) | Learns its own features from raw inputs, at the cost of more training time and parameters |

It also plots each model's decision boundary, prints the readable weights and rules of the classic models, and tests how much data each model needs. A few "quick check" cells use `assert` to confirm key results.

Article: [Machine Learning vs Deep Learning](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/05-machine-learning-vs-deep-learning.md)

## Open the notebook

- **On GitHub:** just open the file. The saved outputs are shown.
- **In VS Code:** open the file, pick your `.venv` as the kernel and run the cells.
- **In Jupyter in the browser:**

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ml_vs_dl.ipynb
```

No API keys or GPU are needed. Everything runs locally in a few seconds.

## Test it

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 05-machine-learning-vs-deep-learning
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.

## Tested with

- Python 3.14.5 (Windows 11)
- numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2
- nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, pytest 9.2.0
- Last tested: 2026-11-17

Accuracies are reproducible because every random step uses a fixed seed (42). Training times are not: they change from run to run and machine to machine.

> This is teaching code. scikit-learn's `MLPClassifier` is a fine way to see how a neural network behaves, but real deep-learning work uses frameworks such as PyTorch, TensorFlow or JAX on GPUs.
