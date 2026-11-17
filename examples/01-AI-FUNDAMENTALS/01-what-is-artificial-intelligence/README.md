# AI Ticket Triage: companion notebook for "What is Artificial Intelligence?"

[`what_is_ai_playbook.ipynb`](what_is_ai_playbook.ipynb) shows three ways of building "intelligent" software side by side, in small cells with a plain-language explanation before each one:

| Approach | What it shows |
|---|---|
| Symbolic (rule-based) AI | Knowledge written by a human as keyword rules |
| Machine learning | Behaviour learned from labelled examples (TF-IDF + logistic regression) |
| AI agent | A perceive → decide → act → learn loop with human escalation |

It also includes experiments (threshold tuning, looking inside the model) and a "try it yourself" cell.

Article: [What is Artificial Intelligence?](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/01-what-is-artificial-intelligence.md)

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
jupyter lab what_is_ai_playbook.ipynb
```

No API keys are needed. Everything runs locally.

## Test it

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 01-what-is-artificial-intelligence
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.

## Tested with

- Python 3.14.5 (Windows 11)
- numpy 2.5.5, scikit-learn 1.9.1
- nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, pytest 9.2.0
- Last tested: 2026-11-17

> This is teaching code, not a production system. The dataset has only 30 training examples.
