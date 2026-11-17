# Reinforcement Learning: Q-Learning in a Custom GridWorld

Companion code for [Reinforcement Learning](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/08-reinforcement-learning.md).

Demonstrates reinforcement learning using **only the Python standard library** (no external game engines or heavy dependencies). We build a 4x4 GridWorld environment from scratch, implement tabular Q-learning with epsilon-greedy exploration, train the agent over 600 episodes, and extract the learned policy.

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook reinforcement_learning.ipynb
```

## Run the tests

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 08-reinforcement-learning
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.