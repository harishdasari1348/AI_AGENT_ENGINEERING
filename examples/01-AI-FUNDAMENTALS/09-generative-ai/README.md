# Generative AI

Companion code for [Generative AI](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/09-generative-ai.md).

This project demonstrates the core mechanisms of **Generative Artificial Intelligence** using standard Python and NumPy, requiring zero external API keys. It builds from first principles:
- Discriminative ($P(Y \mid X)$) versus Generative ($P(X)$) models
- Tokenization, vocabulary indexing, and special sequence boundaries (`<BOS>`, `<EOS>`)
- Bigram Language Model trained on transition counts with Laplace smoothing
- Probability distributions, logits, softmax, and temperature-controlled sampling ($T \to 0$, $T = 0.8$, $T = 1.5$)
- Measuring generative quality with Cross-Entropy Loss and Perplexity
- N-gram Markov Chain text generation with variable context windows ($k=1, 2, 3$)
- Conceptual and architectural bridge to modern Transformer Large Language Models (LLMs)

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook generative_ai.ipynb
```

## Run the tests

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 09-generative-ai
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.