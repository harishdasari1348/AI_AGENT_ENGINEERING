# Applications of AI in Real-World Industries

Companion notebook and tests for [Applications of AI in Real-World Industries](../../../docs/01-AI-FUNDAMENTALS/01-introduction-to-artificial-intelligence/10-applications-of-ai-in-real-world-industries.md).

This project demonstrates practical machine learning workflows across five core sectors:
1. **Healthcare:** Clinical diagnosis using Logistic Regression on biopsy measurements (Wisconsin Breast Cancer dataset).
2. **Finance:** Credit card fraud and anomaly detection using `IsolationForest`.
3. **Retail & E-Commerce:** Customer behavioral segmentation using `KMeans` clustering.
4. **Transportation & Logistics:** Dispatch delivery duration forecasting using Ridge Regression.
5. **Customer Service / NLP:** Automated ticket routing using `CountVectorizer` and `LogisticRegression`.

## Quick start

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ai_industry_applications.ipynb
```

## Run the tests

```bash
cd ../..
pip install -r requirements-test.txt
python -m pytest -k 10-applications-of-ai-in-real-world-industries
```

`examples/tests/test_notebooks.py` runs every example notebook from top to bottom and fails if any cell raises an error (including the notebook's own `assert` quick checks). `-k` runs only this example; leave it off to run them all.
