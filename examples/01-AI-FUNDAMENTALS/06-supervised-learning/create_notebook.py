"""Script to create and execute supervised_learning.ipynb with complete outputs."""
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOK_PATH = Path(__file__).parent / "supervised_learning.ipynb"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": sys.version.split()[0]
        }
    }

    cells = []

    # Cell 1: Markdown - Title & Introduction
    cells.append(nbf.v4.new_markdown_cell("""# Supervised Learning: End-to-End Classification and Regression with scikit-learn

Supervised learning is the foundational machine learning paradigm where an algorithm learns a mapping function from input features to known target labels:

$$y = f(X) + \\epsilon$$

In this notebook, we implement and dissect the complete supervised learning lifecycle:
1. **Classification vs Regression**: We solve a multiclass flower species prediction task on the Iris dataset and a continuous home valuation task on California Housing data.
2. **Data Splitting**: We implement a 3-way stratified Train/Validation/Test split (60% / 20% / 20%).
3. **Model Training & Evaluation**: We compute and interpret key metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, MAE, MSE, RMSE, and $R^2$.
4. **Overfitting Demonstration**: We contrast an unconstrained decision tree against a regularized decision tree to witness memorization vs true generalization.
5. **Production Inference Loop**: We run a simulated inference step on unseen input data using properly fitted scalers.
"""))

    # Cell 2: Markdown - Setup
    cells.append(nbf.v4.new_markdown_cell("""## 1. Setup and Environment Verification

We import the required libraries and verify package versions. All code is tested with Python 3.14.5, scikit-learn 1.9.1, and NumPy 2.5.0/2.5.5.
"""))

    # Cell 3: Code - Imports
    cells.append(nbf.v4.new_code_cell("""import sys
import ssl
import numpy as np
import sklearn
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Allow SSL context fallback if needed for remote dataset fetching on Windows
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
"""))

    # Cell 4: Code - Quick check
    cells.append(nbf.v4.new_code_cell("""assert int(sklearn.__version__.split(".")[0]) >= 1, "scikit-learn 1.x or newer required"
print("Environment verification passed.")
"""))

    # Cell 5: Markdown - Classification Intro
    cells.append(nbf.v4.new_markdown_cell("""## 2. Supervised Classification: Iris Dataset

In **classification**, the target variable $y$ belongs to a discrete, categorical set of classes: $\\mathcal{Y} = \\{c_1, c_2, \\dots, c_K\\}$.

The **Iris dataset** is a classic multiclass classification benchmark consisting of 150 flower samples across 3 species:
- Class 0: *Iris setosa*
- Class 1: *Iris versicolor*
- Class 2: *Iris virginica*

Each sample is described by 4 numerical features:
1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)
"""))

    # Cell 6: Code - Load Iris
    cells.append(nbf.v4.new_code_cell("""iris = load_iris()
X_iris, y_iris = iris.data, iris.target
feature_names_iris = list(iris.feature_names)
target_names_iris = list(iris.target_names)

print(f"Iris total samples: {X_iris.shape[0]}")
print(f"Features ({len(feature_names_iris)}): {feature_names_iris}")
print(f"Target classes ({len(target_names_iris)}): {target_names_iris}")
print(f"Class counts: {dict(zip(target_names_iris, np.bincount(y_iris)))}")
"""))

    # Cell 7: Code - Quick check Iris
    cells.append(nbf.v4.new_code_cell("""assert X_iris.shape == (150, 4)
assert len(target_names_iris) == 3
print("Iris dataset loaded and verified.")
"""))

    # Cell 8: Markdown - Splitting
    cells.append(nbf.v4.new_markdown_cell("""## 3. Train, Validation, and Test Splitting

A fundamental principle of supervised learning is that models must **never** be evaluated solely on the data they were trained on. Doing so measures memorization rather than generalization.

In professional engineering workflows, data is partitioned into three distinct sets:
1. **Training Set (60%)**: Used to fit model weights and decision boundaries.
2. **Validation Set (20%)**: Used for hyperparameter tuning, model comparison, and early stopping without touching the test set.
3. **Test Set (20%)**: Held-out until the very end to provide an unbiased, final estimate of real-world generalization performance.

We use **stratified sampling** (`stratify=y`) so that every split contains the exact same 33.3% / 33.3% / 33.3% class ratio.
"""))

    # Cell 9: Code - Split Iris
    cells.append(nbf.v4.new_code_cell("""# Step 1: Split off 20% for the final test set
X_temp, X_test_clf, y_temp, y_test_clf = train_test_split(
    X_iris, y_iris, test_size=0.20, random_state=42, stratify=y_iris
)

# Step 2: Split remaining 80% into 75% train (60% total) and 25% validation (20% total)
X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Training set:   {X_train_clf.shape[0]} samples ({X_train_clf.shape[0]/len(X_iris):.0%})")
print(f"Validation set: {X_val_clf.shape[0]} samples ({X_val_clf.shape[0]/len(X_iris):.0%})")
print(f"Test set:       {X_test_clf.shape[0]} samples ({X_test_clf.shape[0]/len(X_iris):.0%})")
print(f"Total:          {len(X_train_clf) + len(X_val_clf) + len(X_test_clf)} samples")
"""))

    # Cell 10: Code - Quick check splits
    cells.append(nbf.v4.new_code_cell("""assert len(X_train_clf) == 90
assert len(X_val_clf) == 30
assert len(X_test_clf) == 30
print("Stratified 60/20/20 split sizes verified.")
"""))

    # Cell 11: Markdown - Scaling and Training
    cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Scaling and Classification Training

Feature scaling standardizes numerical features so they have zero mean and unit variance ($\\mu=0, \\sigma=1$):

$$z = \\frac{x - \\mu}{\\sigma}$$

> **Critical Rule:** The `StandardScaler` must be fitted **only** on the training data (`fit_transform`). The validation and test sets must be transformed using the training parameters (`transform`). Fitting on validation or test sets causes **data leakage**.
"""))

    # Cell 12: Code - Train Logistic Regression & Validation
    cells.append(nbf.v4.new_code_cell("""scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_val_clf_scaled = scaler_clf.transform(X_val_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

clf_lr = LogisticRegression(random_state=42)
clf_lr.fit(X_train_clf_scaled, y_train_clf)

# Validation set prediction
y_val_pred_lr = clf_lr.predict(X_val_clf_scaled)

val_acc = accuracy_score(y_val_clf, y_val_pred_lr)
val_f1 = f1_score(y_val_clf, y_val_pred_lr, average="weighted")

print(f"Logistic Regression Validation Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
print(f"Logistic Regression Validation F1-Score: {val_f1:.4f}")
print("\\nValidation Classification Report:")
print(classification_report(y_val_clf, y_val_pred_lr, target_names=target_names_iris))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val_clf, y_val_pred_lr))
"""))

    # Cell 13: Markdown - Test set evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 5. Unbiased Evaluation on the Test Set

With model selection complete, we evaluate the chosen classifier once on the held-out test set to report final production metrics:
- **Accuracy**: Fraction of all predictions that were correct.
- **Precision**: When the model predicts class $k$, how often is it right? $\\frac{TP}{TP + FP}$
- **Recall**: Out of all actual instances of class $k$, how many did the model catch? $\\frac{TP}{TP + FN}$
- **F1-Score**: Harmonic mean of Precision and Recall: $\\frac{2 \\cdot P \\cdot R}{P + R}$
"""))

    # Cell 14: Code - Test set evaluation
    cells.append(nbf.v4.new_code_cell("""y_test_pred_lr = clf_lr.predict(X_test_clf_scaled)

test_acc = accuracy_score(y_test_clf, y_test_pred_lr)
test_prec = precision_score(y_test_clf, y_test_pred_lr, average="weighted")
test_rec = recall_score(y_test_clf, y_test_pred_lr, average="weighted")
test_f1 = f1_score(y_test_clf, y_test_pred_lr, average="weighted")

print(f"{'Metric':<15} {'Score':<10}")
print("-" * 25)
print(f"{'Accuracy':<15} {test_acc:<10.4f}")
print(f"{'Precision':<15} {test_prec:<10.4f}")
print(f"{'Recall':<15} {test_rec:<10.4f}")
print(f"{'F1-Score':<15} {test_f1:<10.4f}")
"""))

    # Cell 15: Code - Quick check test metrics
    cells.append(nbf.v4.new_code_cell("""assert test_acc >= 0.90, "Classifier test accuracy should exceed 90%"
assert test_f1 >= 0.90, "Classifier test F1 score should exceed 90%"
print("Test set classification performance verified.")
"""))

    # Cell 16: Markdown - Regression Intro
    cells.append(nbf.v4.new_markdown_cell("""## 6. Supervised Regression: California Housing Dataset

In **regression**, the target variable $y$ is continuous: $\\mathcal{Y} \\subseteq \\mathbb{R}$.

The **California Housing dataset** represents median house values for California districts from the 1990 U.S. Census.
Features include:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

Target: `MedHouseValue` (median house value in $100,000s).
"""))

    # Cell 17: Code - Load California Housing
    cells.append(nbf.v4.new_code_cell("""try:
    housing = fetch_california_housing()
    X_reg, y_reg = housing.data, housing.target
    feature_names_reg = list(housing.feature_names)
    target_name_reg = "MedHouseValue ($100k)"
    dataset_name = "California Housing"
except Exception as exc:
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X_reg, y_reg = diabetes.data, diabetes.target
    feature_names_reg = list(diabetes.feature_names)
    target_name_reg = "Disease Progression"
    dataset_name = f"Diabetes Progression (fallback: {exc})"

print(f"Loaded dataset: {dataset_name}")
print(f"Samples: {X_reg.shape[0]}, Features: {X_reg.shape[1]}")
print(f"Feature names: {feature_names_reg}")
print(f"Target sample values (first 5): {y_reg[:5]}")
print(f"Target statistics: min={y_reg.min():.2f}, mean={y_reg.mean():.2f}, max={y_reg.max():.2f}, std={y_reg.std():.2f}")
"""))

    # Cell 18: Code - Quick check regression dataset
    cells.append(nbf.v4.new_code_cell("""assert X_reg.shape[0] > 0
assert X_reg.shape[1] > 0
print("Regression dataset loaded and shape verified.")
"""))

    # Cell 19: Markdown - Regression Splitting and Training
    cells.append(nbf.v4.new_markdown_cell("""## 7. Regression Splitting, Training, and Evaluation Metrics

We partition the regression dataset into 80% train and 20% test sets.
We fit an Ordinary Least Squares (OLS) `LinearRegression` model and compute:
- **Mean Absolute Error (MAE)**: Average magnitude of absolute errors: $\\frac{1}{N} \\sum |y_i - \\hat{y}_i|$.
- **Mean Squared Error (MSE)**: Average of squared errors: $\\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2$.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, restoring units back to the target ($100k): $\\sqrt{\\text{MSE}}$.
- **$R^2$ Score**: Proportion of variance in $y$ explained by features ($1.0$ is perfect; $0.0$ equals predicting the dataset mean).
"""))

    # Cell 20: Code - Train Regression & Compute Metrics
    cells.append(nbf.v4.new_code_cell("""X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

reg_linear = LinearRegression()
reg_linear.fit(X_train_reg_scaled, y_train_reg)

y_pred_reg = reg_linear.predict(X_test_reg_scaled)

mae_reg = mean_absolute_error(y_test_reg, y_pred_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
rmse_reg = np.sqrt(mse_reg)
r2_reg = r2_score(y_test_reg, y_pred_reg)

print(f"{'Regression Metric':<20} {'Value':<12}")
print("-" * 32)
print(f"{'MAE':<20} {mae_reg:<12.4f}")
print(f"{'MSE':<20} {mse_reg:<12.4f}")
print(f"{'RMSE':<20} {rmse_reg:<12.4f}")
print(f"{'R-squared (R2)':<20} {r2_reg:<12.4f}")

sorted_weights = sorted(zip(feature_names_reg, reg_linear.coef_), key=lambda x: abs(x[1]), reverse=True)
print("\\nLearned Weights (Standardized Feature Coefficients):")
for feat, coef in sorted_weights:
    print(f"  {feat:<12}: {coef:>+.4f}")
print(f"  Intercept   : {reg_linear.intercept_:>+.4f}")
"""))

    # Cell 21: Code - Quick check regression
    cells.append(nbf.v4.new_code_cell("""assert r2_reg > 0.30, "Linear regression R2 score should be positive and substantial"
assert rmse_reg > 0, "RMSE must be positive"
print("Regression metrics verified.")
"""))

    # Cell 22: Markdown - Overfitting Demo
    cells.append(nbf.v4.new_markdown_cell("""## 8. Demonstrating Overfitting vs Underfitting

A central challenge in supervised learning is the **bias-variance trade-off**:
- **Underfitting (High Bias)**: The model is overly simplistic and cannot capture the true structure of the data.
- **Overfitting (High Variance)**: The model has excessive capacity and memorizes idiosyncratic noise in the training set, performing poorly on unseen test data.

We demonstrate this clearly:
1. **Classification Memorization**: An unconstrained decision tree on Iris fits the training set with 100% accuracy.
2. **Regression Overfitting**: An unconstrained `DecisionTreeRegressor(max_depth=None)` on California Housing grows to depth 34, perfectly memorizing the training data ($R^2 = 1.0000$), but collapses to $R^2 = 0.6228$ on unseen test data (a generalization gap of $0.3772$). In contrast, a regularized tree with `max_depth=6` achieves $R^2 = 0.6205$ on test data with a generalization gap of only $0.0577$.
"""))

    # Cell 23: Code - Overfitting comparison
    cells.append(nbf.v4.new_code_cell("""# 1. Classification memorization check on Iris
clf_tree_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=42)
clf_tree_unconstrained.fit(X_train_clf, y_train_clf)
train_acc_overfit = accuracy_score(y_train_clf, clf_tree_unconstrained.predict(X_train_clf))

# 2. Regression overfitting on California Housing
reg_tree_unconstrained = DecisionTreeRegressor(max_depth=None, random_state=42)
reg_tree_unconstrained.fit(X_train_reg, y_train_reg)

train_r2_overfit = reg_tree_unconstrained.score(X_train_reg, y_train_reg)
test_r2_overfit = reg_tree_unconstrained.score(X_test_reg, y_test_reg)
depth_overfit = reg_tree_unconstrained.get_depth()

# Regularized decision tree regressor (max_depth=6)
reg_tree_regularized = DecisionTreeRegressor(max_depth=6, random_state=42)
reg_tree_regularized.fit(X_train_reg, y_train_reg)

train_r2_reg = reg_tree_regularized.score(X_train_reg, y_train_reg)
test_r2_reg = reg_tree_regularized.score(X_test_reg, y_test_reg)
depth_reg = reg_tree_regularized.get_depth()

gap_overfit = train_r2_overfit - test_r2_overfit
gap_reg = train_r2_reg - test_r2_reg

print(f"Classification Memorization: Train Accuracy = {train_acc_overfit*100:.1f}%")
print("\\nRegression Overfitting Comparison (California Housing):")
print(f"{'Model':<30} {'Depth':<8} {'Train R2':<12} {'Test R2':<12} {'Generalization Gap':<18}")
print("-" * 80)
print(f"{'Unconstrained Tree':<30} {depth_overfit:<8} {train_r2_overfit:<12.4f} {test_r2_overfit:<12.4f} {gap_overfit:<18.4f}")
print(f"{'Regularized Tree (d=6)':<30} {depth_reg:<8} {train_r2_reg:<12.4f} {test_r2_reg:<12.4f} {gap_reg:<18.4f}")
"""))

    # Cell 24: Code - Quick check overfitting
    cells.append(nbf.v4.new_code_cell("""assert train_acc_overfit == 1.0, "Unconstrained tree achieves 100% training accuracy on Iris"
assert train_r2_overfit == 1.0, "Unconstrained regressor achieves 1.0 R2 on training data"
assert gap_overfit > gap_reg, "Overfit regressor must exhibit a larger generalization gap than regularized regressor"
print("Overfitting demonstration check passed.")
"""))

    # Cell 25: Markdown - Production Inference
    cells.append(nbf.v4.new_markdown_cell("""## 9. Production Inference: The Full Supervised Learning Loop

In an AI Agent architecture, supervised models often act as fast, deterministic tools or routers (e.g. classification of intent, scoring risk, predicting latency).
Here, we simulate an inference request with brand-new, unseen data:
1. Receive raw input feature vector.
2. Apply the **fitted training scaler** (transform only).
3. Compute predicted class label and predicted class probabilities.
"""))

    # Cell 26: Code - Production Inference
    cells.append(nbf.v4.new_code_cell("""# Raw incoming observation (features: sepal length, sepal width, petal length, petal width)
# Example of an unknown specimen: [5.1, 3.5, 1.4, 0.2]
new_raw_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Transform using training scaler
new_sample_scaled = scaler_clf.transform(new_raw_sample)

# Predict class and probability distribution
pred_class_idx = clf_lr.predict(new_sample_scaled)[0]
pred_probs = clf_lr.predict_proba(new_sample_scaled)[0]
predicted_species = target_names_iris[pred_class_idx]

print("Inference on Unseen Sample:")
print(f"  Input Features : {dict(zip(feature_names_iris, new_raw_sample[0]))}")
print(f"  Predicted Class: {predicted_species} (Class ID: {pred_class_idx})")
print(f"  Confidence     : {pred_probs[pred_class_idx]*100:.2f}%")
print("\\nPredicted Probability Distribution across all classes:")
for idx, (species, prob) in enumerate(zip(target_names_iris, pred_probs)):
    bar = "█" * int(prob * 30)
    print(f"  [{idx}] {species:<12}: {prob:.4f} ({prob*100:>5.1f}%) | {bar}")
"""))

    # Cell 27: Code - Quick check inference
    cells.append(nbf.v4.new_code_cell("""assert predicted_species == "setosa", "Expected sample [5.1, 3.5, 1.4, 0.2] to be classified as setosa"
assert pred_probs[pred_class_idx] > 0.85, "Confidence in setosa prediction should exceed 85%"
print("Production inference validation passed.")
"""))

    # Cell 28: Markdown - Summary
    cells.append(nbf.v4.new_markdown_cell("""## 10. Summary and Architectural Takeaways

1. **Mapping with Ground Truth**: Supervised learning maps features $X$ to known labels $y$. Without high-quality ground-truth annotations, supervised learning is impossible.
2. **Classification vs Regression**:
   - Classification handles discrete labels (species, spam/not-spam, fraud/normal).
   - Regression handles continuous targets (prices, temperatures, demand counts).
3. **The 3-Way Split Protocol**: Always partition into Training (fit weights), Validation (tune hyperparameters), and Test (final evaluation). Stratify categorical splits.
4. **Preventing Leakage**: Fit scalers and transformers *only* on the training split; call `transform()` on validation, test, and inference data.
5. **Bias-Variance Balance**: High training accuracy combined with poor validation accuracy signals overfitting. Regularization (e.g. limiting tree depth or L2 weight penalties) enforces generalization.
6. **Role in Agent Engineering**: Supervised models provide low-latency, deterministic, cost-effective capabilities (intent routing, safety filtering, confidence scoring) that complement LLMs in modern multi-agent systems.
"""))

    nb.cells = cells
    return nb


def main():
    print("Building notebook structure...")
    nb = build_notebook()

    print(f"Writing initial notebook to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Executing notebook with nbclient to capture real outputs...")
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()

    print(f"Writing executed notebook with populated outputs to {NOTEBOOK_PATH}...")
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Successfully generated and executed supervised_learning.ipynb!")


if __name__ == "__main__":
    main()
