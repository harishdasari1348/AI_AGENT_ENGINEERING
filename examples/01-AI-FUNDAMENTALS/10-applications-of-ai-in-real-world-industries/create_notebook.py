"""Create and execute the ai_industry_applications.ipynb companion notebook.

This script uses nbformat and nbclient to programmatically construct
and run the notebook, capturing all actual execution outputs.
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = NOTEBOOK_DIR / "ai_industry_applications.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.14.5",
    }

    cells = []

    # Title & Introduction
    cells.append(new_markdown_cell("""# Applications of AI in Real-World Industries

Companion notebook for **Track 01: AI and LLM Fundamentals — Module 1.1: Topic 10**.

In this notebook, we implement end-to-end machine learning workflows for five distinct real-world industry applications using Python and `scikit-learn`:

1. **Healthcare:** Clinical diagnosis & tumor classification (`load_breast_cancer` dataset).
2. **Finance:** Credit card transaction anomaly & fraud detection (`IsolationForest`).
3. **Retail & E-Commerce:** Customer behavioral segmentation for targeted marketing (`KMeans`).
4. **Transportation & Logistics:** Dispatch trip duration and ETA estimation (`Ridge` regression).
5. **Customer Service / NLP:** Automated customer inquiry triage & intent routing (`CountVectorizer` + `LogisticRegression`).

Every section follows a standard production pipeline:
- **Problem Formulation:** Defining the operational objective and metrics.
- **Data Engineering:** Preparing structured features and domain splits.
- **Model Training:** Fitting scikit-learn estimators within clean pipelines.
- **Evaluation:** Measuring industry-relevant metrics (Accuracy, Recall, MAE, Silhouette).
- **Assertion Checks:** Verifying mathematical guarantees and output invariants."""))

    # Cell 1: Environment check
    cells.append(new_code_cell("""# Environment Verification
import sys
import numpy as np
import sklearn

print(f"Python version:      {sys.version.split()[0]}")
print(f"NumPy version:       {np.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")

assert sys.version_info >= (3, 10), "Requires Python 3.10+"
assert sklearn.__version__ >= "1.4.0", "Requires scikit-learn 1.4+"""))

    # Section 1: Healthcare
    cells.append(new_markdown_cell("""## 1. Healthcare: Clinical Disease Diagnosis

### Real-World Operational Context
In clinical oncology, pathologists analyze fine needle aspirate (FNA) biopsies. Manual microscopic inspection is time-consuming and subject to inter-observer variability. An AI diagnostic assistant analyzes digitized cell nucleus morphometry (radius, texture, perimeter, concavity) to provide a high-confidence triage score (benign vs. malignant).

In healthcare screening, **Recall (Sensitivity)** on the malignant class is the paramount metric: false negatives (missing a malignancy) have severe clinical consequences, whereas false positives can be re-evaluated through secondary biopsy."""))

    cells.append(new_code_cell("""from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# 1. Load clinical dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names  # 0: malignant, 1: benign

print(f"Dataset shape: {X.shape[0]} patient samples, {X.shape[1]} biological features")
print(f"Class distribution: {np.sum(y == 0)} malignant, {np.sum(y == 1)} benign")

# 2. Split train/test (80/20) stratified to preserve malignancy ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Build diagnostic pipeline (standardization + logistic regression with L2 regularization)
diagnostic_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=1000)
)
diagnostic_pipeline.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = diagnostic_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
# In sklearn breast cancer, 0=malignant, 1=benign. Malignancy recall pos_label=0:
malignant_recall = recall_score(y_test, y_pred, pos_label=0)

print(f"Test Accuracy:         {test_acc * 100:.2f}%")
print(f"Malignancy Recall:     {malignant_recall * 100:.2f}%")
print("\\nConfusion Matrix [rows=True, cols=Pred]:")
print(f"             Pred Malignant  Pred Benign")
print(f"True Malig:       {confusion_matrix(y_test, y_pred)[0,0]:3d}             {confusion_matrix(y_test, y_pred)[0,1]:3d}")
print(f"True Benign:      {confusion_matrix(y_test, y_pred)[1,0]:3d}             {confusion_matrix(y_test, y_pred)[1,1]:3d}")

# 5. Extract top diagnostic risk factors (highest negative weights -> predictive of malignant class 0)
clf = diagnostic_pipeline.named_steps["logisticregression"]
top_malignant_indices = np.argsort(clf.coef_[0])[:3]
print("\\nTop Morphological Features Predictive of Malignancy:")
for idx in top_malignant_indices:
    print(f"  - {feature_names[idx]}: weight = {clf.coef_[0][idx]:.3f}")"""))

    cells.append(new_code_cell("""# Quick-check assertion for Healthcare
assert test_acc >= 0.95, f"Expected test accuracy >= 95%, got {test_acc*100:.1f}%"
assert malignant_recall >= 0.90, f"Expected malignant recall >= 90%, got {malignant_recall*100:.1f}%"
assert confusion_matrix(y_test, y_pred)[0, 1] <= 3, "Too many false negatives for clinical safety!"
print("Assertion check PASSED: Healthcare diagnostic model satisfies clinical accuracy & recall thresholds.")"""))

    # Section 2: Finance
    cells.append(new_markdown_cell("""## 2. Finance: Transaction Fraud & Anomaly Detection

### Real-World Operational Context
Financial institutions process millions of electronic transactions per second. Fraudulent transactions constitute less than 0.5% of volume and constantly shift patterns to evade hand-crafted rules.

**Isolation Forest** isolates anomalies by randomly partitioning feature values. Because anomalous transactions have atypical values (such as an abrupt $3,500 international wire transfer at 3:00 AM from an account that normally spends $40 at local grocery stores), they require far fewer recursive partitions to isolate than normal cluster points."""))

    cells.append(new_code_cell("""from sklearn.ensemble import IsolationForest

# 1. Generate realistic synthetic transaction stream
np.random.seed(42)
n_normal = 1000
n_fraud = 25

# Normal transactions: average $48.50, low frequency, moderate risk score
amounts_normal = np.random.normal(loc=48.5, scale=18.0, size=(n_normal, 1))
amounts_normal = np.clip(amounts_normal, a_min=2.0, a_max=180.0)
velocity_normal = np.random.exponential(scale=1.5, size=(n_normal, 1))  # hours since last txn
risk_score_normal = np.random.beta(a=1.5, b=8.0, size=(n_normal, 1)) * 100

X_normal = np.hstack([amounts_normal, velocity_normal, risk_score_normal])
y_normal = np.zeros(n_normal, dtype=int)  # 0: legitimate

# Fraudulent transactions: high amounts, extreme velocity (seconds apart), elevated risk indicators
amounts_fraud = np.random.uniform(low=1200.0, high=4500.0, size=(n_fraud, 1))
velocity_fraud = np.random.uniform(low=0.01, high=0.10, size=(n_fraud, 1))  # rapid burst
risk_score_fraud = np.random.uniform(low=75.0, high=99.0, size=(n_fraud, 1))

X_fraud = np.hstack([amounts_fraud, velocity_fraud, risk_score_fraud])
y_fraud = np.ones(n_fraud, dtype=int)  # 1: fraudulent ground truth

# Combine stream
X_transactions = np.vstack([X_normal, X_fraud])
y_true = np.concatenate([y_normal, y_fraud])

# 2. Fit IsolationForest (Unsupervised - no labels provided during training)
contamination_rate = n_fraud / len(y_true)  # ~ 2.4% expected anomaly rate
iso_forest = IsolationForest(
    n_estimators=150,
    contamination=contamination_rate,
    random_state=42
)
iso_forest.fit(X_transactions)

# 3. Detect anomalies: -1 is flagged anomaly, +1 is normal
preds = iso_forest.predict(X_transactions)
flagged_as_anomaly = (preds == -1)

# Evaluation against ground truth synthetic fraud labels
true_positives = np.sum(flagged_as_anomaly & (y_true == 1))
false_positives = np.sum(flagged_as_anomaly & (y_true == 0))
fraud_recall = true_positives / n_fraud
precision = true_positives / (true_positives + false_positives)

print(f"Total Transactions Processed: {len(X_transactions)}")
print(f"Transactions Flagged for Review: {np.sum(flagged_as_anomaly)}")
print(f"Actual Fraud Caught (True Positives): {true_positives} / {n_fraud} ({fraud_recall * 100:.1f}%)")
print(f"False Alarm Rate (False Positives): {false_positives} / {n_normal} ({false_positives/n_normal * 100:.2f}%)")
print(f"Precision on Flagged Stream: {precision * 100:.1f}%")"""))

    cells.append(new_code_cell("""# Quick-check assertion for Finance
assert fraud_recall >= 0.88, f"Expected fraud recall >= 88%, got {fraud_recall*100:.1f}%"
assert false_positives <= 10, f"False positives too high: {false_positives}"
print("Assertion check PASSED: Finance Isolation Forest captures over 88% of fraud with low false alarm rate.")"""))

    # Section 3: Retail & E-Commerce
    cells.append(new_markdown_cell("""## 3. Retail & E-Commerce: Customer Behavioral Segmentation

### Real-World Operational Context
Modern e-commerce platforms cannot treat all customers identically. Marketing teams segment customers to personalize email campaigns, allocate loyalty rewards, and minimize churn.

**K-Means clustering** groups customers with similar behavioral footprints across multiple dimensions (annual spend, purchase frequency, average browsing session length) into distinct, actionable personas without requiring pre-labeled training data."""))

    cells.append(new_code_cell("""from sklearn.cluster import KMeans

# 1. Generate synthetic customer profiles across 3 commercial personas
np.random.seed(42)

# Cohort A: Bargain Hunters (low spend, moderate frequency, high browsing duration looking for deals)
spend_a = np.random.normal(loc=120, scale=30, size=150)
freq_a = np.random.normal(loc=3.5, scale=1.0, size=150)
browse_a = np.random.normal(loc=25, scale=5, size=150)

# Cohort B: Busy Everyday Shoppers (moderate spend, high frequency, very quick sessions)
spend_b = np.random.normal(loc=550, scale=80, size=200)
freq_b = np.random.normal(loc=12.0, scale=2.5, size=200)
browse_b = np.random.normal(loc=8, scale=2, size=200)

# Cohort C: High-Value VIPs (high spend, moderate frequency, deep product research)
spend_c = np.random.normal(loc=2200, scale=350, size=100)
freq_c = np.random.normal(loc=6.0, scale=1.5, size=100)
browse_c = np.random.normal(loc=35, scale=8, size=100)

X_customers = np.vstack([
    np.column_stack([spend_a, freq_a, browse_a]),
    np.column_stack([spend_b, freq_b, browse_b]),
    np.column_stack([spend_c, freq_c, browse_c])
])

print(f"Customer records: {X_customers.shape[0]} profiles with 3 behavioral features.")

# 2. Build clustering pipeline with StandardScaler
clustering_pipe = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, random_state=42, n_init="auto")
)
cluster_labels = clustering_pipe.fit_predict(X_customers)

# 3. Analyze cluster personas in original physical units (USD, visits, minutes)
scaler = clustering_pipe.named_steps["standardscaler"]
kmeans = clustering_pipe.named_steps["kmeans"]
unscaled_centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("\\nIdentified Customer Personas (Cluster Centroids):")
print("-" * 75)
print(f"{'Cluster':<10} | {'Annual Spend ($)':<18} | {'Monthly Orders':<16} | {'Avg Browse (min)':<16}")
print("-" * 75)
for i, center in enumerate(unscaled_centers):
    count = np.sum(cluster_labels == i)
    print(f"Persona {i:<3} | ${center[0]:>14.2f}    | {center[1]:>12.1f}     | {center[2]:>12.1f} min  ({count} users)")
print("-" * 75)"""))

    cells.append(new_code_cell("""# Quick-check assertion for Retail
assert len(set(cluster_labels)) == 3, "Expected exactly 3 customer clusters."
assert all(np.sum(cluster_labels == i) >= 80 for i in range(3)), "Clusters are unbalanced or degenerate."
print("Assertion check PASSED: Customer segmentation produced 3 well-formed commercial behavioral cohorts.")"""))

    # Section 4: Transportation & Logistics
    cells.append(new_markdown_cell("""## 4. Transportation & Logistics: Delivery Trip Duration Forecasting

### Real-World Operational Context
Last-mile delivery and freight logistics rely on accurate Estimated Time of Arrival (ETA) predictions. Inaccurate ETAs lead to warehouse bottlenecks, breached Service Level Agreements (SLAs), and dissatisfied consumers.

A **regularized linear regression (Ridge)** model predicts dispatch duration based on measurable physical covariates: route distance, traffic congestion score, cargo package count, and adverse weather conditions."""))

    cells.append(new_code_cell("""from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Generate realistic logistics dispatch telemetry (600 completed delivery routes)
np.random.seed(42)
n_routes = 600

distance_km = np.random.uniform(3.0, 45.0, size=n_routes)
traffic_index = np.random.uniform(1.0, 10.0, size=n_routes)  # 1 = clear highway, 10 = gridlock
package_count = np.random.randint(1, 15, size=n_routes)
weather_severity = np.random.uniform(1.0, 5.0, size=n_routes)  # 1 = clear, 5 = blizzard/torrential

# True physical generation function:
# Duration = 10 min baseline + 1.8 min/km + 4.2 min/traffic_pt + 1.2 min/pkg + 3.5 min/weather_pt + noise
transit_minutes = (
    10.0
    + 1.80 * distance_km
    + 4.20 * traffic_index
    + 1.20 * package_count
    + 3.50 * weather_severity
    + np.random.normal(loc=0.0, scale=4.0, size=n_routes)
)

X_logistics = np.column_stack([distance_km, traffic_index, package_count, weather_severity])
y_logistics = transit_minutes
feature_names_logistics = ["Distance (km)", "Traffic Index (1-10)", "Package Count", "Weather Severity (1-5)"]

# 2. Train/Test Split (75% train, 25% test)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_logistics, y_logistics, test_size=0.25, random_state=42
)

# 3. Train Ridge Regression pipeline
logistics_pipe = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
logistics_pipe.fit(X_train_log, y_train_log)

# 4. Predict and evaluate
y_pred_log = logistics_pipe.predict(X_test_log)
mae = mean_absolute_error(y_test_log, y_pred_log)
rmse = root_mean_squared_error(y_test_log, y_pred_log)
r2 = r2_score(y_test_log, y_pred_log)

print(f"Logistics ETA Model Performance on Unseen Test Routes:")
print(f"  - Mean Absolute Error (MAE):     {mae:.2f} minutes")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
print(f"  - Explained Variance (R² Score): {r2:.3f}")

# 5. Inspect standardized feature importance (coefficients on scaled variables)
ridge_model = logistics_pipe.named_steps["ridge"]
print("\\nFeature Impact on Trip Duration (standardized impact per 1-sigma shift):")
for name, coef in zip(feature_names_logistics, ridge_model.coef_):
    print(f"  - {name:<25}: +{coef:.2f} minutes")"""))

    cells.append(new_code_cell("""# Quick-check assertion for Transportation & Logistics
assert mae < 5.0, f"MAE too high for route scheduling: {mae:.2f} min"
assert r2 > 0.90, f"R² score lower than required: {r2:.3f}"
print("Assertion check PASSED: Logistics ETA model achieves sub-5-minute error and >90% explained variance.")"""))

    # Section 5: Customer Service & NLP
    cells.append(new_markdown_cell("""## 5. Customer Service & NLP: Automated Support Ticket Intent Routing

### Real-World Operational Context
Enterprise customer support departments receive thousands of inbound inquiries daily. Manually reading and dispatching each ticket introduces multi-hour delays. An AI-powered ticket classifier analyzes customer text messages and routes them directly to the appropriate specialist queue:
- **Billing:** Invoices, overcharges, credit card updates, refunds.
- **Technical Support:** Bugs, system crashes, API failures, 500 errors.
- **Account Management:** Password resets, MFA lockouts, user profile updates.
- **Logistics & Returns:** Return labels, damaged delivery, missing shipments.

We build a text classification pipeline using **Bag-of-Words (CountVectorizer)** coupled with **Multinomial Logistic Regression**."""))

    cells.append(new_code_cell("""from sklearn.feature_extraction.text import CountVectorizer

# 1. Representative training corpus of customer inquiries across 4 department queues
training_tickets = [
    # Billing
    "I was charged twice on my monthly credit card statement for subscription",
    "Please send me the VAT receipt and official invoice for last month",
    "How can I request a refund for the billing overcharge?",
    "Need to update the credit card payment method on my billing account",
    "Why did my subscription renewal price increase without notice?",
    "My corporate billing address changed, need an updated invoice",
    
    # Technical Support
    "Application crashes with error 500 when uploading large CSV files",
    "API endpoint returns connection timeout 504 on POST request",
    "Database connection failed with socket error on the production server",
    "The web interface is frozen and javascript console shows uncaught exception",
    "Encountering continuous server error during file export operation",
    "Mobile app terminates unexpectedly upon login button press",
    
    # Account Management
    "Forgot my password and the password reset link email is not arriving",
    "Locked out of my account because of lost two factor authenticator phone",
    "Need to change the primary administrator email on the account",
    "How do I delete my user profile and remove all associated personal data?",
    "Require assistance resetting two-factor authentication credentials",
    "Update my registered username and security recovery phone number",
    
    # Logistics & Returns
    "Where is my package tracking number says delayed in transit",
    "The item arrived broken and damaged in shipping need a replacement",
    "Want to return the delivered product and get a prepaid return label",
    "Package shows delivered by courier but nothing is on my porch",
    "Tracking shows parcel stuck at regional distribution warehouse hub",
    "How do I initiate a return for defective merchandise received today"
]

training_labels = [
    "Billing", "Billing", "Billing", "Billing", "Billing", "Billing",
    "Technical Support", "Technical Support", "Technical Support", "Technical Support", "Technical Support", "Technical Support",
    "Account Management", "Account Management", "Account Management", "Account Management", "Account Management", "Account Management",
    "Logistics & Returns", "Logistics & Returns", "Logistics & Returns", "Logistics & Returns", "Logistics & Returns", "Logistics & Returns"
]

# 2. Build NLP routing pipeline (tokenization + stopword removal + logistic regression)
ticket_router = make_pipeline(
    CountVectorizer(stop_words="english"),
    LogisticRegression(random_state=42)
)
ticket_router.fit(training_tickets, training_labels)

# 3. Test on novel, unseen customer inquiries arriving in production
unseen_customer_inquiries = [
    "My credit card shows duplicate charges from yesterday",
    "The server throws a 500 internal server error during data sync",
    "I am locked out because my 2FA authentication code expired",
    "The delivered box was crushed and the product inside is completely broken",
    "Please email the tax invoice and payment receipt for our records"
]

predicted_departments = ticket_router.predict(unseen_customer_inquiries)
prediction_probabilities = ticket_router.predict_proba(unseen_customer_inquiries)

print("Automated Ticket Triage Results (Production Inquiries):")
print("-" * 80)
for inquiry, dept, probs in zip(unseen_customer_inquiries, predicted_departments, prediction_probabilities):
    conf = np.max(probs) * 100
    print(f"Inquiry:    \\"{inquiry}\\"")
    print(f"Routed To:  [{dept}] (Confidence: {conf:.1f}%)\\n")"""))

    cells.append(new_code_cell("""# Quick-check assertion for Customer Service & NLP
expected_routing = [
    "Billing",
    "Technical Support",
    "Account Management",
    "Logistics & Returns",
    "Billing"
]
assert list(predicted_departments) == expected_routing, (
    f"Mismatch in ticket routing: expected {expected_routing}, got {list(predicted_departments)}"
)
print("Assertion check PASSED: Customer support tickets successfully routed to correct department queues.")"""))

    # Summary table cell
    cells.append(new_markdown_cell("""## 6. Synthesis: Cross-Industry AI Application Matrix

| Industry | Operational Problem | Machine Learning Paradigm | scikit-learn Estimator | Primary Metric |
|---|---|---|---|---|
| **Healthcare** | Tumor Malignancy Diagnosis | Supervised Classification | `LogisticRegression` + `StandardScaler` | Malignancy Recall (Sensitivity) |
| **Finance** | Fraud / Outlier Detection | Unsupervised Anomaly Detection | `IsolationForest` | True Positive Fraud Detection & Low False Alarm Rate |
| **Retail** | Customer Cohort Segmentation | Unsupervised Clustering | `KMeans` + `StandardScaler` | Cluster Balance & Persona Differentiation |
| **Logistics** | Dispatch Duration (ETA) Prediction | Supervised Regression | `Ridge` + `StandardScaler` | MAE (Minutes) & R² Variance Explained |
| **Customer Service** | Inbound Ticket Intent Routing | Supervised NLP Classification | `CountVectorizer` + `LogisticRegression` | Department Triage Accuracy & Confidence |"""))

    nb.cells = cells
    return nb


def main() -> None:
    print(f"Building notebook at {NOTEBOOK_PATH}...")
    nb = build_notebook()
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook written successfully.")

    print("Executing notebook with nbclient to generate and save outputs...")
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
    )
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook executed and saved with outputs successfully.")


if __name__ == "__main__":
    main()
