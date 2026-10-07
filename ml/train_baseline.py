import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "public"
MODEL_DIR = PROJECT_ROOT / "ml"

TRAIN_FILE = DATA_DIR / "MLRan_X_train_RFE.csv"
TEST_FILE = DATA_DIR / "MLRan_X_test_RFE.csv"
LABEL_FILE = DATA_DIR / "MLRan_labels.csv"

MODEL_FILE = MODEL_DIR / "baseline_random_forest.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MLRAN BASELINE MODEL")
print("=" * 70)

print("\nLoading datasets...")

X_train = pd.read_csv(TRAIN_FILE)
X_test = pd.read_csv(TEST_FILE)
labels = pd.read_csv(LABEL_FILE)


# ============================================================
# METADATA COLUMNS
# ============================================================

metadata_columns = [
    "sample_id",
    "sample_type",
    "family_label",
    "type_label"
]


# ============================================================
# CREATE TARGET
# ============================================================

# sample_type:
# 0 = goodware
# 1 = ransomware

y_all = labels.set_index("sample_id")["sample_type"]


# Match labels to the train/test sample IDs
y_train = X_train["sample_id"].map(y_all)
y_test = X_test["sample_id"].map(y_all)


# Check for missing labels
if y_train.isnull().any():
    raise ValueError("Some training samples do not have labels.")

if y_test.isnull().any():
    raise ValueError("Some test samples do not have labels.")


# ============================================================
# CREATE FEATURE MATRICES
# ============================================================

feature_columns = [
    column
    for column in X_train.columns
    if column not in metadata_columns
]


X_train_model = X_train[feature_columns]
X_test_model = X_test[feature_columns]


print("\nDataset information")
print("-" * 70)

print("Training samples:", len(X_train_model))
print("Testing samples :", len(X_test_model))
print("Features        :", len(feature_columns))


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nTraining label distribution")
print("-" * 70)

print(y_train.value_counts())

print("\nTesting label distribution")
print("-" * 70)

print(y_test.value_counts())


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train_model,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning test prediction...")

y_pred = model.predict(X_test_model)


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nConfusion Matrix")
print("-" * 70)

print(cm)


print("\nClassification Report")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Goodware",
            "Ransomware"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    MODEL_FILE
)


print("\nModel saved to:")

print(MODEL_FILE)

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)