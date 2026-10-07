import pandas as pd
from pathlib import Path
import time
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public"
MODEL_DIR = PROJECT_ROOT / "ml"

TRAIN_FILE = DATA_DIR / "MLRan_X_train_RFE.csv"
TEST_FILE = DATA_DIR / "MLRan_X_test_RFE.csv"
LABEL_FILE = DATA_DIR / "MLRan_labels.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("MLRAN MODEL COMPARISON")
print("=" * 75)

print("\nLoading datasets...")

X_train = pd.read_csv(TRAIN_FILE)
X_test = pd.read_csv(TEST_FILE)
labels = pd.read_csv(LABEL_FILE)


# ============================================================
# PREPARE TARGET
# ============================================================

metadata_columns = [
    "sample_id",
    "sample_type",
    "family_label",
    "type_label"
]

# Map sample_id -> sample_type
label_map = labels.set_index("sample_id")["sample_type"]

y_train = X_train["sample_id"].map(label_map)
y_test = X_test["sample_id"].map(label_map)


if y_train.isnull().any():
    raise ValueError("Some training samples do not have labels.")

if y_test.isnull().any():
    raise ValueError("Some testing samples do not have labels.")


# ============================================================
# SELECT 483 BEHAVIOURAL FEATURES
# ============================================================

feature_columns = [
    column
    for column in X_train.columns
    if column not in metadata_columns
]

X_train_model = X_train[feature_columns]
X_test_model = X_test[feature_columns]


print("\nDataset information")
print("-" * 75)
print("Training samples :", len(X_train_model))
print("Testing samples  :", len(X_test_model))
print("Features         :", len(feature_columns))


print("\nTarget distribution")
print("-" * 75)
print("Training:")
print(y_train.value_counts())

print("\nTesting:")
print(y_test.value_counts())


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\n")
    print("=" * 75)
    print(f"TRAINING: {model_name}")
    print("=" * 75)

    start_time = time.perf_counter()

    model.fit(X_train_model, y_train)

    training_time = time.perf_counter() - start_time

    print(f"Training completed in {training_time:.2f} seconds.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_start = time.perf_counter()

    y_pred = model.predict(X_test_model)

    prediction_time = time.perf_counter() - prediction_start

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    cm = confusion_matrix(y_test, y_pred)

    # --------------------------------------------------------
    # Confusion matrix values
    # --------------------------------------------------------

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "False Positives": fp,
        "False Negatives": fn,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time
    })

    trained_models[model_name] = model

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults")
    print("-" * 75)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")

    print(f"False Positives   : {fp}")
    print(f"False Negatives   : {fn}")

    print(f"Training Time     : {training_time:.2f} sec")
    print(f"Prediction Time   : {prediction_time:.4f} sec")

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
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
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 100)
print("FINAL MODEL COMPARISON")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = MODEL_DIR / "model_comparison_results.csv"

results_df.to_csv(
    results_file,
    index=False
)

print("\nComparison results saved to:")
print(results_file)


# ============================================================
# SAVE MODELS
# ============================================================

for model_name, model in trained_models.items():

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    model_file = MODEL_DIR / filename

    joblib.dump(
        {
            "model": model,
            "features": feature_columns
        },
        model_file
    )

    print(f"Saved {model_name}:")
    print(model_file)


print("\n")
print("=" * 75)
print("MODEL COMPARISON COMPLETE")
print("=" * 75)