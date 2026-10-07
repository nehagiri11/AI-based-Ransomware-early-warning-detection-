import pandas as pd
from pathlib import Path
import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "ml" / "lightgbm.pkl"

OUTPUT_FILE = PROJECT_ROOT / "ml" / "lightgbm_feature_importance.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("LIGHTGBM FEATURE IMPORTANCE ANALYSIS")
print("=" * 75)

print("\nLoading model...")

saved_model = joblib.load(MODEL_FILE)

model = saved_model["model"]
features = saved_model["features"]

print("Model loaded successfully.")
print("Number of features:", len(features))


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

feature_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\nTop 30 most important features")
print("-" * 75)

print(
    feature_df.head(30).to_string(
        index=False
    )
)


# ============================================================
# FEATURE CATEGORY SUMMARY
# ============================================================

def get_category(feature):

    if ":" in feature:
        return feature.split(":")[0]

    return "OTHER"


feature_df["Category"] = feature_df["Feature"].apply(
    get_category
)

category_summary = (
    feature_df
    .groupby("Category")["Importance"]
    .sum()
    .sort_values(ascending=False)
)


print("\n")
print("=" * 75)
print("IMPORTANCE BY FEATURE CATEGORY")
print("=" * 75)

print(category_summary)


print("\n")
print("=" * 75)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 75)

print("\nResults saved to:")
print(OUTPUT_FILE)