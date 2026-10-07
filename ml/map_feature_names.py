import json
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSON_FILE = (
    PROJECT_ROOT
    / "data"
    / "public"
    / "RFE_selected_feature_names_dic.json"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "ml"
    / "lightgbm_feature_importance.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "ml"
    / "lightgbm_feature_importance_named.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 75)
print("MAPPING MLRAN FEATURE IDS TO FEATURE NAMES")
print("=" * 75)

print("\nLoading feature mapping...")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    feature_mapping = json.load(f)

print("Mapping entries:", len(feature_mapping))

print("\nLoading feature importance...")

importance_df = pd.read_csv(IMPORTANCE_FILE)

print("Importance rows:", len(importance_df))


# ============================================================
# NORMALIZE MAPPING
# ============================================================

# Convert JSON keys to strings
feature_mapping = {
    str(key): value
    for key, value in feature_mapping.items()
}


# ============================================================
# MAP FEATURE IDS
# ============================================================

importance_df["Feature_ID"] = (
    importance_df["Feature"]
    .astype(str)
)

importance_df["Feature_Name"] = (
    importance_df["Feature_ID"]
    .map(feature_mapping)
)


# ============================================================
# CHECK UNMAPPED FEATURES
# ============================================================

unmapped = importance_df[
    importance_df["Feature_Name"].isna()
]

print("\nMapped features:", len(importance_df) - len(unmapped))
print("Unmapped features:", len(unmapped))


if len(unmapped) > 0:

    print("\nFirst unmapped feature IDs:")
    print(
        unmapped["Feature_ID"]
        .head(20)
        .tolist()
    )


# ============================================================
# CATEGORY
# ============================================================

def get_category(feature):

    if not isinstance(feature, str):
        return "UNKNOWN"

    if ":" in feature:
        return feature.split(":")[0]

    return "OTHER"


importance_df["Category"] = (
    importance_df["Feature_Name"]
    .apply(get_category)
)


# ============================================================
# SORT
# ============================================================

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\n")
print("=" * 75)
print("TOP 30 NAMED FEATURES")
print("=" * 75)

print(
    importance_df[
        [
            "Feature_ID",
            "Feature_Name",
            "Category",
            "Importance"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# CATEGORY SUMMARY
# ============================================================

category_summary = (
    importance_df
    .groupby("Category")["Importance"]
    .sum()
    .sort_values(ascending=False)
)


print("\n")
print("=" * 75)
print("IMPORTANCE BY FEATURE CATEGORY")
print("=" * 75)

print(category_summary)


# ============================================================
# SAVE
# ============================================================

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("Named feature importance saved to:")
print(OUTPUT_FILE)

print("\n")
print("=" * 75)
print("FEATURE MAPPING COMPLETE")
print("=" * 75)