import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public"

TRAIN_FILE = DATA_DIR / "MLRan_X_train_RFE.csv"
FEATURE_MAP_FILE = DATA_DIR / "RFE_selected_feature_names_dic.json"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MLRAN FEATURE ANALYSIS")
print("=" * 70)

X_train = pd.read_csv(TRAIN_FILE)

with open(FEATURE_MAP_FILE, "r", encoding="utf-8") as file:
    feature_map = json.load(file)


# ============================================================
# REMOVE METADATA
# ============================================================

metadata_columns = [
    "sample_id",
    "sample_type",
    "family_label",
    "type_label"
]

feature_columns = [
    column
    for column in X_train.columns
    if column not in metadata_columns
]


print("\nTotal behavioural features:", len(feature_columns))


# ============================================================
# DISPLAY FEATURE MAPPING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE MAPPING")
print("=" * 70)

for feature_id in feature_columns:

    mapped_name = (
        feature_map.get(str(feature_id))
        or feature_map.get(feature_id)
        or "UNKNOWN"
    )

    print(f"{feature_id:>10}  ->  {mapped_name}")


# ============================================================
# SUMMARY
# ============================================================

unknown_features = []

for feature_id in feature_columns:

    mapped_name = (
        feature_map.get(str(feature_id))
        or feature_map.get(feature_id)
    )

    if mapped_name is None:
        unknown_features.append(feature_id)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Behavioural features:", len(feature_columns))
print("Mapped features:", len(feature_columns) - len(unknown_features))
print("Unknown features:", len(unknown_features))

if unknown_features:
    print("\nUnknown feature IDs:")
    print(unknown_features)


print("\nAnalysis complete.")