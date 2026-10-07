import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "public"

train_file = DATA_DIR / "MLRan_X_train_RFE.csv"
test_file = DATA_DIR / "MLRan_X_test_RFE.csv"
labels_file = DATA_DIR / "MLRan_labels.csv"


print("=" * 70)
print("MLRAN DATASET INSPECTION")
print("=" * 70)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading files...")

X_train = pd.read_csv(train_file)
X_test = pd.read_csv(test_file)
labels = pd.read_csv(labels_file)


# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("\nTRAINING DATA")
print("-" * 70)

print("Shape:", X_train.shape)
print("Rows:", X_train.shape[0])
print("Features:", X_train.shape[1])


print("\nTEST DATA")
print("-" * 70)

print("Shape:", X_test.shape)
print("Rows:", X_test.shape[0])
print("Features:", X_test.shape[1])


print("\nLABEL DATA")
print("-" * 70)

print("Shape:", labels.shape)

print("\nLabel columns:")
print(labels.columns.tolist())


# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------

print("\nFEATURE NAMES")
print("-" * 70)

for i, column in enumerate(X_train.columns, start=1):
    print(f"{i:3}. {column}")


# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\nDATA TYPES")
print("-" * 70)

print(X_train.dtypes.value_counts())


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("-" * 70)

missing = X_train.isnull().sum()

print("Total missing values:", missing.sum())

if missing.sum() > 0:
    print("\nColumns with missing values:")
    print(missing[missing > 0])
else:
    print("No missing values found.")


# ------------------------------------------------------------
# Label distribution
# ------------------------------------------------------------

print("\nLABEL DISTRIBUTION")
print("-" * 70)

for column in labels.columns:
    print(f"\nColumn: {column}")
    print(labels[column].value_counts(dropna=False))


# ------------------------------------------------------------
# First five rows
# ------------------------------------------------------------

print("\nFIRST 5 TRAINING ROWS")
print("-" * 70)

print(X_train.head())


print("\nFIRST 5 LABEL ROWS")
print("-" * 70)

print(labels.head())


print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETE")
print("=" * 70)