import pandas as pd
import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TELEMETRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry.csv"
)

BASELINE_DIR = (
    PROJECT_ROOT
    / "ml"
)

BASELINE_FILE = (
    BASELINE_DIR
    / "adaptive_baseline.json"
)


# ============================================================
# LOAD TELEMETRY
# ============================================================

print("=" * 70)
print("ADAPTIVE BASELINE BUILDER")
print("=" * 70)

print("\nLoading telemetry...")

df = pd.read_csv(
    TELEMETRY_FILE
)

print("Telemetry windows:", len(df))


if len(df) < 5:

    raise ValueError(
        "Not enough telemetry windows. "
        "Collect more normal activity first."
    )


# ============================================================
# FEATURES USED FOR BASELINE
# ============================================================

baseline_features = [
    "cpu_percent",
    "memory_percent",
    "process_count",
    "files_created",
    "files_modified",
    "files_deleted",
    "files_renamed",
    "total_file_events"
]


# ============================================================
# CALCULATE BASELINE STATISTICS
# ============================================================

baseline = {}

for feature in baseline_features:

    series = df[feature]

    baseline[feature] = {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75))
    }


# ============================================================
# SAVE BASELINE
# ============================================================

with open(
    BASELINE_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        baseline,
        file,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("NORMAL ACTIVITY BASELINE")
print("=" * 70)

for feature, statistics in baseline.items():

    print("\n" + feature)

    print(
        f"  Mean   : {statistics['mean']:.2f}"
    )

    print(
        f"  Std    : {statistics['std']:.2f}"
    )

    print(
        f"  Median : {statistics['median']:.2f}"
    )

    print(
        f"  Min    : {statistics['min']:.2f}"
    )

    print(
        f"  Max    : {statistics['max']:.2f}"
    )

    print(
        f"  Q25    : {statistics['q25']:.2f}"
    )

    print(
        f"  Q75    : {statistics['q75']:.2f}"
    )


print("\n")
print("=" * 70)
print("BASELINE CREATED")
print("=" * 70)

print("\nSaved to:")
print(BASELINE_FILE)