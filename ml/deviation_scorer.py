import json
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TELEMETRY_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry_prepared.csv"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "ml"
    / "adaptive_baseline.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry_with_deviation.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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
# LOAD DATA
# ============================================================

print("=" * 70)
print("ADAPTIVE DEVIATION SCORER")
print("=" * 70)

print("\nLoading prepared telemetry...")

df = pd.read_csv(
    TELEMETRY_FILE
)

print("Telemetry windows:", len(df))

print(
    "Monitoring sessions:",
    df["session_id"].nunique()
)


print("\nLoading adaptive baseline...")

with open(
    BASELINE_FILE,
    "r",
    encoding="utf-8"
) as file:

    baseline = json.load(file)


# ============================================================
# CALCULATE FEATURE DEVIATIONS
# ============================================================

print("\nCalculating deviations...")

deviation_columns = []


for feature in FEATURES:

    mean = baseline[feature]["mean"]
    std = baseline[feature]["std"]

    deviation_column = (
        feature + "_deviation"
    )

    deviation_columns.append(
        deviation_column
    )

    # --------------------------------------------------------
    # Handle zero standard deviation
    # --------------------------------------------------------

    if std == 0:

        df[deviation_column] = np.where(
            df[feature] == mean,
            0.0,
            1.0
        )

    else:

        df[deviation_column] = (
            abs(
                df[feature] - mean
            )
            / std
        )


# ============================================================
# OVERALL DEVIATION SCORE
# ============================================================

df["overall_deviation"] = (
    df[deviation_columns]
    .mean(axis=1)
)


# ============================================================
# CAP SCORE FOR DISPLAY
# ============================================================

df["deviation_score"] = (
    df["overall_deviation"]
    .clip(
        lower=0,
        upper=10
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DEVIATION RESULTS")
print("=" * 70)

display_columns = [
    "session_id",
    "window_id",
    "global_window_id",
    "cpu_percent",
    "memory_percent",
    "process_count",
    "total_file_events",
    "deviation_score"
]

print(
    df[display_columns]
    .tail(20)
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DEVIATION SUMMARY")
print("=" * 70)

print(
    f"Mean deviation : "
    f"{df['deviation_score'].mean():.3f}"
)

print(
    f"Median deviation : "
    f"{df['deviation_score'].median():.3f}"
)

print(
    f"Maximum deviation : "
    f"{df['deviation_score'].max():.3f}"
)


# ============================================================
# SESSION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SESSION SUMMARY")
print("=" * 70)

session_summary = (
    df.groupby("session_id")
    ["deviation_score"]
    .agg(
        windows="count",
        mean_deviation="mean",
        maximum_deviation="max"
    )
)

print(
    session_summary.to_string()
)


# ============================================================
# SAVE MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("DEVIATION SCORING COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)