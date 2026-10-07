import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry_with_deviation.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry_temporal.csv"
)


# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

# Initial experimental deviation threshold.
# This is NOT a proven ransomware threshold.

DEVIATION_THRESHOLD = 1.5

# Number of consecutive elevated windows required
# for a persistent behavioural signal.

PERSISTENCE_WINDOWS = 3


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TEMPORAL BEHAVIOUR ANALYSIS")
print("=" * 70)

print("\nLoading deviation data...")

df = pd.read_csv(INPUT_FILE)

print("Telemetry windows:", len(df))
print("Monitoring sessions:", df["session_id"].nunique())


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["session_id", "global_window_id"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY ELEVATED DEVIATION
# ============================================================

df["elevated_deviation"] = (
    df["deviation_score"] >= DEVIATION_THRESHOLD
)


# ============================================================
# TEMPORAL PERSISTENCE
# ============================================================

# Persistence is calculated separately inside each
# monitoring session.

df["consecutive_elevated_windows"] = 0


for session_id, session_indices in df.groupby(
    "session_id"
).groups.items():

    consecutive_count = 0

    for index in session_indices:

        if df.loc[index, "elevated_deviation"]:

            consecutive_count += 1

        else:

            consecutive_count = 0

        df.loc[
            index,
            "consecutive_elevated_windows"
        ] = consecutive_count


# ============================================================
# PERSISTENT DEVIATION
# ============================================================

df["persistent_deviation"] = (
    df["consecutive_elevated_windows"]
    >= PERSISTENCE_WINDOWS
)


# ============================================================
# TEMPORAL SCORE
# ============================================================

df["temporal_score"] = (
    df["consecutive_elevated_windows"]
    / PERSISTENCE_WINDOWS
)

df["temporal_score"] = (
    df["temporal_score"]
    .clip(0, 1)
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
print("TEMPORAL ANALYSIS RESULTS")
print("=" * 70)

display_columns = [
    "session_id",
    "window_id",
    "global_window_id",
    "deviation_score",
    "elevated_deviation",
    "consecutive_elevated_windows",
    "persistent_deviation",
    "temporal_score"
]

print(
    df[display_columns]
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TEMPORAL SUMMARY")
print("=" * 70)

elevated_count = (
    df["elevated_deviation"]
    .sum()
)

persistent_count = (
    df["persistent_deviation"]
    .sum()
)

max_consecutive = (
    df["consecutive_elevated_windows"]
    .max()
)

print(
    f"Elevated windows: "
    f"{elevated_count}"
)

print(
    f"Persistent windows: "
    f"{persistent_count}"
)

print(
    f"Maximum consecutive elevated windows: "
    f"{max_consecutive}"
)


# ============================================================
# SESSION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SESSION-WISE SUMMARY")
print("=" * 70)

session_summary = (
    df.groupby("session_id")
    .agg(
        windows=("global_window_id", "count"),
        elevated_windows=("elevated_deviation", "sum"),
        persistent_windows=("persistent_deviation", "sum"),
        maximum_consecutive=(
            "consecutive_elevated_windows",
            "max"
        )
    )
)

print(
    session_summary.to_string()
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("TEMPORAL ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)