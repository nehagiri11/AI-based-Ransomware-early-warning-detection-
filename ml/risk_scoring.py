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
    / "telemetry_temporal.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
    / "telemetry_risk.csv"
)


# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

DEVIATION_WEIGHT = 0.60
TEMPORAL_WEIGHT = 0.40

EARLY_WARNING_THRESHOLD = 0.60


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BEHAVIOURAL RISK SCORING")
print("=" * 70)

print("\nLoading temporal telemetry data...")

df = pd.read_csv(INPUT_FILE)

print("Telemetry windows:", len(df))
print("Monitoring sessions:", df["session_id"].nunique())


# ============================================================
# NORMALIZE DEVIATION
# ============================================================

# deviation_score is currently on a 0-10 scale.
# Convert it to a 0-1 scale.

df["normalized_deviation"] = (
    df["deviation_score"] / 10.0
).clip(0, 1)


# ============================================================
# TEMPORAL SCORE
# ============================================================

df["temporal_score"] = (
    df["temporal_score"]
    .fillna(0)
    .clip(0, 1)
)


# ============================================================
# COMBINED BEHAVIOURAL RISK
# ============================================================

df["behavioural_risk_score"] = (
    DEVIATION_WEIGHT
    * df["normalized_deviation"]
    +
    TEMPORAL_WEIGHT
    * df["temporal_score"]
)

df["behavioural_risk_score"] = (
    df["behavioural_risk_score"]
    .clip(0, 1)
)


# ============================================================
# PERCENTAGE REPRESENTATION
# ============================================================

df["risk_percentage"] = (
    df["behavioural_risk_score"] * 100
).round(2)


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(score):

    if score < 0.30:
        return "LOW"

    elif score < 0.60:
        return "MEDIUM"

    elif score < 0.80:
        return "HIGH"

    else:
        return "CRITICAL"


df["risk_level"] = (
    df["behavioural_risk_score"]
    .apply(classify_risk)
)


# ============================================================
# EARLY WARNING
# ============================================================

# This is an experimental behavioural warning rule.
# It is NOT a confirmed ransomware detection rule.

df["early_warning"] = (
    (df["persistent_deviation"] == True)
    &
    (df["behavioural_risk_score"]
     >= EARLY_WARNING_THRESHOLD)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RISK SCORING RESULTS")
print("=" * 70)

display_columns = [
    "session_id",
    "window_id",
    "global_window_id",
    "deviation_score",
    "temporal_score",
    "behavioural_risk_score",
    "risk_percentage",
    "risk_level",
    "early_warning"
]

print(
    df[display_columns]
    .to_string(index=False)
)


# ============================================================
# RISK SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("RISK SUMMARY")
print("=" * 70)

print(
    f"Average behavioural risk : "
    f"{df['behavioural_risk_score'].mean():.3f}"
)

print(
    f"Maximum behavioural risk : "
    f"{df['behavioural_risk_score'].max():.3f}"
)

print(
    f"LOW risk windows        : "
    f"{(df['risk_level'] == 'LOW').sum()}"
)

print(
    f"MEDIUM risk windows     : "
    f"{(df['risk_level'] == 'MEDIUM').sum()}"
)

print(
    f"HIGH risk windows       : "
    f"{(df['risk_level'] == 'HIGH').sum()}"
)

print(
    f"CRITICAL risk windows   : "
    f"{(df['risk_level'] == 'CRITICAL').sum()}"
)

print(
    f"Early warnings          : "
    f"{df['early_warning'].sum()}"
)


# ============================================================
# SESSION-WISE SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SESSION-WISE RISK SUMMARY")
print("=" * 70)

session_summary = (
    df.groupby("session_id")
    .agg(
        windows=("global_window_id", "count"),
        average_risk=("behavioural_risk_score", "mean"),
        maximum_risk=("behavioural_risk_score", "max"),
        high_or_above=(
            "risk_level",
            lambda x: (
                (x == "HIGH") |
                (x == "CRITICAL")
            ).sum()
        ),
        early_warnings=("early_warning", "sum")
    )
)

print(
    session_summary.to_string()
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("RISK SCORING COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)