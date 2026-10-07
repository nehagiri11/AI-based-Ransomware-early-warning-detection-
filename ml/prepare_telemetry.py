import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "telemetry" / "telemetry.csv"
OUTPUT_FILE = BASE_DIR / "data" / "telemetry" / "telemetry_prepared.csv"

print("=" * 70)
print("PREPARING TELEMETRY SESSIONS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

# Make sure timestamps are chronological
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Detect a new monitoring session whenever window_id
# goes backwards or restarts.
df["session_id"] = (
    df["window_id"].diff() <= 0
).cumsum() + 1

# Create one continuous global window number
df["global_window_id"] = range(1, len(df) + 1)

print(f"\nTotal telemetry rows: {len(df)}")
print(f"Monitoring sessions: {df['session_id'].nunique()}")

print("\nSession information:")
print(
    df.groupby("session_id")
    .agg(
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
        windows=("window_id", "count"),
        first_window=("window_id", "min"),
        last_window=("window_id", "max")
    )
    .to_string()
)

# Save prepared telemetry
df.to_csv(OUTPUT_FILE, index=False)

print("\n")
print("=" * 70)
print("TELEMETRY PREPARATION COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)