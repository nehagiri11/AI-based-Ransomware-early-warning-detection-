import csv
import time
import threading
from pathlib import Path
from collections import Counter
from datetime import datetime

import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MONITORED_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitored_test_dir"
)

TELEMETRY_DIR = (
    PROJECT_ROOT
    / "data"
    / "telemetry"
)

TELEMETRY_FILE = (
    TELEMETRY_DIR
    / "telemetry.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MONITORED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TELEMETRY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SAFE WINDOWS TELEMETRY COLLECTOR")
print("=" * 70)

print("\nMonitoring directory:")
print(MONITORED_DIR)

print("\nTelemetry output:")
print(TELEMETRY_FILE)

print("\nSAFETY:")
print("This program ONLY observes and records activity.")
print("It does NOT modify, delete, encrypt, execute,")
print("download, or create malicious files.")


# ============================================================
# EVENT COUNTERS
# ============================================================

event_counter = Counter()

lock = threading.Lock()


# ============================================================
# FILE EVENT HANDLER
# ============================================================

class SafeFileMonitor(FileSystemEventHandler):

    def on_created(self, event):

        if not event.is_directory:

            with lock:
                event_counter["created"] += 1

            print("[FILE CREATED]", event.src_path)


    def on_modified(self, event):

        if not event.is_directory:

            with lock:
                event_counter["modified"] += 1

            print("[FILE MODIFIED]", event.src_path)


    def on_deleted(self, event):

        if not event.is_directory:

            with lock:
                event_counter["deleted"] += 1

            print("[FILE DELETED]", event.src_path)


    def on_moved(self, event):

        if not event.is_directory:

            with lock:
                event_counter["renamed"] += 1

            print(
                "[FILE RENAMED]",
                event.src_path,
                "->",
                event.dest_path
            )


# ============================================================
# TELEMETRY CSV INITIALIZATION
# ============================================================

CSV_COLUMNS = [
    "timestamp",
    "window_id",
    "cpu_percent",
    "memory_percent",
    "process_count",
    "files_created",
    "files_modified",
    "files_deleted",
    "files_renamed",
    "total_file_events"
]


if not TELEMETRY_FILE.exists():

    with open(
        TELEMETRY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()


# ============================================================
# SYSTEM METRICS
# ============================================================

def get_system_metrics():

    cpu = psutil.cpu_percent(
        interval=0.5
    )

    memory = psutil.virtual_memory()

    process_count = len(
        list(
            psutil.process_iter()
        )
    )

    return {
        "cpu_percent": cpu,
        "memory_percent": memory.percent,
        "process_count": process_count
    }


# ============================================================
# SAVE TELEMETRY WINDOW
# ============================================================

def save_telemetry(
    window_id,
    metrics,
    events
):

    total_file_events = sum(
        events.values()
    )

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "window_id": window_id,

        "cpu_percent":
            round(
                metrics["cpu_percent"],
                2
            ),

        "memory_percent":
            round(
                metrics["memory_percent"],
                2
            ),

        "process_count":
            metrics["process_count"],

        "files_created":
            events.get("created", 0),

        "files_modified":
            events.get("modified", 0),

        "files_deleted":
            events.get("deleted", 0),

        "files_renamed":
            events.get("renamed", 0),

        "total_file_events":
            total_file_events
    }


    with open(
        TELEMETRY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writerow(row)


# ============================================================
# MONITORING LOOP
# ============================================================

def monitoring_loop():

    window_id = 1

    while True:

        time.sleep(5)

        metrics = get_system_metrics()

        with lock:

            events = dict(
                event_counter
            )

            event_counter.clear()


        save_telemetry(
            window_id,
            metrics,
            events
        )


        print("\n")
        print("-" * 70)

        print(
            f"5-SECOND ACTIVITY WINDOW #{window_id}"
        )

        print("-" * 70)

        print(
            "CPU usage        :",
            f"{metrics['cpu_percent']:.1f}%"
        )

        print(
            "Memory usage     :",
            f"{metrics['memory_percent']:.1f}%"
        )

        print(
            "Running processes:",
            metrics["process_count"]
        )

        print(
            "Files created    :",
            events.get("created", 0)
        )

        print(
            "Files modified   :",
            events.get("modified", 0)
        )

        print(
            "Files deleted    :",
            events.get("deleted", 0)
        )

        print(
            "Files renamed    :",
            events.get("renamed", 0)
        )

        print(
            "Total file events:",
            sum(events.values())
        )

        print(
            "Telemetry saved  :",
            TELEMETRY_FILE
        )

        window_id += 1


# ============================================================
# START OBSERVER
# ============================================================

event_handler = SafeFileMonitor()

observer = Observer()

observer.schedule(
    event_handler,
    str(MONITORED_DIR),
    recursive=True
)

observer.start()


monitor_thread = threading.Thread(
    target=monitoring_loop,
    daemon=True
)

monitor_thread.start()


print("\nMonitoring started.")
print("Collecting one telemetry record every 5 seconds.")
print("Press CTRL+C to stop.\n")


# ============================================================
# KEEP PROGRAM RUNNING
# ============================================================

try:

    while True:

        time.sleep(1)


except KeyboardInterrupt:

    print("\nStopping telemetry collector...")

    observer.stop()

    observer.join()

    print("Telemetry collector stopped safely.")