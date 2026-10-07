"""Merge model evaluation outputs into one comparable table."""

import csv
from pathlib import Path

REQUIRED_COLUMNS = {"model", "scenario", "seed", "mae", "rmse"}


def merge_metrics(input_files, output_file="metrics.csv"):
    rows = []
    for filename in input_files:
        with Path(filename).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{filename} missing columns: {sorted(missing)}")
            rows.extend(reader)
    with Path(output_file).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "scenario", "seed", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)
    return rows
