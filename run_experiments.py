"""Build the shared evaluation manifest for frozen models."""

import csv
from pathlib import Path

RANDOM_SEEDS = (7, 17, 27)


def build_evaluation_manifest(models=("median", "rf", "xgboost", "lstm")):
    rows = []
    for model in models:
        rows.append({"model": model, "scenario": "S0", "seed": ""})
        for scenario in ("S1", "S2"):
            for seed in RANDOM_SEEDS:
                rows.append({"model": model, "scenario": scenario, "seed": seed})
        rows.append({"model": model, "scenario": "S3", "seed": ""})
    return rows


def write_manifest(path="experiment_manifest.csv"):
    rows = build_evaluation_manifest()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "scenario", "seed"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write_manifest()
