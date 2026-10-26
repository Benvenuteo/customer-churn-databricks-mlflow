"""Publish small measured artifacts and portable native model examples without raw customer rows."""

import argparse
import json
import shutil
from pathlib import Path

from churn_study.io import write_json


def export(source: Path, report_destination: Path, example_destination: Path | None = None) -> None:
    """Normalize only the local run path, preserving all measured metrics and fingerprints."""
    manifest = json.loads((source / "manifest.json").read_text())
    if any(
        manifest["stages"].get(task, {}).get("status") != "complete"
        for task in ("classification", "regression")
    ):
        raise ValueError("Both learning tasks must be complete before exporting measured results")
    report_destination.mkdir(parents=True, exist_ok=True)
    manifest["settings"]["output"] = "outputs/" + manifest["run_id"]
    manifest["export_note"] = (
        "Local output path normalized; measured metrics and input identities unchanged"
    )
    write_json(report_destination / "manifest.json", manifest)
    for name in (
        "quality.json",
        "shift_audit.json",
        "classification.json",
        "regression.json",
        "regularization.json",
        "automl_classification.json",
        "automl_regression.json",
    ):
        if (source / name).exists():
            shutil.copyfile(source / name, report_destination / name)
    shutil.copytree(source / "report", report_destination / "report", dirs_exist_ok=True)
    if example_destination is not None:
        example_destination.mkdir(parents=True, exist_ok=True)
        for task in ("classification", "regression"):
            shutil.copytree(
                source / "models" / task, example_destination / "models" / task, dirs_exist_ok=True
            )
            shutil.copyfile(source / f"{task}.json", example_destination / f"{task}.json")
        write_json(example_destination / "source_identity.json", manifest["inputs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("report_destination", type=Path)
    parser.add_argument("--examples", type=Path)
    args = parser.parse_args()
    export(args.source, args.report_destination, args.examples)
