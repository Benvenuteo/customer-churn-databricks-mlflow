"""Local commands; cloud notebooks pass their managed Spark session to the same functions."""

import argparse
import json
import logging
import os
from pathlib import Path

from churn_study.config import Settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Customer churn and spending study")
    commands = result.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create explicitly synthetic demonstration CSVs")
    demo.add_argument("--directory", type=Path, default=Path("data/demo"))
    demo.add_argument("--rows", type=int, default=600)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--training", type=Path, required=True)
        command.add_argument("--external", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--run-id", default="churn-v1")
        command.add_argument("--data-kind", choices=("provided", "synthetic"), default="provided")
        command.add_argument("--profile", choices=("full", "smoke"), default="full")
        command.add_argument("--cv-fraction", type=float)
        command.add_argument("--folds", type=int)
        command.add_argument("--bootstrap", type=int)
        command.add_argument("--master", default="local[2]")
        command.add_argument("--partitions", type=int, default=4)
        command.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Resume from prepared silver data")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--task", choices=("classification", "regression", "all"), default="all")
    train.add_argument("--profile", choices=("full", "smoke"), default="full")
    report = commands.add_parser("report", help="Regenerate the self-contained HTML report")
    report.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Apply a saved complete native Spark pipeline")
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--task", choices=("classification", "regression"), required=True)
    score.add_argument("--threshold", type=float)
    return result


def restore(output: Path) -> Settings:
    values = json.loads((output / "manifest.json").read_text())["settings"]
    values["output"] = output
    values["tracking_uri"] = os.environ.get("MLFLOW_TRACKING_URI")
    return Settings(**values)


def main() -> None:
    """Dispatch without starting a JVM for help, demo generation or report rendering."""
    args = parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.command == "demo":
        from churn_study.demo import generate

        print(generate(args.directory, args.rows))
        return
    if args.command == "report":
        from churn_study.report import render_report

        print(render_report(args.output))
        return
    if args.command == "score":
        from churn_study.scoring import score_csv

        score_csv(args.input, args.model, args.output, args.task, args.threshold)
        return
    if args.command == "train":
        settings = restore(args.output)
    else:
        smoke = args.profile == "smoke"
        settings = Settings(
            output=args.output,
            run_id=args.run_id,
            master=args.master,
            partitions=args.partitions,
            seed=args.seed,
            data_kind=args.data_kind,
            folds=args.folds if args.folds is not None else (2 if smoke else 3),
            cv_fraction=args.cv_fraction
            if args.cv_fraction is not None
            else (1.0 if smoke else 0.25),
            bootstrap_repeats=args.bootstrap
            if args.bootstrap is not None
            else (20 if smoke else 200),
            tree_iterations=5 if smoke else 20,
        )
    from churn_study.engine import create_spark, prepare, train_task

    spark = create_spark(settings)
    try:
        if args.command in {"prepare", "run"}:
            prepare(spark, args.training, args.external, settings)
        if args.command in {"train", "run"}:
            tasks = (
                ("classification", "regression")
                if getattr(args, "task", "all") == "all"
                else (args.task,)
            )
            for task in tasks:
                train_task(spark, settings, task, smoke=args.profile == "smoke")
            from churn_study.report import render_report

            print(render_report(settings.output))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
