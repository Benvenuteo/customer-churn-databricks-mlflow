"""Self-contained, measured-result report with matched metric/plot populations."""

import base64
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402


def table(rows: list[dict[str, Any]]) -> str:
    """Escape every source-derived value and keep absent metrics explicit."""
    if not rows:
        return "<p>No results recorded.</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(str(name))}</th>" for name in columns)
    body = []
    for row in rows:
        cells = []
        for name in columns:
            value = row[name]
            formatted = f"{value:,.4f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{html.escape(formatted)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _figure(fig: Any, directory: Path, name: str, alt: str) -> str:
    """Keep a PNG artifact and embed the same bytes so the HTML works as a single file."""
    path = directory / (name + ".png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{encoded}'>"


def render_report(root: Path) -> Path:
    """Render existing artifacts only; this function never trains, tunes or alters a model."""
    manifest = json.loads((root / "manifest.json").read_text())
    quality = json.loads((root / "quality.json").read_text())
    audit = json.loads((root / "shift_audit.json").read_text())
    directory = root / "report"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    synthetic = manifest["settings"]["data_kind"] == "synthetic"
    badge = (
        "SYNTHETIC SMOKE TEST — not results from the supplied dataset"
        if synthetic
        else "Results from the supplied development and external CSV files"
    )
    counts = [{"source": source, **values} for source, values in quality["sources"].items()]
    summary = [
        {
            "source": row["source"],
            "rows": row["rows"],
            "churn_rate": row["mean_churn"],
            "id_churn_correlation": row["id_churn_pearson"],
        }
        for row in audit["source_summary"]
    ]
    content = [
        f"<h1>Customer churn and spending</h1><p class='badge'>{badge}</p>",
        f"<p>Run <code>{html.escape(manifest['run_id'])}</code>. Seed {manifest['settings']['seed']}.</p>",
        "<section><h2>Study questions and protocol</h2><p>Can customer attributes discriminate the supplied churn label, "
        "and estimate recorded total spending? No dated prediction horizon, spending currency or customer lifetime value "
        "is established by these files. CustomerID is excluded from models; Churn is also excluded from spending features.</p>"
        "<p>The development file is stratified 70/15/15. A declared sample of training rows is used for cross-validation; "
        "the winner is refitted on all training rows. Validation selects the classification F1 threshold. Internal test and "
        "the supplied external file do not select candidates or preprocessing. Between-file diagnostics were inspected "
        "during the audit; the external file is a transfer stress test, not an unseen real-world deployment.</p></section>",
        "<section><h2>Data quality and between-file differences</h2>"
        + table(counts)
        + table(summary)
        + "<h3>Churn among records with at least six support calls</h3>"
        + table(audit["support_calls_ge6"])
        + "<p>Shared ID values are not assumed to identify the same person across files.</p>"
        + table([audit["identity_overlap"]]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, name in zip(
        axes.flat, ("age", "support_calls", "payment_delay", "total_spend"), strict=True
    ):
        for source, color in (("development", "#216b8e"), ("external", "#c17124")):
            bins = [
                item
                for item in audit.get("numeric_bins", [])
                if item["source"] == source
                and item["feature"] == name
                and item["lower_bound"] is not None
            ]
            bins.sort(key=lambda item: item["lower_bound"])
            count = sum(item["count"] for item in bins)
            if count:
                axis.plot(
                    [item["lower_bound"] for item in bins],
                    [item["count"] / count for item in bins],
                    marker=".",
                    label=source,
                    color=color,
                )
        axis.set(xlabel=name + " (recorded units)", ylabel="Share of records")
        axis.legend()
    content.append(
        _figure(
            fig,
            directory,
            "source_distributions",
            "Feature distributions in development and external files",
        )
        + "</section>"
    )
    results: dict[str, Any] = {}
    for task in ("classification", "regression"):
        path = root / f"{task}.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        results[task] = result
        content.append(
            f"<section><h2>{task.title()}</h2><p>Selected: <strong>{html.escape(result['winner'])}</strong>. "
            f"CV sample: {result['cv_sample_rows']:,}; full-training refit: {result['full_training_rows']:,} rows.</p>"
        )
        fields = (
            (
                "roc_auc",
                "average_precision",
                "f1",
                "brier",
                "precision_at_top_10pct",
                "lift_at_top_10pct",
            )
            if task == "classification"
            else ("mae", "rmse", "r2")
        )
        metric_rows = []
        for split, evaluation in result["evaluations"].items():
            for label in ("model", "baseline"):
                metric_rows.append(
                    {
                        "partition": split,
                        "estimator": label,
                        **{name: evaluation[label][name] for name in fields},
                    }
                )
        content.append(table(metric_rows))
        content.append(
            "<h3>Cross-validation candidates</h3>"
            + table(
                [
                    {"candidate": item["name"], "mean": item["mean"], "fold_std": item["std"]}
                    for item in result["candidates"]
                ]
            )
        )
        intervals = [
            {"partition": split, "metric": name, **values}
            for split, evaluation in result["evaluations"].items()
            for name, values in evaluation["bootstrap"]["intervals"].items()
        ]
        content.append(
            "<h3>95% confidence intervals</h3>"
            + table(intervals)
            + "<p>Percentile bootstrap of fixed held-out predictions; these intervals exclude training uncertainty.</p>"
        )
        frames = {
            split: pd.read_parquet(root / "predictions" / task / f"{split}.parquet")
            for split in ("test", "external")
        }
        if task == "classification":
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
            for split, frame in frames.items():
                y, score = frame["label"].to_numpy(), frame["score"].to_numpy()
                fpr, tpr, _ = roc_curve(y, score)
                precision, recall, _ = precision_recall_curve(y, score)
                observed, predicted = calibration_curve(y, score, n_bins=10, strategy="quantile")
                axes[0].plot(fpr, tpr, label=f"{split}: AUC {roc_auc_score(y, score):.3f}")
                axes[1].plot(recall, precision, label=split)
                axes[2].plot(predicted, observed, marker=".", label=split)
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[2].plot([0, 1], [0, 1], "--", color="gray")
            for axis, x, y, title in zip(
                axes,
                ("False positive rate", "Recall", "Mean predicted probability"),
                ("True positive rate", "Precision", "Observed churn rate"),
                ("Held-out ROC", "Precision-recall", "Calibration"),
                strict=True,
            ):
                axis.set(xlabel=x, ylabel=y, title=title)
                axis.legend()
            content.append(
                _figure(
                    fig,
                    directory,
                    "classification_diagnostics",
                    "ROC, precision-recall and calibration on the stated held-out populations",
                )
            )
            content.append(
                f"<p>Decision threshold {result['threshold']:.2f} was selected on validation only. "
                "Average precision (AP) is used for the precision-recall summary. Top-decile ties use seeded random ordering.</p>"
            )
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
            for axis, (split, frame) in zip(axes, frames.items(), strict=True):
                sample = frame.sample(min(len(frame), 2000), random_state=42)
                axis.scatter(sample["label"], sample["score"], alpha=0.15, s=10)
                bounds = [
                    min(sample["label"].min(), sample["score"].min()),
                    max(sample["label"].max(), sample["score"].max()),
                ]
                axis.plot(bounds, bounds, "--", color="gray")
                axis.set(
                    xlabel="Observed spending (dataset units)",
                    ylabel="Predicted spending",
                    title=split,
                )
            content.append(
                _figure(
                    fig,
                    directory,
                    "regression_predictions",
                    "Predicted versus observed spending on held-out records",
                )
            )
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
            for axis, (split, frame) in zip(axes, frames.items(), strict=True):
                sample = frame.sample(min(len(frame), 2000), random_state=42)
                residual = sample["label"] - sample["score"]
                axis.scatter(sample["score"], residual, alpha=0.15, s=10)
                axis.axhline(0, color="gray", linestyle="--")
                axis.set(
                    xlabel="Predicted spending", ylabel="Observed minus predicted", title=split
                )
            content.append(
                _figure(
                    fig,
                    directory,
                    "regression_residuals",
                    "Held-out spending residuals versus predictions",
                )
            )
            fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
            for family in ("ridge", "lasso"):
                points = sorted(
                    [item for item in result["candidates"] if item["name"].startswith(family)],
                    key=lambda item: item["reg_param"],
                )
                axis.errorbar(
                    [item["reg_param"] for item in points],
                    [item["mean"] for item in points],
                    yerr=[item["std"] for item in points],
                    marker="o",
                    label=family,
                )
            ols = next(item for item in result["candidates"] if item["name"] == "ols")
            axis.axhline(ols["mean"], color="gray", linestyle="--", label="OLS")
            axis.set(
                xscale="log",
                xlabel="Regularization parameter",
                ylabel="Mean CV RMSE (dataset units)",
            )
            axis.legend()
            content.append(
                _figure(
                    fig,
                    directory,
                    "regularization_cv",
                    "Cross-validation errors for Ridge, Lasso and OLS",
                )
            )
        content.append(
            "<h3>External segment diagnostics</h3>"
            + table(
                [
                    {
                        "feature": row["feature"],
                        "value": row["value"],
                        "rows": row["rows"],
                        **{name: row[name] for name in fields[:3]},
                    }
                    for row in result["evaluations"]["external"]["segments"]
                ]
            )
        )
        diagnostics = sorted(
            result["model_details"]["features"], key=lambda item: abs(item["value"]), reverse=True
        )[:12]
        content.append(
            "<h3>Selected model feature diagnostics</h3>"
            + table(diagnostics)
            + "<p>Tree split importance is descriptive. Linear coefficients use original feature units. Neither establishes causality.</p>"
        )
        content.append("<h3>Model lineage</h3>" + table([result["lineage"]]) + "</section>")
    regularization_file = root / "regularization.json"
    if regularization_file.exists():
        paths = json.loads(regularization_file.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for axis, family in zip(axes, ("ridge", "lasso"), strict=True):
            points = sorted(
                [point for point in paths["paths"] if point["family"] == family],
                key=lambda point: point["reg_param"],
            )
            numeric_names = [
                item["name"]
                for item in points[0]["coefficients"]
                if item["name"].endswith("_imputed")
            ]
            for name in numeric_names:
                axis.plot(
                    [point["reg_param"] for point in points],
                    [
                        next(
                            item["per_training_sd"]
                            for item in point["coefficients"]
                            if item["name"] == name
                        )
                        for point in points
                    ],
                    marker=".",
                    label=name.removesuffix("_imputed"),
                )
            axis.set(
                xscale="log",
                xlabel="Regularization parameter",
                ylabel="Coefficient per training SD",
                title=family,
            )
            axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
        content.append(
            "<section><h2>Training-only coefficient paths</h2><p>These diagnostic refits do not select the winning model. Coefficients are expressed per one training standard deviation, in dataset spending units.</p>"
            + _figure(
                fig,
                directory,
                "coefficient_paths",
                "Ridge and Lasso standardized coefficient paths",
            )
            + table(
                [
                    {key: item[key] for key in ("family", "reg_param", "zero_coefficients")}
                    for item in paths["paths"]
                ]
            )
            + "</section>"
        )
    automl_results = []
    for task in ("classification", "regression"):
        artifact = root / f"automl_{task}.json"
        if artifact.exists():
            result = json.loads(artifact.read_text())
            fields = (
                ("roc_auc", "average_precision", "f1")
                if task == "classification"
                else ("mae", "rmse", "r2")
            )
            automl_results.extend(
                {"task": task, "partition": split, **{name: values[name] for name in fields}}
                for split, values in result["evaluations"].items()
            )
    content.append("<section><h2>Optional Databricks AutoML</h2>")
    if automl_results:
        for task in ("classification", "regression"):
            content.append(table([item for item in automl_results if item["task"] == task]))
        content.append(
            "<p>Feature whitelists and holdout partitions match. AutoML has its recorded time budget; manual CV uses its recorded training-only sample. Optimization budgets differ.</p>"
        )
    else:
        content.append(
            "<p>Not executed in this local run. The supplied Databricks notebook performs the comparison in an authenticated compatible workspace; no AutoML score is claimed here.</p>"
        )
    content.append("</section>")
    if "classification" in results:
        internal = results["classification"]["evaluations"]["test"]["model"]
        external = results["classification"]["evaluations"]["external"]["model"]
        content.append(
            f"<section><h2>Measured transfer result</h2><p>Classification ROC-AUC is {internal['roc_auc']:.4f} on the internal test and {external['roc_auc']:.4f} on the external file. The external top-decile lift is {external['lift_at_top_10pct']:.4f}. A lift below 1 means the selected top decile has a lower observed churn rate than the external population average.</p>"
        )
        if "regression" in results:
            evaluation = results["regression"]["evaluations"]["external"]
            content.append(
                f"<p>External regression RMSE is {evaluation['model']['rmse']:.4f}, versus {evaluation['baseline']['rmse']:.4f} for the training-mean baseline; R² is {evaluation['model']['r2']:.4f}. Negative R² compares against the external file's own mean, which is not the deployable training-mean baseline.</p>"
            )
        content.append(
            "<p>The chosen specification does not demonstrate reliable transfer to this external file. Candidate registration records an experiment; it is not a decision to deploy the model.</p></section>"
        )
    content.append(
        "<section><h2>Interpretation and limits</h2><p>High internal discrimination is not sufficient evidence of "
        "generalization to the other file. Conditional label relationships, ID associations and feature distributions "
        "differ. This is not proof of temporal concept drift, target leakage or causality. The data card does not establish "
        "a real business sampling process or observation horizon. No retention intervention effect or monetary ROI is estimated.</p>"
        "<p>Local Spark/MLflow execution is recorded in the manifest. Databricks notebooks and the optional AutoML "
        "comparison require an authenticated compatible workspace; local execution is not evidence of a cloud run.</p></section>"
    )
    content.append(
        "<section><h2>Reproduction metadata</h2>"
        + table([{"source": name, **identity} for name, identity in manifest["inputs"].items()])
        + table([{key: value for key, value in manifest["environment"].items()}])
        + "<p>Source: Muhammad Shahid Azeem, Customer Churn Dataset, Kaggle version 1. Runtime settings, candidate parameters and exact scores are preserved in the accompanying JSON artifacts.</p></section>"
    )
    style = """body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f2f6f8;color:#173346}main{max-width:1150px;margin:auto;padding:32px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}h1,h2,h3{line-height:1.25}.badge{font-weight:700;color:#91520e}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #dde5e8;text-align:left;vertical-align:top}th{background:#e9f0f3}img{width:100%;height:auto}code{overflow-wrap:anywhere}@media print{body{background:white}main{padding:0}section{break-inside:avoid}}"""
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Customer churn and spending study</title><style>"
        + style
        + "</style><main>"
        + "".join(content)
        + "</main></html>"
    )
    path = directory / "index.html"
    path.write_text(document, encoding="utf-8")
    return path
