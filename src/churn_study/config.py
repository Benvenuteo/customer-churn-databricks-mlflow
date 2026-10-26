"""Configuration and predictive feature contracts shared by local and cloud runs."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RAW_COLUMNS = (
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
)
COLUMNS = (
    "customer_id",
    "age",
    "gender",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "subscription_type",
    "contract_length",
    "total_spend",
    "last_interaction",
    "churn",
)
CATEGORICAL = ("gender", "subscription_type", "contract_length")
NUMERIC = ("age", "tenure", "usage_frequency", "support_calls", "payment_delay", "last_interaction")


def feature_columns(task: str) -> tuple[str, ...]:
    """Explicitly exclude identifiers and the opposite target from regression features."""
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unknown task: {task}")
    return NUMERIC + (("total_spend",) if task == "classification" else ()) + CATEGORICAL


@dataclass(frozen=True)
class Settings:
    """Store all selection/evaluation choices before inspecting model performance."""

    output: Path
    run_id: str = "churn-v1"
    master: str = "local[2]"
    partitions: int = 4
    seed: int = 42
    folds: int = 3
    cv_fraction: float = 0.25
    bootstrap_repeats: int = 200
    tree_iterations: int = 20
    tracking_uri: str | None = None
    experiment_name: str = "customer-churn-spending"
    table_format: str = "parquet"
    registry_prefix: str = "customer_study"
    data_kind: str = "provided"

    def __post_init__(self) -> None:
        if not self.run_id or any(char in self.run_id for char in "/\\\x00"):
            raise ValueError("run_id must be a nonempty path-safe identifier")
        if self.partitions < 1 or self.folds < 2:
            raise ValueError("partitions must be positive and folds must be at least 2")
        if not 0 < self.cv_fraction <= 1:
            raise ValueError("cv_fraction must be in (0, 1]")
        if self.bootstrap_repeats < 0 or self.tree_iterations < 1:
            raise ValueError("bootstrap repeats must be nonnegative and tree iterations positive")
        if self.table_format not in {"parquet", "delta"}:
            raise ValueError("table_format must be parquet or delta")
        if self.data_kind not in {"provided", "synthetic"}:
            raise ValueError("data_kind must be provided or synthetic")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output"] = str(self.output.resolve())
        # Tracking credentials/URLs are environment configuration, not portable run metadata.
        values.pop("tracking_uri")
        return values
