"""Small synthetic smoke-test data, never presented as Kaggle observations."""

import csv
import random
from pathlib import Path

from churn_study.config import RAW_COLUMNS


def generate(directory: Path, rows: int = 600, seed: int = 42) -> tuple[Path, Path]:
    """Generate two labelled files with an explicit between-source shift."""
    if rows < 200:
        raise ValueError("Use at least 200 development rows")
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / "development.csv", directory / "external.csv")
    if any(path.exists() for path in paths):
        raise FileExistsError("Demo files already exist; choose a new directory")
    rng = random.Random(seed)
    for path, count, external in ((paths[0], rows, False), (paths[1], max(100, rows // 4), True)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            for customer in range(1, count + 1):
                tenure, support, delay = rng.randint(1, 60), rng.randint(0, 10), rng.randint(0, 30)
                churn_rate = (
                    (0.61 if external else 1.0) if support >= 6 else (0.2 + 0.3 * (delay > 20))
                )
                writer.writerow(
                    dict(
                        zip(
                            RAW_COLUMNS,
                            (
                                customer,
                                rng.randint(18, 65),
                                rng.choice(("Male", "Female")),
                                tenure,
                                rng.randint(1, 30),
                                support,
                                delay,
                                rng.choice(("Basic", "Standard", "Premium")),
                                rng.choice(("Monthly", "Quarterly", "Annual")),
                                max(
                                    100.0,
                                    min(1000.0, round(250 + tenure * 9 + rng.gauss(0, 80), 2)),
                                ),
                                rng.randint(1, 30),
                                int(rng.random() < churn_rate),
                            ),
                            strict=True,
                        )
                    )
                )
    return paths
