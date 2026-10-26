"""Bounded, reproducible artifact I/O and execution metadata."""

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pyspark


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write valid JSON, rejecting nonfinite float values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(path: Path) -> dict[str, Any]:
    """Hash source bytes without loading the whole CSV into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def environment() -> dict[str, Any]:
    """Record runtime facts without inspecting authentication configuration."""
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=False)
    memory_file = Path("/sys/fs/cgroup/memory.max")
    limit = memory_file.read_text().strip() if memory_file.exists() else "unknown"
    return {
        "python": platform.python_version(),
        "spark": pyspark.__version__,
        "mlflow": mlflow.__version__,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cgroup_memory_limit_bytes": int(limit) if limit.isdigit() else None,
        "java": (java.stderr or java.stdout).splitlines()[0],
    }
