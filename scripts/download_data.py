"""Download only the two known version-1 CSVs, with bounded memory and explicit provenance."""

import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

FILES = {
    "customer_churn_dataset-training-master.csv": 23_448_754,
    "customer_churn_dataset-testing-master.csv": 3_282_220,
}
URL = "https://www.kaggle.com/api/v1/datasets/download/muhammadshahidazeem/customer-churn-dataset?datasetVersionNumber=1"


def download(destination: Path) -> None:
    """Do not overwrite user data or extract arbitrary archive paths."""
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() for name in FILES):
        raise FileExistsError("Source files already exist; use them or choose another directory")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "source.zip"
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                "Download the two version-1 files manually from the dataset card if Kaggle requires authentication"
            ) from exc
        with zipfile.ZipFile(archive) as bundle:
            for name, size in FILES.items():
                info = bundle.getinfo(name)
                if info.file_size != size:
                    raise ValueError(
                        f"Unexpected source size for {name}; verify the dataset version"
                    )
            for name in FILES:
                partial = destination / (name + ".part")
                with bundle.open(name) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                partial.replace(destination / name)
    print(f"Downloaded two version-1 CSVs to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.destination)
