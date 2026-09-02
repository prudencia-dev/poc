"""Télécharge la copie de travail du dataset public AI Incidents."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATASET_REVISION = "a867edbc6ce5380343fff69fb7f97d2288f16ec9"
DATASET_URL = (
    "https://huggingface.co/datasets/butterflylabs/ai-incidents/resolve/"
    f"{DATASET_REVISION}/incidents.csv"
)
DEFAULT_OUTPUT = Path(__file__).parents[1] / "datasets" / "ml" / "ai_incidents.csv"


def download_dataset(output: Path = DEFAULT_OUTPUT) -> Path:
    """Télécharge le fichier en conservant une révision reproductible."""
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "poc-ml/2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        output.write_bytes(response.read())
    print(f"Dataset enregistré : {output} ({output.stat().st_size / 1_000_000:.1f} Mo)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    download_dataset(parser.parse_args().output)
