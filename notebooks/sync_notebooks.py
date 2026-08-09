"""Genere les notebooks a partir des cellules # %% des scripts Python.

Le script Python reste la source de verite. Cette conversion evite que le code
du notebook et celui de la ligne de commande divergent au fil des corrections.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAIRS = {
    ROOT / "ML/01_random_forest_training.py": ROOT / "ML/01_RandomForest_Training.ipynb",
    ROOT / "DL/02_juribert_finetuning.py": ROOT / "DL/02_JuriBERT_FineTuning.ipynb",
}

FINAL_CELLS = {
    "01_random_forest_training.py": "# Demonstration rapide : reduit seulement la recherche d'hyperparametres.\nmetrics = run_pipeline(quick=True)\nmetrics",
    "02_juribert_finetuning.py": "# Smoke test : petit sous-ensemble et une seule epoch.\nnotebook_args = argparse.Namespace(\n    dataset=DEFAULT_DATASET_PATH, text_column=DEFAULT_TEXT_COLUMN,\n    label_column=DEFAULT_LABEL_COLUMN, model_name=DEFAULT_MODEL_NAME,\n    output_dir=DEFAULT_OUTPUT_DIR, epochs=3, batch_size=8,\n    learning_rate=2e-5, max_length=MAX_LENGTH, use_fp16=False,\n    smoke_test=True,\n)\n# Decommenter pour lancer le fine-tuning :\n# validation_metrics = run_pipeline(notebook_args)\n",
}


def markdown_source(lines: list[str]) -> str:
    converted = []
    for line in lines:
        converted.append(re.sub(r"^# ?", "", line))
    return "".join(converted).strip() + "\n"


def make_notebook(source_path: Path) -> dict:
    text = source_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^# %%(?P<markdown> \[markdown\])?(?P<title>.*)$", text, re.MULTILINE))
    cells = []
    for index, match in enumerate(matches):
        start = match.end() + 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        if index == len(matches) - 1 and "\nif __name__ == \"__main__\":" in block:
            block = block.split("\nif __name__ == \"__main__\":", 1)[0]
        if match.group("markdown"):
            cells.append({
                "id": hashlib.sha1(f"{source_path.name}-md-{index}".encode()).hexdigest()[:8],
                "cell_type": "markdown",
                "metadata": {},
                "source": markdown_source(block).splitlines(keepends=True),
            })
        else:
            title = match.group("title").strip()
            if title:
                cells.append({
                    "id": hashlib.sha1(f"{source_path.name}-title-{index}".encode()).hexdigest()[:8],
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## {title}\n"],
                })
            cells.append({
                "id": hashlib.sha1(f"{source_path.name}-code-{index}".encode()).hexdigest()[:8],
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": block.strip().splitlines(keepends=True),
            })
    cells.append({
        "id": hashlib.sha1(f"{source_path.name}-final".encode()).hexdigest()[:8],
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": FINAL_CELLS[source_path.name].splitlines(keepends=True),
    })
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for source, target in PAIRS.items():
        target.write_text(
            json.dumps(make_notebook(source), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"Notebook synchronise : {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
