"""DL niveau 1 — fine-tuning simple de JuriBERT pour PRUDENCIA.

Ce fichier montre uniquement le parcours essentiel : charger le dataset,
préparer les classes, tokeniser, entraîner, évaluer et sauvegarder le modèle.
Il se lance avec le bouton « Run Python File » de VS Code.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# 1. Configuration modifiable avant de cliquer sur « Run Python File »
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "datasets/dl/dl_juribert_training_cases.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs/dl_niveau_1"
MODEL_NAME = "dascim/juribert-base"
TEXT_COLUMN = "Q3"
LABEL_COLUMN = "risk_level_aiact"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
MAX_LENGTH = 256
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-5

LOGGER = logging.getLogger(__name__)


def fixer_graines(seed: int = RANDOM_STATE) -> None:
    """Rend la séparation et l'entraînement aussi reproductibles que possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def charger_et_nettoyer_dataset(path: Path) -> pd.DataFrame:
    """Charge les deux colonnes utiles et retire les lignes inutilisables."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    donnees = pd.read_csv(path, sep=";", encoding="utf-8")
    colonnes_manquantes = {TEXT_COLUMN, LABEL_COLUMN} - set(donnees.columns)
    if colonnes_manquantes:
        raise ValueError(f"Colonnes absentes : {sorted(colonnes_manquantes)}")

    donnees = donnees[[TEXT_COLUMN, LABEL_COLUMN]].dropna().copy()
    donnees[TEXT_COLUMN] = donnees[TEXT_COLUMN].astype(str).str.strip()
    donnees[LABEL_COLUMN] = donnees[LABEL_COLUMN].astype(str).str.strip()
    donnees = donnees[
        (donnees[TEXT_COLUMN] != "") & (donnees[LABEL_COLUMN] != "")
    ].drop_duplicates()
    return donnees.reset_index(drop=True)


def preparer_datasets(
    donnees: pd.DataFrame,
) -> tuple[DatasetDict, dict[int, str], dict[str, int]]:
    """Encode les classes puis crée les jeux d'entraînement et de validation."""
    encodeur = LabelEncoder()
    donnees = donnees.copy()
    donnees["labels"] = encodeur.fit_transform(donnees[LABEL_COLUMN])
    id2label = {index: str(label) for index, label in enumerate(encodeur.classes_)}
    label2id = {label: index for index, label in id2label.items()}

    entrainement, validation = train_test_split(
        donnees,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=donnees["labels"],
    )
    colonnes = [TEXT_COLUMN, "labels"]
    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(
                entrainement[colonnes], preserve_index=False
            ),
            "validation": Dataset.from_pandas(
                validation[colonnes], preserve_index=False
            ),
        }
    )
    return datasets, id2label, label2id


def calculer_metriques(evaluation: Any) -> dict[str, float]:
    """Calcule l'exactitude et la macro-F1 sur le jeu de validation."""
    logits, labels = evaluation
    predictions = np.argmax(logits, axis=-1)
    return {
        "exactitude": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    """Exécute dans l'ordre le pipeline pédagogique de fine-tuning."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    fixer_graines()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("1/5 — Chargement du dataset Prudencia")
    donnees = charger_et_nettoyer_dataset(DATASET_PATH)
    datasets, id2label, label2id = preparer_datasets(donnees)

    LOGGER.info("2/5 — Chargement du tokenizer et de JuriBERT")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modele = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    LOGGER.info("3/5 — Tokenisation des textes")

    def tokeniser(lot: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(lot[TEXT_COLUMN], truncation=True, max_length=MAX_LENGTH)

    tokenises = datasets.map(tokeniser, batched=True)
    colonnes_a_retirer = [
        colonne
        for colonne in tokenises["train"].column_names
        if colonne not in {"input_ids", "attention_mask", "token_type_ids", "labels"}
    ]
    tokenises = tokenises.remove_columns(colonnes_a_retirer)

    arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=RANDOM_STATE,
    )
    trainer = Trainer(
        model=modele,
        args=arguments,
        train_dataset=tokenises["train"],
        eval_dataset=tokenises["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=calculer_metriques,
    )

    LOGGER.info("4/5 — Fine-tuning de JuriBERT")
    trainer.train()
    metriques = trainer.evaluate()

    LOGGER.info("5/5 — Sauvegarde du modèle et des métriques")
    dossier_modele = OUTPUT_DIR / "modele_final"
    trainer.save_model(dossier_modele)
    tokenizer.save_pretrained(dossier_modele)
    (OUTPUT_DIR / "metriques.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Terminé — modèle : %s", dossier_modele)


if __name__ == "__main__":
    main()
