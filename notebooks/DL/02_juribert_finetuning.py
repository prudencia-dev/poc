"""POC Deep Learning — fine-tuning minimal de JuriBERT.

Le script est organise en cellules ``# %%`` pour une execution bloc par bloc.
Le notebook jumeau reprend les memes cellules avec les explications Markdown.

Le corpus fourni est un support pedagogique de POC. Les predictions ne sont pas
des decisions juridiques et doivent etre validees par un expert humain.
"""

# %% [markdown]
# # Classification de cas d'usage IA avec JuriBERT
#
# **Question DL :** un Transformer pre-entraine sur du texte juridique peut-il
# apprendre a classer une courte description de cas d'usage IA parmi cinq
# niveaux (`interdit`, `haut_risque`, `limite`, `minimal`, `hors_champ`) ?
#
# La demarche illustre transfert d'apprentissage, tokenisation, mini-batches,
# fonction de perte, optimiseur, epochs, early stopping et test independant.

# %% 1. Imports et configuration
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

RANDOM_STATE = 42
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    current_dir = Path.cwd()
    SCRIPT_DIR = next(
        (
            candidate
            for candidate in (current_dir, current_dir / "notebooks/DL", current_dir / "DL")
            if (candidate / "02_juribert_finetuning.py").exists()
        ),
        current_dir,
    )
NOTEBOOKS_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = NOTEBOOKS_DIR / "datasets/dl/dl_juribert_training_cases_v2.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/juribert_finetuning"
DEFAULT_MODEL_NAME = "dascim/juribert-base"
DEFAULT_TEXT_COLUMN = "Q3"
DEFAULT_LABEL_COLUMN = "risk_level_aiact"
MAX_LENGTH = 256
LOGGER = logging.getLogger("poc.juribert_dl")


# %% 2. Reproductibilite et materiel
def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    if torch.cuda.is_available():
        return f"CUDA — {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "Apple Silicon — MPS"
    return "CPU"


# %% 3. Chargement robuste du corpus
def load_dataset_file(dataset_path: Path) -> pd.DataFrame:
    """Charge un CSV UTF-8 ; accepte CP1252 pour l'ancien corpus fourni."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    last_error: Exception | None = None
    for encoding in ("utf-8", "cp1252"):
        try:
            dataframe = pd.read_csv(
                dataset_path, sep=None, engine="python", encoding=encoding
            )
            LOGGER.info("Dataset lu en %s : %d lignes, %d colonnes", encoding, *dataframe.shape)
            return dataframe
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Encodage du CSV non reconnu : {dataset_path}") from last_error


def validate_and_clean_dataset(
    dataframe: pd.DataFrame, text_column: str, label_column: str
) -> pd.DataFrame:
    """Controle le schema, nettoie les valeurs et detecte les conflits."""
    missing = [column for column in (text_column, label_column) if column not in dataframe]
    if missing:
        raise ValueError(
            f"Colonnes absentes : {missing}. Colonnes disponibles : {list(dataframe.columns)}"
        )
    cleaned = dataframe[[text_column, label_column]].dropna().copy()
    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()
    cleaned = cleaned[(cleaned[text_column] != "") & (cleaned[label_column] != "")]

    conflicting = cleaned.groupby(text_column)[label_column].nunique()
    conflicting_texts = set(conflicting[conflicting > 1].index)
    if conflicting_texts:
        raise ValueError(f"{len(conflicting_texts)} textes possedent plusieurs labels.")
    cleaned = cleaned.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    counts = cleaned[label_column].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(
            "Chaque classe doit contenir au moins trois exemples pour creer train/validation/test."
        )
    print(counts.to_string())
    return cleaned


# %% 4. Encodage explicite des classes
def encode_labels(
    dataframe: pd.DataFrame, label_column: str
) -> tuple[pd.DataFrame, LabelEncoder, dict[int, str], dict[str, int]]:
    encoder = LabelEncoder()
    encoded = dataframe.copy()
    encoded["labels"] = encoder.fit_transform(encoded[label_column])
    id2label = {index: str(label) for index, label in enumerate(encoder.classes_)}
    label2id = {label: index for index, label in id2label.items()}
    print("Mapping des classes :", id2label)
    return encoded, encoder, id2label, label2id


# %% 5. Trois ensembles : train, validation et test independant
def split_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70 % train, 15 % validation, 15 % test avec stratification."""
    train, temporary = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dataframe["labels"],
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temporary["labels"],
    )
    LOGGER.info("Train=%d, validation=%d, test=%d", len(train), len(validation), len(test))
    return tuple(part.reset_index(drop=True) for part in (train, validation, test))  # type: ignore[return-value]


# %% 6. Conversion au format Hugging Face et tokenisation
def create_huggingface_datasets(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    text_column: str,
) -> DatasetDict:
    def convert(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(frame[[text_column, "labels"]], preserve_index=False)
    return DatasetDict({
        "train": convert(train),
        "validation": convert(validation),
        "test": convert(test),
    })


def tokenize_datasets(
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    text_column: str,
    max_length: int,
) -> DatasetDict:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)

    tokenized = datasets.map(tokenize_batch, batched=True, desc="Tokenisation")
    removable = [
        column
        for column in tokenized["train"].column_names
        if column not in {"input_ids", "attention_mask", "token_type_ids", "labels"}
    ]
    return tokenized.remove_columns(removable) if removable else tokenized


# %% 7. Modele pre-entraine et transfert d'apprentissage
def load_model_and_tokenizer(
    model_name: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer


# %% 8. Metriques adaptees au desequilibre multiclasses
def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


# %% 9. Parametres d'entrainement et early stopping
def build_training_arguments(
    output_dir: Path,
    *, epochs: int,
    batch_size: int,
    learning_rate: float,
    use_fp16: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        fp16=bool(use_fp16 and torch.cuda.is_available()),
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
    )


def build_trainer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    datasets: DatasetDict,
    arguments: TrainingArguments,
) -> Trainer:
    return Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )


# %% 10. Evaluation finale reservee au jeu de test
def evaluate_on_test(
    trainer: Trainer,
    test_dataset: Dataset,
    id2label: dict[int, str],
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    output = trainer.predict(test_dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    label_ids = sorted(id2label)
    names = [id2label[index] for index in label_ids]
    report_dict = classification_report(
        labels, predictions, labels=label_ids, target_names=names,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        labels, predictions, labels=label_ids, target_names=names, zero_division=0,
    )
    matrix = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=label_ids),
        index=[f"reel_{name}" for name in names],
        columns=[f"predit_{name}" for name in names],
    )
    print(report_text)
    print(matrix)
    return report_dict, report_text, matrix


# %% 11. Sauvegarde et inference sur un nouvel exemple
def save_artifacts(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    encoder: LabelEncoder,
    report_dict: dict[str, Any],
    report_text: str,
    matrix: pd.DataFrame,
    output_dir: Path,
) -> None:
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"classes": encoder.classes_.tolist()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    matrix.to_csv(output_dir / "confusion_matrix.csv")


def predict_text(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    id2label: dict[int, str],
    max_length: int,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=True
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    prediction = int(np.argmax(probabilities))
    return {
        "predicted_label": id2label[prediction],
        "probabilities": {id2label[i]: float(value) for i, value in enumerate(probabilities)},
    }


# %% 12. Orchestration du fine-tuning
def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seeds()
    LOGGER.info("Materiel : %s", detect_device())
    raw = load_dataset_file(args.dataset)
    cleaned = validate_and_clean_dataset(raw, args.text_column, args.label_column)
    if args.smoke_test:
        cleaned = pd.concat(
            [
                group.sample(min(8, len(group)), random_state=RANDOM_STATE)
                for _, group in cleaned.groupby(args.label_column)
            ],
            ignore_index=True,
        )
    encoded, encoder, id2label, label2id = encode_labels(cleaned, args.label_column)
    train, validation, test = split_dataframe(encoded)
    datasets = create_huggingface_datasets(train, validation, test, args.text_column)
    model, tokenizer = load_model_and_tokenizer(args.model_name, id2label, label2id)
    tokenized = tokenize_datasets(datasets, tokenizer, args.text_column, args.max_length)
    arguments = build_training_arguments(
        args.output_dir,
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_fp16=args.use_fp16,
    )
    trainer = build_trainer(model, tokenizer, tokenized, arguments)
    trainer.train()
    validation_metrics = trainer.evaluate(tokenized["validation"])
    report_dict, report_text, matrix = evaluate_on_test(trainer, tokenized["test"], id2label)
    save_artifacts(trainer, tokenizer, encoder, report_dict, report_text, matrix, args.output_dir)
    example = "Un systeme utilise la reconnaissance faciale pour identifier des personnes dans un lieu public."
    print(json.dumps(predict_text(example, trainer.model, tokenizer, id2label, args.max_length), indent=2, ensure_ascii=False))
    return validation_metrics


# %% 13. Interface de ligne de commande
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Sous-ensemble et une epoch.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_pipeline(parse_arguments())
