"""
PRUDENCIA - Fine-Tuning supervisé de JuriBERT
=============================================

Ce script entraîne un modèle JuriBERT pour une tâche de classification de textes
juridiques dans le cadre du projet PRUDENCIA.

Objectif :
    À partir d'un texte décrivant un projet ou un cas d'usage IA, prédire une
    classe de risque ou une catégorie définie dans le dataset annoté.

Modèle par défaut :
    dascim/juribert-base

Le modèle peut être remplacé à l'exécution, par exemple :
    --model-name dascim/juribert-small

Format minimal attendu du CSV :
    text,label
    "Un système analyse automatiquement les candidatures...",haut_risque
    "Un assistant recommande des contenus culturels...",risque_limite

Les noms des colonnes sont configurables avec :
    --text-column
    --label-column

Exécution :
    python 02_juribert_finetuning.py \
        --dataset ../datasets/deep_learning/prudencia_annotated_cases.csv

Exemple plus léger pour une machine sans GPU puissant :
    python 02_juribert_finetuning.py \
        --dataset ../datasets/deep_learning/prudencia_annotated_cases.csv \
        --model-name dascim/juribert-small \
        --batch-size 4 \
        --epochs 3

Dépendances principales :
    pandas
    numpy
    scikit-learn
    torch
    transformers
    datasets
    accelerate

Remarque :
    Le fine-tuning d'un Transformer est nettement plus coûteux que
    l'entraînement d'un Random Forest. Un GPU CUDA est recommandé, mais le
    script peut fonctionner sur CPU avec un petit modèle et un petit dataset.
"""

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


# ---------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("prudencia.juribert")

DEFAULT_MODEL_NAME = "dascim/juribert-base"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "label"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# 2. ARGUMENTS DU SCRIPT
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Lit les paramètres transmis dans la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning supervisé de JuriBERT pour PRUDENCIA."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Chemin du dataset CSV annoté.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=(
            "Nom ou chemin du modèle Hugging Face. "
            f"Valeur par défaut : {DEFAULT_MODEL_NAME}"
        ),
    )

    parser.add_argument(
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help="Nom de la colonne contenant les textes.",
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="Nom de la colonne contenant les classes.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/juribert_finetuning"),
        help="Dossier de sauvegarde du modèle et des résultats.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre maximal d'époques d'entraînement.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Taille des lots d'entraînement et d'évaluation.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Taux d'apprentissage.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Régularisation L2 appliquée pendant l'entraînement.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Longueur maximale des séquences tokenisées.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Nombre d'évaluations sans amélioration avant arrêt anticipé.",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Nombre de lots accumulés avant mise à jour des poids.",
    )

    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Active la précision mixte FP16 si un GPU CUDA est disponible.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche davantage d'informations.",
    )

    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    """Configure les messages affichés dans le terminal."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    """Fixe les graines aléatoires pour améliorer la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. CHARGEMENT ET CONTRÔLE DU DATASET
# ---------------------------------------------------------------------------

def load_dataset_file(dataset_path: Path) -> pd.DataFrame:
    """
    Charge un fichier CSV.

    Le séparateur est détecté automatiquement afin d'accepter une virgule ou
    un point-virgule.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {dataset_path.resolve()}"
        )

    LOGGER.info("Chargement du dataset : %s", dataset_path.resolve())

    try:
        dataframe = pd.read_csv(
            dataset_path,
            sep=None,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(
            f"Impossible de lire le fichier CSV : {dataset_path}"
        ) from exc

    if dataframe.empty:
        raise ValueError("Le dataset ne contient aucune ligne.")

    LOGGER.info(
        "Dataset chargé : %s lignes et %s colonnes.",
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


def validate_and_clean_dataset(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """
    Vérifie et nettoie le dataset annoté.

    Traitements appliqués :
    - contrôle des colonnes obligatoires ;
    - suppression des lignes sans texte ou sans classe ;
    - conversion des valeurs en chaînes de caractères ;
    - suppression des textes vides ;
    - suppression des doublons exacts texte/classe.
    """
    missing_columns = [
        column
        for column in (text_column, label_column)
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            + ", ".join(missing_columns)
        )

    cleaned = dataframe[[text_column, label_column]].copy()
    initial_count = len(cleaned)

    cleaned = cleaned.dropna(subset=[text_column, label_column])

    cleaned[text_column] = cleaned[text_column].astype(str).str.strip()
    cleaned[label_column] = cleaned[label_column].astype(str).str.strip()

    cleaned = cleaned[
        (cleaned[text_column] != "")
        & (cleaned[label_column] != "")
    ]

    cleaned = cleaned.drop_duplicates(
        subset=[text_column, label_column]
    ).reset_index(drop=True)

    removed_count = initial_count - len(cleaned)

    if removed_count:
        LOGGER.warning(
            "%s ligne(s) supprimée(s) pendant le nettoyage.",
            removed_count,
        )

    if len(cleaned) < 10:
        LOGGER.warning(
            "Le dataset contient moins de 10 exemples. "
            "Les métriques seront peu représentatives."
        )

    if cleaned[label_column].nunique() < 2:
        raise ValueError(
            "La colonne cible doit contenir au moins deux classes."
        )

    class_counts = cleaned[label_column].value_counts()

    LOGGER.info(
        "Répartition des classes :\n%s",
        class_counts.to_string(),
    )

    rare_classes = class_counts[class_counts < 2]
    if not rare_classes.empty:
        LOGGER.warning(
            "Certaines classes contiennent moins de deux exemples : %s",
            rare_classes.to_dict(),
        )

    return cleaned


# ---------------------------------------------------------------------------
# 4. ENCODAGE DES CLASSES
# ---------------------------------------------------------------------------

def encode_labels(
    dataframe: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, LabelEncoder, dict[int, str], dict[str, int]]:
    """
    Transforme les noms de classes en identifiants numériques.

    Exemple :
        risque_minimal -> 0
        haut_risque    -> 1
    """
    label_encoder = LabelEncoder()

    encoded = dataframe.copy()
    encoded["labels"] = label_encoder.fit_transform(
        encoded[label_column]
    )

    id2label = {
        index: str(label)
        for index, label in enumerate(label_encoder.classes_)
    }

    label2id = {
        label: index
        for index, label in id2label.items()
    }

    LOGGER.info("Correspondance des classes : %s", id2label)

    return encoded, label_encoder, id2label, label2id


# ---------------------------------------------------------------------------
# 5. SÉPARATION ENTRAÎNEMENT / VALIDATION
# ---------------------------------------------------------------------------

def split_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare les données en jeux d'entraînement et de validation.

    Une stratification est utilisée lorsque chaque classe contient au moins
    deux exemples.
    """
    class_counts = dataframe["labels"].value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    train_df, validation_df = train_test_split(
        dataframe,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe["labels"] if can_stratify else None,
    )

    LOGGER.info("Jeu d'entraînement : %s lignes.", len(train_df))
    LOGGER.info("Jeu de validation    : %s lignes.", len(validation_df))
    LOGGER.info("Stratification       : %s.", can_stratify)

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
    )


def create_huggingface_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    text_column: str,
) -> DatasetDict:
    """Convertit les DataFrames pandas au format Hugging Face Dataset."""
    train_dataset = Dataset.from_pandas(
        train_df[[text_column, "labels"]],
        preserve_index=False,
    )

    validation_dataset = Dataset.from_pandas(
        validation_df[[text_column, "labels"]],
        preserve_index=False,
    )

    return DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )


# ---------------------------------------------------------------------------
# 6. TOKENIZER ET TOKENISATION
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Charge le tokenizer associé au modèle JuriBERT."""
    LOGGER.info("Chargement du tokenizer : %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    LOGGER.info("Tokenizer chargé.")
    return tokenizer


def tokenize_datasets(
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    text_column: str,
    max_length: int,
) -> DatasetDict:
    """
    Tokenise les textes.

    La troncature limite les textes trop longs. Le padding dynamique sera
    appliqué plus tard par DataCollatorWithPadding.
    """
    if max_length > 512:
        LOGGER.warning(
            "La longueur maximale demandée est supérieure à 512 tokens. "
            "JuriBERT étant basé sur BERT, 512 est généralement la limite."
        )

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    tokenized = datasets.map(
        tokenize_batch,
        batched=True,
        desc="Tokenisation des textes",
    )

    columns_to_remove = [
        column
        for column in tokenized["train"].column_names
        if column not in {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
        }
    ]

    if columns_to_remove:
        tokenized = tokenized.remove_columns(columns_to_remove)

    return tokenized


# ---------------------------------------------------------------------------
# 7. CHARGEMENT DU MODÈLE
# ---------------------------------------------------------------------------

def load_model(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
) -> AutoModelForSequenceClassification:
    """
    Charge JuriBERT avec une tête de classification.

    La tête de classification est initialisée pour le nombre de classes du
    dataset. Elle sera apprise pendant le fine-tuning.
    """
    LOGGER.info("Chargement du modèle : %s", model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    LOGGER.info(
        "Modèle chargé avec %s classes.",
        num_labels,
    )

    return model


# ---------------------------------------------------------------------------
# 8. MÉTRIQUES
# ---------------------------------------------------------------------------

def compute_metrics(
    evaluation_prediction: Any,
) -> dict[str, float]:
    """
    Calcule les métriques utilisées pendant l'évaluation du Trainer.

    Le Macro-F1 donne le même poids à chaque classe.
    Le rappel par classe sera calculé séparément après l'entraînement.
    """
    logits, labels = evaluation_prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ---------------------------------------------------------------------------
# 9. CONFIGURATION DE L'ENTRAÎNEMENT
# ---------------------------------------------------------------------------

def build_training_arguments(
    args: argparse.Namespace,
) -> TrainingArguments:
    """Crée les hyperparamètres utilisés par Hugging Face Trainer."""
    fp16_enabled = bool(args.use_fp16 and torch.cuda.is_available())

    if args.use_fp16 and not torch.cuda.is_available():
        LOGGER.warning(
            "FP16 demandé, mais aucun GPU CUDA n'est disponible. "
            "FP16 est désactivé."
        )

    return TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=fp16_enabled,
        report_to="none",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# 10. CRÉATION DU TRAINER
# ---------------------------------------------------------------------------

def build_trainer(
    *,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenized_datasets: DatasetDict,
    training_arguments: TrainingArguments,
    early_stopping_patience: int,
) -> Trainer:
    """Construit le Trainer Hugging Face."""
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    callbacks = []

    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
            )
        )

    return Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# 11. ÉVALUATION DÉTAILLÉE
# ---------------------------------------------------------------------------

def evaluate_detailed(
    trainer: Trainer,
    validation_dataset: Dataset,
    id2label: dict[int, str],
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    """
    Produit un rapport détaillé après l'entraînement.

    Le rapport inclut notamment la précision, le rappel et le F1-score pour
    chaque classe, ce qui permet de vérifier spécifiquement les classes
    critiques comme haut_risque ou interdit.
    """
    prediction_output = trainer.predict(validation_dataset)

    predictions = np.argmax(
        prediction_output.predictions,
        axis=-1,
    )

    labels = prediction_output.label_ids
    label_ids = sorted(id2label.keys())
    target_names = [id2label[index] for index in label_ids]

    report_dict = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=label_ids,
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=[f"réel_{label}" for label in target_names],
        columns=[f"prédit_{label}" for label in target_names],
    )

    print("\nRAPPORT DE CLASSIFICATION")
    print("=" * 80)
    print(report_text)

    print("MATRICE DE CONFUSION")
    print("=" * 80)
    print(confusion_dataframe)
    print()

    return report_dict, report_text, confusion_dataframe


# ---------------------------------------------------------------------------
# 12. SAUVEGARDE DES ARTEFACTS
# ---------------------------------------------------------------------------

def save_artifacts(
    *,
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    label_encoder: LabelEncoder,
    id2label: dict[int, str],
    label2id: dict[str, int],
    evaluation_metrics: dict[str, Any],
    report_dict: dict[str, Any],
    report_text: str,
    confusion_dataframe: pd.DataFrame,
    output_dir: Path,
    configuration: dict[str, Any],
) -> None:
    """Sauvegarde le modèle, le tokenizer, les métriques et la configuration."""
    final_model_dir = output_dir / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    label_mapping = {
        "classes": label_encoder.classes_.tolist(),
        "id2label": {
            str(key): value
            for key, value in id2label.items()
        },
        "label2id": label2id,
    }

    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            label_mapping,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            evaluation_metrics,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (output_dir / "classification_report.json").write_text(
        json.dumps(
            report_dict,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    confusion_dataframe.to_csv(
        output_dir / "confusion_matrix.csv",
        encoding="utf-8",
    )

    (output_dir / "training_configuration.json").write_text(
        json.dumps(
            configuration,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    LOGGER.info("Modèle final sauvegardé : %s", final_model_dir.resolve())
    LOGGER.info("Résultats sauvegardés : %s", output_dir.resolve())


# ---------------------------------------------------------------------------
# 13. INFÉRENCE
# ---------------------------------------------------------------------------

def predict_text(
    *,
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    id2label: dict[int, str],
    max_length: int,
) -> dict[str, Any]:
    """
    Classe un nouveau texte avec le modèle entraîné.

    Retourne :
    - la classe prédite ;
    - la probabilité associée à chaque classe.
    """
    model.eval()

    device = next(model.parameters()).device

    encoded_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )

    encoded_inputs = {
        key: value.to(device)
        for key, value in encoded_inputs.items()
    }

    with torch.no_grad():
        outputs = model(**encoded_inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0].detach().cpu().numpy()

    predicted_id = int(np.argmax(probabilities))

    return {
        "predicted_label": id2label[predicted_id],
        "probabilities": {
            id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


# ---------------------------------------------------------------------------
# 14. PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    """Exécute le pipeline complet de fine-tuning."""
    args = parse_arguments()
    configure_logging(args.verbose)
    set_random_seeds()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    LOGGER.info("Matériel utilisé : %s", device_name)

    dataframe = load_dataset_file(args.dataset)

    cleaned_dataframe = validate_and_clean_dataset(
        dataframe,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    encoded_dataframe, label_encoder, id2label, label2id = encode_labels(
        cleaned_dataframe,
        label_column=args.label_column,
    )

    train_dataframe, validation_dataframe = split_dataframe(
        encoded_dataframe
    )

    datasets = create_huggingface_datasets(
        train_dataframe,
        validation_dataframe,
        text_column=args.text_column,
    )

    tokenizer = load_tokenizer(args.model_name)

    tokenized_datasets = tokenize_datasets(
        datasets,
        tokenizer=tokenizer,
        text_column=args.text_column,
        max_length=args.max_length,
    )

    model = load_model(
        model_name=args.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_arguments = build_training_arguments(args)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        training_arguments=training_arguments,
        early_stopping_patience=args.early_stopping_patience,
    )

    LOGGER.info("Début du fine-tuning.")
    training_result = trainer.train()
    LOGGER.info("Fine-tuning terminé.")

    trainer.log_metrics(
        "train",
        training_result.metrics,
    )

    trainer.save_metrics(
        "train",
        training_result.metrics,
    )

    evaluation_metrics = trainer.evaluate(
        tokenized_datasets["validation"]
    )

    trainer.log_metrics(
        "validation",
        evaluation_metrics,
    )

    trainer.save_metrics(
        "validation",
        evaluation_metrics,
    )

    report_dict, report_text, confusion_dataframe = evaluate_detailed(
        trainer,
        tokenized_datasets["validation"],
        id2label=id2label,
    )

    training_configuration = {
        "model_name": args.model_name,
        "dataset": str(args.dataset.resolve()),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "number_of_examples": len(encoded_dataframe),
        "number_of_train_examples": len(train_dataframe),
        "number_of_validation_examples": len(validation_dataframe),
        "number_of_labels": len(id2label),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "early_stopping_patience": args.early_stopping_patience,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": bool(args.use_fp16 and torch.cuda.is_available()),
        "device": device_name,
        "random_state": RANDOM_STATE,
    }

    save_artifacts(
        trainer=trainer,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        id2label=id2label,
        label2id=label2id,
        evaluation_metrics=evaluation_metrics,
        report_dict=report_dict,
        report_text=report_text,
        confusion_dataframe=confusion_dataframe,
        output_dir=args.output_dir,
        configuration=training_configuration,
    )

    example_text = (
        "Le système utilise des données biométriques afin d'identifier "
        "automatiquement les personnes dans un espace accessible au public."
    )

    prediction = predict_text(
        text=example_text,
        model=trainer.model,
        tokenizer=tokenizer,
        id2label=id2label,
        max_length=args.max_length,
    )

    print("EXEMPLE D'INFÉRENCE")
    print("=" * 80)
    print("Texte :", example_text)
    print(json.dumps(prediction, indent=4, ensure_ascii=False))
    print()

    LOGGER.info("Pipeline JuriBERT terminé avec succès.")


if __name__ == "__main__":
    main()
