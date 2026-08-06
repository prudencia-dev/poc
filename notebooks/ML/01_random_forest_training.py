"""
PRUDENCIA - Entraînement du modèle Machine Learning Random Forest
=================================================================

Ce script constitue une version pédagogique et autonome du pipeline ML utilisé
dans le projet PRUDENCIA.

Objectif :
    Prédire le niveau de risque AI Act d'un projet IA à partir de variables
    décrivant le secteur, le rôle de l'organisation, les données utilisées et
    le type de système d'intelligence artificielle.

Étapes principales :
    1. Charger le dataset CSV
    2. Vérifier et préparer les données
    3. Séparer les variables explicatives de la cible
    4. Encoder les variables catégorielles
    5. Séparer les données en ensembles d'entraînement et de test
    6. Entraîner un RandomForestClassifier
    7. Évaluer les performances du modèle
    8. Afficher l'importance des variables
    9. Sauvegarder le pipeline entraîné
   10. Réaliser une prédiction sur un nouvel exemple

Exécution :
    python 01_random_forest_training.py --dataset chemin/vers/dataset.csv

Exemple :
    python 01_random_forest_training.py \
        --dataset ../datasets/machine_learning/prudencia_ml_dataset.csv

Dépendances :
    pandas
    scikit-learn
    matplotlib
    joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Variable que le modèle doit prédire.
TARGET_COLUMN = "risk_level_aiact"

# Variables explicatives retenues pour la version actuelle du modèle.
FEATURE_COLUMNS = [
    "secteur_grp",
    "role",
    "donnees_perso",
    "donnees_sensibles",
    "type_ia_norm",
]

# Hyperparamètres simples et compréhensibles pour la certification.
DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGGER = logging.getLogger("prudencia.random_forest")


# ---------------------------------------------------------------------------
# 2. OUTILS DE CONFIGURATION
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    """Configure l'affichage des messages dans le terminal."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Lit les arguments transmis lors de l'exécution du script."""
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle Random Forest de PRUDENCIA."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Chemin du fichier CSV utilisé pour l'entraînement.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/random_forest"),
        help="Dossier dans lequel enregistrer le modèle et les résultats.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_MODEL_PARAMS["n_estimators"],
        help="Nombre d'arbres de la forêt.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profondeur maximale des arbres. Par défaut : aucune limite.",
    )

    parser.add_argument(
        "--class-weight",
        choices=["balanced", "balanced_subsample"],
        default=None,
        help="Pondération optionnelle des classes en cas de déséquilibre.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche davantage d'informations pendant l'exécution.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. CHARGEMENT ET CONTRÔLE DU DATASET
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Charge le fichier CSV.

    Le séparateur est détecté automatiquement afin d'accepter les fichiers
    utilisant une virgule ou un point-virgule.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Le dataset est introuvable : {dataset_path.resolve()}"
        )

    LOGGER.info("Chargement du dataset : %s", dataset_path.resolve())

    try:
        dataframe = pd.read_csv(dataset_path, sep=None, engine="python")
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


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes nécessaires sont présentes."""
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes du dataset : "
            + ", ".join(missing_columns)
        )


def prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données nécessaires au modèle.

    Choix pédagogiques :
    - seules les colonnes utiles sont conservées ;
    - les lignes sans cible sont supprimées ;
    - les valeurs manquantes des variables explicatives sont remplacées par
      la chaîne 'inconnu' ;
    - toutes les variables sont converties en texte, car elles représentent
      ici des catégories.
    """
    validate_dataset(dataframe)

    prepared = dataframe[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    initial_row_count = len(prepared)
    prepared = prepared.dropna(subset=[TARGET_COLUMN])

    removed_rows = initial_row_count - len(prepared)
    if removed_rows:
        LOGGER.warning(
            "%s ligne(s) supprimée(s), car la cible était absente.",
            removed_rows,
        )

    for column in FEATURE_COLUMNS:
        prepared[column] = (
            prepared[column]
            .fillna("inconnu")
            .astype(str)
            .str.strip()
            .replace("", "inconnu")
        )

    prepared[TARGET_COLUMN] = (
        prepared[TARGET_COLUMN]
        .astype(str)
        .str.strip()
    )

    if prepared[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "La cible doit contenir au moins deux classes différentes."
        )

    LOGGER.info(
        "Classes détectées : %s",
        sorted(prepared[TARGET_COLUMN].unique().tolist()),
    )

    LOGGER.info(
        "Répartition des classes :\n%s",
        prepared[TARGET_COLUMN].value_counts().to_string(),
    )

    return prepared


# ---------------------------------------------------------------------------
# 4. CONSTRUCTION DU PIPELINE
# ---------------------------------------------------------------------------

def build_pipeline(
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    class_weight: str | None = None,
) -> Pipeline:
    """
    Construit le pipeline complet.

    Le pipeline contient :
    1. Un OneHotEncoder pour convertir les catégories en colonnes numériques.
    2. Un RandomForestClassifier pour apprendre la relation entre les
       caractéristiques d'un projet et son niveau de risque AI Act.

    L'intérêt du Pipeline est de conserver ensemble le prétraitement et le
    modèle. Lors d'une future prédiction, il n'est donc pas nécessaire
    d'encoder manuellement les données.
    """
    categorical_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=DEFAULT_MODEL_PARAMS["min_samples_split"],
        min_samples_leaf=DEFAULT_MODEL_PARAMS["min_samples_leaf"],
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=DEFAULT_MODEL_PARAMS["n_jobs"],
    )

    return Pipeline(
        steps=[
            ("preprocessor", categorical_preprocessor),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------------------------
# 5. SÉPARATION TRAIN / TEST
# ---------------------------------------------------------------------------

def split_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en deux ensembles.

    - Train : données utilisées pour entraîner le modèle.
    - Test  : données jamais vues pendant l'entraînement, utilisées pour
              évaluer sa capacité de généralisation.

    La stratification conserve approximativement la même proportion de classes
    dans les deux ensembles.
    """
    features = dataframe[FEATURE_COLUMNS]
    target = dataframe[TARGET_COLUMN]

    class_counts = target.value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2

    if not can_stratify:
        LOGGER.warning(
            "Stratification désactivée : au moins une classe contient "
            "moins de deux exemples."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target if can_stratify else None,
    )

    LOGGER.info("Données d'entraînement : %s lignes.", len(x_train))
    LOGGER.info("Données de test        : %s lignes.", len(x_test))
    LOGGER.info("Stratification         : %s.", can_stratify)

    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. ENTRAÎNEMENT
# ---------------------------------------------------------------------------

def train_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Entraîne le pipeline sur les données d'entraînement."""
    LOGGER.info("Début de l'entraînement du Random Forest.")
    pipeline.fit(x_train, y_train)
    LOGGER.info("Entraînement terminé.")

    return pipeline


# ---------------------------------------------------------------------------
# 7. ÉVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], str, pd.DataFrame]:
    """
    Calcule les principales métriques de classification.

    Macro-moyenne :
        chaque classe a le même poids, même si certaines classes sont moins
        nombreuses que d'autres.

    Weighted-moyenne :
        les classes les plus représentées ont davantage de poids.
    """
    predictions = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    labels = sorted(
        set(y_test.astype(str).tolist())
        | set(pd.Series(predictions).astype(str).tolist())
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=[f"réel_{label}" for label in labels],
        columns=[f"prédit_{label}" for label in labels],
    )

    LOGGER.info("Accuracy        : %.4f", metrics["accuracy"])
    LOGGER.info("Precision macro : %.4f", metrics["precision_macro"])
    LOGGER.info("Recall macro    : %.4f", metrics["recall_macro"])
    LOGGER.info("F1-score macro  : %.4f", metrics["f1_macro"])
    LOGGER.info("F1 pondéré      : %.4f", metrics["f1_weighted"])

    print("\nRAPPORT DE CLASSIFICATION")
    print("=" * 70)
    print(report)

    print("MATRICE DE CONFUSION")
    print("=" * 70)
    print(confusion_dataframe)
    print()

    return metrics, report, confusion_dataframe


def save_confusion_matrix_plot(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Enregistre une représentation graphique de la matrice de confusion."""
    predictions = pipeline.predict(x_test)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45,
    )

    display.ax_.set_title("Matrice de confusion - Random Forest")
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)

    LOGGER.info("Graphique enregistré : %s", output_path.resolve())


# ---------------------------------------------------------------------------
# 8. IMPORTANCE DES VARIABLES
# ---------------------------------------------------------------------------

def get_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """
    Retourne l'importance calculée pour chaque variable encodée.

    Une importance élevée signifie que la variable a souvent été utile aux
    arbres pour séparer les classes. Elle ne prouve pas une relation de cause
    à effet.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier: RandomForestClassifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_dataframe = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return importance_dataframe


def save_feature_importance_plot(
    importance_dataframe: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Enregistre un graphique des variables les plus importantes."""
    top_features = importance_dataframe.head(top_n).sort_values(
        "importance",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top_features["feature"], top_features["importance"])
    axis.set_title(f"Top {min(top_n, len(top_features))} des variables importantes")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Variable encodée")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    LOGGER.info("Graphique enregistré : %s", output_path.resolve())


# ---------------------------------------------------------------------------
# 9. SAUVEGARDE DES RÉSULTATS
# ---------------------------------------------------------------------------

def save_training_artifacts(
    *,
    pipeline: Pipeline,
    metrics: dict[str, float],
    report: str,
    confusion_dataframe: pd.DataFrame,
    importance_dataframe: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Sauvegarde le modèle, les métriques et les tableaux d'analyse."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "random_forest_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    confusion_path = output_dir / "confusion_matrix.csv"
    importance_path = output_dir / "feature_importances.csv"

    joblib.dump(pipeline, model_path)

    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path.write_text(report, encoding="utf-8")
    confusion_dataframe.to_csv(confusion_path, encoding="utf-8")
    importance_dataframe.to_csv(
        importance_path,
        index=False,
        encoding="utf-8",
    )

    LOGGER.info("Pipeline sauvegardé : %s", model_path.resolve())
    LOGGER.info("Métriques sauvegardées : %s", metrics_path.resolve())
    LOGGER.info("Rapport sauvegardé : %s", report_path.resolve())
    LOGGER.info("Matrice sauvegardée : %s", confusion_path.resolve())
    LOGGER.info("Importances sauvegardées : %s", importance_path.resolve())


# ---------------------------------------------------------------------------
# 10. PRÉDICTION SUR UN NOUVEL EXEMPLE
# ---------------------------------------------------------------------------

def predict_new_project(
    pipeline: Pipeline,
    project: dict[str, str],
) -> dict[str, Any]:
    """
    Prédit le niveau de risque d'un nouveau projet IA.

    Le dictionnaire doit contenir les mêmes variables que celles utilisées
    pendant l'entraînement.
    """
    missing_features = [
        feature for feature in FEATURE_COLUMNS if feature not in project
    ]

    if missing_features:
        raise ValueError(
            "Variables absentes du projet à prédire : "
            + ", ".join(missing_features)
        )

    project_dataframe = pd.DataFrame(
        [{feature: project[feature] for feature in FEATURE_COLUMNS}]
    )

    predicted_class = pipeline.predict(project_dataframe)[0]
    probabilities = pipeline.predict_proba(project_dataframe)[0]

    classifier: RandomForestClassifier = pipeline.named_steps["classifier"]

    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(
            classifier.classes_,
            probabilities,
            strict=True,
        )
    }

    return {
        "predicted_risk_level": str(predicted_class),
        "probabilities": probability_by_class,
    }


# ---------------------------------------------------------------------------
# 11. PROGRAMME PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    """Exécute le pipeline ML complet."""
    args = parse_arguments()
    configure_logging(args.verbose)

    dataframe = load_dataset(args.dataset)
    prepared_dataframe = prepare_dataset(dataframe)

    x_train, x_test, y_train, y_test = split_dataset(prepared_dataframe)

    pipeline = build_pipeline(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=args.class_weight,
    )

    trained_pipeline = train_model(
        pipeline,
        x_train,
        y_train,
    )

    metrics, report, confusion_dataframe = evaluate_model(
        trained_pipeline,
        x_test,
        y_test,
    )

    importance_dataframe = get_feature_importances(trained_pipeline)

    print("VARIABLES LES PLUS IMPORTANTES")
    print("=" * 70)
    print(importance_dataframe.head(15).to_string(index=False))
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_training_artifacts(
        pipeline=trained_pipeline,
        metrics=metrics,
        report=report,
        confusion_dataframe=confusion_dataframe,
        importance_dataframe=importance_dataframe,
        output_dir=args.output_dir,
    )

    save_confusion_matrix_plot(
        trained_pipeline,
        x_test,
        y_test,
        args.output_dir / "confusion_matrix.png",
    )

    save_feature_importance_plot(
        importance_dataframe,
        args.output_dir / "feature_importances.png",
    )

    # Exemple de prédiction : les valeurs devront être adaptées aux catégories
    # réellement présentes dans le dataset PRUDENCIA.
    example_project = {
        "secteur_grp": "sante",
        "role": "fournisseur",
        "donnees_perso": "oui",
        "donnees_sensibles": "oui",
        "type_ia_norm": "classification",
    }

    prediction = predict_new_project(
        trained_pipeline,
        example_project,
    )

    print("EXEMPLE DE PRÉDICTION")
    print("=" * 70)
    print(json.dumps(prediction, indent=4, ensure_ascii=False))
    print()

    LOGGER.info("Pipeline ML terminé avec succès.")


if __name__ == "__main__":
    main()
