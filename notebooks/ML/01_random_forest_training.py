"""POC ML — classification de la severite d'incidents lies a l'IA.

Ce fichier est volontairement organise avec des cellules ``# %%``. Il peut etre
execute integralement en ligne de commande ou bloc par bloc dans VS Code/Spyder.
Le notebook jumeau contient exactement les memes cellules.

Source des donnees : Butterfly Labs AI Incident Database, licence CC BY 4.0.
La cible ``severity`` est une annotation heuristique. Le modele reproduit cette
annotation ; il ne constitue ni un avis juridique ni une mesure absolue du risque.
"""

# %% [markdown]
# # AI Incident Severity Classifier
#
# **Question ML :** peut-on predire la severite d'un incident lie a l'IA
# (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) a partir de metadonnees disponibles au
# moment de sa publication ?
#
# Choix pedagogiques : classification multiclasses, donnees mixtes, baseline,
# comparaison de modeles, validation croisee, recherche d'hyperparametres,
# jeu de test temporel, metriques multiclasses et interpretabilite.

# %% 1. Imports et configuration
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = "severity"
DATASET_URL = "https://incidents.butterflylabs.org/api/export/incidents.csv"
if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
else:
    current_dir = Path.cwd()
    SCRIPT_DIR = next(
        (
            candidate
            for candidate in (current_dir, current_dir / "notebooks/ML", current_dir / "ML")
            if (candidate / "01_random_forest_training.py").exists()
        ),
        current_dir,
    )
NOTEBOOKS_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET_PATH = NOTEBOOKS_DIR / "datasets/ml/ai_incidents_butterfly.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs/ai_incident_classifier"

CATEGORICAL_FEATURES = ["category", "source_name", "publication_year"]
NUMERIC_FEATURES = [
    "publication_month",
    "title_length",
    "summary_length",
    "developer_count",
    "deployer_count",
    "harmed_party_count",
    "tag_count",
    "category_confidence",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LOGGER = logging.getLogger("poc.ai_incident_ml")


# %% 2. Acquisition reproductible des donnees
def download_dataset(dataset_path: Path, *, force: bool = False) -> Path:
    """Telecharge et met en cache le CSV public avec une identification claire."""
    if dataset_path.exists() and not force:
        LOGGER.info("Dataset deja present : %s", dataset_path)
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Telechargement du dataset : %s", DATASET_URL)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "PRUDENCIA-certification-POC/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        dataset_path.write_bytes(response.read())
    return dataset_path


def load_raw_dataset(dataset_path: Path) -> pd.DataFrame:
    """Charge le CSV et controle les colonnes indispensables."""
    dataframe = pd.read_csv(dataset_path)
    required = {
        "id", "title", "summary", "datePublished", "category", "severity",
        "sourceName", "developers", "deployers", "harmedParties", "tags",
        "categoryConfidence",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du dataset : {missing}")
    if dataframe.empty:
        raise ValueError("Le dataset est vide.")
    LOGGER.info("Dataset brut : %d lignes, %d colonnes", *dataframe.shape)
    return dataframe


# %% 3. Exploration et qualite des donnees
def audit_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Retourne les indicateurs utiles a commenter devant le jury."""
    audit = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "duplicate_ids": int(dataframe["id"].duplicated().sum()),
        "missing_target": int(dataframe[TARGET_COLUMN].isna().sum()),
        "target_distribution": dataframe[TARGET_COLUMN].value_counts().to_dict(),
    }
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit


# %% 4. Feature engineering sans fuite de cible
def count_pipe_values(series: pd.Series) -> pd.Series:
    """Compte des entites separees par | ; une valeur absente vaut zero."""
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.map(lambda value: 0 if not value else len(value.split("|")))


def prepare_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Construit des variables tabulaires et exclut les champs de fuite.

    ``severityConfidence``, ``classifierReason`` et le texte integral ne sont
    pas utilises : ils ont contribue directement a produire l'annotation cible.
    Les longueurs des textes sont conservees comme simples metadonnees.
    """
    prepared = dataframe.copy()
    prepared = prepared.drop_duplicates(subset="id")
    prepared = prepared.dropna(subset=[TARGET_COLUMN, "datePublished"])
    prepared[TARGET_COLUMN] = prepared[TARGET_COLUMN].astype(str).str.upper().str.strip()
    prepared = prepared[prepared[TARGET_COLUMN].isin({"LOW", "MEDIUM", "HIGH", "CRITICAL"})]

    published = pd.to_datetime(prepared["datePublished"], errors="coerce", utc=True)
    prepared["published_at"] = published
    prepared["publication_year"] = published.dt.year.astype("Int64").astype(str)
    prepared["publication_month"] = published.dt.month
    prepared["source_name"] = prepared["sourceName"].fillna("UNKNOWN").astype(str)
    prepared["category"] = prepared["category"].fillna("OTHER").astype(str)
    prepared["title_length"] = prepared["title"].fillna("").astype(str).str.len()
    prepared["summary_length"] = prepared["summary"].fillna("").astype(str).str.len()
    prepared["developer_count"] = count_pipe_values(prepared["developers"])
    prepared["deployer_count"] = count_pipe_values(prepared["deployers"])
    prepared["harmed_party_count"] = count_pipe_values(prepared["harmedParties"])
    prepared["tag_count"] = count_pipe_values(prepared["tags"])
    prepared["category_confidence"] = pd.to_numeric(
        prepared["categoryConfidence"], errors="coerce"
    )
    prepared = prepared.dropna(subset=["published_at"])
    return prepared.sort_values("published_at").reset_index(drop=True)


# %% 5. Decoupage temporel train/test
def temporal_train_test_split(
    dataframe: pd.DataFrame, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserve les incidents les plus recents pour une evaluation realiste."""
    cut = int(len(dataframe) * (1 - test_fraction))
    train, test = dataframe.iloc[:cut], dataframe.iloc[cut:]
    x_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]
    missing_test_classes = set(y_train.unique()) - set(y_test.unique())
    if missing_test_classes:
        LOGGER.warning("Classes absentes du test temporel : %s", missing_test_classes)
    LOGGER.info("Train=%d, test=%d, date de coupure=%s", len(train), len(test), test["published_at"].min())
    return x_train, x_test, y_train, y_test


# %% 6. Pretraitement commun aux modeles
def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    """Baseline et deux familles de modeles aux biais differents."""
    return {
        "dummy": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=1,
            )),
        ]),
    }


# %% 7. Comparaison par validation croisee
def compare_models(
    candidates: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, x_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"}, n_jobs=1,
        )
        rows.append({
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
        })
    results = pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False)
    print(results.to_string(index=False))
    return results


# %% 8. Optimisation mesuree du Random Forest
def tune_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series, *, quick: bool = False
) -> RandomizedSearchCV:
    pipeline = build_candidates()["random_forest"]
    distributions = {
        "classifier__n_estimators": [150, 250, 400],
        "classifier__max_depth": [None, 12, 24],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }
    search = RandomizedSearchCV(
        pipeline,
        distributions,
        n_iter=3 if quick else 15,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    LOGGER.info("Meilleur macro-F1 CV=%.3f ; %s", search.best_score_, search.best_params_)
    return search


# %% 9. Evaluation finale sur le test jamais utilise
def evaluate_model(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], str, pd.DataFrame]:
    predictions = model.predict(x_test)
    labels = [label for label in ["LOW", "MEDIUM", "HIGH", "CRITICAL"] if label in set(y_test) | set(predictions)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=[f"reel_{label}" for label in labels],
        columns=[f"predit_{label}" for label in labels],
    )
    print(json.dumps(metrics, indent=2))
    print(report)
    return metrics, report, matrix


# %% 10. Interpretabilite globale sans pretendre a la causalite
def compute_permutation_importance(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    result = permutation_importance(
        model, x_test, y_test, scoring="f1_macro", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


# %% 11. Sauvegarde des artefacts et graphiques
def save_artifacts(
    *, model: Pipeline, metrics: dict[str, float], report: str,
    matrix: pd.DataFrame, importance: pd.DataFrame,
    comparison: pd.DataFrame, output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "ai_incident_severity_pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    importance.to_csv(output_dir / "permutation_importance.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de confusion — test temporel")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    plot_data = importance.sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(plot_data["feature"], plot_data["importance_mean"], xerr=plot_data["importance_std"])
    plt.xlabel("Baisse du macro-F1 apres permutation")
    plt.title("Importance par permutation (non causale)")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=150)
    plt.close()


# %% 12. Orchestration complete
def run_pipeline(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *, quick: bool = False, force_download: bool = False,
) -> dict[str, float]:
    download_dataset(dataset_path, force=force_download)
    raw = load_raw_dataset(dataset_path)
    audit_dataset(raw)
    prepared = prepare_dataset(raw)
    x_train, x_test, y_train, y_test = temporal_train_test_split(prepared)
    candidates = build_candidates()
    comparison = compare_models(candidates, x_train, y_train)
    search = tune_random_forest(x_train, y_train, quick=quick)
    model = search.best_estimator_
    metrics, report, matrix = evaluate_model(model, x_test, y_test)
    importance = compute_permutation_importance(model, x_test, y_test)
    print(importance.to_string(index=False))
    save_artifacts(
        model=model, metrics=metrics, report=report, matrix=matrix,
        importance=importance, comparison=comparison, output_dir=output_dir,
    )
    reloaded = joblib.load(output_dir / "ai_incident_severity_pipeline.joblib")
    assert len(reloaded.predict(x_test.head(1))) == 1
    return metrics


# %% 13. Execution en ligne de commande
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true", help="Recherche reduite pour une demonstration rapide.")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cli_args = parse_arguments()
    run_pipeline(
        cli_args.dataset, cli_args.output_dir,
        quick=cli_args.quick, force_download=cli_args.force_download,
    )
