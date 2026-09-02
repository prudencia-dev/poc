# %%
# Gestion des annotations de type pour compatibilité avec les versions antérieures de Python
from __future__ import annotations

# Importations des bibliothèques standard
import argparse # Pour gérer les arguments de la ligne de commande
import json # Pour la sérialisation des métriques en JSON
from pathlib import Path # Pour gérer les chemins de fichiers de manière indépendante du système d'exploitation

# %%
# Importations tierces
import joblib # Pour la sérialisation du modèle entraîné - Sauvegarde du pipeline de machine learning
import pandas as pd # Pour la manipulation des données - Lire CSV et gérer les DataFrames,

# Import seulement un élément pour le prétraitement des données - Sklearn fournit des outils pour diviser les données, transformer les colonnes, gérer les valeurs manquantes, encoder les variables catégorielles, normaliser les variables numériques, entraîner un modèle de forêt aléatoire et évaluer ses performances.
from sklearn.model_selection import train_test_split # Pour diviser le dataset en ensembles d'entraînement et de test
from sklearn.compose import ColumnTransformer # Pour appliquer différentes transformations à différentes colonnes du DataFrame
from sklearn.impute import SimpleImputer # Pour gérer les valeurs manquantes dans les données
from sklearn.preprocessing import OneHotEncoder # Pour encoder les variables catégorielles ou textuelles en format numérique
from sklearn.preprocessing import StandardScaler # Normaliser les variables numériques
from sklearn.ensemble import RandomForestClassifier # Pour entraîner un modèle de forêt aléatoire pour la classification et Prédiction
from sklearn.metrics import classification_report, confusion_matrix, f1_score # Pour évaluer les performances du modèle - Générer un rapport de classification, une matrice de confusion et calculer le score F1
from sklearn.pipeline import Pipeline # Pour créer un pipeline de machine learning qui enchaîne les étapes de prétraitement et d'entraînement du modèle

# %%
# Définition des constantes pour le traitement des données
CIBLE = "gravite" # La variable cible que le modèle doit prédire
VARIABLES_CATEGORIELLES = ["categorie", "nom_source"] # Les colonnes textuelles qui seront transformées en variables numériques avec OneHotEncoder
# Définition des colonnes numériques du dataset qui seront normalisées pour l'entraînement du modèle
VARIABLES_NUMERIQUES = [
    "annee_survenue", "annee_publication", "longueur_titre",
    "nombre_developpeurs", "nombre_deployeurs",
    "nombre_parties_lesees", "nombre_etiquettes",
]

COLONNES_REQUISES = {
    "identifiant", "titre", "date_survenue", "date_publication",
    "nom_source", "categorie", "gravite", "developpeurs", "deployeurs",
    "parties_lesees", "etiquettes",
}

# %%
# Fonctions utilitaires pour le traitement des données et l'entraînement du modèle
def compter_elements(value: object) -> int:

    # Compte les éléments des listes JSON et des textes séparés par une barre verticale.
    if pd.isna(value) or not str(value).strip():
        return 0
    texte = str(value).strip()
    try:
        elements = json.loads(texte)
    except json.JSONDecodeError:
        elements = [element.strip() for element in texte.split("|")]
    if not isinstance(elements, list):
        elements = [elements]
    return len([element for element in elements if str(element).strip()])

# %%
# Fonction pour préparer le dataset avant l'entraînement du modèle
def preparer_dataset(chemin: Path) -> pd.DataFrame:

    # Crée des variables simples sans employer les champs qui révèlent la cible.
    donnees = pd.read_csv(chemin)
    donnees = donnees.drop_duplicates(subset=["identifiant"]).copy()
    colonnes_absentes = sorted(COLONNES_REQUISES - set(donnees.columns))

    # Si des colonnes requises sont manquantes, lève une exception avec un message d'erreur indiquant quelles colonnes sont absentes
    if colonnes_absentes:
        raise ValueError(f"Colonnes absentes : {colonnes_absentes}")
    for source, output in (
        ("date_survenue", "annee_survenue"),
        ("date_publication", "annee_publication"),
    ):
        donnees[output] = pd.to_datetime(donnees[source], errors="coerce").dt.year # Convertit les colonnes de date en années et gère les erreurs de conversion
    donnees["longueur_titre"] = donnees["titre"].fillna("").str.len()

    # Crée de nouvelles colonnes pour compter les éléments des listes : développeurs, déployeurs, parties lésées et étiquettes.
    for source, output in (
        ("developpeurs", "nombre_developpeurs"),
        ("deployeurs", "nombre_deployeurs"),
        ("parties_lesees", "nombre_parties_lesees"),
        ("etiquettes", "nombre_etiquettes"),
    ):
        donnees[output] = donnees[source].map(compter_elements)

    # Sélectionne les colonnes pertinentes et supprime les lignes dont la gravité est absente.
    colonnes = VARIABLES_CATEGORIELLES + VARIABLES_NUMERIQUES + [CIBLE]

    # Prépare le DataFrame final en sélectionnant les colonnes pertinentes et en gérant les valeurs manquantes
    donnees_preparees = donnees[colonnes].dropna(subset=[CIBLE]).copy()
    for colonne in VARIABLES_CATEGORIELLES:
        donnees_preparees[colonne] = donnees_preparees[colonne].fillna("non renseigné").astype(str)
    return donnees_preparees


# %%
# Fonction pour construire le pipeline de machine learning
def construire_pipeline() -> Pipeline:

    # Réunit la préparation des variables et le Random Forest
    preprocessor = ColumnTransformer(
        [
            (
                "categories",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), # Pipeline pour gérer les colonnes catégorielles : imputation des valeurs manquantes avec la valeur la plus fréquente, puis encodage OneHot pour transformer les catégories en variables binaires
                VARIABLES_CATEGORIELLES,
            ),
            (
                "nombres",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), # Pipeline pour gérer les colonnes numériques : imputation des valeurs manquantes avec la médiane, puis normalisation des variables numériques
                VARIABLES_NUMERIQUES,
            ),
        ]
    )

    # Définition du modèle de forêt aléatoire avec des paramètres spécifiques pour l'entraînement
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced", # Utilise un nombre d'arbres de 200, une taille minimale de feuille de 2 et un poids de classe équilibré pour gérer les classes déséquilibrées dans le dataset
        random_state=42, n_jobs=-1, # Utilise un état aléatoire fixe pour la reproductibilité et utilise tous les cœurs disponibles pour l'entraînement du modèle
    )
    return Pipeline([("preparation", preprocessor), ("modele", model)]) # Crée un pipeline final qui combine le préprocesseur et le modèle de forêt aléatoire pour l'entraînement et la prédiction


# %%
# Fonction principale pour exécuter l'entraînement, l'évaluation et la sauvegarde des résultats
def executer(jeu_donnees: Path, repertoire_sortie: Path) -> None:

    # Entraîne, évalue et sauvegarde les résultats essentiels
    donnees = preparer_dataset(jeu_donnees)

    # Sépare les données en ensembles d'entraînement et de test, en stratifiant par la variable cible pour maintenir la distribution des classes
    x, y = donnees.drop(columns=CIBLE), donnees[CIBLE]
    x_entrainement, x_evaluation, y_entrainement, y_evaluation = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y # Utilise 20% des données pour le test, un état aléatoire fixe pour la reproductibilité et stratifie par la variable cible pour maintenir la distribution des classes
    )

    # Construit le pipeline de machine learning en combinant le prétraitement des données et le modèle de forêt aléatoire
    pipeline = construire_pipeline()

    # Entraîne le pipeline sur les données d'entraînement et effectue des prédictions sur les données de test
    pipeline.fit(x_entrainement, y_entrainement)

    # Effectue des prédictions sur l'ensemble de test pour évaluer les performances du modèle
    predictions = pipeline.predict(x_evaluation)
    repertoire_sortie.mkdir(parents=True, exist_ok=True) # Crée le répertoire de sortie s'il n'existe pas déjà

    # Sauvegarde le pipeline entraîné, les métriques d'évaluation et la matrice de confusion dans le répertoire de sortie spécifié
    joblib.dump(pipeline, repertoire_sortie / "foret_aleatoire_incidents_ia.joblib")
    metriques = {
        "nombre_exemples": len(donnees),
        "nombre_exemples_evaluation": len(y_evaluation),
        "f1_macro": float(f1_score(y_evaluation, predictions, average="macro")),
    }
    # Sauvegarde les métriques d'évaluation dans un fichier JSON et le rapport de classification dans un fichier texte
    (repertoire_sortie / "metriques.json").write_text(
        json.dumps(metriques, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Sauvegarde le rapport de classification dans un fichier texte pour une analyse plus détaillée des performances du modèle sur chaque classe
    (repertoire_sortie / "rapport_classification.txt").write_text(
        classification_report(y_evaluation, predictions, zero_division=0), encoding="utf-8"
    )
    classes = sorted(y.unique())

    # Sauvegarde la matrice de confusion dans un fichier CSV pour visualiser les performances du modèle sur chaque classe
    pd.DataFrame(
        confusion_matrix(y_evaluation, predictions, labels=classes),
        index=[f"reel_{classe}" for classe in classes],
        columns=[f"predit_{classe}" for classe in classes],
    ).to_csv(repertoire_sortie / "matrice_confusion.csv")
    print(json.dumps(metriques, indent=2, ensure_ascii=False))


# %%
# Fonction utilitaire pour trouver le dataset par défaut
def jeu_donnees_par_defaut() -> Path:
    # Calcule le chemin depuis ce fichier, et non depuis le dossier du terminal.
    dossier_notebooks = Path(__file__).resolve().parents[1]
    return dossier_notebooks / "datasets/ml/ai_incidents_fr.csv"


def repertoire_sortie_par_defaut() -> Path:
    # Place les résultats dans le dépôt PoC, quel que soit le dossier du terminal.
    racine_projet = Path(__file__).resolve().parents[2]
    return racine_projet / "outputs/incidents_ia"


# %%
# Point d'entrée du script pour l'exécution en ligne de commande
if __name__ == "__main__":
    # Gestion des arguments de la ligne de commande pour spécifier le chemin du dataset et le répertoire de sortie
    parser = argparse.ArgumentParser(description=__doc__)
    # Ajoute un argument pour spécifier le chemin du dataset à utiliser pour l'entraînement et l'évaluation du modèle, avec une valeur par défaut si aucun chemin n'est fourni
    parser.add_argument(
        "--jeu-donnees",
        dest="jeu_donnees",
        type=Path,
        default=jeu_donnees_par_defaut(),
    )
    # Ajoute un argument pour spécifier le répertoire de sortie où les résultats seront sauvegardés, avec une valeur par défaut si aucun répertoire n'est fourni
    parser.add_argument(
        "--repertoire-sortie",
        type=Path,
        default=repertoire_sortie_par_defaut(),
    )
    args = parser.parse_args()

    # Exécute la fonction principale avec les arguments fournis par l'utilisateur ou les valeurs par défaut
    executer(args.jeu_donnees, args.repertoire_sortie)
