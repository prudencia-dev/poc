# PRUDENCIA – Notebooks Machine Learning & Deep Learning

## Présentation

Ce dossier regroupe les notebooks et les scripts Python utilisés pour illustrer les étapes d'entraînement des modèles de Machine Learning et de Deep Learning développés dans le cadre du projet **PRUDENCIA**.

Contrairement au code de l'application (FastAPI, Streamlit, PostgreSQL, Docker...), ces notebooks ont un objectif exclusivement pédagogique.

Ils permettent de démontrer, étape par étape, la méthodologie employée pour :

- préparer les données ;
- entraîner les modèles ;
- évaluer leurs performances ;
- interpréter les résultats ;
- sauvegarder les modèles entraînés.

Ces notebooks constituent un support de démonstration pour la certification **Développeur en Intelligence Artificielle**.

---

# Objectifs pédagogiques

Les notebooks illustrent les principales étapes d'un pipeline de Data Science :

1. Chargement des données
2. Exploration du dataset
3. Prétraitement
4. Visualisation
5. Entraînement
6. Évaluation
7. Interprétation
8. Sauvegarde du modèle

Chaque notebook contient :

- des cellules Markdown expliquant les concepts ;
- du code Python abondamment commenté ;
- des visualisations réalisées avec Matplotlib et Seaborn ;
- les métriques principales utilisées en Machine Learning.

L'objectif est de rendre chaque étape compréhensible et reproductible.

---

# Arborescence

```
notebooks/

├── README.md

├── requirements-notebooks.txt

├── datasets/
│   ├── ml/
│   └── dl/

├── ML/
│   ├── 01_RandomForest_Training.ipynb
│   └── 01_random_forest_training.py

└── DL/
    ├── 02_JuriBERT_FineTuning.ipynb
    └── 02_juribert_finetuning.py
```

---

# Notebook Machine Learning

Le notebook Machine Learning présente un pipeline complet utilisant un modèle **Random Forest**.

Les principales étapes sont :

- chargement du dataset ;
- exploration des données ;
- préparation des variables ;
- séparation Train/Test ;
- entraînement du modèle ;
- calcul des métriques ;
- matrice de confusion ;
- importance des variables ;
- sauvegarde du modèle.

Les bibliothèques utilisées sont notamment :

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

# Notebook Deep Learning

Le notebook Deep Learning présente un exemple complet de Fine-Tuning du modèle **JuriBERT**.

Les principales étapes sont :

- chargement du corpus annoté ;
- tokenisation ;
- création du dataset ;
- configuration du modèle ;
- entraînement ;
- évaluation ;
- sauvegarde du modèle.

Les bibliothèques utilisées sont notamment :

- Pandas
- PyTorch
- Hugging Face Transformers
- Datasets
- Accelerate
- Matplotlib
- Seaborn

---

# Jeux de données

Les datasets utilisés pour les démonstrations sont volontairement indépendants de l'application PRUDENCIA.

Deux catégories sont distinguées :

## Machine Learning

Les données tabulaires correspondent aux réponses du questionnaire de conformité.

## Deep Learning

Les données textuelles correspondent à des cas juridiques annotés destinés au Fine-Tuning de JuriBERT.

---

# Objectif de la certification

Ces notebooks ont été réalisés afin de démontrer la maîtrise :

- du Machine Learning ;
- du Deep Learning ;
- des étapes d'un pipeline de Data Science ;
- de l'utilisation de Jupyter Notebook.

Ils sont indépendants de l'architecture logicielle de PRUDENCIA et servent exclusivement de support pédagogique et de démonstration technique.

---

# Auteur

Projet : **PRUDENCIA**

Certification : **Développeur en Intelligence Artificielle**

Auteur : **Jean-Philippe**

Version : **1.0**