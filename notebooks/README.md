# POC Machine Learning et Deep Learning

Deux pipelines autonomes, sans interface graphique ni dependance a l'application
PRUDENCIA, destines a expliquer une demarche complete devant le jury de la
certification **Developpeur en Intelligence Artificielle**.

Chaque POC existe sous deux formes synchronisees :

- un script Python organise en cellules `# %%`, executable en entier ou bloc par bloc ;
- un notebook Jupyter contenant les memes cellules et davantage de Markdown.

Le script Python est la source de verite. Apres une modification, regenerer les
notebooks avec :

```bash
python notebooks/sync_notebooks.py
```

## Installation

Python 3.11 ou 3.12 est recommande.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows : .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r notebooks/requirements.txt
```

## POC 1 — Machine Learning

**Probleme :** predire la severite (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) d'un
incident lie a l'intelligence artificielle a partir de metadonnees tabulaires.

Le pipeline illustre : acquisition et audit des donnees, feature engineering,
prevention de la fuite de cible, test temporel, baseline, regression logistique,
Random Forest, validation croisee, optimisation, macro-F1, matrice de confusion,
importance par permutation, sauvegarde et rechargement.

Source : [Butterfly Labs AI Incident Database](https://huggingface.co/datasets/butterflylabs/ai-incidents),
licence CC BY 4.0. Le CSV est telecharge et mis en cache au premier lancement.
La severite est une annotation heuristique : le POC reproduit cette annotation
et ne fournit ni avis juridique ni mesure absolue du risque.

La provenance, le dictionnaire des colonnes, la justification du choix, les
biais et une proposition d'explication orale sont détaillés dans
[ML/DATASET_CARD.md](ML/DATASET_CARD.md).

```bash
python notebooks/ML/01_random_forest_training.py --quick
```

Retirer `--quick` pour effectuer la recherche d'hyperparametres complete.

## POC 2 — Deep Learning

**Probleme :** fine-tuner JuriBERT afin de classer la description textuelle d'un
cas d'usage IA parmi `interdit`, `haut_risque`, `limite`, `minimal` et
`hors_champ`.

Le pipeline illustre : transfert d'apprentissage, tokenisation, encodage des
classes, trois jeux stratifies, mini-batches, epochs, learning rate, early
stopping sur validation, evaluation finale sur test, sauvegarde et inference.

```bash
python notebooks/DL/02_juribert_finetuning.py --smoke-test
```

Retirer `--smoke-test` pour utiliser tout le corpus et trois epochs. Un GPU est
recommande ; le smoke test peut fonctionner sur CPU mais reste plus lent que le
POC ML.

Le script accepte un futur dataset sans modification du code :

```bash
python notebooks/DL/02_juribert_finetuning.py \
  --dataset chemin/corpus.csv \
  --text-column text \
  --label-column label
```

Les resultats ne constituent pas une decision juridique et doivent etre relus
par un expert humain.

## Arborescence

```text
notebooks/
├── ML/
│   ├── 01_random_forest_training.py
│   └── 01_RandomForest_Training.ipynb
├── DL/
│   ├── 02_juribert_finetuning.py
│   └── 02_JuriBERT_FineTuning.ipynb
├── datasets/
├── requirements.txt
└── sync_notebooks.py
```

Auteur : Jean-Philippe — Projet de certification PRUDENCIA.
