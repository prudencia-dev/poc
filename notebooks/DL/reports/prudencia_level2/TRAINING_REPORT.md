# Rapport d'entraînement — POC JuriBERT niveau 2

## Résumé

Le modèle `dascim/juribert-base` a été fine-tuné pour classer des descriptions
de cas d'usage IA dans cinq niveaux de risque. L'entraînement reprend la configuration
conseillée dans PRUDENCIA niveau 2, avec un jeu de test indépendant ajouté pour éviter
d'évaluer le modèle sur les données utilisées par l'early stopping.

- Date UTC : 2026-08-11T16:42:18.267076+00:00
- Matériel : Apple Silicon — MPS
- Durée : 464.2 secondes
- Époques demandées : 10
- Époques réalisées : 9.00
- Meilleur checkpoint : `checkpoint-90`
- Meilleur macro-F1 de validation : 0.818

## Dataset

- Corpus : `dl_juribert_training_cases_v2.csv`
- Texte : `Q3`
- Cible : `risk_level_aiact`
- Total nettoyé : 163
- Train : 114
- Validation : 24
- Test : 25

- `haut_risque` : 68 exemples
- `interdit` : 29 exemples
- `limite` : 29 exemples
- `minimal` : 27 exemples
- `hors_champ` : 10 exemples

Le découpage est stratifié et reproductible avec la graine 42. La validation sert à
sélectionner le meilleur checkpoint ; le test reste isolé jusqu'à l'évaluation finale.

## Hyperparamètres et justification

| Hyperparamètre | Valeur | Justification |
|---|---:|---|
| Modèle | `dascim/juribert-base` | Transformer préentraîné sur du français juridique |
| Époques max. | 10 | L'early stopping évite d'exécuter les époques inutiles |
| Batch réel | 4 | Compatible avec la mémoire disponible |
| Accumulation | 2 | Simule un batch effectif plus grand |
| Batch effectif | 8 | Stabilise les gradients |
| Learning rate | 2e-05 | Valeur classique pour le fine-tuning de BERT |
| Weight decay | 0.01 | Régularisation L2 |
| Warmup ratio | 0.1 | Montée progressive du learning rate au démarrage |
| Longueur max. | 256 tokens | Compromis information, mémoire et durée |
| Patience | 3 | Arrêt après trois validations sans progrès |
| Métrique de sélection | macro-F1 | Donne le même poids aux classes rares et majoritaires |
| Poids de classes | True | Pénalise davantage les erreurs sur les classes rares |

Ces valeurs correspondent à la configuration niveau 2 de PRUDENCIA. Il ne s'agit pas
d'un GridSearch exhaustif : l'optimisation repose sur le checkpoint au meilleur
macro-F1, l'early stopping, le warmup et la pondération des classes.

## Résultats

### Validation au meilleur checkpoint

- Accuracy : 0.833
- Macro-F1 : 0.818
- Loss : 0.667

### Test indépendant

- Accuracy : 0.880
- Précision macro : 0.930
- Rappel macro : 0.890
- Macro-F1 : 0.902
- F1 pondéré : 0.880

| Classe | Précision | Rappel | F1 | Support test |
|---|---:|---:|---:|---:|
| `haut_risque` | 0.818 | 0.900 | 0.857 | 10 |
| `hors_champ` | 1.000 | 1.000 | 1.000 | 1 |
| `interdit` | 0.833 | 1.000 | 0.909 | 5 |
| `limite` | 1.000 | 0.800 | 0.889 | 5 |
| `minimal` | 1.000 | 0.750 | 0.857 | 4 |

## Interprétation et limites

- Le corpus ne contient que 163 exemples : les métriques
  par classe ont une forte variance, particulièrement pour `hors_champ`.
- La pondération réduit le biais vers `haut_risque`, mais ne remplace pas l'ajout de
  données annotées et relues par un expert.
- Le test mesure la reproductibilité du POC, pas une validité juridique générale.
- Les prédictions doivent rester une aide à l'analyse avec validation humaine.

## Artefacts

- `learning_curves.png` : loss et macro-F1 par époque ;
- `confusion_matrix.png` et `.csv` : erreurs entre classes ;
- `training_history.csv` : historique brut du Trainer ;
- `validation_metrics.json` et `test_metrics.json` : résultats structurés ;
- `training_configuration.json` : paramètres reproductibles ;
- `final_model/` : modèle et tokenizer sauvegardés.
