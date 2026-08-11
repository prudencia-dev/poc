# Fiche descriptive — Butterfly Labs AI Incident Database

## 1. Identification

| Élément | Valeur |
|---|---|
| Nom | Butterfly Labs AI Incident Database |
| Producteur | Butterfly Labs |
| Page du dataset | https://huggingface.co/datasets/butterflylabs/ai-incidents |
| Site du producteur | https://incidents.butterflylabs.org |
| Téléchargement CSV | https://incidents.butterflylabs.org/api/export/incidents.csv |
| Licence annoncée | Creative Commons Attribution 4.0 — CC BY 4.0 |
| Langue principale | Anglais |
| Format utilisé | CSV UTF-8 |
| Date du snapshot du POC | 9 août 2026 |
| Taille du snapshot | 6 220 observations, 21 colonnes, 9 693 751 octets |
| Empreinte SHA-256 | `b7aab0a130e111d72bd56286469ff38e0557ec98c16cf0a6315ecc52c31ccce6` |

Le fichier n'est pas versionné dans Git car la base est mise à jour régulièrement
et pèse plusieurs mégaoctets. Le script le télécharge et le met en cache au
premier lancement. L'URL, la date et l'empreinte permettent d'identifier
précisément la version utilisée pour les résultats présentés.

## 2. Finalité du dataset

Le dataset rassemble des incidents, défaillances, controverses et publications
relatifs à des systèmes d'intelligence artificielle. Les sujets couvrent
notamment les modèles de langage, les agents, les systèmes autonomes, la
robotique, les deepfakes, la surveillance, les biais, la transparence, la vie
privée et l'accessibilité.

Chaque observation décrit un incident ou un signal de risque avec son titre, un
résumé, sa source, sa catégorie et un niveau de sévérité. La base est alimentée
à partir de plusieurs sources publiques, parmi lesquelles AI Incident Database,
arXiv, NVD/CVE, MITRE ATLAS et des organismes de recherche ou de la société
civile.

## 3. Problème de Machine Learning retenu

Le POC formule une tâche de **classification supervisée multiclasses** :

> Prédire le niveau de sévérité d'un incident lié à l'IA à partir de métadonnées
> disponibles au moment de sa publication.

La cible est `severity`, qui contient quatre classes ordonnées par gravité :

| Classe | Nombre | Part du snapshot |
|---|---:|---:|
| `LOW` | 2 866 | 46,1 % |
| `MEDIUM` | 2 109 | 33,9 % |
| `HIGH` | 1 169 | 18,8 % |
| `CRITICAL` | 76 | 1,2 % |
| **Total** | **6 220** | **100,0 %** |

Cette distribution est déséquilibrée. L'accuracy ne suffit donc pas : le POC
privilégie le macro-F1, qui donne le même poids à chaque classe, et présente
également les scores par classe ainsi que la matrice de confusion.

## 4. Pourquoi ce dataset a été choisi

### Proximité avec le thème du projet

Le sujet reste directement lié à l'analyse du risque des systèmes d'IA, sans
réutiliser le questionnaire ou les règles métier de PRUDENCIA. Le POC ML est
donc indépendant du projet applicatif, tout en restant cohérent avec la
présentation générale consacrée au risque et à la gouvernance de l'IA.

### Problème réellement adapté au Machine Learning tabulaire

Les observations comportent un mélange de variables catégorielles, numériques
et temporelles. Elles permettent de démontrer concrètement :

- l'analyse exploratoire et la qualité des données ;
- les valeurs manquantes et le feature engineering ;
- l'encodage des catégories avec `OneHotEncoder` ;
- la comparaison d'une baseline, d'une régression logistique et d'une forêt
  aléatoire ;
- la validation croisée et la recherche d'hyperparamètres ;
- le déséquilibre des classes et le choix du macro-F1 ;
- l'interprétation par importance de permutation ;
- la sauvegarde d'un pipeline complet et réutilisable.

### Volume exploitable pour une démonstration

Avec 6 220 lignes dans le snapshot, la base est assez grande pour effectuer une
séparation train/test, une validation croisée et une optimisation raisonnable,
tout en restant suffisamment légère pour être exécutée sur un ordinateur
portable pendant une soutenance.

### Traçabilité et licence

Le dataset est publiquement accessible, téléchargeable en CSV et accompagné
d'une licence CC BY 4.0. Son URL et son empreinte sont conservées pour assurer la
traçabilité de l'expérience.

### Potentiel pédagogique des limites

La forte rareté de la classe `CRITICAL`, les données manquantes et la nature
heuristique de la cible constituent des limites réelles. Elles permettent
d'expliquer au jury qu'un score global ne suffit pas et qu'un modèle doit être
évalué au regard de la provenance et des biais de ses données.

## 5. Dictionnaire des 21 colonnes sources

| Colonne | Description synthétique | Utilisation dans le POC |
|---|---|---|
| `id` | Identifiant unique de l'observation | Contrôle des doublons, puis exclu |
| `title` | Titre de l'incident | Longueur uniquement |
| `summary` | Résumé textuel | Longueur uniquement |
| `dateOccurred` | Date supposée de l'événement | Exclue |
| `datePublished` | Date de publication | Année, mois et séparation temporelle |
| `dateIngested` | Date d'intégration dans la base | Exclue |
| `sourceUrl` | URL de la source | Exclue |
| `sourceName` | Nom de la source | Variable catégorielle |
| `category` | Catégorie de risque ou d'incident | Variable catégorielle |
| `severity` | Sévérité `LOW` à `CRITICAL` | **Cible à prédire** |
| `developers` | Développeurs ou organisations associés | Nombre d'entités |
| `deployers` | Déployeurs associés | Nombre d'entités |
| `harmedParties` | Parties potentiellement affectées | Nombre d'entités |
| `location` | Localisation | Exclue : 100 % manquante dans le snapshot |
| `sector` | Secteur | Exclue : 99,1 % manquante |
| `tags` | Étiquettes complémentaires | Nombre d'étiquettes |
| `imageUrl` | Illustration associée | Exclue |
| `categoryConfidence` | Confiance de l'annotation de catégorie | Variable numérique |
| `severityConfidence` | Confiance de l'annotation de sévérité | Exclue : fuite de cible |
| `classifierModel` | Modèle ayant participé à l'annotation | Exclue : information du processus d'annotation |
| `classifierReason` | Justification de la sévérité | Exclue : fuite directe de cible |

## 6. Variables effectivement fournies au modèle

Le pipeline construit onze variables :

- catégorielles : `category`, `source_name`, `publication_year` ;
- numériques : `publication_month`, `title_length`, `summary_length`,
  `developer_count`, `deployer_count`, `harmed_party_count`, `tag_count`,
  `category_confidence`.

Le texte intégral de `title` et `summary` n'est volontairement pas utilisé. Le
POC reste ainsi un projet ML tabulaire et ne fait pas doublon avec le POC Deep
Learning/NLP. Seules leurs longueurs sont utilisées comme métadonnées simples.

Les champs `severityConfidence` et `classifierReason` sont écartés car ils ont
été produits en même temps que la cible. Les conserver donnerait au modèle un
indice direct sur la réponse attendue et créerait une fuite de cible.

## 7. Préparation et protocole d'évaluation

1. téléchargement ou lecture du cache local ;
2. validation du schéma et contrôle des identifiants dupliqués ;
3. suppression des lignes sans cible ou sans date de publication exploitable ;
4. création des variables numériques, catégorielles et temporelles ;
5. tri chronologique ;
6. entraînement sur les 80 % d'observations les plus anciennes ;
7. test final sur les 20 % les plus récentes ;
8. validation croisée effectuée uniquement dans l'ensemble d'entraînement ;
9. comparaison au modèle naïf majoritaire ;
10. évaluation finale unique sur le test temporel.

Dans le snapshot utilisé, 59 lignes, soit 0,9 %, ne possèdent pas de date de
publication. Après préparation, 6 161 observations sont exploitables : 4 928
pour l'entraînement et 1 233 pour le test. La date de coupure du test est le
22 juin 2026.

Le découpage temporel a été préféré à un découpage aléatoire : il simule mieux
l'utilisation réelle du modèle, qui doit généraliser vers des incidents futurs,
et réduit le risque de mélanger des périodes très proches entre train et test.

## 8. Qualité, biais et limites

- La cible `severity` est une **annotation heuristique**. Elle ne représente pas
  une vérité juridique ou scientifique absolue.
- La classe `CRITICAL` ne représente que 1,2 % du snapshot. Ses métriques sont
  donc instables et doivent être commentées séparément.
- La base agrège des sources hétérogènes. Le style éditorial et la sélection des
  sujets peuvent varier selon la source.
- `arXiv` et AI Incident Database représentent une part importante des
  observations ; le dataset n'est pas un échantillon uniforme de tous les
  systèmes d'IA déployés dans le monde.
- Certaines observations décrivent des publications, vulnérabilités ou signaux
  de risque plutôt que des dommages définitivement établis.
- Les champs `developers`, `deployers`, `harmedParties` et `tags` sont manquants
  dans environ 73 % à 82 % des lignes.
- Le dataset évolue : un téléchargement ultérieur peut modifier le nombre de
  lignes et les résultats. C'est la raison de la date et de l'empreinte du
  snapshot.
- Le modèle sert à démontrer une démarche ML. Il ne doit pas déclencher seul une
  décision de conformité, une accusation ou une action affectant une personne.

## 9. Résultat du run de validation du POC

Sur le test temporel du snapshot :

- accuracy : 0,761 ;
- macro-F1 : 0,461 ;
- F1 pondéré : 0,736.

Le score pondéré est nettement supérieur au macro-F1 parce que le modèle réussit
mieux sur les classes majoritaires. La classe `CRITICAL` ne compte que six
exemples dans le test et reste difficile à détecter. Cette différence montre
pourquoi il faut regarder les métriques par classe et non annoncer uniquement
l'accuracy.

## 10. Citation et attribution

Attribution conseillée dans le dossier ou pendant la soutenance :

> Butterfly Labs, *AI Incident Database*, dataset public sous licence CC BY 4.0,
> snapshot téléchargé le 9 août 2026 depuis
> https://incidents.butterflylabs.org/api/export/incidents.csv.

La page descriptive officielle doit également être citée :
https://huggingface.co/datasets/butterflylabs/ai-incidents.

## 11. Proposition d'explication orale courte

> J'ai choisi ce dataset parce qu'il traite directement des risques et incidents
> liés aux systèmes d'IA, tout en étant indépendant des données métier de
> PRUDENCIA. Ses 6 220 observations offrent un vrai problème de classification
> multiclasses avec des variables catégorielles, numériques et temporelles. Il
> permet de démontrer le prétraitement, la comparaison de modèles, la validation
> croisée, le déséquilibre des classes et l'interprétabilité. J'ai écarté les
> champs pouvant révéler directement la sévérité afin d'éviter la fuite de cible,
> et j'ai réservé les incidents les plus récents au test. Enfin, je précise que
> la sévérité est une annotation heuristique et que la classe critique est très
> rare : le modèle est un POC pédagogique, pas un outil de décision autonome.
