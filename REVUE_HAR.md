# Revue de `HAR.ipynb` : qualités, défauts et ordre de résolution

Revue du 21/10/2026, faite sur le commit `6b24db6`. Le notebook a été relancé de bout en bout (Python 3.14, scikit-learn 1.9.1) : il tourne sans erreur en environ 1 min 40.

Les références de cellules donnent la **section** et un **extrait de texte** à chercher avec Ctrl+F dans Jupyter.

---

## Qualités (à garder et à mettre en avant à l'oral)

- **Validation croisée par sujet** (`GroupKFold` sur `subject`), bien justifiée en §3.1. C'est le point de méthode le plus important sur ce dataset, et beaucoup de notebooks Kaggle le ratent.
- **Pas de fuite de données** : scaler, PCA et SelectKBest sont tous dans des `Pipeline`, donc ajustés dans chaque pli. Le modèle est choisi sur la validation croisée, et le jeu de test ne sert qu'une fois, à la fin (§6).
- **Question de recherche claire** (quel effet a la réduction de dimension ?). Elle répond à l'exigence d'originalité de la consigne.
- **Deux métriques justifiées** : accuracy et F1 macro (§2.1).
- **État de l'art sérieux** : Anguita et al. (2013), et Gao et al. (2026), qui est récent et directement comparable (même découpage par sujet, validation croisée groupée). La référence Gao existe bien, et le chiffre 95,55 % est cohérent avec son résumé.
- **Conforme à la consigne sur la forme** : 0,64 Mo (< 10 Mo), lien vers les données, données non incluses, 561 variables et 10 299 lignes.
- **Reproductible** : la relance complète passe, et `random_state=42` est fixé sur la RF.
- **Graphiques utiles** : courbe de variance cumulée de la PCA avec le seuil de 95 %, et matrice de confusion.
- **Limites reconnues honnêtement** (dimensions différentes entre PCA et SelectKBest, limites de l'étude).

---

## Ordre de résolution

L'ordre est choisi pour **éviter de refaire le travail** : d'abord les corrections qui ne touchent pas aux résultats, puis le code, puis les nouvelles expériences, et **en tout dernier** la relance finale et la mise à jour des chiffres. Chaque modification du code fait bouger les résultats.

> **Si le temps manque**, le minimum pour rendre un notebook cohérent, c'est les étapes **1, 2 et 12**.

### Étape 1 : corrections de forme immédiates (~10 min, aucun impact sur les résultats)

- [x] **Restes de copier-coller :**
  - §4 (intro « Feature Selection ») : `data leakage.akage.` → `data leakage.`
  - §7 (Conclusion, dernier paragraphe) : `predictive performance.n performance.` → `predictive performance.`
- [x] **DOI cassé** (section References) : `https://doi.org/10.3390/app162010046p162010046` → `https://doi.org/10.3390/app162010046`
- [x] **Fautes :**

  | Section | Faute | Correction |
  |---|---|---|
  | §2 (intro) | `insure class distribution` | `ensure` |
  | §2.2 | `in an reasonable range` | `a reasonable` |
  | §2.4 | `expermients` | `experiments` |
  | §2.5 | `obvisously` | `obviously` |
  | §3.1 | `may benefits form` | `may benefit from` |

- [ ] **Membres du groupe** (première cellule) : vérifier que tous les noms y sont. La consigne l'exige.
- [x] **§2, texte après `train_df.info()`** : la phrase « We also check for constant features. There are none. » arrive avant le code qui fait la vérification. La déplacer après la cellule « Identify constant features ».
- [x] **§2.1** : « The classes are balanced » contredit « the imbalance is really moderate ». Écrire plutôt « The classes are approximately balanced (13.4 % to 19.1 %) ».
- [x] **Supprimer la cellule de code vide** à la fin du notebook.

### Étape 2 : remettre le notebook dans un état cohérent (~5 min)

- [x] **Kernel → Restart Kernel and Run All Cells.** La cellule §3.1 (`lr_pipeline`) a le numéro d'exécution `[29]`, alors que l'évaluation sur le test (§6) a `[25]`. Le score de test vient donc peut-être d'une **ancienne définition** de `lr_pipeline`.
- [x] **Sauvegarder et commiter** ce point de départ propre.

### Étape 3 : factoriser le code (critère « Code readability », ~30 min)

À faire **avant** d'ajouter des expériences : elles en profiteront directement.

- [ ] **Remplacer les 9 blocs `cross_validate` + `print` copiés-collés** par une fonction qui stocke les résultats dans un dictionnaire :

  ```python
  results = {}

  def evaluate(name, steps, n_jobs=-1):
      scores = cross_validate(
          Pipeline(steps), X_train, y_train,
          groups=groups, cv=cv,
          scoring=["accuracy", "f1_macro"],
          n_jobs=n_jobs,
      )
      results[name] = scores
      acc = scores["test_accuracy"]
      print(f"{name}: accuracy {acc.mean():.2%} ± {acc.std():.2%}, "
            f"F1-macro {scores['test_f1_macro'].mean():.2%}, "
            f"fit {scores['fit_time'].mean():.2f}s")
      return scores
  ```

  Pour la RF, appeler avec `n_jobs=1` (la RF est déjà parallélisée en interne, comme dans le code actuel).
- [ ] **Construire le tableau récapitulatif de §5 à partir de `results`.** Cela supprime le double tableau (§3.5 et §5) et le `"≈102"` codé en dur. Pour la dimension réelle de la PCA, passer `return_estimator=True` à `cross_validate` et lire `est["pca"].n_components_` dans chaque pli : la PCA y garde 101 ou 102 composantes selon le pli, pas 103 (ce chiffre vient de la PCA d'exploration de §2.4, ajustée sur tout le train ; avant la suppression des 21 colonnes dupliquées, c'était 100 ou 101 contre 102).
- [x] **Afficher l'écart-type dans tous les tableaux** (il manque dans le tableau LR + SelectKBest de §4.1). Fait avec `ddof=1` (division par k − 1), convention annoncée en §3.1.
- [ ] **Regrouper les imports sklearn** dans la première cellule de code.
- [x] **Code mort** : `X_train_pca` et `X_test_pca` (§2.5) ne sont utilisés par aucun modèle. Soit les supprimer, soit garder seulement l'affichage des dimensions en précisant que les pipelines refont la PCA dans chaque pli. Transformer le jeu de test en §2 sème le doute sur une fuite de données. *Fait : §2.5 supprimée (code et texte), `X_test_scaled` retiré de §2.2, et un paragraphe à la fin de §2.4 explique que cette PCA sert seulement à l'exploration et que les pipelines refont scaler + PCA dans chaque pli. Le jeu de test n'apparaît plus qu'en §6.*

### Étape 4 : compléter le prétraitement (critère « Preprocessing », ~30 min)

- [x] **21 colonnes strictement identiques** à d'autres (par exemple `tGravityAccMag-*` = `tBodyAccMag-*`). `train_df.duplicated()` ne teste que les lignes, il faut aussi `X_train.T.duplicated()`. **À décider en groupe :**
  - soit les supprimer (recommandé), mais alors tous les « 561 » du texte deviennent **540** ;
  - soit les signaler et justifier pourquoi on les garde.

  *Fait (option « supprimer ») : en §2, après le test des variables constantes, une cellule liste les 21 copies et la variable qu'elles recopient, puis une autre les retire de `X_train` et de `X_test` (colonnes repérées sur le train seulement). Deux familles : les 13 `tGravityAccMag-*` = `tBodyAccMag-*`, et `-sma()` = `-mean()` pour les 8 autres signaux de magnitude (une magnitude est positive, donc la moyenne de sa valeur absolue est sa moyenne). Le texte justifie la suppression (poids doublé dans la PCA, les distances du SVM et le tirage des variables de la RF ; deux places sur k prises dans SelectKBest) et explique pourquoi la faire avant la validation croisée ne crée pas de fuite. Tous les « 561 » qui désignent l'entrée des modèles sont passés à 540, et 64,3 % à 63,0 %. Les « 561 » qui décrivent le dataset (intro, `info()`, Gao et al.) restent. Chiffres de §2 mis à jour : 1 746 paires corrélées sur 145 530 (1,20 %), 103 composantes sur 540 (80,9 %). Le code de §5 lit la dimension avec `X_train.shape[1]`.*
  - *À dire à l'oral : sur les 561 variables, SelectKBest aurait mis 6 copies dans son top 50, 11 dans le top 100 et 16 dans le top 200.*
- [ ] **84 noms de colonnes `bandsEnergy`** sont dupliqués dans la source, et pandas leur ajoute `.1`/`.2`. Le mentionner.
- [ ] **§2.2** : préciser que les auteurs du dataset ont déjà normalisé les variables dans **[-1, 1]**. La standardisation reste utile (moyennes et variances différentes), mais la phrase actuelle est vague.
- [ ] **Exploration par sujet** : 21 sujets en entraînement, 9 en test, sans recouvrement. Le découpage officiel est déjà fait par sujet, ce qui renforce l'argument de la validation croisée groupée.
- [ ] **Visualisation 2D** : nuage de points sur les 2 premières composantes PCA (ou t-SNE), coloré par activité. On y voit la séparation entre activités statiques et dynamiques, et le chevauchement SITTING/STANDING, ce qui prépare l'analyse d'erreurs (étape 8).

### Étape 5 : comparer PCA et SelectKBest à dimension égale (critère « Evaluation », ~30 min, **gros impact**)

Aujourd'hui, la comparaison oppose 102 composantes à 50/100/200 variables : la conclusion « la sélection de variables préserve mieux la performance » est biaisée. Premiers résultats, obtenus avec un script à part (à refaire dans le notebook) :

| Modèle | PCA(100) | KBest(100) | PCA(200) | KBest(200) |
|---|---|---|---|---|
| LR | **91,86 %** | 91,02 % | **93,17 %** | 92,40 % |
| RF | 85,36 % | **88,48 %** | 85,90 % | **91,60 %** |

- **Pour la LR, la PCA bat la sélection de variables** à dimension égale.
- **Pour la RF, c'est l'inverse.** La meilleure méthode dépend du modèle, et c'est le résultat le plus intéressant du projet.

- [ ] **Évaluer les deux méthodes sur la même grille** (par exemple k = 25, 50, 100, 150, 200, 300), pour les 3 modèles.
- [ ] **Tracer l'accuracy en fonction du nombre de dimensions** : une courbe par couple méthode/modèle.
- [ ] **Réécrire §3.6, §4, §5.1 et la conclusion** en conséquence.

### Étape 6 : régler les hyperparamètres (critère « Model », ~30 min)

Aucun hyperparamètre n'est réglé. Premier test : le gain est faible, donc le classement tient. Le jury attendra quand même la démarche, et « le réglage apporte peu » est en soi un résultat à présenter.

| Configuration | Accuracy en validation croisée |
|---|---|
| LR, C = 1 (actuel) | 93,45 % |
| LR, C = 0,1 | 93,48 % |
| SVC rbf, C = 1 (actuel) | 92,33 % |
| SVC rbf, C = 10 | 92,43 % |
| SVM linéaire, C = 0,1 | 93,39 % |

- [ ] **`GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro")`** puis `.fit(X_train, y_train, groups=groups)`, avec les grilles suivantes :
  - LR : `C` ;
  - SVM : `C` et `gamma` ;
  - RF : `max_features` et `n_estimators`.
- [ ] **Traiter aussi `n_components` (PCA) et `k` (SelectKBest)** comme des hyperparamètres, plutôt que les seuils arbitraires 95 % et 50/100/200.
- [ ] **Signaler que `best_score_` est légèrement optimiste**, puisqu'il est mesuré sur la même validation croisée que celle qui a servi à choisir. La validation croisée imbriquée est la solution rigoureuse ; au minimum, le mentionner.

### Étape 7 : présenter les résultats honnêtement (critère « Evaluation », ~15 min)

- [ ] **L'écart-type entre plis est d'environ 4 pt** : la LR va de 0,88 à 0,98 selon le pli. L'écart LR/SVM (1,1 pt) est donc plus petit que la variabilité. Nuancer « Logistic Regression achieved the highest… » : les modèles sont proches.
- [ ] **Comparer les modèles pli par pli** : ils voient les mêmes sujets à chaque pli, donc regarder l'écart dans chaque pli est plus solide que de comparer des moyennes.
- [ ] **Graphique de §5** : ajouter des barres d'erreur (écart-type), et remplacer `plt.ylim(0, 1)` par une échelle qui montre les différences, par exemple `(0.8, 1)`, en le signalant. *Barres d'erreur faites ; reste l'échelle.*

### Étape 8 : analyser les erreurs (critère « Evaluation », ~20 min)

- [ ] **Ajouter un texte après la matrice de confusion (§6).** Aujourd'hui, il n'y a aucun commentaire.
  - SITTING (rappel 0,88) et STANDING (précision 0,89) concentrent la majorité des erreurs.
  - Explication physique : la posture est quasi statique dans les deux cas, et seule l'orientation de la gravité diffère légèrement.
  - Anguita et al. relèvent la même difficulté : le relier à l'état de l'art.
- [ ] **Expliquer pourquoi le test (95,5 %) dépasse la validation croisée (93,5 %)** :
  - le modèle final est entraîné sur 21 sujets, contre environ 17 par pli ;
  - les 9 sujets du test sont peut-être plus faciles ;
  - la forte variabilité entre plis montre que la difficulté dépend du sujet.

### Étape 9 : mesurer le coût de calcul (~10 min)

- [ ] **L'efficacité est l'argument principal de la réduction de dimension, mais elle n'est jamais mesurée.** `cross_validate` renvoie déjà `fit_time` et `score_time` : les ajouter aux tableaux (la fonction `evaluate` de l'étape 3 les affiche). Par exemple, la RF + PCA(200) met environ 9,6 s par pli, contre 0,8 s pour la LR.

### Étape 10 (optionnelle) : une sélection de variables qui tient compte de la redondance

- [ ] **SelectKBest avec ANOVA est univarié et ignore la redondance**, alors que c'est la forte corrélation (2 281 paires au-dessus de 0,95, §2.3) qui motive l'étude. Le top 200 contient probablement beaucoup de variables redondantes.
  - Pistes : filtre de corrélation avant SelectKBest, régression logistique L1, `RFE`, `mutual_info_classif`.
- [ ] **Montrer quelles variables sont retenues** (top 10 des scores F, importances de la RF). C'est un apport d'interprétabilité, et ça s'explique bien à l'oral.

### Étape 11 : structure et rédaction du rapport (~20 min)

- [ ] **Déplacer « State of the Art and Related Work » dans l'introduction** (après la question de recherche). Il est actuellement après la conclusion, alors que la consigne demande de résumer l'état de l'art et de justifier le choix de l'approche.
- [ ] **Numéroter les titres orphelins** : « Example with tBodyAcc-std()-X… », « Random Forest Results », « Performance–Dimensionality Trade-off ».
- [ ] **Placer la sélection de variables (§4) dans le plan de la consigne** : Training ou Evaluation.
- [ ] **Alléger les répétitions** :
  - « We use the same five-fold subject-aware cross-validation… » revient environ 8 fois : le dire une fois en §3.1 ;
  - « These results suggest that… » revient de nombreuses fois.
- [x] **Conclusion** : bien distinguer ce qui a marché, ce qui n'a pas marché et ce que vous feriez avec plus de temps (structure demandée par la consigne).

  *Fait sur les résultats actuels du notebook (540 variables, avant les étapes 3 et 5 à 10) : §7 réécrite en cinq parties (What we did, What worked, What did not work, Comparison with existing work, If we had more time). Les chiffres viennent des sorties actuelles (LR 93,52 % en validation croisée, 95,42 % sur le test). Le constat principal se lit déjà dans les sorties : à environ 100 dimensions, PCA(≈102) bat SelectKBest(100) pour la LR (92,08 % contre 90,80 %) et le SVM (90,88 % contre 89,20 %), et perd pour la RF (85,27 % contre 89,24 %). La conclusion corrige donc l'idée « la sélection de variables préserve mieux la performance » de §3.6, §4 et §5.1, qui restent à réécrire (étape 5). Autres points tirés des sorties : écarts entre modèles (1,2 et 1,7 pt) inférieurs à l'écart-type entre plis (4,4 à 5,1 pt) ; 73 des 135 erreurs de test entre SITTING et STANDING, et seulement 4 entre activités statiques et dynamiques. Comparaison avec l'existant : 95,42 % contre ≈ 96 % (Anguita) et 95,55 % (Gao), moins de 20 fenêtres de test d'écart, donc résultats jugés équivalents. Le 93,89 % de Gao à 100 variables (section SOTA) n'a pas pu être vérifié (page MDPI inaccessible) : la conclusion ne le cite pas. À reprendre après les étapes 5 à 8 si leurs résultats changent le constat.*

### Étape 12 : relance finale et mise à jour des chiffres (**en dernier**, ~20 min)

- [ ] **Kernel → Restart Kernel and Run All Cells.**
- [ ] **Remettre à jour tous les chiffres du texte.** Ils sont actuellement périmés : ils ne correspondent pas aux sorties du code et viennent d'une ancienne version.

  | Mesure | Écrit | Affiché par le code | Où dans le texte |
  |---|---|---|---|
  | LR (561) accuracy | 93,42 % | 93,46 % | §3.6, §5.1, §6 Trade-off, §7, SOTA |
  | LR (561) F1 macro | 93,30 % | 93,36 % | §3.6, §5.1 |
  | LR + PCA accuracy | 91,94 % | 91,91 % | §3.2, §6 Trade-off |
  | LR + PCA F1 macro | 91,80 % | 91,75 % | §3.2 |
  | Baisse d'accuracy LR avec PCA | 1,48 pt | 1,55 pt | §3.2, §3.6 |
  | Baisse de F1 LR avec PCA | 1,50 pt | 1,61 pt | §3.2 |
  | RF + PCA accuracy | 85,19 % | 85,13 % | Random Forest Results, §4.3 |
  | RF + PCA F1 macro | 84,66 % | 84,65 % | Random Forest Results |
  | Baisse d'accuracy RF avec PCA | 6,56 pt | 6,62 pt | Random Forest Results |
  | LR + KBest(200) accuracy | 92,42 % | 92,41 % | §4.1, §6 Trade-off, §7, SOTA |
  | LR + KBest(200) F1 macro | 92,23 % | 92,21 % | §4.1 |
  | Baisse d'accuracy LR avec KBest | « 1 pt » | 1,05 pt | §4.1, §6 Trade-off, SOTA |

  La colonne « Affiché par le code » date d'avant la suppression des 21 colonnes dupliquées (étape 4). Avec 540 variables, la relance donne par exemple 93,52 % pour la LR en validation croisée, 92,08 % pour la LR + PCA et 95,42 % sur le test. Ces valeurs changeront de toute façon après les étapes 3 à 6. **Le mieux** est de faire afficher les phrases clés par le code (f-strings avec `:.2%`), ce qui évite ce décalage. Les scores de la LR varient aussi d'environ ±0,1 pt selon la machine (la relance donne 93,45 % en validation croisée et 95,45 % sur le test).
- [ ] **Vérifier la taille** du notebook (< 10 Mo) et que les noms de tous les membres y sont.
- [ ] **Préparer les slides en PDF** (recommandées par la consigne) à partir des graphiques finaux.
