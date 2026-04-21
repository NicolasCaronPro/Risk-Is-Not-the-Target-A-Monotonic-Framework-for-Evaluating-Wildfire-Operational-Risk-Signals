# Risk Is Not the Target: A Monotonic Framework for Evaluating Wildfire Operational Risk Signals

## Objectif du dépôt
Ce dépôt accompagne l'article **"Risk Is Not the Target"** et propose un cadre d'évaluation pour les systèmes de risque incendie de forêt orientés opérations.

L'idée centrale est la suivante :
- un bon signal de risque opérationnel n'est pas seulement un signal qui "prédit" des événements,
- c'est un signal dont l'échelle ordinale (faible → fort) reflète réellement la charge opérationnelle observée sur le terrain.

Ce dépôt permet de :
- reproduire la logique de scoring monotone,
- comparer plusieurs types de systèmes de risque,
- analyser la cohérence opérationnelle d'un signal de risque continu.

## Abstract
Evaluating wildfire risk systems using standard machine-learning metrics such as F1-score or IoU is fundamentally flawed: these metrics assess event prediction accuracy, not the operational coherence of a continuous risk signal. This work proposes a novel monotonic evaluation framework that measures whether increases in a predicted risk score consistently correspond to increases in observed operational load, such as number of fires, intervention time, and deployed resources. Moreover, we compare three structurally different approaches on the French Alpes-Maritimes department: the expert-based DFE index, GRU-based predictive models, and FARS, a hybrid multi-agent system combining predictive AI with LLM-based reasoning. Experimental results reveal that the DFE, despite poor classification metrics, exhibits the most balanced monotonic behavior across the full risk scale. GRU models achieve strong local monotonicity but fail to produce well-distributed risk levels. FARS inherits and reveals the structural limitations of upstream signals rather than correcting them. The central finding is a paradigm shift: a good risk model does not predict fires accurately, but one whose ordinal scale meaningfully explains operational dynamics, as proved in this paper.

## Contenu du dépôt
- `monotonic_score.py` : implémentation principale des métriques de monotonie et des fonctions d'analyse.
- `USAGE.ipynb` : notebook d'exemples pour exécuter le pipeline et interpréter les résultats.
- `README.md` : présentation du projet et guide de prise en main.

## Installation
1. Cloner le dépôt :
   ```bash
   git clone <url-du-repo>
   cd Risk-Is-Not-the-Target-A-Monotonic-Framework-for-Evaluating-Wildfire-Operational-Risk-Signals
   ```
2. Installer les dépendances Python nécessaires (dans votre environnement) :
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

> Si d'autres dépendances spécifiques apparaissent dans votre version locale du notebook, ajoutez-les selon les imports de `USAGE.ipynb`.

## Utilisation rapide
### Option 1 — via le notebook
1. Ouvrir `USAGE.ipynb` dans Jupyter Lab/Notebook.
2. Exécuter les cellules dans l'ordre.
3. Observer :
   - la distribution des niveaux de risque,
   - les relations monotones entre score de risque et charge opérationnelle,
   - les comparaisons entre approches (DFE / GRU / FARS).

### Option 2 — via le script Python
Vous pouvez importer les fonctions depuis `monotonic_score.py` dans votre propre pipeline :

```python
from monotonic_score import *

# Exemple conceptuel :
# - risk_scores : scores continus prédits
# - operational_targets : variables de charge opérationnelle observées
# Utiliser ensuite les fonctions du module pour calculer et agréger les indicateurs monotones.
```

## Interprétation des résultats
Le framework met l'accent sur la question opérationnelle :
- **Quand le niveau de risque augmente, la charge réelle augmente-t-elle aussi de manière cohérente ?**

Un système peut donc :
- avoir de bons scores de classification classiques mais être peu utile en pilotage opérationnel,
- ou au contraire afficher des performances de classification modestes tout en étant robuste sur l'axe monotone (cas typique mis en avant pour le DFE).

## Cas d'usage visé
Ce travail est utile pour :
- services incendie et secours,
- décideurs opérationnels,
- équipes IA qui conçoivent des systèmes d'alerte graduée,
- chercheurs en évaluation de modèles de risque temporels/spatiaux.

## Citation
Si vous utilisez ce dépôt dans un travail académique, merci de citer l'article associé (à compléter avec la référence bibliographique officielle).
