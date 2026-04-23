# Risk Is Not the Target: A Monotonic Framework for Evaluating Wildfire Operational Risk Signals

## Repository Objective
This repository supports the paper **"Risk Is Not the Target"** and introduces an evaluation framework for wildfire risk systems from an operational perspective.

Core idea:
- a good operational risk signal is not only a signal that predicts events,
- it is a signal whose ordinal scale (low → high) consistently reflects real operational load in the field.

This repository helps you:
- reproduce the monotonic scoring logic,
- compare different types of wildfire risk systems,
- evaluate whether a continuous risk score is operationally coherent.

## Abstract
Evaluating wildfire risk systems using standard machine-learning metrics such as F1-score or IoU is fundamentally flawed: these metrics assess event prediction accuracy, not the operational coherence of a continuous risk signal. This work proposes a novel monotonic evaluation framework that measures whether increases in a predicted risk score consistently correspond to increases in observed operational load, such as number of fires, intervention time, and deployed resources. Moreover, we compare three structurally different approaches on the French Alpes-Maritimes department: the expert-based DFE index, GRU-based predictive models, and FARS, a hybrid multi-agent system combining predictive AI with LLM-based reasoning. Experimental results reveal that the DFE, despite poor classification metrics, exhibits the most balanced monotonic behavior across the full risk scale. GRU models achieve strong local monotonicity but fail to produce well-distributed risk levels. FARS inherits and reveals the structural limitations of upstream signals rather than correcting them. The central finding is a paradigm shift: a good risk model does not predict fires accurately, but one whose ordinal scale meaningfully explains operational dynamics, as proved in this paper.

## Repository Contents
- `monotonic_score.py`: main implementation of monotonic metrics and analysis functions.
- `USAGE.ipynb`: example notebook to run the workflow and interpret outputs.
- `README.md`: project overview and usage guide.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Risk-Is-Not-the-Target-A-Monotonic-Framework-for-Evaluating-Wildfire-Operational-Risk-Signals
   ```
2. Install Python dependencies in your environment:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

> If your local notebook version requires additional dependencies, install them based on imports in `USAGE.ipynb`.

## Quick Start
### Option 1 — Notebook workflow
1. Open `USAGE.ipynb` in JupyterLab/Notebook.
2. Run cells in order.
3. Analyze:
   - risk-level distribution,
   - monotonic relationships between risk score and operational load,
   - model comparisons (DFE / GRU / FARS).

### Option 2 — Python script workflow
You can import functions from `monotonic_score.py` into your own pipeline:

```python
from monotonic_score import *

# Conceptual example:
# - risk_scores: predicted continuous risk values
# - operational_targets: observed operational load variables
# Then use module functions to compute and aggregate monotonic indicators.
```

## How to Interpret Results
The framework focuses on one operational question:
- **When risk increases, does real operational load also increase consistently?**

Therefore, a model can:
- achieve strong classification metrics but still be weak for operational decision support,
- or show modest classification performance while being robust on monotonic behavior (as highlighted for DFE).

## Intended Use Cases
This work is useful for:
- fire and rescue services,
- operational decision-makers,
- AI teams building graded alert systems,
- researchers working on temporal/spatial risk evaluation.

## Citation
If you use this repository in academic work, please cite the associated paper (to be completed with official bibliographic reference).
