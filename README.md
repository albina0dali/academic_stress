# Academic Stress Classification — Explainable ML Framework

## Overview

This repository contains the complete source code for a comparative machine learning study on academic stress classification. The framework integrates:

- Five classification algorithms evaluated across eight performance metrics
- SMOTE-based class imbalance correction with structured ablation analysis
- 10×5 Repeated Stratified K-Fold cross-validation
- Friedman + Nemenyi statistical significance testing
- SHAP-based explainability (TreeSHAP)

---

## Datasets

| | Dataset 1 | Dataset 2 |
|---|---|---|
| **Source** | Kaggle — [student-stress-factors-a-comprehensive-analysis](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis) | Kaggle — [student-stress-factors](https://www.kaggle.com/datasets/samyakb/student-stress-factors) |
| **Records** | 777 (filtered: age 18–22) | 1,100 |
| **Features** | 23 binary symptom indicators | 20 quantitative features |
| **Target** | Stress type: Eustress / Distress / No Stress | Stress severity: Low / Medium / High |
| **Class balance** | Severely imbalanced (91.4% Eustress) | Approximately balanced |
| **Role in study** | Ablation study (SMOTE) | Main evaluation dataset |

Place both CSV files in the root directory before running the notebook:

```
Stress_Dataset.csv
StressLevelDataset.csv
stress_level.ipynb
```

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap scipy scikit-posthocs
```

Tested with Python 3.10.

| Package | Version |
|---|---|
| scikit-learn | 1.3+ |
| imbalanced-learn | 0.11+ |
| shap | 0.44+ |
| scikit-posthocs | 0.7+ |
| scipy | 1.11+ |

---

## Notebook Structure

The notebook `stress_level.ipynb` is organized sequentially — run all cells top to bottom.

| Cell | Description |
|---|---|
| **IMPORTS & SETTINGS** | Libraries, global constants, `RANDOM_STATE = 42` |
| **LOAD & INSPECT DATA** | Load both CSVs, print shapes and class distributions |
| **PREPROCESSING** | Age filtering (Dataset 1), label encoding, z-score normalisation, SMOTE (train partition only) |
| **DEFINE MODELS & TRAIN** | LR, RF, GB, SVM-RBF, MLP — all with default hyperparameters, `random_state=42` |
| **FIGURE 1** | Class distribution and SMOTE-balancing effect |
| **FIGURE 2** | Spearman rank correlation heatmap (Dataset 2) |
| **FIGURE 3** | Feature distribution boxplots by stress class |
| **FIGURE 4** | Comparative bar chart — 5 metrics × 5 models |
| **FIGURE 5** | ROC curves — one-vs-rest, macro-averaged AUC |
| **FIGURE 6** | Normalised confusion matrices — all five classifiers |
| **FIGURE 7** | Cross-validation boxplots (10×5 RSKF) |
| **FIGURE 8** | Ablation study — accuracy and per-class recall across 4 configurations |
| **FIGURE 9** | Friedman + Nemenyi post-hoc pairwise p-value heatmap |
| **FIGURE 10** | SHAP beeswarm plot — High Stress class |
| **FIGURE 11** | Global SHAP feature importance bar chart |
| **FIGURE 12** | Learning curves — bias–variance characterisation |

All figures are saved as high-resolution PNG files (`dpi=300`) in the working directory.

---

## Key Results

| Model | Accuracy | AUC-ROC | F1 | Log Loss |
|---|---|---|---|---|
| **Random Forest** | **0.8909** | 0.9830 | **0.8907** | **0.2057** |
| Logistic Regression | 0.8818 | **0.9853** | 0.8817 | 0.2930 |
| Gradient Boosting | 0.8727 | 0.9825 | 0.8722 | 0.3323 |
| SVM (RBF) | 0.8773 | 0.9845 | 0.8771 | 0.2752 |
| MLP Neural Network | 0.8727 | 0.9836 | 0.8721 | 0.5244 |

Friedman test: χ² = 68.97, p < 0.001 — performance differences are statistically significant.

Top SHAP predictors of stress severity: **self-esteem**, **sleep quality**, **bullying**, **anxiety level**.

---

## Reproducibility

All stochastic components use `random_state = 42`. Results are fully reproducible on any platform with the package versions listed above.

---
