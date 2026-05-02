## 🎓 An Explainable Machine Learning Framework for Academic Stress Classification Among University Students

A comprehensive, publication-ready research project implementing an explainable ML framework for classifying academic stress levels among university students. The project covers dual-dataset analysis, feature engineering, class balancing, five-model comparison, SHAP-based interpretability, and cross-dataset synthesis.

> 📄 **Full Research Paper:** See [`research_paper.md`](research_paper.md) for the complete publication-quality paper including detailed methodology, figure captions with interpretations, theoretical grounding, and cross-dataset comparison analysis.

## 📊 Project Overview

The framework is applied to two structurally different datasets:
- **Academic Stress Level dataset** (1 100 students) — social-pressure stressors on ordinal scales
- **Student Mental Health dataset** (842 students) — binary clinical condition indicators

**Key findings:**
- Nearly 64% of students fall into the High stress category (Academic Stress dataset)
- **`peer_x_competition`** (peer pressure × competition) is the strongest predictor, accounting for ~31% of Extra Trees Gini importance — grounded in social-evaluation threat theory
- Extra Trees classifier achieves **96.3% accuracy** on the Academic Stress dataset; cross-dataset comparison on the Mental Health dataset (88–91%) quantifies the difficulty premium of clinical classification
- SHAP analysis confirms engineered interaction features capture genuine signal, not statistical artefacts
- Bootstrap oversampling (training-only) prevents data leakage while balancing all three stress classes

## 🗂️ Repository Contents

| File | Description |
|------|-------------|
| [`research_paper.md`](research_paper.md) | Full publication-quality research paper with detailed figure captions, methodology, and discussion |
| [`students_stress_level.ipynb`](students_stress_level.ipynb) | Primary ML notebook — Academic Stress Level dataset with annotated figure captions |
| [`mental_health.ipynb`](mental_health.ipynb) | Secondary ML notebook — Student Mental Health dataset with annotated figure captions |
| [`StressLevelDataset.csv`](StressLevelDataset.csv) | Kaggle Student Stress Factors dataset (1 100 rows, 21 features) |
| [`Stress_Dataset.csv`](Stress_Dataset.csv) | Academic stress survey dataset (842 rows, 26 features) |

## 🛠️ Technologies Used

- **Python** — Core language
- **Pandas & NumPy** — Data manipulation and feature engineering
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning modeling, evaluation, and cross-validation
- **imbalanced-learn** — Bootstrap oversampling for class balancing
- **SHAP** — SHapley Additive exPlanations for model interpretability
- **joblib** — Model serialization

## 🔬 Methodology

1. **Data Preprocessing**
   - Missing value imputation (median for continuous; 'Unknown' category for nominal)
   - Label encoding of categorical variables (appropriate for tree-based models)
   - 5-point stress scale collapsed to 3 clinically actionable classes: Low (1–2), Medium (3), High (4–5)

2. **Feature Engineering** (5 derived features)
   - `total_pressure` = peer_pressure + parental_pressure (cumulative load)
   - `peer_x_competition` = peer_pressure × competition (synergistic amplification — #1 SHAP feature)
   - `parent_x_competition` = parental_pressure × competition
   - `avg_stress_peer` = group-mean stress by peer-pressure tier (population prior)
   - `avg_stress_parent` = group-mean stress by parental-pressure tier

3. **Class Balancing** — Bootstrap oversampling applied **exclusively to the training partition** (preventing data leakage), equalising Low / Medium / High stress classes

4. **Modeling** — Five classifiers with diverse inductive biases:
   - Extra Trees (randomised thresholds, 500 trees)
   - Gradient Boosting (stagewise residual fitting, lr=0.05, depth=5)
   - Random Forest (optimal splits + feature subsets, 500 trees)
   - SVM (RBF kernel, C=5, gamma='scale')
   - Soft-Voting Ensemble (ET + GBM + RF probability averaging)

5. **Evaluation** — Stratified 5-fold cross-validation + held-out 20% test set; Accuracy, F1, Precision, Recall (weighted)

6. **Explainability** — SHAP TreeExplainer on Extra Trees model; global feature ranking + class-level directional analysis

## 📈 Results Summary

| Model | Accuracy | F1 | Precision | Recall | CV (mean±SD) |
|-------|----------|----|-----------|--------|--------------|
| **Extra Trees** | **96.3%** | **96.2%** | **96.4%** | **96.3%** | 95.8%±0.9% |
| Random Forest | 95.1% | 95.0% | 95.2% | 95.1% | 94.9%±1.1% |
| Voting Ensemble | 94.7% | 94.6% | 94.8% | 94.7% | 94.2%±1.0% |
| Gradient Boosting | 93.8% | 93.7% | 93.9% | 93.8% | 93.5%±1.2% |
| SVM | ~89.4% | ~89.2% | ~89.5% | ~89.4% | ~88.8%±1.8% |

## 📐 Top SHAP Features (Academic Stress Dataset)

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | `peer_x_competition` | ~31% | Multiplicative stress amplification in competitive peer environments |
| 2 | `total_pressure` | ~18% | Cumulative interpersonal demand exceeding coping threshold |
| 3 | `avg_stress_peer` | ~15% | Population-level calibration signal for peer-pressure tier |
| 4 | `peer_pressure` | ~12% | Direct social-comparison threat appraisal |
| 5 | `competition` | ~9% | Academic environment competitiveness moderator |
