## 🎓 Academic Stress Among Students: ML-Based Stress Prediction
A comprehensive data science project focused on analyzing and predicting academic stress levels among students. Includes data preprocessing, exploratory data analysis (EDA), class balancing with SMOTE, and multiple machine learning classification models to predict stress levels based on social, behavioral, and environmental factors.

## 📊 Project Overview
The goal of this project is to perform a detailed Exploratory Data Analysis (EDA) on student stress survey data and implement multiple classification models to predict stress levels. It covers everything from data cleaning and feature engineering to model comparison and evaluation.
Key findings include:

Nearly 64% of students fall into the High stress category
Peer pressure × Competition is the strongest predictor of stress (engineered interaction feature)
Extra Trees classifier achieves 96.3% accuracy — the best among all tested models

## 🛠️ Technologies Used

Python (Core language)
Pandas & NumPy (Data manipulation and feature engineering)
Matplotlib & Seaborn (Data visualization)
Scikit-learn (Machine learning modeling and evaluation)
Imbalanced-learn / SMOTE (Handling class imbalance)

## 🔬 Methodology

1. **Data Preprocessing** — Encoding categorical variables, engineering interaction features (Peer × Competition, Parental × Competition, Total Pressure)
2. **EDA** — Distribution analysis, correlation heatmaps, stress breakdowns by environment, habits, and coping strategy
3. **Class Balancing** — SMOTE applied to training set to equalize Low / Medium / High stress classes
4. **Modeling** — Five classifiers trained and compared: Extra Trees, Gradient Boosting, Random Forest, SVM, Voting Ensemble
5. **Evaluation** — 5-fold cross-validation + held-out test set scored on Accuracy, F1, Precision, and Recall
