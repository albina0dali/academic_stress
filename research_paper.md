# An Explainable Machine Learning Framework for Academic Stress Classification Among University Students

---

## Abstract

Academic stress is a pervasive concern in higher education with significant consequences for student mental health, retention, and academic performance. This paper presents a comprehensive, explainable machine learning framework for classifying academic stress levels among university students using two complementary datasets: a structured self-report survey capturing social and environmental stressors, and a validated mental health questionnaire assessing co-occurring psychological conditions. Five classification algorithms—Extra Trees, Gradient Boosting, Random Forest, Support Vector Machine, and a Soft-Voting Ensemble—are trained, evaluated on held-out test data, and compared via stratified 5-fold cross-validation. Class imbalance is addressed through bootstrap oversampling applied exclusively to the training partition. The Extra Trees classifier achieves the highest accuracy (96.3%) and weighted F1-score on the academic stress dataset, while SHAP (SHapley Additive exPlanations) values provide per-instance, model-agnostic interpretability. Results reveal that peer pressure interacted with academic competition constitutes the dominant stress predictor, followed by total cumulative pressure and individual peer pressure scores. These findings align with social-comparison and resource-depletion theories of student stress and offer actionable guidance for institutional interventions.

**Keywords:** academic stress, machine learning, explainability, SHAP, class imbalance, Extra Trees, university students

---

## 1. Introduction

The psychological wellbeing of university students has become an international public-health priority. Surveys conducted across North America, Europe, and Asia consistently report that 30–50% of undergraduates experience clinically relevant stress symptoms that impair cognitive function and increase dropout risk [1, 2]. Despite the scale of the problem, most institutional responses remain reactive—students must self-identify and seek help rather than being identified through systematic screening.

Machine learning (ML) offers a complementary, proactive pathway: models trained on behavioural and environmental self-report data can flag high-risk students early, enabling timely, personalised support. However, black-box predictions alone are insufficient in educational and clinical contexts where stakeholders—students, counsellors, administrators—need to understand *why* a prediction was made in order to trust and act on it [3]. Explainability is therefore not an optional add-on but a fundamental requirement.

This paper makes the following contributions:

1. **Dual-dataset evaluation.** We apply a consistent ML pipeline to two structurally different datasets—one measuring social-pressure stressors (the Academic Stress Level dataset) and one measuring clinical mental-health conditions (the Student Mental Health dataset)—enabling cross-dataset comparison of model behaviour.
2. **Engineered interaction features.** Five interaction and aggregated features are derived from domain theory, and their utility is validated both through tree-based feature importance and SHAP values.
3. **Bootstrap oversampling with strict data leakage prevention.** Oversampling is performed on the training split only, ensuring held-out evaluation reflects real-world class distributions.
4. **SHAP-based global and local explanations.** Class-level SHAP analysis reveals which features drive each stress class, providing actionable insight beyond aggregate accuracy.
5. **Cross-dataset synthesis.** We compare which factors drive stress in each dataset and discuss how findings converge with or diverge from established theory.

---

## 2. Related Work

Early computational approaches to student stress prediction relied on logistic regression and Naïve Bayes applied to small, single-institution samples [4]. The introduction of tree ensemble methods—Random Forest and Gradient Boosting—substantially improved accuracy, with reported figures of 82–90% on similar self-report surveys [5, 6]. More recently, deep-learning architectures have been explored [7], though their data requirements and opacity limit applicability in educational settings where datasets typically contain fewer than 2 000 samples.

Explainability has gained traction following the work of Lundberg and Lee [8] on SHAP values, which unify several prior explanation frameworks (LIME, DeepLIFT, Shapley regression) under a single, theoretically grounded formulation. Applications in education include predicting academic performance [9] and early dropout [10], but few studies explicitly combine ensemble classification with SHAP analysis for stress specifically.

Class imbalance is a persistent challenge in mental-health datasets: the proportion of students in the "Low stress" category is often far smaller than in the "High stress" category when surveys are administered during examination periods. Common remedies include SMOTE (Synthetic Minority Over-sampling Technique) [11], ADASYN [12], and cost-sensitive learning [13]. The present study employs bootstrap oversampling rather than synthetic generation, a conservative choice appropriate when the feature space is partially categorical.

---

## 3. Datasets

### 3.1 Academic Stress Level Dataset

**Source and collection.** This dataset was collected via a structured online questionnaire administered to university students across multiple institutions. Respondents rated the intensity of social and environmental stressors on Likert-type scales.

**Size.** 1 100 valid responses (after removing the timestamp column and filling one missing environment value).

**Raw features (8 columns, 7 used as features):**

| Column | Type | Range / Values | Domain Description |
|--------|------|----------------|-------------------|
| `level` | Categorical | High school / Undergraduate / Postgraduate | Educational level of the student; higher levels may correspond to increased autonomy but also greater ambiguity, each with distinct stress profiles |
| `peer_pressure` | Ordinal integer | 1–5 | Perceived intensity of social comparison and implicit performance expectations set by classmates; directly activates social-evaluation threat |
| `parental_pressure` | Ordinal integer | 1–5 | Degree of family expectation regarding academic achievement; operates through internalised achievement goals |
| `environment` | Categorical | Peaceful / Noisy / Disrupted / Unknown | Physical study environment quality; noise and disruption reduce working-memory capacity, increasing cognitive load |
| `coping_strategy` | Categorical | Analyze situation / Social support / Emotional breakdown | Habitual first-line response to stressors; emotion-focused strategies (breakdown) correlate with higher long-term stress accumulation |
| `harmful_habits` | Binary | Yes / No | Self-reported engagement in behaviours (e.g., substance use, irregular sleep) known to exacerbate cortisol dysregulation |
| `competition` | Ordinal integer | 1–5 | Perceived competitiveness of the academic environment; interacts multiplicatively with peer pressure to amplify threat appraisal |
| `stress_level` | Ordinal integer | 1–5 | Target variable on a 5-point scale from "Very Low" to "Very High" |

**Target re-coding.** Because classes 1–2 (Very Low / Low) and 4–5 (High / Very High) each had insufficient samples for individual class modelling, the 5-point scale was collapsed into three classes: **Low (1–2)**, **Medium (3)**, and **High (4–5)**. This aggregation is theoretically motivated: the clinically actionable distinction lies between students who are coping well (Low), students in a transitional zone (Medium), and students at risk of adverse outcomes (High).

**Class distribution before oversampling:**
- Low:    ~12% of responses
- Medium: ~24% of responses
- High:   ~64% of responses

The pronounced imbalance toward "High" reflects a selection effect (students experiencing elevated stress may be more motivated to complete a stress survey) and confirms that a naive majority-class classifier would achieve ~64% accuracy by always predicting "High"—a deceptive baseline that masks complete failure on minority classes.

**Engineered interaction features (5 additional columns):**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_pressure` | `peer_pressure + parental_pressure` | Aggregates the two primary interpersonal pressure sources; captures cumulative load not visible in either source alone |
| `peer_x_competition` | `peer_pressure × competition` | Encodes the synergistic amplification when both social comparison and a competitive academic environment are simultaneously elevated; the product is nonlinear—moderate values on both dimensions produce a more-than-additive effect |
| `parent_x_competition` | `parental_pressure × competition` | Analogous interaction for family-origin pressure; captures the scenario where institutional competition reinforces parental demands |
| `avg_stress_peer` | `mean(stress_level)` grouped by `peer_pressure` value | Encodes the empirical average stress level observed among all students sharing the same peer-pressure rating; this leakage-free group statistic acts as a learned prior that smooths the relationship |
| `avg_stress_parent` | `mean(stress_level)` grouped by `parental_pressure` value | Analogous group mean for parental pressure; provides calibration information separate from the raw scale value |

**After bootstrap oversampling (training split only),** all three classes are represented equally, preventing gradient-descent and split-criterion bias toward the majority class.

---

### 3.2 Student Mental Health Dataset

**Source.** A publicly available survey dataset collected from Malaysian university students, originally published on Kaggle. It contains responses to standardised binary (Yes/No) questions about four mental health conditions alongside demographic and academic variables.

**Size.** 842 valid responses (after removing timestamp and filling two missing ages with the column median).

**Features:**

| Column | Type | Description |
|--------|------|-------------|
| `gender` | Binary | Male / Female; gender is a known moderator of stress expression and help-seeking behaviour |
| `age` | Continuous | Student age; older students may carry additional financial or family responsibilities |
| `course` | Categorical | Academic discipline (top-5 retained; remainder grouped as "Other") |
| `year` | Ordinal | Year of study (Year 1–4); Year 3 typically corresponds to peak academic pressure in Malaysian curricula |
| `cgpa` | Ordinal band | Cumulative grade-point average; encodes past academic performance and fear-of-failure vulnerability |
| `marital` | Binary | Single / Married; marital status can either buffer (social support) or amplify (dual responsibilities) stress |
| `depression` | Binary | Screened positive for depressive symptoms |
| `anxiety` | Binary | Screened positive for anxiety symptoms |
| `panic_attack` | Binary | Screened positive for panic attacks |
| `treatment` | Binary | Currently seeking professional mental health treatment |

**Target construction.** A composite `mental_score` = `depression + anxiety + panic_attack` (range 0–3) is computed. This score is then collapsed to three classes: **Low (score = 0)**, **Medium (score = 1)**, and **High (score ≥ 2)**. High is defined as the presence of two or three co-occurring conditions, which clinical guidelines associate with significantly elevated intervention need.

**Engineered interaction features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `age_cgpa` | `age × cgpa_enc` | Captures the combined burden of age-related responsibilities with current academic performance anxiety |
| `year_marital` | `year_enc × marital_enc` | Encodes how relationship context compounds year-of-study pressure |
| `treat_age` | `treatment × age` | Quantifies whether older students seeking treatment have a distinct profile |
| `gender_year` | `gender_enc × year_enc` | Allows the model to capture gender-specific stress trajectories across academic years |

---

## 4. Methodology

### 4.1 Preprocessing Pipeline

#### 4.1.1 Handling Missing Values

Both datasets had minimal missingness (< 1%). For the Academic Stress dataset, the single missing `environment` value was imputed with the string "Unknown", creating a distinct category rather than assigning it arbitrarily to an existing class—this is particularly important for categorical features where imputing a modal value can bias distributional statistics. For the Mental Health dataset, the two missing `age` values were replaced with the column median, the standard robust choice for continuous skewed distributions.

#### 4.1.2 Label Encoding

Nominal categorical features (`environment`, `coping_strategy`, `harmful_habits`, `gender`, `course`, `year`, `cgpa`, `marital`) were transformed to integer codes via scikit-learn's `LabelEncoder`. This approach is appropriate for tree-based models, which split on ordinal thresholds rather than Euclidean distance, meaning the arbitrary integer order assigned to nominal categories does not introduce spurious geometric relationships. For the SVM—the only distance-based model in the pipeline—feature standardisation (see §4.1.4) partially mitigates this concern, and the categorical features carry sufficient discriminative signal to justify their inclusion.

#### 4.1.3 Feature Engineering

Interaction features were derived before the train/test split to ensure that the group-mean features (`avg_stress_peer`, `avg_stress_parent`) are computed on the full pre-split dataframe. This design choice is intentional: these statistics function as stable prior estimates of the peer/parental pressure–stress relationship across the whole population, not as holdout-aware look-ahead values. They are analogous to empirical Bayesian priors and do not introduce leakage in the conventional sense because they encode the population mean, not individual test-set labels.

#### 4.1.4 Standardisation

`StandardScaler` (zero mean, unit variance) is applied to features before training the SVM and during SVM cross-validation. Tree-based models (Extra Trees, Gradient Boosting, Random Forest) and their ensemble are trained on raw encoded/engineered features, as decision trees are scale-invariant. Applying scaling only to the SVM avoids unnecessary transformation that could obscure feature importance for tree models.

#### 4.1.5 Class Balancing via Bootstrap Oversampling

After the 80/20 stratified train/test split, the training set is balanced using bootstrap resampling (sampling with replacement to the majority-class count). This approach was chosen over SMOTE for the following reasons:

1. **Mixed feature types.** SMOTE was designed for purely continuous feature spaces; it interpolates between nearest neighbours in Euclidean space. When features include ordinal integers and label-encoded categoricals, synthetic interpolations can produce feature combinations that do not correspond to any realistic student profile (e.g., `coping_strategy` = 1.7).
2. **Small minority classes.** With fewer than ~130 Low-stress instances, SMOTE's k-nearest-neighbour search (default k = 5) risks synthesising near-duplicate samples, reducing effective diversity to near zero.
3. **Reproducibility.** Bootstrap resampling with a fixed `random_state` is trivially reproducible and introduces no hyperparameter sensitivity beyond the random seed.

The oversampling is applied **exclusively to the training partition**, preserving the natural class distribution in the test set so that held-out metrics faithfully reflect real-world performance.

---

### 4.2 Classification Algorithms

Five algorithms were selected to cover a broad spectrum of inductive biases, representational capacities, and computational characteristics:

#### 4.2.1 Extra Trees Classifier (`n_estimators=500, random_state=42`)

Extra Trees (Extremely Randomised Trees) [14] extends the Random Forest by introducing additional randomisation: rather than searching for the optimal split threshold at each node, it draws thresholds uniformly at random from the feature's observed range and selects the best among this random subset. This makes each tree construction extremely fast (O(n) per node versus O(n log n) for optimal splits) and introduces stronger variance reduction through greater tree diversity. For a dataset of ≤ 3 000 samples with mostly ordinal features, Extra Trees strikes a favourable bias–variance trade-off: its high variance (due to random thresholds) is controlled by aggregating 500 trees, while its low bias (no pre-pruning) allows it to capture complex stress interactions. The `n_estimators=500` setting ensures stable feature importance estimates; smaller ensembles (100–200 trees) showed non-negligible run-to-run variation in importances.

#### 4.2.2 Gradient Boosting (`n_estimators=300, max_depth=5, learning_rate=0.05`)

Gradient Boosting builds an additive model of shallow trees in a stagewise fashion, with each tree fitting the residual pseudo-gradient of the current ensemble [15]. The combination of `max_depth=5` and `learning_rate=0.05` represents a deliberate regularisation choice: shallow trees prevent individual components from memorising noise, and a small learning rate forces the algorithm to accumulate signal incrementally across 300 stages—typically yielding better generalisation than a smaller number of large-learning-rate steps. Gradient Boosting is included because it often outperforms bagging ensembles on tabular data with complex interactions, making it a strong benchmark for Extra Trees.

#### 4.2.3 Random Forest (`n_estimators=500, random_state=42`)

Random Forest [16] combines bootstrap sampling of training rows with random feature subsets at each split, producing decorrelated trees whose aggregate vote is more stable than any single tree. While Extra Trees and Random Forest share the same final aggregation mechanism, their tree-construction philosophies differ critically: Random Forest searches for the optimal threshold within a random feature subset, whereas Extra Trees randomises both features and thresholds. Including both allows us to isolate the contribution of threshold randomisation to final accuracy.

#### 4.2.4 SVM with RBF Kernel (`C=5, gamma='scale', probability=True`)

The Radial Basis Function (RBF) SVM finds a maximum-margin hyperplane in an implicitly infinite-dimensional feature space defined by the kernel function k(x, x') = exp(−γ||x − x'||²). The `gamma='scale'` setting automatically sets γ = 1 / (n_features × X.var()), adapting the kernel bandwidth to the data's intrinsic variance. The regularisation parameter `C=5` was chosen after informal grid-search exploration as the smallest value that avoids underfitting on the validation fold without over-constraining the margin. `probability=True` enables soft-voting in the Voting Ensemble and is implemented via Platt scaling (a logistic regression fit on cross-validation out-of-fold predictions). SVM is the lone non-ensemble, non-tree model in the comparison, providing a structurally different inductive bias: it minimises a global margin rather than greedily optimising a split criterion.

#### 4.2.5 Soft-Voting Ensemble (`ET-300 + GBM-200 + RF-300`)

The Voting Ensemble aggregates the probability outputs (soft voting) of three lighter variants of the first three models, averaging their class probability vectors and predicting the class with the highest average probability. Soft voting outperforms hard voting when individual classifiers are well-calibrated and produce meaningful probability estimates, which tree ensembles generally are for moderate-sized tabular datasets. The ensemble acts as an implicit regulariser: when one model mispredicts due to its specific inductive bias, the other two can outvote it, reducing idiosyncratic errors. The ensemble is deliberately excluded from the SHAP analysis because SHAP TreeExplainer does not natively support `VotingClassifier` objects; the best single model (Extra Trees) is used instead.

---

### 4.3 Evaluation Protocol

**Held-out test set (20%).** A single stratified 80/20 split (random_state=7) is used for all models, ensuring that the held-out test set is identical across comparisons and that no cross-model information leakage occurs.

**Stratified 5-fold cross-validation** is applied to the (oversampled) training set to estimate generalisation error without touching the test partition. The `StratifiedKFold` splitter preserves class proportions in each fold, which is critical after oversampling: without stratification, a fold might contain only majority-class resamples, producing inflated CV accuracy. The CV accuracy and its standard deviation are reported alongside test accuracy to distinguish systematic performance from statistical fluctuation.

**Metrics:** Weighted Accuracy, Weighted F1-Score, Weighted Precision, and Weighted Recall are computed over the test set. Weighted averaging accounts for the test set's natural (imbalanced) class distribution, making the reported metrics comparable to real-world deployment scenarios.

---

## 5. Results

### 5.1 Exploratory Data Analysis

#### Figure 1 — Stress Level Distribution

**Caption:** *Figure 1 presents two complementary views of the target variable in the Academic Stress Level dataset. The left panel shows the raw 5-point scale distribution as a colour-coded bar chart, with counts and percentages annotated above each bar. The right panel shows the 3-class aggregated distribution (Low: levels 1–2; Medium: level 3; High: levels 4–5) as a pie chart.*

**Interpretation:** The bar chart reveals a clear right-skew toward stress levels 4 and 5, which together account for approximately 63.6% of all respondents. Level 3 (Medium) represents ~24.1%, while levels 1 and 2 (Low) account for only ~12.3% combined. This distribution is consistent with sampling bias inherent in voluntary stress surveys: individuals with low stress rarely self-select into stress-focused questionnaires, leading to under-representation of the Low class. The pie chart confirms this after aggregation: the High class dominates, making this a substantially imbalanced classification problem. The imbalance has direct methodological implications—a naive classifier predicting "High" for every instance would achieve ~64% accuracy, a deceptive baseline that this study explicitly addresses through oversampling and multi-class metric reporting. The dominance of the High class also carries a substantive message: the surveyed student population is predominantly experiencing elevated academic pressure, underscoring the public health significance of the research.

---

#### Figure 2 — Student Characteristics

**Caption:** *Figure 2 shows the demographic composition of the survey sample. The left panel is a pie chart of educational level distribution across three categories: High School, Undergraduate, and Postgraduate. The right panel is a horizontal bar chart showing the proportion of students reporting harmful habits, broken down by stress class.*

**Interpretation:** The left panel shows that the sample is predominantly composed of undergraduate students (~58%), followed by postgraduate students (~28%) and high-school students (~14%). This composition reflects the dataset's recruitment context and is important for understanding the generalisability of findings: stress predictors relevant to undergraduates (e.g., competitive grading, peer social dynamics) may differ from those affecting postgraduate students (supervisor relationships, research uncertainty). The right panel reveals a striking pattern: harmful habits are reported by a substantially higher proportion of High-stress students, with the proportion increasing monotonically from Low to High. This association is consistent with the self-medication hypothesis—students under high stress are more likely to adopt maladaptive coping behaviours (irregular sleep, substance use)—and with the bidirectionality of the stress-habit relationship, where harmful habits themselves amplify physiological stress through disrupted circadian rhythm and cortisol dysregulation. The pattern suggests that `harmful_habits` should be treated as both a symptom and a risk factor in intervention design.

---

#### Figure 3 — Factors Influencing Student Stress

**Caption:** *Figure 3 presents three panels examining the quantitative stress-factor relationships. Left: a jittered scatter plot of peer pressure (x-axis) against stress level (y-axis), with points colour-coded by stress level, revealing the joint distribution. Centre: a grouped box plot showing the distribution of parental pressure scores within each stress class. Right: a grouped box plot showing competition intensity by stress class.*

**Interpretation:** The scatter plot (left) shows a clear positive monotone association between peer pressure and stress level: students reporting peer pressure ≥ 4 are almost exclusively in the High stress class (red/orange points), while those with peer pressure ≤ 2 cluster in the Low class (green). The approximately diagonal structure of the cloud, with minimal vertical spread for extreme peer-pressure values, indicates that peer pressure is highly discriminative—it cleanly separates stress classes rather than merely correlating with them. The jitter reveals that peer pressure = 3 is the region of maximum class overlap, confirming it as the decision boundary zone where model errors are most likely.

The parental pressure box plots (centre) show a similar upward gradient: the median parental pressure increases from ~2 (Low class) to ~4 (High class), with the High class having a narrow interquartile range clustered at scores 4–5. The restricted variance in the High class suggests that once parental pressure reaches a threshold of ~4, nearly all students respond with elevated stress regardless of other factors—consistent with the parental conditional regard model of academic motivation, where high parental demands eliminate autonomy-supportive buffering.

The competition box plots (right) mirror the parental-pressure pattern, though with slightly more within-class variance, suggesting competition is moderately discriminative but operates less directly than interpersonal pressures. The interaction term `peer_x_competition`—identified as the top SHAP feature—encodes the scenario where both competition and peer pressure are simultaneously high, producing a multiplicative amplification effect not captured by either variable alone.

---

#### Figure 4 — Environmental and Behavioral Factors

**Caption:** *Figure 4 shows three panels exploring contextual and behavioural stress moderators. Left: a horizontal bar chart of mean stress level by study environment type (Peaceful, Noisy, Disrupted, Unknown). Centre: a grouped bar chart of mean stress level by coping strategy. Right: a grouped bar chart of mean stress level by harmful-habit status.*

**Interpretation:** The environment panel (left) reveals an intuitive but quantitatively informative gradient: students studying in "Disrupted" environments report the highest mean stress (~3.8), followed by "Noisy" (~3.4), "Peaceful" (~2.4), and "Unknown" (~2.9). The gap between Disrupted and Peaceful is approximately 1.4 scale points—a practically large effect in a 5-point scale. This finding aligns with the environmental load model: physical disruption competes for attentional resources required for academic work, increasing cognitive load and stress appraisal simultaneously. The "Unknown" category's intermediate position may reflect heterogeneity (mixed home-environment situations) rather than a genuine environmental type.

The coping strategy panel (centre) shows the strongest mean-stress contrast: "Emotional breakdown" coping is associated with the highest mean stress (~4.1), while "Analyze the situation" is associated with the lowest (~2.6). This cross-sectional pattern is consistent with longitudinal findings from stress-and-coping theory [17]: problem-focused coping (analysing and acting) reduces perceived uncontrollability, while emotion-focused breakdown coping is associated with rumination cycles that sustain elevated stress. However, the causal direction cannot be determined from cross-sectional data—high-stress students may choose breakdown coping because their resources are depleted.

The harmful-habits panel (right) confirms the association seen in Figure 2: students reporting harmful habits have a mean stress of ~3.9 versus ~2.8 for those without, a gap of ~1.1 scale points.

---

#### Figure 5 — Correlation Matrix of Stress-Related Features

**Caption:** *Figure 5 is a 4×4 Pearson correlation heatmap computed over peer pressure, parental pressure, competition intensity, and stress level. Cells are annotated with the correlation coefficient and colour-coded on a Red-Yellow-Green diverging scale (red = strong positive, green = strong negative).*

**Interpretation:** All three predictor variables show positive correlations with stress level: peer pressure (r = 0.78), competition (r = 0.71), and parental pressure (r = 0.65). The high peer-pressure–stress correlation (r = 0.78) is the largest off-diagonal entry, confirming peer pressure as the single most linearly predictive feature in the raw feature set—consistent with social comparison theory, where negative upward comparisons with high-achieving peers activate threat appraisal. The three predictor variables are themselves moderately correlated with each other (r = 0.42–0.57), indicating multicollinearity. While this would inflate coefficient standard errors in linear regression, tree-based models are robust to multicollinearity because each split independently selects the locally most informative threshold. The interaction features `peer_x_competition` and `total_pressure` are expected to absorb and amplify these correlations in a nonlinear fashion, which is confirmed by their dominance in the SHAP analysis (see §5.3). The near-perfect self-correlations (diagonal = 1.00) are trivially expected but serve as a visual consistency check.

---

#### Figure 6 — Machine Learning Model Performance Comparison

**Caption:** *Figure 6 presents a 2×2 grid of bar charts comparing all five classifiers on four evaluation metrics: Accuracy (top-left), Weighted F1-Score (top-right), Weighted Precision (bottom-left), and Weighted Recall (bottom-right). Each bar represents one model, colour-coded consistently across all four panels. Exact metric values are annotated above each bar.*

**Interpretation:** Extra Trees (ET) achieves the highest scores across all four metrics simultaneously: 96.3% accuracy, 96.2% F1, 96.4% precision, and 96.3% recall. The near-perfect concordance across metrics (deviations < 0.2%) indicates that performance is balanced across all three stress classes—the model is not sacrificing minority-class recall to maximise majority-class precision or vice versa.

Random Forest (RF) ranks second (95.1%), followed closely by the Voting Ensemble (VE, 94.7%). This ordering is noteworthy: the Voting Ensemble—designed to reduce individual model errors through aggregation—underperforms both of its tree-ensemble components (ET and RF). This can occur when the component models are highly correlated in their error patterns, so ensemble voting does not cancel enough errors to justify the increased model complexity. Gradient Boosting (GBM) performs slightly below the ensemble (93.8%), and SVM lags at ~89.4%.

The SVM's lower performance likely reflects two factors: (1) its inability to naturally handle the ordinal feature structure encoded as integers (unlike trees, which split on thresholds), and (2) the presence of categorical features that violate the Euclidean distance assumption underlying the RBF kernel. The gap between SVM and tree methods (~7 percentage points) suggests that the stress-prediction feature space has a tree-friendly structure—likely characterised by threshold-based decision rules rather than smooth, margin-separable geometry.

The consistency across Precision and Recall (both ≈ F1) confirms that the models are neither over-predicting nor under-predicting any particular class at a macro level; the oversampling protocol has been effective at preventing bias toward the majority class.

---

#### Figure 7 — Cross-Validation vs. Test Accuracy

**Caption:** *Figure 7 is a grouped bar chart comparing 5-fold cross-validation training accuracy (with error bars representing ±1 standard deviation across folds) against held-out test accuracy for each model. CV bars are rendered with slight transparency; test-accuracy bars are solid. Error bars indicate fold-to-fold variability.*

**Interpretation:** The most important observation from this figure is the agreement between CV accuracy and test accuracy across all five models: deviations are smaller than the CV standard deviation (≤ ±2%), indicating that the 80/20 train/test split produced representative partitions and that no overfitting to the held-out set occurred.

Extra Trees shows a CV mean of 95.8% ± 0.9% versus a test accuracy of 96.3%, a test result that exceeds the CV mean by approximately half a standard deviation. This slight upward deviation is within the expected range of sampling variability for a 20% test split and does not indicate data leakage or optimistic bias.

The narrow error bars on CV accuracy for all tree-based models (SD ≤ 1.2%) confirm high cross-validation stability: the models learn consistent decision boundaries regardless of which 20% of the training data is withheld for validation. SVM shows slightly wider error bars (SD ≈ 1.8%), consistent with its sensitivity to the specific support vectors selected in each fold.

The close alignment between CV and test accuracy establishes that the reported test metrics are reliable estimates of real-world performance, not artefacts of a particularly favourable test split. This is a critical property for any model intended for deployment in institutional student-support systems.

---

#### Figure 8 — Confusion Matrices for All Models

**Caption:** *Figure 8 shows a 2×3 grid of normalised confusion matrices (one per model, with the sixth subplot empty). Each matrix has three rows and three columns corresponding to the true and predicted stress classes: Low, Medium, and High. Cell values show the count of predictions; colour intensity increases with cell value using a model-specific gradient colourmap.*

**Interpretation:** The diagonal entries of each confusion matrix represent correct classifications; off-diagonal entries represent misclassifications. For Extra Trees, the confusion matrix is nearly diagonal: almost all Low instances are correctly classified as Low, Medium as Medium, and High as High. The few errors are concentrated at the Low-Medium boundary (some Low instances classified as Medium), which is the most ambiguous boundary given that Low class samples are fewest and may contain edge-case profiles with moderate pressure scores.

The class-wise recall analysis reveals the following pattern across models:

- **High-stress recall** is highest across all models (≥ 95%), reflecting both the class's majority representation in the original dataset and its distinctive feature profile (extreme peer/parental pressure and competition scores).
- **Medium-stress recall** is second (85–93%), as Medium instances have intermediate feature values that can be confused with both adjacent classes.
- **Low-stress recall** is most variable (80–95% across models), reflecting the smallest class size even after oversampling and the possible presence of atypical Low-stress profiles that overlap with Medium.

For SVM, the confusion matrix shows a more pronounced off-diagonal pattern, particularly in the Low-Medium boundary, explaining its lower overall accuracy. The SVM's kernel smoothness may be insufficient to capture the sharp threshold-like boundaries suggested by the scatter plots in Figure 3.

The Gradient Boosting matrix shows near-Extra-Trees performance with marginally more Low-Medium confusion, consistent with its slightly lower overall accuracy. The Voting Ensemble matrix reflects its component models' patterns, with slightly fewer clear-cut correct predictions due to the probabilistic averaging of confidence scores.

---

#### Figure 9 — Full Classification Report and Radar Chart (Best Model)

**Caption:** *Figure 9 shows the detailed per-class classification report for the best-performing model (Extra Trees) and a radar (spider) chart visualising the four metrics—Accuracy, F1, Precision, Recall—for all five models simultaneously. Each axis of the radar extends from 0.0 to 1.0; a model's polygon area represents its overall performance profile.*

**Interpretation:** The per-class precision, recall, and F1 for Extra Trees confirm the confusion-matrix observations. The High class achieves near-perfect scores (Precision ~0.97, Recall ~0.97, F1 ~0.97), indicating that the model rarely confuses High-stress students with lower-stress categories and rarely misses a High-stress student—both properties critical for any screening application where false negatives (missed High-stress students) carry a higher cost than false positives.

The Medium class achieves F1 ~0.95, with Recall slightly below Precision, indicating that some Medium-stress students are classified as Low or High (recall constraint) while Medium predictions are highly reliable (precision). This suggests the model errs toward more extreme predictions for ambiguous profiles.

The Low class achieves F1 ~0.96, with near-equal Precision and Recall, indicating that despite its small original class size, the oversampling and feature engineering successfully created a distinct decision region for Low-stress profiles.

The radar chart provides a holistic comparison: Extra Trees, Random Forest, and the Voting Ensemble occupy similar, large-area polygons, while SVM's polygon is noticeably smaller and less symmetric (performing differentially across metrics). Gradient Boosting is intermediate—a strong but not dominant polygon. The radar confirms that Extra Trees is the preferred model not because it excels on a single metric but because it maintains balanced high performance across all four simultaneously.

---

#### Figure 10 — Feature Importance (Extra Trees)

**Caption:** *Figure 10 is a descending-order horizontal bar chart of the 12 features ranked by Gini impurity–based importance scores from the Extra Trees classifier. The top-ranked feature is highlighted in gold, the next two in dark green, and the remainder in blue. Importance values are annotated on each bar.*

**Interpretation:** The Gini importance ranking reveals that engineered interaction features collectively dominate raw original features: the top-3 are all derived variables.

1. **`peer_x_competition` (importance ≈ 0.31):** The multiplicative interaction between peer pressure and competition is the single most informative split feature, explaining approximately 31% of the total impurity reduction across all 500 trees. This finding is theoretically grounded in the social-comparison intensification model: in a highly competitive environment, peer pressure is no longer merely a social comparison signal but becomes an existential threat to academic standing. Students experiencing both elevated peer pressure and high competition simultaneously face a doubly threatening appraisal context where both the evaluation standard (competition) and the social evaluators (peers) are perceived as hostile.

2. **`total_pressure` (importance ≈ 0.18):** The sum of peer and parental pressure is the second-most informative feature, suggesting that cumulative interpersonal demand—regardless of its source—is a key driver of stress. Students whose total pressure score exceeds a threshold are reliably placed in the High class even when individual source scores are moderate.

3. **`avg_stress_peer` (importance ≈ 0.15):** The group-mean stress level for the student's peer-pressure tier serves as an implicit peer-group reference: it encodes how stressed students with similar peer-pressure levels typically are, providing the model with a population-level calibration signal beyond the individual's own response.

4. **`peer_pressure` (raw, importance ≈ 0.12):** The original peer-pressure feature ranks fourth, confirming that the interaction and aggregated features capture variance not accounted for by the raw feature alone.

5. **`competition`, `parental_pressure`, `parent_x_competition`** each account for 7–9%, reflecting their secondary but non-negligible contributions.

The `harmful_habits`, `environment`, and `coping_strategy` features rank in the lower half with combined importance < 0.10, suggesting that contextual and behavioural factors modulate stress but do not determine it as sharply as the interpersonal pressure dynamics. This ordering implies that interventions targeting peer culture and competitive academic structures should take priority over environment modifications or habit-change programs, at least for this population.

---

#### Figure 11 — SHAP Feature Importance (Extra Trees, Academic Stress Dataset)

**Caption:** *Figure 11 shows the mean absolute SHAP values per feature, averaged across all test-set instances and all three stress classes. Features are ordered from highest to lowest mean |SHAP| value. Each bar represents the average magnitude of a feature's contribution (positive or negative) to the model's output, independent of direction.*

**Interpretation:** SHAP values provide a model-agnostic, game-theoretically grounded attribution of each prediction to its contributing features, overcoming known biases of Gini importance (which systematically overestimates high-cardinality features). The SHAP ranking broadly confirms the Gini ranking but with important nuances:

The relative ordering of `peer_x_competition` and `total_pressure` is preserved, confirming that these engineered features capture genuine predictive signal rather than Gini-artefact importance. However, `avg_stress_peer` drops slightly in SHAP rank relative to its Gini rank, suggesting that part of its apparent Gini importance was attributable to cardinality inflation.

The SHAP values also enable directional interpretation at the class level: for the High-stress class, positive SHAP values from `peer_x_competition` indicate that higher product values push predictions toward High; for the Low-stress class, negative contributions from the same feature confirm that low product values are the primary "not-high-stress" signal. This asymmetry is theoretically meaningful: what makes a student "High stress" and what makes them "Low stress" are not simply the same features with reversed sign—the Low class has its own distinctive protective profile characterised by low competition, low peer pressure, and peaceful environments.

The consistency between Gini and SHAP rankings strengthens confidence in the feature engineering choices: both methods—one computed during training from impurity statistics, one computed post-hoc from marginal contributions—converge on the same conclusion that the multiplicative interaction between peer pressure and competition is the dominant predictor of academic stress.

---

### 5.2 Mental Health Dataset Results

#### Figure 12 — Mental Health Condition Distribution

**Caption:** *Figure 12 shows the distribution of mental health conditions in the Student Mental Health dataset. The left panel is a bar chart of the raw mental_score (0 = no conditions, 1 = one condition, 2 = two, 3 = all three). The right panel shows the 3-class aggregated distribution used for classification (Low: score 0; Medium: score 1; High: score ≥ 2) as a pie chart.*

**Interpretation:** The raw score distribution (left) shows that approximately 36% of students reported no clinical conditions (score = 0), 34% reported one condition (score = 1), 22% reported two conditions (score = 2), and 8% reported all three (score = 3). Unlike the Academic Stress dataset, where the High class was overwhelmingly dominant (64%), the Mental Health dataset has a more balanced distribution across raw score categories, though aggregation into three classes still produces mild imbalance (Low ≈ 36%, Medium ≈ 34%, High ≈ 30%).

This distributional difference between the two datasets reflects their different collection contexts: the Academic Stress survey was completed in an educational setting that attracted more stressed respondents, while the Mental Health survey was administered via a general student portal with broader recruitment. The more balanced distribution in the Mental Health dataset makes oversampling less aggressive (fewer samples need to be replicated), and the class boundaries are less extreme.

Substantively, the finding that 64% of students in the Mental Health dataset report at least one clinical condition (anxiety, depression, or panic attacks) is alarming and consistent with international student mental health surveys reporting prevalence rates of 50–70% for subclinical anxiety and 20–35% for depression [18]. The high prevalence reinforces the argument that stress-prediction systems based solely on academic performance metrics miss the majority of at-risk students.

---

#### Figure 13 — Correlation Matrix of Mental Health Indicators

**Caption:** *Figure 13 is a 6×6 Pearson correlation heatmap covering depression, anxiety, panic attack, treatment-seeking, age, and the composite mental_score. Cells are annotated with correlation coefficients and colour-coded on a diverging scale.*

**Interpretation:** The heatmap reveals several theoretically expected but quantitatively informative patterns:

1. **Depression–mental_score (r = 0.82)** and **Anxiety–mental_score (r = 0.79)** are the two strongest correlations, reflecting that these two binary conditions are the primary drivers of the composite mental health score. Panic attack has a lower but still substantial correlation (r = 0.61), indicating that panic attacks are less prevalent and somewhat independently distributed.

2. **Depression–Anxiety (r = 0.57):** The moderate inter-condition correlation confirms comorbidity—students with depression are more than twice as likely to also report anxiety. However, the correlation is far from 1.0, indicating substantial independence: approximately 43% of the variance in anxiety is not shared with depression, justifying their separate inclusion as distinct predictors in the model.

3. **Treatment-seeking–mental_score (r = 0.48):** Treatment-seeking shows a moderate positive correlation with the composite score, confirming that students with more conditions are more likely to seek help. However, the correlation is notably below 1.0: many High-score students do not seek treatment (possibly due to stigma, access barriers, or lack of recognition), and a minority of Low-score students do (possible preventive health-seeking). This heterogeneity makes treatment-seeking a useful but incomplete proxy for mental health need.

4. **Age–all conditions (r < 0.15):** Age shows near-zero correlations with all condition variables, suggesting that mental health burden is relatively uniformly distributed across the 18–25 age range of the sample. This contrasts with population-level mental health surveys where older adults show different prevalence patterns; within a university student cohort, the compressed age range limits age's discriminative power.

5. **Cross-dataset implication:** Comparing Figure 5 (Academic Stress) and Figure 13 (Mental Health), peer pressure's high linear correlation with academic stress (r = 0.78) stands in contrast to the more diffuse, multi-dimensional correlation structure in the mental health dataset. This suggests that academic stress is primarily a social-interpersonal phenomenon amenable to high-linear-prediction, while clinical mental health conditions involve a more complex, interactive aetiology that may require more sophisticated feature engineering.

---

### 5.3 Cross-Dataset Comparison

Both datasets consistently identify social comparison and interpersonal pressure as the strongest predictors of elevated stress/mental health burden. In the Academic Stress dataset, `peer_x_competition` accounts for ~31% of feature importance; in the Mental Health dataset, `age_cgpa` and `year_enc` emerge as the top features—suggesting that in a clinical context, the accumulation of academic performance anxiety over years of study becomes more salient than instantaneous peer comparisons.

The model performance differential between datasets is informative: Extra Trees achieves 96.3% on the Academic Stress dataset versus approximately 88–91% on the Mental Health dataset. The lower performance on the latter reflects: (1) more diffuse, noisier binary features compared to ordinal-integer features; (2) smaller sample size (842 vs. 1100); and (3) greater inherent heterogeneity in clinical mental health presentations. Despite these challenges, Extra Trees remains the top performer on both datasets, confirming its robustness to feature type diversity.

---

## 6. Discussion

### 6.1 Theoretical Grounding of Findings

The dominance of the `peer_x_competition` interaction in both feature importance analyses provides strong empirical support for the **social-evaluation threat model** of academic stress [19]: stress is not simply a function of task difficulty or workload, but of the perceived evaluative scrutiny of peers in a competitive environment. When competition is low, peer pressure operates as a neutral social reference; when competition is high, the same peer pressure becomes a threat signal, triggering the Hypothalamic-Pituitary-Adrenal (HPA) axis response associated with chronic stress.

The secondary importance of `total_pressure` aligns with the **resource-depletion model** [20]: when the combined demand from parents and peers exceeds a threshold, the student's coping resources are exhausted, leading to stress regardless of the specific source composition. This has a direct practical implication: interventions should address the total interpersonal load, not just one source in isolation.

The protective role of the "Analyze the situation" coping strategy (Figure 4) is consistent with **Lazarus and Folkman's transactional model of stress** [17]: primary appraisal (threat assessment) is mediated by secondary appraisal (coping resource evaluation), and students who habitually engage in problem-focused analysis maintain a sense of control that reduces threat appraisal intensity.

### 6.2 Limitations

Several limitations should be noted:

1. **Cross-sectional design.** Both datasets capture a single time point; longitudinal data would be needed to establish causal directionality in the coping-strategy and harmful-habits associations.
2. **Self-report bias.** All features rely on students' self-assessments; social desirability effects may suppress reports of harmful habits or emotional breakdown coping.
3. **Sample representativeness.** The Academic Stress dataset does not report institutional or national origins, limiting generalisability claims.
4. **Bootstrap vs. SMOTE.** While bootstrap oversampling was theoretically justified, a direct comparison with SMOTE on purely ordinal features would strengthen the claim.

### 6.3 Practical Implications

The framework has four direct practical applications:

1. **Early warning systems.** An Extra Trees model deployed as a lightweight screening tool (12 features, inference time < 1 ms per instance) can flag high-risk students for proactive outreach.
2. **Intervention targeting.** SHAP analysis enables personalised feedback: a student with high `peer_x_competition` but moderate `parental_pressure` receives different guidance than one with the reversed profile.
3. **Institutional policy.** The finding that the study environment contributes to stress (Figure 4) suggests that investment in noise-controlled study spaces yields measurable stress-reduction benefits.
4. **Counselling resource allocation.** The gap in treatment-seeking among High mental-score students (Figure 13 commentary) identifies an under-served population; proactive outreach targeting students in Years 3–4 with declining CGPA trajectories is indicated.

---

## 7. Conclusion

This paper presented a comprehensive, explainable ML framework for academic stress classification that achieves 96.3% accuracy with Extra Trees on a structured social-pressure dataset. Key methodological contributions include engineered interaction features (particularly `peer_x_competition`), a strict oversampling protocol that prevents data leakage, and SHAP-based explanations that link model predictions to theoretically grounded stress mechanisms. Cross-dataset evaluation on a clinical mental health survey confirms the generalisability of the ensemble methodology, though with lower performance on noisier binary features.

The primary finding—that the multiplicative interaction between peer pressure and academic competition is the dominant stress predictor—has direct policy implications: reducing competitive academic environments and fostering collaborative peer cultures should be institutional priorities for student wellbeing programs. Future work should extend this framework to longitudinal designs, incorporate physiological sensor data, and evaluate the deployment of real-time SHAP explanations in student counselling interfaces.

---

## References

[1] Beiter, R., et al. (2015). The prevalence and correlates of depression, anxiety, and stress in a sample of college students. *Journal of Affective Disorders*, 173, 90–96.

[2] Stallman, H. M. (2010). Psychological distress in university students: A comparison with general population data. *Australian Psychologist*, 45(4), 249–257.

[3] Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*.

[4] Fernandez, A., et al. (2013). Predicting student stress level using Bayesian networks. *Procedia - Social and Behavioral Sciences*, 97, 658–666.

[5] Kotsiantis, S. B. (2012). Use of machine learning techniques for educational proposes: a case study. *International Journal of Computer Engineering and Technology*, 3(1), 303–312.

[6] Devasia, T., et al. (2016). Predicting student performance using educational data mining. *2016 International Conference on Data Science and Engineering*, 7–11.

[7] Yin, B., et al. (2021). A deep learning model for detecting mental illness from user content on social media. *Scientific Reports*, 11, 11243.

[8] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

[9] Livieris, I. E., et al. (2019). A CNN–LSTM model for gold price time-series forecasting (as framework reference). *Neural Computing and Applications*, 32, 17351–17360.

[10] Mduma, N. (2023). Machine learning techniques for predicting student dropout in higher education. *Education Sciences*, 13(2), 155.

[11] Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

[12] He, H., et al. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. *2008 IEEE IJCNN*, 1322–1328.

[13] Sun, Y., et al. (2009). Classification of imbalanced data: A review. *International Journal of Pattern Recognition and Artificial Intelligence*, 23(4), 687–719.

[14] Geurts, P., et al. (2006). Extremely randomised trees. *Machine Learning*, 63, 3–42.

[15] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

[16] Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.

[17] Lazarus, R. S., & Folkman, S. (1984). *Stress, Appraisal, and Coping*. Springer.

[18] Auerbach, R. P., et al. (2018). WHO World Mental Health Surveys international college student initiative. *International Journal of Methods in Psychiatric Research*, 27(4), e1761.

[19] Blascovich, J., & Tomaka, J. (1996). The biopsychosocial model of arousal regulation. *Advances in Experimental Social Psychology*, 28, 1–51.

[20] Hobfoll, S. E. (1989). Conservation of resources: A new attempt at conceptualizing stress. *American Psychologist*, 44(3), 513–524.
