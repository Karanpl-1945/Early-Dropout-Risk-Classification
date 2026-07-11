# Early Student Dropout Risk Classification
## Complete Technical Documentation — Design Decisions & Methodology

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Leakage Investigation](#2-data-leakage-investigation)
3. [Feature Engineering](#3-feature-engineering)
4. [Train/Test Split](#4-traintest-split)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Class Imbalance Handling](#6-class-imbalance-handling)
7. [Baseline Model Comparison](#7-baseline-model-comparison)
8. [Imbalance Handling Comparison](#8-imbalance-handling-comparison)
9. [Hyperparameter Tuning with Optuna](#9-hyperparameter-tuning-with-optuna)
10. [Threshold Selection](#10-threshold-selection)
11. [Final Model Selection](#11-final-model-selection)
12. [Stacking Ensemble](#12-stacking-ensemble)
13. [Probability Calibration](#13-probability-calibration)
14. [Fairness and Subgroup Analysis](#14-fairness-and-subgroup-analysis)
15. [Explainability — SHAP](#15-explainability--shap)
16. [Explainability — LIME](#16-explainability--lime)
17. [Actionable Intervention Recommendations](#17-actionable-intervention-recommendations)
18. [Limitations and Future Work](#18-limitations-and-future-work)

---

## 1. Project Overview

### Problem
Schools need to identify students at risk of dropping out **before** it happens — early enough for counselors and teachers to intervene. A reactive system (acting after dropout) is useless. The goal is a proactive early-warning system.

### Dataset
- **Source:** Portuguese secondary school student performance data
- **Size:** 649 students, 34 original columns
- **Target:** `Dropped_Out` (boolean) — approximately 15% positive rate (imbalanced)
- **Features:** Demographics, family background, behavioural patterns, academic performance

### Why This Is Hard
- **Class imbalance:** Only ~15% of students drop out — a naive model that predicts "no dropout" for everyone gets 85% accuracy while being completely useless
- **Cost asymmetry:** Missing a student who will drop out (False Negative) is far more costly than flagging one who won't (False Positive)
- **Interpretability requirement:** A probability score alone is not enough — counselors need to know *why* a student is flagged

### Evaluation Metric: F2-Score
We use **F2-score** (not accuracy or F1) throughout the project.

```
F2 = (1 + 4) × Precision × Recall / (4 × Precision + Recall)
```

F2 weights **recall twice as heavily as precision**. This is domain-appropriate: missing a dropout (false negative) has severe real-world consequences, while a false positive means a student gets extra support they may not need — a much smaller cost.

---

## 2. Data Leakage Investigation

### What Was Found
The original baseline notebook showed **100% accuracy** — an immediate red flag. Investigation revealed that `Final_Grade = 0` for **100% of students who dropped out**.

```
Students with Final_Grade=0         : 15
Of those, who dropped out           : 15 (100%)
Among dropouts, fraction with grade=0: 100%
```

### Why This Is Leakage
`Final_Grade` is 0 *because* the student dropped out — it is not a predictor of dropout, it is a consequence of it. In a real deployment scenario, `Final_Grade` would not be available at the time of prediction (you want to identify at-risk students during the semester, before they drop out).

Including it gives the model a cheat code — it achieves 100% accuracy but is completely useless in practice.

### Fix
`Final_Grade` is removed from all features. `Grade_1` (first semester) and `Grade_2` (second semester) are retained as legitimate early-warning academic signals — these are available before dropout occurs.

---

## 3. Feature Engineering

Seven new features were derived from the existing columns to give the model richer, more explicit signals.

### Features Added

| Feature | Formula | Rationale |
|---|---|---|
| `Grade_Trend` | Grade_2 − Grade_1 | Captures trajectory — a declining student is more at risk than a consistently low student |
| `Low_Grade_Flag` | 1 if Grade_1 ≤ 5 or Grade_2 ≤ 5 | Hard binary indicator of severe academic underperformance |
| `Total_Alcohol` | Weekend + Weekday alcohol | Combined substance risk signal; both together matter more than either alone |
| `Max_Parent_Edu` | max(Mother_Edu, Father_Edu) | The higher-educated parent drives academic support; max is more decisive than average |
| `Academic_Risk` | Failures × 3 + Absences / 5 | Composite early-warning index combining the two strongest behavioural risk factors |
| `Study_Efficiency` | Grade_Avg / (Study_Time + ε) | Grades earned per unit of study effort — a student studying a lot but achieving little is flagged |
| `Grade_Avg` | (Grade_1 + Grade_2) / 2 | Used internally to compute Study_Efficiency — **not included as a model feature** |

### Why Grade_Avg Was Dropped as a Feature
`Grade_Avg = (Grade_1 + Grade_2) / 2` is a perfect linear combination of `Grade_1` and `Grade_2`. Since both raw grades are kept in the feature set, `Grade_Avg` adds zero new information to the model. It only splits the importance signal across three correlated columns. It is computed internally for `Study_Efficiency` but excluded from `X`.

### Why Average Parent Education Was Replaced by Maximum
Initially `(Mother_Edu + Father_Edu) / 2` was used. Domain reasoning suggests that the higher-educated parent has a disproportionate influence on academic guidance and aspiration. A family with one highly educated parent provides meaningfully different support than a family where both parents have low education. The maximum captures this asymmetry better than the average.

---

## 4. Train/Test Split

- **Split:** 80% training, 20% test
- **Method:** `train_test_split` with `stratify=y`
- **Why stratified:** With only ~15% positive rate, a random split could produce a test set with very few or very many dropout cases by chance. Stratification ensures both sets maintain the same class ratio.
- **Test set discipline:** The test set is used exactly once — at the very end after all model decisions are made. It is never used for hyperparameter selection, threshold selection, or any model comparison.

---

## 5. Preprocessing Pipeline

A `ColumnTransformer` applies different transformations to different column types:

### Categorical Columns (17 columns)
**One-Hot Encoding (OHE)** with `drop='first'`:
- Converts nominal categories (e.g., `Gender: F/M`, `School: GP/MS`) into binary columns
- `drop='first'` removes one category per feature to avoid multicollinearity (dummy variable trap)
- `handle_unknown='ignore'` prevents errors if unseen categories appear at prediction time

### Numerical Columns (22 columns)
**StandardScaler:**
- Scales features to zero mean and unit variance
- Required for Logistic Regression (distance-based) in baseline comparison
- Tree models (XGBoost, RF) don't strictly need scaling but it doesn't hurt and ensures consistent treatment across all models

### Why the Preprocessor Lives Inside the Pipeline
The preprocessor is **not** fitted on the full training set and then passed around. Instead, it is embedded inside every `ImbPipeline` alongside the sampler and classifier. This ensures:
- In cross-validation, the preprocessor is fitted on training folds only — never sees validation fold data
- After Optuna tuning, the final preprocessor is fitted on the full training set inside the final pipeline
- No data leakage from the scaling statistics

---

## 6. Class Imbalance Handling

### The Problem
With ~15% dropout rate, training a model without any imbalance handling causes it to optimise for the majority class. It learns to say "not at risk" for most students, producing very low recall on the minority (dropout) class.

### Why SMOTENC — Not BorderlineSMOTE or Standard SMOTE

#### Standard SMOTE
- Creates synthetic minority samples by interpolating between real samples
- Works correctly for **purely numerical** data
- **Problem for this dataset:** Interpolating between categorical values produces meaningless results (e.g., 0.4 × "teacher" + 0.6 × "other" has no meaning)

#### BorderlineSMOTE
- A variant of SMOTE that focuses on borderline samples near the decision boundary
- Also requires **numerical input** — it computes k-nearest neighbours using Euclidean distance
- **Cannot be applied before OHE** on raw data containing categorical strings
- **Applied after OHE (old approach):** Creates fractional values in binary OHE columns (e.g., `Gender_M = 0.3`). A student cannot be 30% male. This violates the categorical constraint.

#### Why SMOTENC Is Correct
**SMOTENC (SMOTE for Nominal and Continuous features)** is specifically designed for datasets with mixed types:
- **Numerical columns:** Interpolated normally between neighbours (standard SMOTE behaviour)
- **Categorical columns:** The synthetic sample inherits the **most frequent category** among k nearest neighbours — no interpolation, no fractional values

Critically, SMOTENC operates on **raw data before OHE**, so the correct pipeline order is:

```
Raw Data → SMOTENC → OneHotEncoder → Model    ✓ Correct

Raw Data → OneHotEncoder → BorderlineSMOTE → Model    ✗ Interpolates binary columns
```

Our dataset has 17 categorical columns including School, Gender, Address, Mother_Job, Father_Job, Parental_Status, and others. SMOTENC handles all of them correctly.

### Why Inside the Pipeline (Not Before)
SMOTENC is placed inside `ImbPipeline` for every model and every cross-validation fold. This is critical:
- During `fit`, SMOTENC resamples the training fold
- During `predict`/`predict_proba`, SMOTENC is automatically skipped — it never touches test data
- This prevents synthetic samples from leaking into validation/test folds, which would inflate performance estimates

### class_weight vs SMOTE-based approaches
`class_weight='balanced'` is a simpler alternative — it adjusts the loss function to penalise misclassifying the minority class more heavily. It does not create synthetic samples. While useful, our comparison (Part 8) showed that SMOTENC achieves higher F2-score by actually augmenting the minority class representation rather than just reweighting.

---

## 7. Baseline Model Comparison

### Models Compared
Five models are evaluated in stratified 5-fold cross-validation:

| Model | Why Included |
|---|---|
| Logistic Regression | Linear baseline — interpretable, good calibration |
| Decision Tree | Non-linear but interpretable; prone to overfitting |
| Random Forest | Ensemble of trees — robust, handles non-linearity |
| Gradient Boosting | Sequential ensemble — strong performer |
| XGBoost | State-of-the-art gradient boosting with regularisation |

### Why 5-Fold Stratified CV
- A single train/validation split gives a noisy estimate of generalisation performance
- 5-fold CV uses all training data for both training and validation across folds
- `stratify` ensures each fold maintains the class ratio

### Key Results
Random Forest (F2 = 0.803) and XGBoost (F2 = 0.787) consistently outperformed the others. Both are tree-based ensemble models — suitable for SHAP TreeExplainer, robust to the feature scale, and strong on tabular data.

---

## 8. Imbalance Handling Comparison

A dedicated comparison isolates the effect of the imbalance strategy by keeping all other variables fixed (Random Forest, default hyperparameters, same CV):

| Strategy | Description |
|---|---|
| No handling | Raw class distribution, no adjustment |
| class_weight=balanced | Loss reweighting only |
| SMOTE after OHE (old) | Synthetic samples in OHE space |
| BorderlineSMOTE after OHE | Borderline samples in OHE space |
| SMOTENC before OHE (correct) | Synthetic samples in raw feature space |

This comparison justifies the SMOTENC choice with actual F2 numbers rather than just theoretical reasoning.

---

## 9. Hyperparameter Tuning with Optuna

### Why Optuna over GridSearchCV / RandomizedSearchCV

**GridSearchCV:** Exhaustively tries every combination. With 9 XGBoost parameters each with multiple values, the search space is enormous. Completely impractical.

**RandomizedSearchCV:** Randomly samples parameter combinations. Better than grid search, but treats each trial independently — it has no memory of which combinations worked well and cannot focus on promising regions.

**Optuna (TPE — Tree-structured Parzen Estimator):**
- Bayesian optimisation — each trial uses the results of all previous trials to choose the next set of parameters
- Builds a probabilistic model of which parameter values produce good scores
- Focuses sampling in the most promising regions of the search space
- Handles continuous parameters (e.g., `learning_rate`) without binning — unlike grid search which requires a predefined list of values
- 60 trials with Optuna typically outperforms 200 trials with random search

### Two Separate Studies
Both XGBoost and Random Forest are tuned independently with their own 60-trial Optuna studies. The winner is selected automatically based on the best CV F2-score from each study.

### XGBoost Parameters Tuned

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of trees |
| `max_depth` | Tree depth — controls complexity |
| `min_child_weight` | Minimum samples in leaf — prevents overfitting |
| `learning_rate` | Shrinkage factor — controls contribution of each tree |
| `gamma` | Minimum loss reduction for a split |
| `subsample` | Fraction of training samples per tree |
| `colsample_bytree` | Fraction of features per tree |
| `reg_alpha` | L1 regularisation |
| `reg_lambda` | L2 regularisation |

`reg_alpha` and `reg_lambda` are particularly important for small datasets (649 students) — they prevent the model from memorising training data.

### Random Forest Parameters Tuned
`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

---

## 10. Threshold Selection

### The Default Threshold Problem
Classification models output a probability (e.g., 0.73). A threshold converts this to a binary prediction. The default threshold of 0.50 means: predict dropout if probability > 50%.

For imbalanced datasets with asymmetric costs, 0.50 is rarely optimal. A lower threshold catches more dropouts (higher recall) at the cost of more false positives.

### Why F2-Score for Threshold Selection (Not F1)
Initially, F1-optimal threshold selection was used. This pushed the threshold to 0.97 — an extremely conservative predictor that **increased false negatives from 2 to 7**. F1 balances precision and recall equally, which does not reflect the domain cost structure.

**F2-score** (β=2) weights recall twice as heavily as precision. The F2-optimal threshold finds the cutoff that maximises this domain-appropriate metric — prioritising catching dropouts over avoiding false alarms.

### Why Validation Set (Not Test Set)
Threshold selection is done on a held-out validation split carved from the training data:

```
Training data (80%) → split → Sub-train (64%) + Validation (16%)
```

The model is refitted on sub-train, probabilities are evaluated on validation, and the F2-optimal threshold is found on validation. Only then is this threshold applied once to the test set.

**Why this matters:** If the threshold were selected on the test set, the test set would have influenced a model decision — it would no longer be a true holdout and would give an optimistically biased performance estimate.

---

## 11. Final Model Selection

After both Optuna studies complete:
- The XGBoost study's best CV F2 score is compared against the Random Forest study's best CV F2 score
- The winner is selected automatically
- The winning pipeline is retrained on the **full training set** (not just the sub-train used for threshold selection)
- After fitting, the preprocessor's fitted state is extracted to transform data for SHAP and LIME

---

## 12. Stacking Ensemble

### What Is Stacking
Stacking trains multiple base models and combines their predictions using a meta-learner. The meta-learner learns *when to trust each base model* rather than just averaging.

### Base Models
- **XGBoost** — strong gradient boosting
- **Random Forest** — diverse bagging ensemble
- **ExtraTreesClassifier** — more randomised variant of RF; adds diversity

Diversity among base models is key — if all base models make the same errors, stacking cannot improve. XGBoost, RF, and ExtraTrees make errors in different situations, giving the meta-learner something to work with.

### Meta-Learner
**Logistic Regression** with `class_weight='balanced'`:
- Simple and interpretable
- Takes the probability outputs of base models as inputs
- Learns a weighted combination of their predictions
- Avoids overfitting at the meta-level

### Why SMOTENC in the Stacking Pipeline
The same methodology applies — SMOTENC → OHE → StackingClassifier. The stacking classifier receives preprocessed (OHE'd, scaled) features after the pipeline preprocessor step.

---

## 13. Probability Calibration

### The Problem
Tree-based models (XGBoost, RF) are often **overconfident** — they tend to push probabilities toward the extremes (close to 0 or 1) more than the true underlying rate justifies. A model that says "98% dropout risk" when the true rate is 70% leads to misplaced confidence in intervention decisions.

### Why This Matters
- Risk tiers (LOW / MEDIUM / HIGH) are assigned from raw probabilities
- Counselors make decisions based on probability statements
- A school principal reading "Student X has 98% dropout risk" expects that to be true ~98% of the time

### Fix: Platt Scaling (Sigmoid Calibration)
`CalibratedClassifierCV` with `method='sigmoid'` and `cv='prefit'`:
- Fits a logistic (sigmoid) layer on top of the existing model's probability outputs
- Trained on the same validation set used for threshold selection
- Maps overconfident scores to more reliable probability estimates
- Does not change the ranking order of predictions — only rescales the probability values

### Reliability Diagram
A reliability diagram (calibration curve) plots mean predicted probability vs actual fraction of positives. A perfectly calibrated model's points fall exactly on the diagonal. Points below the diagonal indicate overconfidence; points above indicate underconfidence.

### Brier Score
The Brier score measures mean squared error of probability predictions (lower = better). Improvement in Brier score after calibration quantifies how much more reliable the probabilities become.

---

## 14. Fairness and Subgroup Analysis

### Why Fairness Analysis Is Necessary
A model with good overall recall can still systematically miss dropout-risk students from specific demographic groups. Before deploying any education tool, it is ethically essential to verify the model is not biased by gender, school, age, or location.

### Subgroups Analysed
- **Gender:** Male vs Female
- **School:** GP vs MS
- **Address:** Urban (U) vs Rural (R)
- **Age Group:** 15–17, 18–19, 20+

### Key Metric: Recall Per Subgroup
For each subgroup, recall measures: "Of all students in this group who actually dropped out, what fraction did the model correctly flag?" A gap in recall between subgroups indicates potential bias.

### Findings
The model achieved balanced recall across gender. However, performance varied across school, address, and age:
- The model performs better for MS school than GP school
- The model performs better for rural students than urban students
- Younger students (15–17) had the lowest recall

The largest gap was in the Address subgroup — urban students had lower recall and a higher false-negative rate.

### Why Group-Wise Thresholds Were Not Applied
Applying a lower threshold for flagged subgroups (to increase their recall) was considered but rejected. Several subgroups contain very few dropout cases in the validation set (e.g., GP school: 7 dropouts, Age 15–17: 12 dropouts). With so few positive examples, any threshold selected on the validation set would overfit to noise and not generalise reliably. The global threshold selected on validation data is used for all students. Subgroup metrics are reported as a fairness diagnostic only.

### Bias Threshold
A recall gap > 0.15 across subgroups is flagged as potential bias requiring investigation.

---

## 15. Explainability — SHAP

### What Is SHAP
**SHAP (SHapley Additive exPlanations)** is based on cooperative game theory. It assigns each feature a contribution value for a specific prediction:
- **Positive SHAP value:** Feature pushes prediction toward dropout
- **Negative SHAP value:** Feature pushes prediction away from dropout
- SHAP values sum to the difference between the model's prediction and the average prediction (base rate)

### Why TreeExplainer
For tree-based models (XGBoost, RF), `shap.TreeExplainer` computes exact SHAP values efficiently by traversing tree paths. It is faster and more accurate than the model-agnostic KernelExplainer.

### Plots Produced

| Plot | What It Shows |
|---|---|
| Global bar chart | Mean absolute SHAP value per feature — overall feature importance |
| Beeswarm plot | Distribution of SHAP values per feature across all students — shows direction and spread |
| Waterfall plot | Individual student breakdown — exactly how each feature contributed to their specific prediction |
| Dependence plot | How SHAP value for a feature changes with feature value — reveals non-linear relationships |

### Waterfall: High-Risk vs Low-Risk Comparison
Two waterfall plots are generated — one for the highest-risk student and one for the lowest-risk student. A side-by-side bar chart then compares their top SHAP drivers. This reveals what makes a student high-risk vs low-risk in terms of specific feature contributions.

---

## 16. Explainability — LIME

### What Is LIME
**LIME (Local Interpretable Model-agnostic Explanations)** fits a simple linear model in the neighbourhood of a specific data point. It is:
- **Model-agnostic:** Works with any classifier
- **Local:** Explains one prediction at a time, not the global model

### Why LIME Alongside SHAP
SHAP and LIME approach explainability differently:
- SHAP uses game-theoretic exact attributions from the tree structure
- LIME uses a locally fitted linear approximation

When both agree on which features drove a prediction, the finding is more trustworthy. When they disagree, it signals the model's decision boundary is complex in that region and warrants closer inspection.

LIME also provides an independent cross-check that does not rely on tree structure — useful for validating SHAP findings from a different methodological angle.

---

## 17. Actionable Intervention Recommendations

### The Gap Between XAI and Action
SHAP can tell a counselor that `Academic_Risk` has a SHAP value of +0.45 for a specific student. A counselor who is not a data scientist cannot translate that into action.

### What Was Built
A rule table maps each risk-driving feature to a concrete intervention:

| Feature Condition | Recommended Action |
|---|---|
| Grade_1 or Grade_2 ≤ 8 | Assign peer tutor immediately |
| Number_of_Absences > 10 | Contact parents, set up attendance monitoring plan |
| Number_of_Failures ≥ 2 | Refer to remedial class, schedule monthly progress reviews |
| Study_Time = 1 (very low) | Enrol in structured study skills workshop |
| Total_Alcohol ≥ 6 | Refer to school counsellor for welfare check |
| Wants_Higher_Education = no | Career counselling and motivational support |
| Health_Status ≤ 2 | Connect with school health services |
| Family_Relationship ≤ 2 | Involve school social worker |
| Going_Out ≥ 4 | Time management discussion |

For each high-risk student, only the interventions whose conditions are actually triggered appear — the system does not give generic advice.

### Output Format
```
Student #7  |  Risk: HIGH  |  P(dropout) = 97%  |  Actual: DROPOUT
═══════════════════════════════════════════════════════════════════
  [Grade_2 = 3  |  SHAP +1.95]
    → Very low Grade 2 — schedule urgent academic support.

  [Academic_Risk = 9.4  |  SHAP +0.18]
    → High combined Academic Risk — prioritise for weekly check-ins.
```

This makes the model output directly actionable for non-technical school staff.

---

## 18. Limitations and Future Work

### Current Limitations

**Dataset size:** 649 students is small. Model performance estimates have wide confidence intervals. Subgroup analysis (particularly for smaller groups like older students) has high variance.

**Single school system:** The dataset comes from Portuguese secondary schools. The dropout risk patterns (which features matter most) may differ significantly across different countries, education systems, and cultural contexts. The model should not be applied to other school systems without revalidation.

**SMOTENC validation set:** The same validation set is used for both threshold selection and Platt calibration. Ideally, a separate calibration set would be used. With 649 students, carving out three separate subsets (train, threshold-val, calibration-val, test) is not feasible without making each set too small to be reliable.

**Static model:** The model is trained once and applied statically. In practice, dropout risk changes throughout the school year. A model trained on one academic year's data may not reflect changes in student circumstances.

**Subgroup fairness gaps:** The model shows lower recall for urban students and younger students (15–17). Group-wise threshold tuning was not applied due to insufficient data per subgroup. This is an unresolved fairness limitation.

**SMOTENC in stacking:** The StackingClassifier's internal cross-validation for out-of-fold predictions does not re-apply SMOTENC at each internal fold — it uses the already-resampled data passed to it by the pipeline. This is a minor methodological imperfection.

### Future Work

1. **Larger dataset** — Collect data from multiple schools and academic years. Larger data would enable reliable subgroup-wise threshold tuning and better generalisation.

2. **Group-wise threshold calibration** — With sufficient data per subgroup, apply separate thresholds for urban/rural, GP/MS, and age groups to equalise recall across subgroups.

3. **Longitudinal modelling** — Update risk scores throughout the academic year as new data arrives (attendance records, mid-term grades) rather than making a single prediction at the start.

4. **SMOTENC + BorderlineSMOTE hybrid** — Explore whether a borderline-focused variant of SMOTENC (focusing synthetic generation on borderline minority samples) would improve performance further.

5. **Calibration holdout set** — With a larger dataset, use a separate calibration set distinct from the threshold validation set for Platt scaling.

6. **Explainability for counselors** — Build a simple web interface that shows a student's risk tier, SHAP waterfall, and intervention recommendations without requiring any technical knowledge.

7. **Feedback loop** — Track which interventions were applied and whether they successfully prevented dropout. Use this data to validate and improve the recommendation rules.

---

## Summary Table

| Design Decision | Choice Made | Rejected Alternative | Reason |
|---|---|---|---|
| Leakage fix | Remove Final_Grade | Keep Final_Grade | It is caused by dropout, not a predictor of it |
| Parent education | Max(mother, father) | Average | One highly-educated parent provides disproportionate support |
| Grade_Avg | Dropped from features | Kept | Perfect linear combination of Grade_1 and Grade_2; adds no information |
| Imbalance handling | SMOTENC before OHE | BorderlineSMOTE after OHE | SMOTENC handles categorical features correctly; BorderlineSMOTE creates fractional binary values |
| Evaluation metric | F2-score | F1, Accuracy | Recall is twice as important — missing a dropout is costlier than a false alarm |
| Hyperparameter tuning | Optuna (TPE) | RandomizedSearchCV | Bayesian search learns from previous trials; random search does not |
| Threshold selection | F2-optimal on validation set | F1-optimal on test set | F1 increases FN; test set must remain untouched |
| Model explainability | SHAP + LIME | Single method | Independent methods cross-validate each other |
| Group-wise thresholds | Not applied | Applied per subgroup | Subgroups too small; would overfit to validation noise |
| Calibration | Platt scaling | No calibration | Tree models are overconfident; probabilities need to be trustworthy |
