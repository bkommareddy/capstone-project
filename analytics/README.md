# Module 2 — Analytics Pipeline

Profiles and cleans the Titanic dataset, tells a visual data story about survival,
then builds, tunes, and evaluates a full classification + regression modeling
pipeline on the same cleaned data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` needs internet access the first time (`sns.load_dataset` fetches
and caches the dataset). `titanic.csv`, committed in this folder, is the
offline fallback — grading can run `02_modeling.py` from it directly even
with no network access, and `02_modeling.py` never calls `sns.load_dataset()`
itself.

Outputs (all in this folder):
- `titanic.csv` — the one cleaned dataset both parts work from
- `charts/*.png` — every chart described below
- `eda_report.txt` / `modeling_report.txt` — full captured console output from each script
- `best_pipeline.joblib` — the saved, fitted, end-to-end best pipeline

---

## Part A — Profiling, cleaning, and the data story

### Missing values

| Column | % missing | Strategy |
|---|---|---|
| `deck` | 77.22% | **Dropped the column.** Far above the 30% band where imputation becomes unreliable, and the information is largely redundant with `pclass`/`fare` (cabin deck roughly tracks ticket class). |
| `age` | 19.87% | **Imputed with the median** (28.0). Falls in the 5–30% band; median is robust to the right-skew we later confirm in `fare`-adjacent numeric columns. |
| `embarked` | 0.22% | **Dropped the 2 affected rows.** Under 5%, per the threshold rule. |
| `embark_town` | 0.22% | Same 2 rows as `embarked` (they're missing together) — no additional rows dropped. |

Shape after cleaning: **889 rows × 14 columns** (from the original 891×15).

### Univariate analysis — age & fare

![age/fare histograms and box plots](charts/01_univariate_age_fare.png)

- **IQR outliers**: `age` has **65** outliers (bounds [2.50, 54.50]); `fare` has **114** outliers (bounds [−26.76, 65.66]) — fare has nearly double the outlier count relative to its scale, consistent with a long right tail of expensive first-class tickets.
- **Fare skew**: mean = 32.10, median = 14.45, mode = 8.05. Since **mean > median > mode**, `fare` is clearly **right-skewed** — a small number of very expensive tickets pull the mean well above the typical (median) fare.

### Bivariate analysis

- **Survival by sex**: female **0.740**, male **0.189** — sex is the single strongest visible predictor of survival ("women and children first" plays out directly in the data).
- **Survival by pclass**: 1st **0.626**, 2nd **0.473**, 3rd **0.242** — survival drops steadily with ticket class, i.e. with cabin proximity to lifeboats and boarding priority.
- **Survival by sex + pclass**: female/1st **0.967**, female/2nd **0.921**, female/3rd **0.500**, male/1st **0.369**, male/2nd **0.157**, male/3rd **0.135**. The two factors compound: a 1st-class woman was almost certain to survive (97%), while a 3rd-class man had only a 13.5% chance — a ~7x gap driven by the interaction of sex and class, not either alone.

**Correlation heatmap** (6×6, on `survived, pclass, age, sibsp, parch, fare`; `adult_male`/`alone` excluded as redundant derived flags):

![correlation heatmap](charts/02_correlation_heatmap.png)

Top 2 strongest correlations by |r|:
1. **`pclass` ↔ `fare`: r = −0.548** — expected and mechanical: lower `pclass` number (better class) means a higher fare, so the two are strongly (negatively, since 1=best) correlated by construction of the ticketing system.
2. **`sibsp` ↔ `parch`: r = 0.415** — passengers traveling with more siblings/spouses also tend to travel with more parents/children, i.e. family size components move together, as you'd expect from people traveling as family units rather than independently.

### Multivariate data story (4 charts)

**Chart A — Survival rate by class and sex**
![survival by class and sex](charts/03_survival_by_class_sex.png)
Survival rate is highest for women across every class and drops sharply for men, but class still matters within each sex — women in 3rd class survive far less often than women in 1st/2nd. This confirms sex and class act as compounding, not competing, factors.

**Chart B — Age distribution by survival and sex**
![age by survival and sex](charts/04_age_by_survival_sex.png)
Median ages look broadly similar across survival outcomes for both sexes, meaning age alone is a much weaker survival signal here than sex or class — consistent with the near-zero `age`↔`survived` correlation (−0.07) seen in the heatmap.

**Chart C — Age vs fare, colored by survival, styled by class**
![age vs fare survival](charts/05_age_vs_fare_survival.png)
Survivors (colored points) cluster more densely at higher fares, and higher-fare points are overwhelmingly 1st-class markers — visually tying together the fare/class/survival relationship already seen numerically, while age shows no clear separation between survivors and non-survivors.

**Chart D — Survival rate by embarkation port**
![survival by embarked](charts/06_survival_by_embarked.png)
Passengers who embarked at Cherbourg ('C') have a noticeably higher survival rate than those from Southampton ('S') or Queenstown ('Q') — this likely reflects that Cherbourg boarders skewed toward 1st class rather than embarkation port having any causal effect on survival itself.

### Exploratory standardization check (EDA-only)

Before: `age` mean = 29.315, std = 12.985; `fare` mean = 32.097, std = 49.698.
After z-scoring both: mean ≈ 0.0, std ≈ 1.0 for both columns — confirming the transform is correct.
**This standardized version is not saved into `titanic.csv` or used downstream** — the modeling pipeline in Part B fits its own `StandardScaler` on the training split only, as required.

---

## Part B — Predictive modeling

### Train/test split

Stratified 80/20 split on `survived` (train: 711 rows, test: 178 rows). Justification: the overall class balance is imbalanced (**61.8% did not survive / 38.2% survived**), so a plain random split risks over/under-representing the minority (survived) class in either split and skewing evaluation metrics; stratifying preserves the ~62/38 ratio in both train and test (confirmed: train 61.7/38.3, test 61.8/38.2).

### Preprocessing

Built with a `ColumnTransformer` + `Pipeline` so fit/transform separation is structural, not manual:
- Numeric (`age`, `sibsp`, `parch`, `fare`): median imputation → `StandardScaler`
- Categorical (`sex`, `embarked`): most-frequent imputation → `OneHotEncoder`
- `pclass`: passed through unchanged (already a small ordinal integer)

### Classifier comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | **0.861** |
| Decision Tree | 0.764 | 0.760 | 0.559 | 0.644 | 0.837 |
| Random Forest | 0.809 | 0.766 | **0.721** | **0.742** | 0.825 |

### Imbalance handling comparison (Random Forest)

Training-fold class balance: 61.7% / 38.3%.

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | 0.766 | 0.721 | **0.742** |
| `class_weight='balanced'` | 0.762 | 0.706 | 0.733 |
| SMOTE (train fold only) | 0.731 | 0.721 | 0.726 |

**Conclusion**: the baseline (no special handling) actually edged out both imbalance-correction strategies on F1. The imbalance here (62/38) is mild enough that Random Forest's built-in robustness to moderate imbalance already handles it reasonably well; `class_weight='balanced'` and SMOTE both trade a bit of precision for essentially flat or slightly lower recall in this case, rather than clearly helping. This suggests imbalance-correction is most valuable at more extreme skews than the one present in this dataset.

### Hyperparameter tuning (GridSearchCV, Random Forest)

Best params: `max_depth=4, max_features='sqrt', n_estimators=200` (best CV F1 = 0.747).
**OOB score of the tuned Random Forest: 0.820.**

### Regression side-task — predicting fare

MAE = 21.099, RMSE = 41.702, R² = 0.348, Adjusted R² = 0.321.

![residual plot](charts/09_residual_plot.png)

### Final model comparison and recommendation

| Metric group | Metric | Value |
|---|---|---|
| **Classification** (Random Forest, recommended) | Accuracy | 0.809 |
| | Precision | 0.766 |
| | Recall | 0.721 |
| | F1 | 0.742 |
| | AUC | 0.825 |
| **Regression** (fare prediction) | MAE | 21.099 |
| | RMSE | 41.702 |
| | R² | 0.348 |
| | Adjusted R² | 0.321 |

*(Classification and regression metrics are on different scales and are not directly comparable — kept as two separate metric groups above, not a single merged scale.)*

**Recommendation**: deploy the **Random Forest** classifier. It ties Logistic Regression on accuracy (0.809) but has the best F1 (0.742) of the three, with the best recall (0.721) — meaning it misses fewer actual survivors than the alternatives, which matters if false negatives (predicting someone didn't survive when they did) are the costlier error in this use case. Logistic Regression has a higher AUC (0.861 vs 0.825), so it ranks probabilities slightly better overall, but Random Forest's higher F1 at the default threshold makes it the stronger choice for a fixed-threshold deployed classifier. Decision Tree trails both on every metric and is kept mainly for its interpretable visualization.

### Saved pipeline

`best_pipeline.joblib` contains the complete fitted `Pipeline` (preprocessing `ColumnTransformer` + the Random Forest estimator) via `joblib.dump`. Reload check in `02_modeling.py` confirms `joblib.load(...).predict(...)` works directly on raw (unpreprocessed) test rows and reproduces correct predictions.
