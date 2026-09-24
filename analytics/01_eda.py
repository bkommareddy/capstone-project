

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CHART_DIR = "charts"

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def main():
    import os
    os.makedirs(CHART_DIR, exist_ok=True)

    log_file = open("eda_report.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    section("1. LOAD + PROFILE")
    df = sns.load_dataset("titanic")

    print("\ndf.info()")
    df.info()

    print("\ndf.describe()")
    print(df.describe(include="all"))

    print(f"\nShape: {df.shape}")

    print("\nMissing values (%):")
    missing_pct = (df.isna().mean() * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    print(missing_pct)
    df.to_csv("titanic_raw.csv", index=False)
    section("2. MISSING VALUE HANDLING")

    df_clean = df.copy()

    for col, pct in missing_pct.items():
        if col == "deck":
            df_clean = df_clean.drop(columns=["deck"])
        elif col == "age":
            median_age = df_clean["age"].median()
            df_clean["age"] = df_clean["age"].fillna(median_age)
        elif col in ("embarked", "embark_town"):
            df_clean = df_clean.dropna(subset=[col])
        else:
            if pct < 5:
                before = len(df_clean)
                df_clean = df_clean.dropna(subset=[col])
            elif pct <= 30:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_val = df_clean[col].median()
                else:
                    fill_val = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(fill_val)
            else:
                df_clean = df_clean.drop(columns=[col])
    print(f"\nShape after cleaning: {df_clean.shape}")
    print("Remaining missing values:\n", df_clean.isna().sum()[df_clean.isna().sum() > 0])
    df_clean.to_csv("titanic.csv", index=False)
    print("\nSaved cleaned data -> titanic.csv")
    section("3. UNIVARIATE ANALYSIS — age & fare")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0]).set_title("Age — histogram")
    sns.boxplot(x=df_clean["age"], ax=axes[0, 1]).set_title("Age — box plot")
    sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0]).set_title("Fare — histogram")
    sns.boxplot(x=df_clean["fare"], ax=axes[1, 1]).set_title("Fare — box plot")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/01_univariate_age_fare.png", dpi=120)
    plt.close()

    def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        return len(outliers), lower, upper

    age_out_n, age_lo, age_hi = iqr_outliers(df_clean["age"])
    fare_out_n, fare_lo, fare_hi = iqr_outliers(df_clean["fare"])
    print(f"\nAge IQR outliers: {age_out_n} (bounds: [{age_lo:.2f}, {age_hi:.2f}])")
    print(f"Fare IQR outliers: {fare_out_n} (bounds: [{fare_lo:.2f}, {fare_hi:.2f}])")

    fare_mean = df_clean["fare"].mean()
    fare_median = df_clean["fare"].median()
    fare_mode = df_clean["fare"].mode().iloc[0]
    print(f"\nFare — mean: {fare_mean:.2f}, median: {fare_median:.2f}, mode: {fare_mode:.2f}")
    skew_direction = "right-skewed" if fare_mean > fare_median else (
        "left-skewed" if fare_mean < fare_median else "symmetric"
    )
    print(f"Fare distribution: {skew_direction}")
    section("4. BIVARIATE ANALYSIS")

    survival_by_sex = df_clean[df_clean["sex"] == "female"]["survived"].mean(), \
                       df_clean[df_clean["sex"] == "male"]["survived"].mean()
    print(f"\nSurvival rate by sex — female: {survival_by_sex[0]:.3f}, male: {survival_by_sex[1]:.3f}")

    survival_by_class = {}
    for pc in sorted(df_clean["pclass"].unique()):
        survival_by_class[pc] = df_clean[df_clean["pclass"] == pc]["survived"].mean()
    print(f"Survival rate by pclass: { {k: round(v,3) for k,v in survival_by_class.items()} }")

    print("\nSurvival rate by sex AND pclass:")
    for sex in ["female", "male"]:
        for pc in sorted(df_clean["pclass"].unique()):
            mask = (df_clean["sex"] == sex) & (df_clean["pclass"] == pc)
            rate = df_clean[mask]["survived"].mean()
            print(f"  sex={sex}, pclass={pc}: survival rate = {rate:.3f} (n={mask.sum()})")

    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr_matrix = df_clean[corr_cols].corr()
    print("\nCorrelation matrix (6x6):\n", corr_matrix.round(3))

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap (survived, pclass, age, sibsp, parch, fare)")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/02_correlation_heatmap.png", dpi=120)
    plt.close()
    pairs = []
    for i, c1 in enumerate(corr_cols):
        for j, c2 in enumerate(corr_cols):
            if i < j:
                pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\nTop 2 strongest correlations (by |r|):")
    for c1, c2, r in pairs[:2]:
        print(f"  {c1} <-> {c2}: r = {r:.3f}")
    section("5. MULTIVARIATE DATA STORY")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", ax=ax)
    ax.set_title("Chart A: Survival rate by class and sex")
    ax.set_ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/03_survival_by_class_sex.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df_clean, x="survived", y="age", hue="sex", ax=ax)
    ax.set_title("Chart B: Age distribution by survival and sex")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/04_age_by_survival_sex.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", style="pclass", alpha=0.7, ax=ax)
    ax.set_title("Chart C: Age vs fare, colored by survival, styled by class")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/05_age_vs_fare_survival.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    embark_survival = df_clean.groupby("embarked")["survived"].mean().sort_values(ascending=False)
    embark_survival.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Chart D: Survival rate by embarkation port")
    ax.set_ylabel("Survival rate")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/06_survival_by_embarked.png", dpi=120)
    plt.close()

    print("Saved multivariate charts.")
    section("6. STANDARDIZATION CHECK")

    print("\nBefore standardization:")
    print(df_clean[["age", "fare"]].agg(["mean", "std"]).round(3))

    age_z = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
    fare_z = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()
    standardized = pd.DataFrame({"age_z": age_z, "fare_z": fare_z})

    print("\nAfter standardization (z-score):")
    print(standardized.agg(["mean", "std"]).round(3))
    print("\n(Means ~0 and stds ~1 as expected — confirms the z-score transform is correct. "
          "This standardized version is NOT saved into titanic.csv or used downstream; "
          "the modeling pipeline in 02_modeling.py fits its own StandardScaler on the train split only.)")

    print("\nDone.")
    sys.stdout = sys.__stdout__
    log_file.close()

if __name__ == "__main__":
    main()
