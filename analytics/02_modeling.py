import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

CHART_DIR = "charts"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)

    log_file = open("modeling_report.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)

    section("Load data")
    df = pd.read_csv("titanic.csv")
    print(f"Dataset shape: {df.shape}")

    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target = "survived"

    X = df[features]
    y = df[target]

    section("Train/test split")
    print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    numeric_features = ["age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]
    passthrough_features = ["pclass"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("pass", "passthrough", passthrough_features),
    ])

    section("Train classifiers")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    fitted_pipelines = {}
    metrics = []

    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n{name}")
        print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
        print(
            f"Accuracy={acc:.3f}  Precision={precision:.3f}  "
            f"Recall={recall:.3f}  F1={f1:.3f}  AUC={auc:.3f}"
        )

        metrics.append({
            "model": name,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        })

        RocCurveDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            ax=ax,
            name=name,
        )

    ax.set_title("ROC Curves")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/07_roc_curves.png", dpi=120)
    plt.close()

    comparison_df = pd.DataFrame(metrics).set_index("model").round(3)
    print("\nModel comparison:")
    print(comparison_df)

    dt_pipeline = fitted_pipelines["Decision Tree"]
    dt_model = dt_pipeline.named_steps["classifier"]
    onehot = (
        dt_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    feature_names = (
        numeric_features
        + list(onehot.get_feature_names_out(categorical_features))
        + passthrough_features
    )

    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=["Died", "Survived"],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/08_decision_tree.png", dpi=120)
    plt.close()

    section("Imbalance handling")

    imbalance_results = []

    y_pred = fitted_pipelines["Random Forest"].predict(X_test)
    imbalance_results.append({
        "strategy": "baseline",
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    })

    balanced_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    balanced_pipeline.fit(X_train, y_train)
    y_pred_balanced = balanced_pipeline.predict(X_test)

    imbalance_results.append({
        "strategy": "class_weight=balanced",
        "precision": precision_score(y_test, y_pred_balanced),
        "recall": recall_score(y_test, y_pred_balanced),
        "f1": f1_score(y_test, y_pred_balanced),
    })

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_processed,
        y_train,
    )

    rf_smote = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_smote.fit(X_train_resampled, y_train_resampled)

    y_pred_smote = rf_smote.predict(X_test_processed)

    imbalance_results.append({
        "strategy": "SMOTE",
        "precision": precision_score(y_test, y_pred_smote),
        "recall": recall_score(y_test, y_pred_smote),
        "f1": f1_score(y_test, y_pred_smote),
    })

    imbalance_df = pd.DataFrame(imbalance_results).set_index("strategy").round(3)
    print(imbalance_df)

    section("Random Forest tuning")

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                oob_score=True,
                random_state=42,
                bootstrap=True,
            ),
        ),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [4, 8, None],
        "classifier__max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.3f}")

    best_rf_pipeline = grid_search.best_estimator_
    oob_score = best_rf_pipeline.named_steps["classifier"].oob_score_
    print(f"OOB score: {oob_score:.3f}")

    section("Fare regression")

    reg_features = ["pclass", "age", "sibsp", "parch", "survived"]
    reg_categorical = ["sex", "embarked"]

    X_reg = df[reg_features + reg_categorical]
    y_reg = df["fare"]

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42,
    )

    reg_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), reg_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical),
    ])

    reg_pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression()),
    ])

    reg_pipeline.fit(X_reg_train, y_reg_train)
    y_reg_pred = reg_pipeline.predict(X_reg_test)

    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    r2 = r2_score(y_reg_test, y_reg_pred)

    n = X_reg_test.shape[0]
    p = X_reg_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    print(
        f"MAE={mae:.3f}  RMSE={rmse:.3f}  "
        f"R2={r2:.3f}  Adjusted R2={adjusted_r2:.3f}"
    )

    residuals = y_reg_test - y_reg_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_reg_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/09_residual_plot.png", dpi=120)
    plt.close()

    median_prediction = np.median(y_reg_pred)
    low_std = residuals[y_reg_pred < median_prediction].std()
    high_std = residuals[y_reg_pred >= median_prediction].std()

    print(f"Low predicted-fare residual std: {low_std:.2f}")
    print(f"High predicted-fare residual std: {high_std:.2f}")

    section("Final results")

    print("Classification metrics:")
    print(comparison_df)

    print(
        f"\nRegression metrics: MAE={mae:.3f}, RMSE={rmse:.3f}, "
        f"R2={r2:.3f}, Adjusted R2={adjusted_r2:.3f}"
    )

    best_classifier = comparison_df["f1"].idxmax()
    print(f"\nBest classifier by F1: {best_classifier}")

    section("Save pipeline")

    best_pipeline = fitted_pipelines[best_classifier]
    joblib.dump(best_pipeline, "best_pipeline.joblib")

    reloaded = joblib.load("best_pipeline.joblib")
    sample = X_test.iloc[:3]

    print(f"Predictions: {reloaded.predict(sample).tolist()}")
    print(f"Actual: {y_test.iloc[:3].tolist()}")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
