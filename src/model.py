"""Churn classification: logistic regression vs random forest.

End-to-end script: loads the synthetic dataset, cleans it, engineers
features, trains two pipelines (standardized logistic regression and
random forest) on a stratified train/test split, prints comparison
metrics, and saves evaluation charts to images/.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import clean_customers

RANDOM_STATE = 42
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "customers.csv"
IMAGES_DIR = REPO_ROOT / "images"

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "balance",
    "num_products",
    "has_credit_card",
    "is_active_member",
    "estimated_salary",
]
CATEGORICAL_FEATURES = ["geography", "gender"]


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    df = clean_customers(pd.read_csv(DATA_PATH))
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churned"]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_names


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    X, y = load_features()
    print(f"Dataset: {len(X):,} customers | churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"Train: {len(X_train):,} (churn {y_train.mean():.2%}) | "
        f"Test: {len(X_test):,} (churn {y_test.mean():.2%})"
    )

    models = build_models(build_preprocessor())
    results: dict[str, dict] = {}
    probabilities: dict[str, np.ndarray] = {}

    print("\n" + "=" * 64)
    print(f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'ROC-AUC':>10}")
    print("=" * 64)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_prob)
        results[name] = metrics
        probabilities[name] = y_prob
        print(
            f"{name:<22}"
            f"{metrics['accuracy']:>8.3f}"
            f"{metrics['precision']:>8.3f}"
            f"{metrics['recall']:>8.3f}"
            f"{metrics['f1']:>8.3f}"
            f"{metrics['roc_auc']:>10.3f}"
        )
    print("=" * 64)

    # --- 5-fold stratified CV ROC-AUC on the training set ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\n5-fold stratified CV ROC-AUC (train):")
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"  {name:<22} mean={scores.mean():.3f}  std={scores.std():.3f}")

    # --- Threshold note: what happens if we lower the decision threshold? ---
    rf_prob = probabilities["Random Forest"]
    print("\nRandom Forest @ different thresholds (precision/recall trade-off):")
    print(f"{'Threshold':<10}{'Precision':>10}{'Recall':>8}{'F1':>8}")
    for thr in (0.3, 0.4, 0.5, 0.6):
        pred = (rf_prob >= thr).astype(int)
        print(
            f"{thr:<10}"
            f"{precision_score(y_test, pred, zero_division=0):>10.3f}"
            f"{recall_score(y_test, pred):>8.3f}"
            f"{f1_score(y_test, pred):>8.3f}"
        )
    print(
        "\nNote: lowering the threshold lifts recall (catches more churners) at "
        "the cost of precision. For a retention campaign where a cheap email is "
        "the intervention, recall-first (threshold ~0.3-0.4) is usually preferred."
    )

    # --- Feature importance (random forest) ---
    rf_pipe = models["Random Forest"]
    importances = rf_pipe.named_steps["model"].feature_importances_
    names = feature_names(rf_pipe.named_steps["preprocess"])
    fi = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("\nRandom Forest feature importance (top 8):")
    for _, row in fi.head(8).iterrows():
        print(f"  {row['feature']:<22} {row['importance']:.3f}")

    # --- Logistic regression coefficients (direction of effect) ---
    lr_pipe = models["Logistic Regression"]
    coefs = lr_pipe.named_steps["model"].coef_[0]
    lr_names = feature_names(lr_pipe.named_steps["preprocess"])
    lr_fi = (
        pd.DataFrame({"feature": lr_names, "coefficient": coefs})
        .sort_values("coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    print("\nLogistic Regression coefficients (top 8 by |coef|):")
    for _, row in lr_fi.head(8).iterrows():
        print(f"  {row['feature']:<22} {row['coefficient']:+.3f}")

    # ================= Charts =================
    plt.rcParams["figure.dpi"] = 150

    # 1. ROC curves (both models)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Models (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "roc_curves.png")
    plt.close(fig)

    # 2. Confusion matrix heatmap (random forest @ 0.5)
    cm = results["Random Forest"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: Retained", "Predicted: Churned"],
        yticklabels=["Actual: Retained", "Actual: Churned"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Random Forest (threshold 0.5, test set)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix.png")
    plt.close(fig)

    # 3. Feature importance (top 10, random forest)
    top = fi.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], color="#2a7cc7")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Top 10 Feature Importances — Random Forest")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "feature_importance.png")
    plt.close(fig)

    print("\nCharts saved to images/: roc_curves.png, confusion_matrix.png, feature_importance.png")


if __name__ == "__main__":
    main()
