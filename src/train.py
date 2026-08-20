"""
JobSentinel - Model Training Pipeline

Baseline model comparison for fraudulent job posting detection.

Models:
1. Logistic Regression
2. Linear SVM

Evaluation:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Because fraudulent jobs are a minority class,
we focus strongly on Precision, Recall, F1 and PR-AUC.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD FEATURES
# ============================================================

def load_features():

    print("=" * 70)
    print("JobSentinel - Model Training")
    print("=" * 70)

    print("\nLoading feature matrices...")

    X_train = load_npz(
        FEATURE_DIR / "X_train.npz"
    )

    X_test = load_npz(
        FEATURE_DIR / "X_test.npz"
    )

    y_train = np.load(
        FEATURE_DIR / "y_train.npy"
    )

    y_test = np.load(
        FEATURE_DIR / "y_test.npy"
    )

    print("\nFeatures loaded successfully.")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nTraining class distribution:")

    print(
        pd.Series(y_train)
        .value_counts()
        .sort_index()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 3. CREATE MODELS
# ============================================================

def create_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            solver="liblinear",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=10000,
            random_state=42
        ),

    }

    return models


# ============================================================
# 4. GET CONTINUOUS MODEL SCORES
# ============================================================

def get_prediction_scores(
    model,
    X
):

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(X)
        )

        return probabilities[:, 1]

    if hasattr(
        model,
        "decision_function"
    ):

        return model.decision_function(
            X
        )

    return model.predict(X)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    scores = get_prediction_scores(
        model,
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        scores
    )

    pr_auc = average_precision_score(
        y_test,
        scores
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nMetrics:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraudulent"
            ],
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


# ============================================================
# 6. SAVE BEST MODEL
# ============================================================

def save_best_model(
    model,
    model_name
):

    model_path = (
        MODEL_DIR
        / "best_model.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    metadata = {
        "model_name": model_name,
        "model_path": str(model_path),
    }

    metadata_path = (
        MODEL_DIR
        / "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nBest model saved to:"
        f"\n{model_path}"
    )


# ============================================================
# 7. SAVE COMPARISON RESULTS
# ============================================================

def save_results(results):

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False
    )

    output_path = (
        REPORT_DIR
        / "model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nModel comparison saved to:"
        f"\n{output_path}"
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "pr_auc"
            ]
        ].to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# 8. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_features()


    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    results = []

    trained_models = {}


    # --------------------------------------------------------
    # Train models
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("\n")

        print("#" * 70)
        print(
            f"Training: {model_name}"
        )
        print("#" * 70)

        model.fit(
            X_train,
            y_train
        )

        trained_models[
            model_name
        ] = model

        metrics = evaluate_model(
            model_name,
            model,
            X_test,
            y_test
        )

        results.append(
            metrics
        )


    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    results_df = save_results(
        results
    )


    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_model_name = (
        results_df.iloc[0]["model"]
    )

    best_model = (
        trained_models[
            best_model_name
        ]
    )


    print("\n" + "=" * 70)

    print(
        f"BEST BASELINE MODEL: "
        f"{best_model_name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    save_best_model(
        best_model,
        best_model_name
    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":
    main()