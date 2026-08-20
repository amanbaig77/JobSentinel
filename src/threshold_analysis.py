"""
JobSentinel - Threshold Analysis & Error Analysis

Purpose:
1. Analyze the best baseline model (Linear SVM)
2. Test different decision thresholds
3. Compare Precision / Recall / F1
4. Identify false positives and false negatives
5. Save detailed error-analysis reports

Important:
The model is NOT retrained here.
We are analyzing the already trained Linear SVM.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD MODEL + TEST DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("JobSentinel - Threshold & Error Analysis")
    print("=" * 70)

    print("\nLoading trained model...")

    model_path = (
        MODEL_DIR
        / "best_model.joblib"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Best model not found. "
            "Run train.py first."
        )

    model = joblib.load(
        model_path
    )

    print(
        "Model loaded successfully."
    )

    print(
        "\nModel type:",
        type(model).__name__
    )


    # --------------------------------------------------------
    # Load test features
    # --------------------------------------------------------

    print(
        "\nLoading test features..."
    )

    X_test = load_npz(
        FEATURE_DIR
        / "X_test.npz"
    )

    y_test = np.load(
        FEATURE_DIR
        / "y_test.npy"
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    return (
        model,
        X_test,
        y_test
    )


# ============================================================
# 3. GET MODEL SCORES
# ============================================================

def get_scores(
    model,
    X_test
):

    print(
        "\nGenerating model scores..."
    )

    if hasattr(
        model,
        "decision_function"
    ):

        scores = model.decision_function(
            X_test
        )

    elif hasattr(
        model,
        "predict_proba"
    ):

        scores = model.predict_proba(
            X_test
        )[:, 1]

    else:

        raise ValueError(
            "Model does not provide "
            "decision_function or predict_proba."
        )

    return scores


# ============================================================
# 4. THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    scores,
    y_test
):

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Important:
    #
    # Linear SVM's decision_function is NOT a probability.
    #
    # We are using its decision score and testing different
    # decision boundaries.
    # --------------------------------------------------------

    thresholds = [
        -1.0,
        -0.75,
        -0.50,
        -0.25,
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
        1.25,
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            scores >= threshold
        ).astype(int)

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

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions
            ).ravel()
        )

        results.append({

            "threshold": threshold,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "true_negatives": tn,

            "false_positives": fp,

            "false_negatives": fn,

            "true_positives": tp,
        })


    results_df = pd.DataFrame(
        results
    )

    print(
        "\nThreshold comparison:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# 5. FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    results_df
):

    print("\n" + "=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    best_row = (
        results_df
        .sort_values(
            "f1_score",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "\nBest threshold based on F1:"
    )

    print(
        f"Threshold : "
        f"{best_row['threshold']}"
    )

    print(
        f"Precision : "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1-score  : "
        f"{best_row['f1_score']:.4f}"
    )

    print(
        f"FP        : "
        f"{int(best_row['false_positives'])}"
    )

    print(
        f"FN        : "
        f"{int(best_row['false_negatives'])}"
    )

    print(
        f"TP        : "
        f"{int(best_row['true_positives'])}"
    )

    return best_row


# ============================================================
# 6. ERROR ANALYSIS
# ============================================================

def error_analysis(
    scores,
    y_test,
    threshold
):

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    predictions = (
        scores >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # False Positives
    #
    # Legitimate jobs predicted as fraudulent
    # --------------------------------------------------------

    false_positive_indices = np.where(
        (y_test == 0)
        & (predictions == 1)
    )[0]


    # --------------------------------------------------------
    # False Negatives
    #
    # Fraudulent jobs predicted as legitimate
    # --------------------------------------------------------

    false_negative_indices = np.where(
        (y_test == 1)
        & (predictions == 0)
    )[0]


    # --------------------------------------------------------
    # True Positives
    # --------------------------------------------------------

    true_positive_indices = np.where(
        (y_test == 1)
        & (predictions == 1)
    )[0]


    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\nFalse Positives:",
        len(false_positive_indices)
    )

    print(
        "False Negatives:",
        len(false_negative_indices)
    )

    print(
        "True Positives:",
        len(true_positive_indices)
    )


    # --------------------------------------------------------
    # Save error index reports
    # --------------------------------------------------------

    false_positive_df = pd.DataFrame({

        "test_index":
        false_positive_indices,

        "true_label":
        y_test[false_positive_indices],

        "prediction":
        predictions[false_positive_indices],

        "model_score":
        scores[false_positive_indices],

    })


    false_negative_df = pd.DataFrame({

        "test_index":
        false_negative_indices,

        "true_label":
        y_test[false_negative_indices],

        "prediction":
        predictions[false_negative_indices],

        "model_score":
        scores[false_negative_indices],

    })


    false_positive_path = (
        REPORT_DIR
        / "false_positives.csv"
    )

    false_negative_path = (
        REPORT_DIR
        / "false_negatives.csv"
    )


    false_positive_df.to_csv(
        false_positive_path,
        index=False
    )

    false_negative_df.to_csv(
        false_negative_path,
        index=False
    )


    print(
        "\nFalse-positive report saved:"
    )

    print(
        false_positive_path
    )


    print(
        "\nFalse-negative report saved:"
    )

    print(
        false_negative_path
    )


    return (
        false_positive_df,
        false_negative_df
    )


# ============================================================
# 7. FINAL CONFUSION MATRIX
# ============================================================

def final_evaluation(
    scores,
    y_test,
    threshold
):

    predictions = (
        scores >= threshold
    ).astype(int)

    print("\n" + "=" * 70)
    print("FINAL THRESHOLD EVALUATION")
    print("=" * 70)

    print(
        "\nClassification Report:"
    )

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

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


# ============================================================
# 8. SAVE THRESHOLD CONFIG
# ============================================================

def save_threshold(
    threshold
):

    threshold_path = (
        MODEL_DIR
        / "decision_threshold.txt"
    )

    with open(
        threshold_path,
        "w"
    ) as file:

        file.write(
            str(threshold)
        )

    print(
        "\nDecision threshold saved to:"
    )

    print(
        threshold_path
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        model,
        X_test,
        y_test
    ) = load_data()


    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    scores = get_scores(
        model,
        X_test
    )


    # --------------------------------------------------------
    # Threshold testing
    # --------------------------------------------------------

    results_df = threshold_analysis(
        scores,
        y_test
    )


    # --------------------------------------------------------
    # Save threshold results
    # --------------------------------------------------------

    threshold_results_path = (
        REPORT_DIR
        / "threshold_comparison.csv"
    )

    results_df.to_csv(
        threshold_results_path,
        index=False
    )

    print(
        "\nThreshold results saved to:"
    )

    print(
        threshold_results_path
    )


    # --------------------------------------------------------
    # Best threshold
    # --------------------------------------------------------

    best_row = find_best_threshold(
        results_df
    )

    best_threshold = float(
        best_row["threshold"]
    )


    # --------------------------------------------------------
    # Error analysis
    # --------------------------------------------------------

    error_analysis(
        scores,
        y_test,
        best_threshold
    )


    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    final_evaluation(
        scores,
        y_test,
        best_threshold
    )


    # --------------------------------------------------------
    # Save threshold
    # --------------------------------------------------------

    save_threshold(
        best_threshold
    )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD & ERROR ANALYSIS COMPLETE")
    print("=" * 70)


# ============================================================
# 10. RUN
# ============================================================

if __name__ == "__main__":
    main()