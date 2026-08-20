"""
JobSentinel - Prediction Pipeline

Responsible for:

1. Loading the trained ML pipeline
2. Preparing a single job posting
3. Creating model features
4. Producing the SVM prediction
5. Producing LOW / MEDIUM / HIGH risk
6. Detecting strong scam indicators

The ML model remains the primary classifier.

The risk-signal layer provides additional explanation
for obviously suspicious job postings.
"""

from pathlib import Path
import json
import re

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT / "models"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_pipeline():

    model = joblib.load(
        MODEL_DIR / "best_model.joblib"
    )

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    encoder = joblib.load(
        MODEL_DIR / "categorical_encoder.joblib"
    )

    scaler = joblib.load(
        MODEL_DIR / "numeric_scaler.joblib"
    )

    with open(
        MODEL_DIR / "feature_config.json",
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)


    threshold_path = (
        MODEL_DIR / "decision_threshold.txt"
    )

    with open(
        threshold_path,
        "r",
        encoding="utf-8"
    ) as file:

        threshold = float(
            file.read().strip()
        )


    return (
        model,
        vectorizer,
        encoder,
        scaler,
        config,
        threshold
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def count_urls(text):

    pattern = (
        r"https?://\S+|www\.\S+"
    )

    return len(
        re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )
    )


def count_emails(text):

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    return len(
        re.findall(
            pattern,
            text
        )
    )


def count_phone_numbers(text):

    pattern = (
        r"(?<!\d)"
        r"(?:\+?\d[\d\s().-]{7,}\d)"
        r"(?!\d)"
    )

    return len(
        re.findall(
            pattern,
            text
        )
    )


def uppercase_ratio(text):

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:

        return 0.0

    uppercase = sum(
        char.isupper()
        for char in letters
    )

    return (
        uppercase / len(letters)
    )


def count_suspicious_terms(
    text,
    suspicious_terms
):

    text = text.lower()

    return sum(
        text.count(term.lower())
        for term in suspicious_terms
    )


# ============================================================
# PREPARE JOB
# ============================================================

def prepare_job(
    job_data,
    config
):

    text_columns = (
        config["text_columns"]
    )

    categorical_columns = (
        config["categorical_columns"]
    )

    missing_columns = (
        config["missing_indicator_columns"]
    )

    suspicious_terms = (
        config["suspicious_terms"]
    )


    df = pd.DataFrame(
        [job_data]
    )


    # --------------------------------------------------------
    # Ensure columns
    # --------------------------------------------------------

    all_columns = (
        text_columns
        + categorical_columns
        + missing_columns
    )

    for column in all_columns:

        if column not in df.columns:

            df[column] = ""


    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    for column in text_columns:

        df[column] = (
            df[column]
            .apply(clean_text)
        )


    # --------------------------------------------------------
    # Combined text
    # --------------------------------------------------------

    df["combined_text"] = (
        df[text_columns]
        .agg(
            " ".join,
            axis=1
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


    # --------------------------------------------------------
    # Numeric text features
    # --------------------------------------------------------

    for column in text_columns:

        df[
            f"{column}_length"
        ] = (
            df[column]
            .str.len()
        )

        df[
            f"{column}_word_count"
        ] = (
            df[column]
            .str.split()
            .str.len()
        )

        df[
            f"{column}_url_count"
        ] = (
            df[column]
            .apply(count_urls)
        )

        df[
            f"{column}_email_count"
        ] = (
            df[column]
            .apply(count_emails)
        )

        df[
            f"{column}_phone_count"
        ] = (
            df[column]
            .apply(count_phone_numbers)
        )

        df[
            f"{column}_uppercase_ratio"
        ] = (
            df[column]
            .apply(uppercase_ratio)
        )


    # --------------------------------------------------------
    # Suspicious terms
    # --------------------------------------------------------

    df["suspicious_term_count"] = (
        df["combined_text"]
        .apply(
            lambda text:
            count_suspicious_terms(
                text,
                suspicious_terms
            )
        )
    )


    # --------------------------------------------------------
    # Missing features
    # --------------------------------------------------------

    for column in missing_columns:

        df[
            f"has_{column}"
        ] = (
            df[column]
            .notna()
            .astype(int)
        )


    return df


# ============================================================
# CREATE FEATURES
# ============================================================

def create_features(
    df,
    vectorizer,
    encoder,
    scaler,
    config
):

    text_columns = (
        config["text_columns"]
    )

    categorical_columns = (
        config["categorical_columns"]
    )

    numeric_columns = (
        config["numeric_columns"]
    )


    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    X_text = vectorizer.transform(
        df["combined_text"]
    )


    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    categorical_data = (
        df[categorical_columns]
        .fillna("Unknown")
        .astype(str)
    )

    X_categorical = (
        encoder.transform(
            categorical_data
        )
    )


    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    X_numeric = (
        df[numeric_columns]
        .fillna(0)
        .astype(float)
    )

    X_numeric = scaler.transform(
        X_numeric
    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    from scipy.sparse import hstack
    from scipy.sparse import csr_matrix

    X_numeric_sparse = csr_matrix(
        X_numeric
    )

    X = hstack(
        [
            X_text,
            X_categorical,
            X_numeric_sparse
        ],
        format="csr"
    )


    return X


# ============================================================
# STRONG RISK SIGNALS
# ============================================================

STRONG_RISK_TERMS = [

    "registration fee",
    "registration fees",

    "joining fee",
    "joining fees",

    "application fee",
    "application fees",

    "processing fee",
    "processing fees",

    "security deposit",

    "pay money",
    "pay a fee",
    "pay fee",

    "pay upfront",
    "advance payment",

    "guaranteed income",
    "guaranteed salary",
    "guaranteed earnings",

    "no interview",
    "without interview",

    "whatsapp",
    "telegram",

    "limited vacancies",

    "immediate joining",

    "earn money",

    "no experience required",

    "send your bank details",
    "bank account details",

    "credit card details",

    "otp",
    "one time password"
]


# ============================================================
# DETECT RISK SIGNALS
# ============================================================

def detect_risk_signals(
    job_data
):

    text_parts = []

    for value in job_data.values():

        if value:

            text_parts.append(
                str(value)
            )


    combined_text = " ".join(
        text_parts
    ).lower()


    matched = []

    for term in STRONG_RISK_TERMS:

        if term in combined_text:

            matched.append(
                term
            )


    return matched


# ============================================================
# PREDICT
# ============================================================

def predict_job(
    job_data
):

    (
        model,
        vectorizer,
        encoder,
        scaler,
        config,
        threshold
    ) = load_pipeline()


    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_job(
        job_data,
        config
    )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = create_features(
        df,
        vectorizer,
        encoder,
        scaler,
        config
    )


    # --------------------------------------------------------
    # SVM decision score
    # --------------------------------------------------------

    score = model.decision_function(
        X
    )[0]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    is_fraud = (
        score >= threshold
    )


    if is_fraud:

        prediction = "FRAUDULENT"

    else:

        prediction = "LEGITIMATE"


    # --------------------------------------------------------
    # Detect strong scam indicators
    # --------------------------------------------------------

    risk_signals = detect_risk_signals(
        job_data
    )

    signal_count = len(
        risk_signals
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    # Fraud prediction + at least 2 strong signals
    # = HIGH RISK

    if is_fraud and signal_count >= 2:

        risk_level = "HIGH"

        interpretation = (
            "The model classified this posting as fraudulent "
            "and multiple strong scam indicators were detected. "
            "Do not make payments or share sensitive personal "
            "or financial information before independently "
            "verifying the employer."
        )


    # Fraud prediction but fewer strong indicators
    # = MEDIUM

    elif is_fraud:

        risk_level = "MEDIUM"

        interpretation = (
            "The model detected suspicious patterns in this "
            "job posting. The employer and job posting should "
            "be independently verified before applying."
        )


    # Very suspicious posting even if model is below threshold

    elif signal_count >= 4:

        risk_level = "HIGH"

        interpretation = (
            "Multiple strong scam indicators were detected "
            "in this job posting. Independently verify the "
            "employer before applying."
        )


    # Some signals

    elif signal_count >= 2:

        risk_level = "MEDIUM"

        interpretation = (
            "Some potentially suspicious patterns were "
            "detected. Verify the employer and job posting "
            "before applying."
        )


    # Normal

    else:

        risk_level = "LOW"

        interpretation = (
            "The model did not detect strong fraud patterns "
            "in this job posting. You should still verify "
            "the employer and job opportunity independently."
        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "prediction":
            prediction,

        "decision_score":
            float(score),

        "threshold":
            float(threshold),

        "risk_level":
            risk_level,

        "model":
            type(model).__name__,

        "interpretation":
            interpretation,

        "risk_signals":
            risk_signals
    }