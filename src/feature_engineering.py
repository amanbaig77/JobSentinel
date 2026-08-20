"""
JobSentinel - Feature Engineering Pipeline

Converts cleaned job postings into machine-learning features.

Feature groups:
1. TF-IDF text features
2. Categorical metadata features
3. Numeric risk / behavioral features
4. Missing-value indicators

The TF-IDF vocabulary is fitted ONLY on the training data
to prevent data leakage.
"""

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "jobs_cleaned.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "fraudulent"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "department",
]

MISSING_INDICATOR_COLUMNS = [
    "company_profile",
    "salary_range",
    "requirements",
    "benefits",
    "department",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "employment_type",
]

SUSPICIOUS_TERMS = [
    "urgent",
    "immediate",
    "work from home",
    "earn money",
    "easy money",
    "no experience",
    "quick money",
    "guaranteed",
    "investment",
    "cash",
    "wire transfer",
    "bank account",
    "western union",
    "money transfer",
    "commission",
    "telegram",
    "whatsapp",
    "bitcoin",
    "crypto",
]


# ============================================================
# 3. TEXT PATTERNS
# ============================================================

EMAIL_PATTERN = (
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

URL_PATTERN = (
    r"https?://\S+|www\.\S+"
)

PHONE_PATTERN = (
    r"(?<!\d)"
    r"(?:\+?\d[\d\s().-]{7,}\d)"
    r"(?!\d)"
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Convert missing/non-string values into clean strings.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def count_urls(text):
    """
    Count URLs in text.
    """
    return len(
        re.findall(
            URL_PATTERN,
            text,
            flags=re.IGNORECASE
        )
    )


def count_emails(text):
    """
    Count email addresses in text.
    """
    return len(
        re.findall(
            EMAIL_PATTERN,
            text
        )
    )


def count_phone_numbers(text):
    """
    Count phone-like patterns in text.
    """
    return len(
        re.findall(
            PHONE_PATTERN,
            text
        )
    )


def uppercase_ratio(text):
    """
    Calculate ratio of uppercase letters
    among all alphabetic characters.
    """

    letters = [
        char for char in text
        if char.isalpha()
    ]

    if not letters:
        return 0.0

    uppercase = sum(
        char.isupper()
        for char in letters
    )

    return uppercase / len(letters)


def count_suspicious_terms(text):
    """
    Count occurrences of manually selected
    suspicious job-posting terms.
    """

    text = text.lower()

    return sum(
        text.count(term)
        for term in SUSPICIOUS_TERMS
    )


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():
    """
    Load cleaned dataset.
    """

    print("=" * 70)
    print("JobSentinel - Feature Engineering")
    print("=" * 70)

    print("\nLoading dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


# ============================================================
# 6. CREATE TEXT FEATURES
# ============================================================

def create_text_features(df):
    """
    Create a single combined text field
    from the important job-posting text columns.
    """

    print("\nCreating combined text...")

    for column in TEXT_COLUMNS:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["combined_text"] = (
        df[TEXT_COLUMNS]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("Combined text created.")

    return df


# ============================================================
# 7. CREATE NUMERIC RISK FEATURES
# ============================================================

def create_numeric_features(df):
    """
    Create interpretable numeric/risk features.
    """

    print("\nCreating behavioral and risk features...")

    # --------------------------------------------
    # Text length
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_length"] = (
            df[column]
            .str.len()
        )

    # --------------------------------------------
    # Word count
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_word_count"] = (
            df[column]
            .str.split()
            .str.len()
        )

    # --------------------------------------------
    # URLs
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_url_count"] = (
            df[column]
            .apply(count_urls)
        )

    # --------------------------------------------
    # Emails
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_email_count"] = (
            df[column]
            .apply(count_emails)
        )

    # --------------------------------------------
    # Phone numbers
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_phone_count"] = (
            df[column]
            .apply(count_phone_numbers)
        )

    # --------------------------------------------
    # Uppercase ratio
    # --------------------------------------------

    for column in TEXT_COLUMNS:

        df[f"{column}_uppercase_ratio"] = (
            df[column]
            .apply(uppercase_ratio)
        )

    # --------------------------------------------
    # Suspicious terms
    # --------------------------------------------

    df["suspicious_term_count"] = (
        df["combined_text"]
        .apply(count_suspicious_terms)
    )

    print("Risk features created.")

    return df


# ============================================================
# 8. CREATE MISSINGNESS FEATURES
# ============================================================

def create_missing_features(df):
    """
    Create indicators showing whether important
    fields were provided by the job poster.
    """

    print("\nCreating missing-value indicators...")

    for column in MISSING_INDICATOR_COLUMNS:

        df[f"has_{column}"] = (
            df[column]
            .notna()
            .astype(int)
        )

    print("Missing-value indicators created.")

    return df


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

def stratified_split(df, test_size=0.20, random_state=42):
    """
    Perform a stratified train/test split manually.

    This avoids requiring the entire feature matrix
    to be created before splitting.
    """

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN]
    )

    print("\nDataset split:")
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)

    print("\nTraining target distribution:")
    print(
        train_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .round(4)
    )

    print("\nTesting target distribution:")
    print(
        test_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .round(4)
    )

    return train_df, test_df


# ============================================================
# 10. TF-IDF
# ============================================================

def build_tfidf(train_df, test_df):
    """
    Fit TF-IDF ONLY on training text.

    This is important because fitting TF-IDF on the
    complete dataset would introduce information leakage.
    """

    print("\nBuilding TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000,
    )

    X_train_text = vectorizer.fit_transform(
        train_df["combined_text"]
    )

    X_test_text = vectorizer.transform(
        test_df["combined_text"]
    )

    print(
        "TF-IDF training shape:",
        X_train_text.shape
    )

    print(
        "TF-IDF testing shape:",
        X_test_text.shape
    )

    return (
        vectorizer,
        X_train_text,
        X_test_text
    )


# ============================================================
# 11. CATEGORICAL FEATURES
# ============================================================

def build_categorical_features(train_df, test_df):
    """
    One-hot encode categorical job metadata.
    """

    print("\nBuilding categorical features...")

    train_cat = (
        train_df[CATEGORICAL_COLUMNS]
        .fillna("Unknown")
        .astype(str)
    )

    test_cat = (
        test_df[CATEGORICAL_COLUMNS]
        .fillna("Unknown")
        .astype(str)
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )

    X_train_cat = encoder.fit_transform(
        train_cat
    )

    X_test_cat = encoder.transform(
        test_cat
    )

    print(
        "Categorical training shape:",
        X_train_cat.shape
    )

    print(
        "Categorical testing shape:",
        X_test_cat.shape
    )

    return (
        encoder,
        X_train_cat,
        X_test_cat
    )


# ============================================================
# 12. NUMERIC FEATURES
# ============================================================

def get_numeric_columns(df):
    """
    Automatically identify engineered numeric features.
    """

    excluded = {
        TARGET_COLUMN
    }

    numeric_columns = []

    for column in df.columns:

        if column in excluded:
            continue

        if column == "combined_text":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            numeric_columns.append(column)

    return numeric_columns


def build_numeric_features(
    train_df,
    test_df,
    numeric_columns
):
    """
    Scale engineered numeric features.
    """

    print("\nBuilding numeric features...")

    X_train_num = (
        train_df[numeric_columns]
        .fillna(0)
        .astype(float)
    )

    X_test_num = (
        test_df[numeric_columns]
        .fillna(0)
        .astype(float)
    )

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(
        X_train_num
    )

    X_test_num = scaler.transform(
        X_test_num
    )

    print(
        "Numeric feature count:",
        len(numeric_columns)
    )

    return (
        scaler,
        X_train_num,
        X_test_num
    )


# ============================================================
# 13. COMBINE ALL FEATURES
# ============================================================

def combine_features(
    X_text,
    X_categorical,
    X_numeric
):
    """
    Combine sparse TF-IDF, categorical and numeric features.
    """

    from scipy.sparse import csr_matrix

    X_numeric_sparse = csr_matrix(
        X_numeric
    )

    X_combined = hstack(
        [
            X_text,
            X_categorical,
            X_numeric_sparse
        ],
        format="csr"
    )

    return X_combined


# ============================================================
# 14. SAVE FEATURES
# ============================================================

def save_features(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Save sparse feature matrices and targets.
    """

    print("\nSaving feature matrices...")

    save_npz(
        OUTPUT_DIR / "X_train.npz",
        X_train
    )

    save_npz(
        OUTPUT_DIR / "X_test.npz",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    print("Feature matrices saved successfully.")


# ============================================================
# 15. SAVE TRANSFORMERS
# ============================================================

def save_transformers(
    vectorizer,
    encoder,
    scaler,
    numeric_columns
):
    """
    Save preprocessing objects for future prediction.
    """

    print("\nSaving preprocessing pipelines...")

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    joblib.dump(
        encoder,
        MODEL_DIR / "categorical_encoder.joblib"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "numeric_scaler.joblib"
    )

    config = {
        "text_columns": TEXT_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "missing_indicator_columns": MISSING_INDICATOR_COLUMNS,
        "numeric_columns": numeric_columns,
        "suspicious_terms": SUSPICIOUS_TERMS,
    }

    with open(
        MODEL_DIR / "feature_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print("Preprocessing objects saved.")


# ============================================================
# 16. MAIN PIPELINE
# ============================================================

def main():

    # Load
    df = load_data()

    # Target validation
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found in dataset."
        )

    # Text
    df = create_text_features(df)

    # Numeric/risk
    df = create_numeric_features(df)

    # Missingness
    df = create_missing_features(df)

    # Split FIRST
    train_df, test_df = stratified_split(
        df
    )

    # TF-IDF
    (
        vectorizer,
        X_train_text,
        X_test_text
    ) = build_tfidf(
        train_df,
        test_df
    )

    # Categorical
    (
        encoder,
        X_train_cat,
        X_test_cat
    ) = build_categorical_features(
        train_df,
        test_df
    )

    # Numeric
    numeric_columns = get_numeric_columns(
        train_df
    )

    (
        scaler,
        X_train_num,
        X_test_num
    ) = build_numeric_features(
        train_df,
        test_df,
        numeric_columns
    )

    # Combine
    print("\nCombining feature groups...")

    X_train = combine_features(
        X_train_text,
        X_train_cat,
        X_train_num
    )

    X_test = combine_features(
        X_test_text,
        X_test_cat,
        X_test_num
    )

    # Targets
    y_train = train_df[
        TARGET_COLUMN
    ].to_numpy()

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy()

    print("\nFinal feature shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nTarget shapes:")
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # Save
    save_features(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_transformers(
        vectorizer,
        encoder,
        scaler,
        numeric_columns
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":
    main()