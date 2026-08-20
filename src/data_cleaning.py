import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "DataSet.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "jobs_cleaned.csv"


def load_data():
    """Load the original raw dataset."""
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Original shape: {df.shape}")
    return df


def clean_data(df):
    """Clean the dataset without modifying the original raw file."""

    # 1. Remove dataset-construction metadata
    if "in_balanced_dataset" in df.columns:
        df = df.drop(columns=["in_balanced_dataset"])

    # 2. Remove exact duplicate rows
    before_duplicates = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_duplicates - len(df)

    # 3. Convert target from f/t to 0/1
    df["fraudulent"] = df["fraudulent"].map({
        "f": 0,
        "t": 1
    })

    # 4. Make sure target conversion succeeded
    if df["fraudulent"].isnull().any():
        raise ValueError("Unexpected values found in 'fraudulent' column.")

    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Cleaned shape: {df.shape}")

    return df


def save_data(df):
    """Save the cleaned dataset."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to:")
    print(PROCESSED_DATA_PATH)


def main():
    df = load_data()
    df = clean_data(df)
    save_data(df)

    print("\nCleaning completed successfully!")


if __name__ == "__main__":
    main()