
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_FILE = RAW_DATA_DIR / "bank-full.csv"

# Official UCI dataset
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "222/bank+marketing.zip"
)

# Expected columns in bank-full.csv
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education",
    "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]


def download_dataset():

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print("Dataset already exists. Skipping download.")
        return DATA_FILE

    print("Downloading UCI Bank Marketing dataset...")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        zip_content = response.read()

    print("Download completed!")

    # Open the outer ZIP archive
    with zipfile.ZipFile(io.BytesIO(zip_content)) as outer_zip:

        bank_zip_name = next(
            (
                name for name in outer_zip.namelist()
                if Path(name).name.lower() == "bank.zip"
            ),
            None
        )

        if bank_zip_name:

            # Read the nested bank.zip
            nested_content = outer_zip.read(bank_zip_name)

            with zipfile.ZipFile(
                io.BytesIO(nested_content)
            ) as inner_zip:

                csv_name = next(
                    (
                        name for name in inner_zip.namelist()
                        if Path(name).name.lower() == "bank-full.csv"
                    ),
                    None
                )

                if not csv_name:
                    raise FileNotFoundError(
                        "bank-full.csv not found inside bank.zip"
                    )

                csv_content = inner_zip.read(csv_name)

        else:
            # Support an archive containing the CSV directly
            csv_name = next(
                (
                    name for name in outer_zip.namelist()
                    if Path(name).name.lower() == "bank-full.csv"
                ),
                None
            )

            if not csv_name:
                raise FileNotFoundError(
                    "bank-full.csv not found in UCI archive"
                )

            csv_content = outer_zip.read(csv_name)

    DATA_FILE.write_bytes(csv_content)

    print("Dataset extracted to:", DATA_FILE)

    return DATA_FILE


def load_and_preprocess_data():

    print("\nLoading Bank Marketing dataset...")

    # Download if necessary
    file_path = download_dataset()

    # Read semicolon-separated CSV
    df = pd.read_csv(file_path, sep=";")

    print("\nOriginal dataset shape:", df.shape)

    # Validate columns
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Dataset columns do not match the expected schema."
        )

    # Validate target
    if not df["y"].isin(["yes", "no"]).all():
        raise ValueError("Unexpected target values found.")

    # Check missing values
    if df.isnull().values.any():
        raise ValueError("Missing values found in the dataset.")

    # Remove exact duplicate records
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        print("Removing duplicate records:", duplicate_count)
        df = df.drop_duplicates().reset_index(drop=True)

    # Remove duration to prevent data leakage
    df = df.drop(columns=["duration"])

    # Separate features and target
    X = df.drop(columns=["y"])

    # Encode target: no = 0, yes = 1
    y = df["y"].map({
        "no": 0,
        "yes": 1
    })

    if y.isnull().any():
        raise ValueError("Target encoding failed.")

    # Validate feature and target dimensions
    assert len(X) == len(y)
    assert "duration" not in X.columns
    assert "y" not in X.columns

    print("\nData validation successful!")
    print("Features:", X.shape)
    print("Target:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nTarget percentage:")
    print(y.value_counts(normalize=True) * 100)

    # Split into numerical and categorical columns
    numerical_features = [
        "age", "balance", "day",
        "campaign", "pdays", "previous"
    ]

    categorical_features = [
        "job", "marital", "education",
        "default", "housing", "loan",
        "contact", "month", "poutcome"
    ]

    # Validate feature definitions
    assert set(numerical_features + categorical_features) == set(X.columns)

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain-test split completed!")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        numerical_features,
        categorical_features
    )


if __name__ == "__main__":

    load_and_preprocess_data()
