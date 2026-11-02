import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(file_path):
    """Load the bank transaction dataset."""

    data = pd.read_csv(file_path)

    return data


def prepare_data(data):
    """Clean and prepare transaction data."""

    # Remove duplicate records
    data = data.drop_duplicates()

    # Handle missing values
    data = data.dropna()

    # Separate input features and target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale transaction amount
    scaler = StandardScaler()

    if "Amount" in X_train.columns:
        X_train = X_train.copy()
        X_test = X_test.copy()

        X_train["Amount"] = scaler.fit_transform(
            X_train[["Amount"]]
        ).ravel()

        X_test["Amount"] = scaler.transform(
            X_test[["Amount"]]
        ).ravel()

    return X_train, X_test, y_train, y_test, scaler