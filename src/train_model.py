import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_dataset, prepare_data
from evaluation import evaluate_model


def train_model():

    print("Loading dataset...")

    data = load_dataset("data/creditcard.csv")

    print("Preparing transaction data...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")

    results = evaluate_model(model, X_test, y_test)

    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    print("\nMODEL TRAINING COMPLETED")

    print("Accuracy:", results["accuracy"], "%")
    print("Precision:", results["precision"], "%")
    print("Recall:", results["recall"], "%")
    print("F1 Score:", results["f1_score"], "%")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    train_model()