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

    print("Evaluating independent test data...")

    results = evaluate_model(model, X_test, y_test)

    os.makedirs("model", exist_ok=True)

    # Save trained model and scaler
    joblib.dump(model, "model/fraud_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    # Save independent test evaluation results
    evaluation_results = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "confusion_matrix": results["confusion_matrix"],
        "classification_report": results["classification_report"],
        "total_transactions": len(y_test),
        "correct_predictions": int(
            (model.predict(X_test) == y_test).sum()
        )
    }

    joblib.dump(
        evaluation_results,
        "model/evaluation_results.pkl"
    )

    print("\nMODEL TRAINING COMPLETED")

    print("Accuracy:", results["accuracy"], "%")
    print("Precision:", results["precision"], "%")
    print("Recall:", results["recall"], "%")
    print("F1 Score:", results["f1_score"], "%")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nIndependent test results saved successfully.")


if __name__ == "__main__":
    train_model()