import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"


def load_model():

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Trained model files are missing. Train the model first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict_transaction(transaction, model, scaler):

    transaction = pd.DataFrame([transaction])

    # Match the exact training feature order
    transaction = transaction.reindex(
        columns=model.feature_names_in_
    )

    # Scale transaction amount
    if "Amount" in transaction.columns:
        transaction["Amount"] = scaler.transform(
            transaction[["Amount"]]
        ).ravel()

    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability) * 100, 2),
        "status": "Fraudulent" if prediction == 1 else "Genuine"
    }