import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Model Reports",
    page_icon="📈",
    layout="wide"
)

# Project directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load CSS
CSS_PATH = ROOT_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# Import model loader
sys.path.insert(0, str(ROOT_DIR / "src"))

from prediction import load_model

# Page title
st.title("📈 Model Performance Reports")
st.markdown("### Machine Learning Evaluation")

st.divider()

DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"

try:
    if not DATA_PATH.exists():
        st.error("Dataset not found.")
        st.stop()

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv(DATA_PATH)

    data = load_data()

    # Load trained model
    model, scaler = load_model()

    st.success("Trained Model Loaded Successfully")

    # Prepare test data
    data = data.drop_duplicates().dropna()

    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_test = X_test.copy()

    if "Amount" in X_test.columns:
        X_test["Amount"] = scaler.transform(
            X_test[["Amount"]]
        ).ravel()

    # Predictions
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    precision = precision_score(
        y_test, predictions, zero_division=0
    ) * 100
    recall = recall_score(
        y_test, predictions, zero_division=0
    ) * 100
    f1 = f1_score(
        y_test, predictions, zero_division=0
    ) * 100

    # Performance metrics
    st.subheader("Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy:.2f}%")
    col2.metric("Precision", f"{precision:.2f}%")
    col3.metric("Recall", f"{recall:.2f}%")
    col4.metric("F1 Score", f"{f1:.2f}%")

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_test, predictions)

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Genuine", "Actual Fraud"],
        columns=["Predicted Genuine", "Predicted Fraud"]
    )

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Model Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraud"],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    st.divider()

    # Explanation
    st.subheader("Understanding the Metrics")

    st.write("""
    - **Accuracy:** Overall percentage of correct predictions.
    - **Precision:** How many predicted fraud transactions were actually fraud.
    - **Recall:** How many actual fraud transactions the model detected.
    - **F1 Score:** Balance between precision and recall.
    - **Confusion Matrix:** Shows correct and incorrect predictions.
    """)

    st.caption(
        "For educational and demonstration purposes only. "
        "This model is not validated for real banking decisions."
    )

except FileNotFoundError as error:
    st.error(str(error))

except Exception as error:
    st.error(f"An error occurred: {error}")
