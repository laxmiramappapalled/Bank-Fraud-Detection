import streamlit as st
from pathlib import Path
import sys

# Page configuration
st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="🔍",
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

# Import prediction functions
sys.path.insert(0, str(ROOT_DIR / "src"))

from prediction import load_model, predict_transaction

# Page title
st.title("🔍 Fraud Transaction Prediction")

st.markdown("""
### AI-Powered Transaction Analysis

Enter transaction details below to check whether
a transaction is potentially fraudulent or genuine.
""")

st.divider()

# Load trained model
try:
    model, scaler = load_model()

    st.success("Machine Learning Model Loaded Successfully")

    st.subheader("Transaction Details")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            transaction_time = st.number_input(
                "Transaction Time",
                min_value=0.0,
                value=0.0
            )

        with col2:
            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0
            )

        st.divider()

        st.subheader("PCA Transaction Features")

        st.info(
            "V1 to V28 are anonymized PCA features "
            "from the original credit card dataset. "
            "For this demonstration, default values are zero."
        )

        transaction = {
            "Time": transaction_time,
            "Amount": transaction_amount
        }

        columns = st.columns(4)

        for i in range(1, 29):
            with columns[(i - 1) % 4]:
                transaction[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.4f",
                    key=f"feature_{i}"
                )

        st.divider()

        submit = st.form_submit_button(
            "🔍 Predict Transaction",
            use_container_width=True
        )

    # Prediction result
    if submit:

        with st.spinner("Analyzing transaction..."):

            result = predict_transaction(
                transaction,
                model,
                scaler
            )

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if result["prediction"] == 1:
                st.error("🚨 Fraudulent Transaction")
            else:
                st.success("✅ Genuine Transaction")

        with col2:
            st.metric(
                "Fraud Probability",
                f"{result['fraud_probability']:.2f}%"
            )

        st.progress(
            min(result["fraud_probability"] / 100, 1.0)
        )

        st.caption(
            "This prediction is for educational and "
            "demonstration purposes only."
        )

except FileNotFoundError as error:
    st.error(str(error))

except Exception as error:
    st.error(f"An error occurred: {error}")