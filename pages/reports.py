import streamlit as st
import pandas as pd
import joblib
import plotly.figure_factory as ff

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Fraud Detection Reports",
    page_icon="📑",
    layout="wide"
)

# --------------------------------
# PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "creditcard_sample.csv"
MODEL_PATH = BASE_DIR / "model" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
CSS_PATH = BASE_DIR / "assets" / "style.css"

# --------------------------------
# LOAD CSS
# --------------------------------

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# --------------------------------
# PAGE TITLE
# --------------------------------

st.title("📑 Fraud Detection Reports")
st.subheader("Machine Learning Model Performance Analysis")

st.markdown("---")

# --------------------------------
# CHECK FILES
# --------------------------------

if not DATA_PATH.exists():
    st.error("Dataset not found.")
    st.stop()

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    st.error("Trained model or scaler file not found.")
    st.stop()

# --------------------------------
# LOAD DATA AND MODEL
# --------------------------------

try:
    data = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# --------------------------------
# PREPARE DATA
# --------------------------------

if "Class" not in data.columns:
    st.error("Class column is missing from dataset.")
    st.stop()

X = data.drop(columns=["Class"])

y = data["Class"].astype(int)

X = X.reindex(columns=model.feature_names_in_)

if "Amount" in X.columns:
    X = X.copy()
    X["Amount"] = scaler.transform(
        X[["Amount"]]
    ).ravel()

# --------------------------------
# MODEL PREDICTIONS
# --------------------------------

try:
    y_pred = model.predict(X)

except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# --------------------------------
# CALCULATE METRICS
# --------------------------------

accuracy = accuracy_score(y, y_pred) * 100

precision = precision_score(
    y, y_pred, zero_division=0
) * 100

recall = recall_score(
    y, y_pred, zero_division=0
) * 100

f1 = f1_score(
    y, y_pred, zero_division=0
) * 100

# --------------------------------
# DISPLAY METRICS
# --------------------------------

st.markdown("### 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}%")
col2.metric("Precision", f"{precision:.2f}%")
col3.metric("Recall", f"{recall:.2f}%")
col4.metric("F1 Score", f"{f1:.2f}%")

st.markdown("---")

# --------------------------------
# CONFUSION MATRIX
# --------------------------------

st.markdown("### 🔍 Confusion Matrix")

cm = confusion_matrix(
    y,
    y_pred,
    labels=[0, 1]
)

fig = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted Genuine", "Predicted Fraud"],
    y=["Actual Genuine", "Actual Fraud"],
    colorscale="Blues",
    showscale=True
)

fig.update_layout(
    title="Confusion Matrix",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------
# CLASSIFICATION REPORT
# --------------------------------

st.markdown("---")

st.markdown("### 📋 Classification Report")

report = classification_report(
    y,
    y_pred,
    target_names=["Genuine", "Fraud"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# --------------------------------
# SUMMARY
# --------------------------------

st.markdown("---")

st.markdown("### 📝 Evaluation Summary")

st.write(f"Total Transactions Evaluated: {len(data):,}")

st.write(
    f"Correct Predictions: {(y == y_pred).sum():,}"
)

st.write(
    f"Incorrect Predictions: {(y != y_pred).sum():,}"
)

st.info(
    "Note: These evaluation results are calculated "
    "using the sample dataset included with this application. "
    "They are intended for educational and demonstration purposes."
)

# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.caption(
    "Bank Fraud Detection System | "
    "Machine Learning Project | "
    "Developed by Laxmi Ramappa Palled"
)