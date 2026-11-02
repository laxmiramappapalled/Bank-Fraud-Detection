import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from pathlib import Path
from io import StringIO

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

RESULTS_PATH = BASE_DIR / "model" / "evaluation_results.pkl"
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
st.subheader("Independent Test Performance Analysis")

st.markdown("---")

# --------------------------------
# CHECK RESULTS FILE
# --------------------------------

if not RESULTS_PATH.exists():
    st.error("Evaluation results not found. Please train the model first.")
    st.stop()

# --------------------------------
# LOAD EVALUATION RESULTS
# --------------------------------

try:
    results = joblib.load(RESULTS_PATH)

except Exception as e:
    st.error(f"Error loading evaluation results: {e}")
    st.stop()

# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

st.markdown("### 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{results['accuracy']:.2f}%")
col2.metric("Precision", f"{results['precision']:.2f}%")
col3.metric("Recall", f"{results['recall']:.2f}%")
col4.metric("F1 Score", f"{results['f1_score']:.2f}%")

st.markdown("---")

# --------------------------------
# CONFUSION MATRIX
# --------------------------------

st.markdown("### 🔍 Confusion Matrix")

cm = results["confusion_matrix"]

fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Predicted Genuine", "Predicted Fraud"],
        y=["Actual Genuine", "Actual Fraud"],
        colorscale="Blues",
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        hovertemplate=(
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}<extra></extra>"
        )
    )
)

fig.update_layout(
    title="Independent Test Confusion Matrix",
    xaxis_title="Predicted Class",
    yaxis_title="Actual Class",
    height=450
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

report = results["classification_report"]

try:
    report_df = pd.read_fwf(
        StringIO(report),
        index_col=0
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

except Exception:
    st.code(report)

# --------------------------------
# EVALUATION SUMMARY
# --------------------------------

st.markdown("---")

st.markdown("### 📝 Evaluation Summary")

st.write(
    f"Total Transactions Evaluated: "
    f"{results['total_transactions']:,}"
)

st.write(
    f"Correct Predictions: "
    f"{results['correct_predictions']:,}"
)

st.write(
    f"Incorrect Predictions: "
    f"{results['total_transactions'] - results['correct_predictions']:,}"
)

st.info(
    "These results are calculated using an independent "
    "20% test dataset that was not used during model training. "
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