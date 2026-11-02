import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Transaction Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# LOAD CSS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CSS_PATH = BASE_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# --------------------------------
# LOAD SAMPLE DATASET
# --------------------------------

DATA_PATH = BASE_DIR / "data" / "creditcard_sample.csv"

st.title("📊 Transaction Dashboard")
st.subheader("Bank Fraud Detection - Analytics Overview")

st.markdown("---")

if not DATA_PATH.exists():
    st.error("Dataset not found. Please check the data folder.")
    st.stop()

try:
    data = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# --------------------------------
# DATA VALIDATION
# --------------------------------

if "Class" not in data.columns or "Amount" not in data.columns:
    st.error("Dataset is missing required columns.")
    st.stop()

data["Class"] = data["Class"].astype(int)

# --------------------------------
# CALCULATE STATISTICS
# --------------------------------

total_transactions = len(data)

fraud_transactions = int((data["Class"] == 1).sum())

genuine_transactions = int((data["Class"] == 0).sum())

fraud_percentage = (
    (fraud_transactions / total_transactions) * 100
    if total_transactions > 0 else 0
)

genuine_percentage = (
    (genuine_transactions / total_transactions) * 100
    if total_transactions > 0 else 0
)

# --------------------------------
# DISPLAY METRICS
# --------------------------------

st.markdown("### 📈 Transaction Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Transactions",
    f"{fraud_transactions:,}"
)

col3.metric(
    "Genuine Transactions",
    f"{genuine_transactions:,}"
)

col4.metric(
    "Fraud Rate",
    f"{fraud_percentage:.2f}%"
)

st.markdown("---")

# --------------------------------
# CHARTS
# --------------------------------

st.markdown("### 📊 Transaction Analysis")

chart_col1, chart_col2 = st.columns(2)

# Fraud vs Genuine Pie Chart

with chart_col1:

    st.markdown("#### Fraud vs Genuine Transactions")

    transaction_counts = pd.DataFrame({
        "Transaction Type": ["Genuine", "Fraud"],
        "Count": [
            genuine_transactions,
            fraud_transactions
        ]
    })

    fig_pie = px.pie(
        transaction_counts,
        names="Transaction Type",
        values="Count",
        hole=0.45,
        title="Transaction Distribution"
    )

    fig_pie.update_traces(
        textinfo="percent+label"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# Transaction Amount Histogram

with chart_col2:

    st.markdown("#### Transaction Amount Distribution")

    fig_hist = px.histogram(
        data,
        x="Amount",
        nbins=40,
        title="Transaction Amount Analysis",
        labels={
            "Amount": "Transaction Amount"
        }
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

st.markdown("---")

# --------------------------------
# FRAUD ANALYSIS
# --------------------------------

st.markdown("### 🔍 Fraud Analysis")

fraud_data = data[data["Class"] == 1]

genuine_data = data[data["Class"] == 0]

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:

    st.markdown("#### Fraud Transaction Details")

    if len(fraud_data) > 0:

        st.write(
            "Average Fraud Transaction Amount:"
        )

        st.metric(
            "Average Amount",
            f"{fraud_data['Amount'].mean():.2f}"
        )

        st.metric(
            "Maximum Amount",
            f"{fraud_data['Amount'].max():.2f}"
        )

    else:
        st.info("No fraud transactions found.")

with analysis_col2:

    st.markdown("#### Genuine Transaction Details")

    if len(genuine_data) > 0:

        st.metric(
            "Average Amount",
            f"{genuine_data['Amount'].mean():.2f}"
        )

        st.metric(
            "Maximum Amount",
            f"{genuine_data['Amount'].max():.2f}"
        )

    else:
        st.info("No genuine transactions found.")

st.markdown("---")

# --------------------------------
# DATA PREVIEW
# --------------------------------

st.markdown("### 📋 Transaction Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
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
