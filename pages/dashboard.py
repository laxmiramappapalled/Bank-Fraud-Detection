import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Transaction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load CSS
ROOT_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT_DIR / "assets" / "style.css"

if CSS_PATH.exists():
    with open(CSS_PATH, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# Dashboard title
st.title("📊 Transaction Dashboard")
st.markdown("### Bank Fraud Detection - Analytics Overview")

st.divider()

# Dataset path
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"

if not DATA_PATH.exists():
    st.error("Dataset not found. Please check the data folder.")
    st.stop()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

# Transaction statistics
total_transactions = len(data)
fraud_transactions = int((data["Class"] == 1).sum())
genuine_transactions = int((data["Class"] == 0).sum())
fraud_rate = (fraud_transactions / total_transactions) * 100

# Transaction overview
st.subheader("Transaction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
col3.metric("Genuine Transactions", f"{genuine_transactions:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

st.divider()

# Transaction distribution
st.subheader("Transaction Distribution")

distribution = pd.DataFrame({
    "Transaction Type": ["Genuine", "Fraud"],
    "Count": [genuine_transactions, fraud_transactions]
})

fig_pie = px.pie(
    distribution,
    names="Transaction Type",
    values="Count",
    hole=0.45,
    title="Genuine vs Fraud Transactions"
)

st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# Transaction amount analysis
st.subheader("Transaction Amount Analysis")

fig_amount = px.histogram(
    data,
    x="Amount",
    color="Class",
    nbins=50,
    title="Transaction Amount Distribution",
    labels={
        "Amount": "Transaction Amount",
        "Class": "Transaction Type"
    }
)

st.plotly_chart(fig_amount, use_container_width=True)

st.divider()

# Dataset preview
st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)

st.caption("Source: Credit Card Fraud Detection Dataset")
st.caption("For educational and demonstration purposes only.")
