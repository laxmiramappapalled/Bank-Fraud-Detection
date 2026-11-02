import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"

with open(css_path, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

st.title("🏦 Bank Fraud Detection System")

st.markdown("""
### AI-Powered Transaction Monitoring

Welcome to the Bank Fraud Detection application.

This application uses Machine Learning techniques
to analyze banking transactions and identify
potentially fraudulent activities.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("📊 Transaction Analysis")

with col2:
    st.success("🤖 Machine Learning")

with col3:
    st.warning("🔍 Fraud Detection")

st.divider()

st.subheader("Project Overview")

st.write(
    "Use the navigation pages on the left to explore "
    "the dashboard, predict transactions, and view reports."
)

st.caption("Project Status: Development in Progress")
st.caption("For educational and demonstration purposes only.")