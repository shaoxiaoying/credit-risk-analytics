"""Streamlit explorer for the prepared UCI credit risk dataset."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data/processed/credit_clients_clean.csv"

st.set_page_config(page_title="Credit Risk Explorer", page_icon="📊", layout="wide")
st.title("Credit Risk Explorer")
st.caption("Historical UCI sample • descriptive exploration, not a lending decision tool")
if not CSV.exists():
    st.info("Prepare the data first by running `python scripts/run_analysis.py` from the project folder.")
    st.stop()

df = pd.read_csv(CSV)
rate = df.default_payment_next_month.mean()
a, b, c = st.columns(3)
a.metric("Clients in sample", f"{len(df):,}")
b.metric("Observed default rate", f"{rate:.1%}")
c.metric("Source period", "2005–2006")

st.subheader("Explore observed default rates")
left, right = st.columns(2)
with left:
    statuses = sorted(df.pay_0.unique())
    selected = st.multiselect("Recent repayment status codes", statuses, default=statuses)
    view = df[df.pay_0.isin(selected)]
    chart = view.groupby("pay_0").default_payment_next_month.agg(["mean", "count"]).reset_index()
    st.bar_chart(chart.set_index("pay_0")["mean"], y_label="Observed default rate")
    st.dataframe(chart.rename(columns={"pay_0":"PAY_0 status", "mean":"Default rate", "count":"Clients"}), hide_index=True)
with right:
    bands = pd.cut(df.limit_bal, [-float("inf"), 50000, 100000, 200000, 500000, float("inf")],
                   labels=["< NT$50k", "NT$50k–99k", "NT$100k–199k", "NT$200k–499k", "NT$500k+"])
    band_summary = df.assign(credit_limit_band=bands).groupby("credit_limit_band", observed=False).default_payment_next_month.agg(["mean", "count"])
    st.bar_chart(band_summary["mean"], y_label="Observed default rate")
    st.dataframe(band_summary.rename(columns={"mean":"Default rate", "count":"Clients"}), hide_index=True)

st.subheader("Sample records")
st.dataframe(df.drop(columns=["client_id"], errors="ignore").head(20), use_container_width=True, hide_index=True)
st.caption("Rates describe this historical sample and are not causal or predictive statements about an individual.")
