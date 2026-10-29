"""FraudLens AI - Streamlit frontend (Phase 1: rule-based MVP).

Run with:  streamlit run app.py (from the project root)
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.src import __version__
from backend.src.data_loader import DataLoadError, load_csv, load_sample
from backend.src.detection_engine import (
    RULE_NAMES,
    DetectionConfig,
    analyze,
    explain_score,
)
from backend.src.reporting import DISCLAIMER, analyzed_to_csv

st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")

LEVEL_ORDER = ["High", "Medium", "Low", "None"]
LEVEL_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2a9d8f", "None": "#9aa5b1"}


# ---------------------------------------------------------------- data + config
@st.cache_data(show_spinner=False)
def _cached_sample():
    return load_sample()


@st.cache_data(show_spinner="Analyzing transactions...")
def _cached_analyze(df: pd.DataFrame, cfg_dict: dict):
    return analyze(df, DetectionConfig(**cfg_dict))


def sidebar_config() -> DetectionConfig:
    d = DetectionConfig()
    with st.sidebar.expander("Detection thresholds", expanded=False):
        st.caption("Changes re-run the analysis immediately.")
        large = st.number_input("R001 amount / baseline ratio", 1.0, 100.0, d.large_ratio_threshold, 0.5)
        burst = st.number_input("R002 transactions in last hour", 1, 100, d.burst_threshold)
        np_ratio = st.number_input("R003 new-payee ratio", 1.0, 100.0, d.new_payee_ratio_threshold, 0.5)
        np_abs = st.number_input("R003 new-payee absolute amount (INR)", 0.0, 10_000_000.0,
                                 d.new_payee_absolute_amount, 1000.0)
        dup_win = st.number_input("R004 duplicate window (minutes)", 1, 1440, d.duplicate_window_minutes)
        n_start, n_end = st.slider("R005 night window (hours)", 0, 24, (d.night_start_hour, d.night_end_hour))
        n_ratio = st.number_input("R005 min amount / baseline ratio", 1.0, 100.0, d.night_min_ratio, 0.5)
        med = st.slider("Medium risk starts at score", 1, 99, d.medium_threshold)
        high = st.slider("High risk starts at score", 2, 100, d.high_threshold)
        if high <= med:
            st.warning("High threshold must be above Medium; using defaults.")
            med, high = d.medium_threshold, d.high_threshold
    return DetectionConfig(
        large_ratio_threshold=float(large), burst_threshold=int(burst),
        new_payee_ratio_threshold=float(np_ratio), new_payee_absolute_amount=float(np_abs),
        duplicate_window_minutes=int(dup_win), night_start_hour=int(n_start), night_end_hour=int(n_end),
        night_min_ratio=float(n_ratio), medium_threshold=int(med), high_threshold=int(high),
    )


def load_data():
    st.sidebar.header("Data")
    source = st.sidebar.radio("Source", ["Synthetic sample data", "Upload CSV"])
    if source == "Synthetic sample data":
        return _cached_sample()
    up = st.sidebar.file_uploader("CSV file (max 5 MB)", type=["csv"])
    if up is None:
        st.sidebar.info("Upload a CSV, or switch to the sample data.")
        return None
    try:
        return load_csv(up, filename=up.name)
    except DataLoadError as exc:
        st.sidebar.error(str(exc))
        return None


# ---------------------------------------------------------------- helpers
def money_by_currency(totals: dict) -> str:
    if not totals:
        return "-"
    return " | ".join(f"{cur} {val:,.2f}" for cur, val in sorted(totals.items()))


def show_validation(val) -> bool:
    """Display validation outcome. Returns True if analysis can proceed."""
    if val.file_errors:
        for e in val.file_errors:
            st.error(e)
        st.caption("Required columns: transaction_id, timestamp, amount, currency, account_avg_amount, "
                   "transactions_last_hour, is_new_payee, merchant_category, channel.")
        return False
    if not val.row_issues.empty:
        st.warning(f"{val.invalid_row_count} of {val.total_rows} rows were excluded because they need correction.")
        with st.expander("Rows requiring correction"):
            st.dataframe(val.row_issues, hide_index=True)
    if len(val.clean) == 0:
        st.error("No valid rows remain after validation.")
        return False
    return True


# ---------------------------------------------------------------- pages
def page_overview(res):
    df = res.data
    st.subheader("Overview")
    flagged = df[df["n_rules"] > 0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", f"{len(df):,}")
    c2.metric("High risk", int((df["risk_level"] == "High").sum()))
    c3.metric("Medium risk", int((df["risk_level"] == "Medium").sum()))
    c4.metric("Low risk", int((df["risk_level"] == "Low").sum()))
    st.caption("Total value (per currency, never converted): " + money_by_currency(res.currency_totals))

    left, right = st.columns(2)
    dist = df["risk_level"].value_counts().reindex(LEVEL_ORDER, fill_value=0).reset_index()
    dist.columns = ["risk_level", "count"]
    fig = px.bar(dist, x="risk_level", y="count", color="risk_level", color_discrete_map=LEVEL_COLORS,
                 title="Risk level distribution")
    fig.update_layout(showlegend=False)
    left.plotly_chart(fig)

    daily = df.assign(day=df["timestamp"].dt.date).groupby("day").size().reset_index(name="transactions")
    right.plotly_chart(px.line(daily, x="day", y="transactions", title="Daily transaction volume"))

    st.markdown("**Recent alerts**")
    if flagged.empty:
        st.info("No rules were triggered.")
    else:
        recent = flagged.sort_values("timestamp", ascending=False).head(8)
        st.dataframe(recent[["transaction_id", "timestamp", "amount", "currency", "risk_score",
                             "risk_level", "rules_triggered"]], hide_index=True)
    st.caption(DISCLAIMER)


def page_explorer(res):
    df = res.data
    st.subheader("Transaction Explorer")
    f1, f2, f3 = st.columns(3)
    query = f1.text_input("Search transaction ID")
    levels = f2.multiselect("Risk level", LEVEL_ORDER, default=LEVEL_ORDER)
    channels = f3.multiselect("Channel", sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))
    g1, g2, g3 = st.columns(3)
    dmin, dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    date_range = g1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    amt_lo, amt_hi = float(df["amount"].min()), float(df["amount"].max())
    amount_range = g2.slider("Amount range (all currencies)", amt_lo, max(amt_hi, amt_lo + 1.0), (amt_lo, max(amt_hi, amt_lo + 1.0)))
    rules = g3.multiselect("Triggered rule", list(RULE_NAMES))

    mask = df["risk_level"].isin(levels) & df["channel"].isin(channels)
    mask &= df["amount"].between(*amount_range)
    if query:
        mask &= df["transaction_id"].str.contains(query.strip(), case=False, regex=False)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= df["timestamp"].dt.date.between(date_range[0], date_range[1])
    if rules:
        mask &= df["rules_triggered"].apply(lambda s: any(r in s.split(";") for r in rules))
    view = df[mask]

    s1, s2, s3 = st.columns(3)
    sort_col = s1.selectbox("Sort by", ["risk_score", "timestamp", "amount"])
    ascending = s2.checkbox("Ascending", value=False)
    page_size = s3.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    view = view.sort_values(sort_col, ascending=ascending)
    pages = max(1, -(-len(view) // page_size))
    page = st.number_input(f"Page (1-{pages})", 1, pages, 1)
    chunk = view.iloc[(page - 1) * page_size: page * page_size]
    st.caption(f"{len(view):,} matching transactions")
    cols = ["transaction_id", "timestamp", "amount", "currency", "account_avg_amount", "transactions_last_hour",
            "is_new_payee", "channel", "risk_score", "risk_level", "rules_triggered"]
    st.dataframe(chunk[cols], hide_index=True)


def page_investigation(res):
    df = res.data
    st.subheader("Alert Investigation")
    flagged = df[df["n_rules"] > 0].sort_values("risk_score", ascending=False)
    if flagged.empty:
        st.info("No alerts to investigate.")
        return
    labels = {r.transaction_id: f"{r.transaction_id}  |  score {r.risk_score:g}  |  {r.risk_level}"
              for r in flagged.itertuples()}
    tid = st.selectbox("Select an alert", list(labels), format_func=labels.get)
    row = df[df["transaction_id"] == tid].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Illustrative risk score", f"{row['risk_score']:g} / 100")
    c2.metric("Risk level", row["risk_level"])
    c3.metric("Amount", f"{row['currency']} {row['amount']:,.2f}")
    st.caption("The score is an illustrative indicator, not a probability of fraud.")

    st.markdown("**Original transaction values**")
    original = {k: (str(v) if k == "timestamp" else v) for k, v in row.items()
                if k not in ("findings", "risk_score", "risk_level", "rules_triggered", "n_rules")}
    st.dataframe(pd.DataFrame([original]), hide_index=True)

    st.markdown("**Triggered rules (deterministic)**")
    for f in row["findings"]:
        with st.expander(f"{f['rule_id']} - {f['rule_name']}  (+{f['points']:g}, {f['severity']})", expanded=True):
            st.write(f["explanation"])
            st.code(json.dumps(f["evidence"], indent=2, default=str), language="json")

    st.markdown("**How the score was calculated**")
    st.text(explain_score(row["findings"], res.config))
    st.info("AI explanation (Gemma 3:4B) arrives in Phase 2. Human review workflow and case notes arrive in Phase 3.")


def page_analytics(res, val):
    df = res.data
    st.subheader("Analytics")
    a, b = st.columns(2)
    rule_counts = (df["rules_triggered"].str.split(";").explode().replace("", pd.NA).dropna()
                   .value_counts().rename_axis("rule").reset_index(name="count"))
    if rule_counts.empty:
        a.info("No rules triggered.")
    else:
        a.plotly_chart(px.bar(rule_counts, x="rule", y="count", title="Rule trigger frequency"))
    cur = b.selectbox("Currency for amount distribution", sorted(df["currency"].unique()))
    b.plotly_chart(px.histogram(df[df["currency"] == cur], x="amount", nbins=30, title=f"Amounts ({cur})"))

    trend = df.assign(day=df["timestamp"].dt.date).groupby("day")["risk_score"].mean().reset_index()
    st.plotly_chart(px.line(trend, x="day", y="risk_score", title="Average risk score per day"))

    st.markdown("**Data-quality summary**")
    st.write(f"Rows read: {val.total_rows:,} | valid: {len(val.clean):,} | excluded: {val.invalid_row_count:,}")
    if not val.row_issues.empty:
        st.dataframe(val.row_issues.groupby(["column", "problem"]).size().reset_index(name="rows"), hide_index=True)
    st.markdown("**Active detection configuration**")
    st.json(res.config.to_dict())
    st.caption("Statistical anomaly detection is planned for Phase 4.")


def page_reports(res):
    st.subheader("Reports")
    include_desc = st.checkbox("Include free-text descriptions in export", value=False)
    st.download_button("Download analyzed transactions (CSV)", analyzed_to_csv(res.data, include_desc),
                       file_name="fraudlens_analysis.csv", mime="text/csv")
    st.caption(f"Analysis time: {res.analyzed_at} | Ruleset version: {res.ruleset_version}")
    st.caption(DISCLAIMER)
    st.info("Investigation summary reports arrive in Phase 3.")


def page_settings(res, val):
    st.subheader("Settings and System Health")
    st.write(f"**Application version:** {__version__}  |  **Ruleset version:** {res.ruleset_version}")
    st.write("**AI status:** not connected yet (local Gemma 3:4B via Ollama is added in Phase 2).")
    st.write(f"**Dataset validation:** {len(val.clean):,} valid of {val.total_rows:,} rows.")
    st.markdown("**Privacy and limitations**")
    st.markdown(
        "- Everything runs locally. No data is sent to external services.\n"
        "- Sample data is entirely fictional.\n"
        "- Rule-based scores are illustrative, not fraud probabilities, and have not been evaluated on real data.\n"
        "- This is an educational prototype, not a certified banking security product."
    )


# ---------------------------------------------------------------- main
def main():
    st.title("🔍 FraudLens AI")
    st.caption("Explainable financial fraud detection and investigation - educational prototype. "
               "Flags are for human review and never confirm fraud.")
    val = load_data()
    cfg = sidebar_config()
    if val is None:
        st.info("Choose a data source in the sidebar.")
        return
    if not show_validation(val):
        return
    res = _cached_analyze(val.clean, cfg.to_dict())

    page = st.sidebar.radio("Page", ["Overview", "Transaction Explorer", "Alert Investigation",
                                     "Analytics", "Reports", "Settings"])
    {
        "Overview": lambda: page_overview(res),
        "Transaction Explorer": lambda: page_explorer(res),
        "Alert Investigation": lambda: page_investigation(res),
        "Analytics": lambda: page_analytics(res, val),
        "Reports": lambda: page_reports(res),
        "Settings": lambda: page_settings(res, val),
    }[page]()


if __name__ == "__main__":
    main()
