import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# -----------------------------
# Cached loaders (define BEFORE use)
# -----------------------------
@st.cache_data
def load_snapshot():
    return pd.read_csv("latest_snapshot.csv")

@st.cache_data
def load_history():
    return pd.read_csv("dataset_full.csv")

# -----------------------------
# Controls (sidebar)
# -----------------------------
st.sidebar.header("Controls")
AUTO_REFRESH = st.sidebar.toggle("Auto-refresh (every 5 min)", value=False)
REFRESH_SECONDS = 300

show_debug = st.sidebar.toggle("Show debug", value=False)

# -----------------------------
# Load data (once)
# -----------------------------
df = load_snapshot()
history = load_history()

if show_debug:
    st.subheader("DEBUG: Risk bucket counts")
    st.dataframe(df["risk_bucket"].value_counts())

# -----------------------------
# Basic cleaning
# -----------------------------
df["predicted_RUL"] = pd.to_numeric(df["predicted_RUL"], errors="coerce")
df = df.dropna(subset=["asset_id", "predicted_RUL", "risk_bucket"])

# -----------------------------
# Filters
# -----------------------------
all_buckets = sorted(df["risk_bucket"].unique().tolist())
bucket_filter = st.sidebar.multiselect("Risk bucket", options=all_buckets, default=all_buckets)

min_rul = float(df["predicted_RUL"].min())
max_rul = float(df["predicted_RUL"].max())
rul_range = st.sidebar.slider("Predicted RUL (days)", min_value=min_rul, max_value=max_rul, value=(min_rul, max_rul))

view = df[
    (df["risk_bucket"].isin(bucket_filter)) &
    (df["predicted_RUL"] >= rul_range[0]) &
    (df["predicted_RUL"] <= rul_range[1])
].copy()

# -----------------------------
# Title + KPIs
# -----------------------------
st.title("Predictive Maintenance — RUL & Risk Dashboard")
st.caption("Asset-level and fleet-level predictive maintenance using vibration features and predicted remaining useful life.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Assets Monitored", df["asset_id"].nunique())
col2.metric("High Risk (RED)", (df["risk_bucket"] == "RED - Immediate Action").sum())
col3.metric("Avg Predicted RUL (days)", round(df["predicted_RUL"].mean(), 1))
col4.metric("Assets in View", view["asset_id"].nunique())

st.divider()

# -----------------------------
# Fleet ranking + Top risky
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Fleet Risk Ranking (Filtered View)")
    cols = ["asset_id", "predicted_RUL", "risk_bucket", "RMS", "Kurtosis", "Fault_Energy"]
    st.dataframe(view.sort_values("predicted_RUL")[cols], use_container_width=True)

with right:
    st.subheader("Top 10 Most Urgent")
    top10 = df.sort_values("predicted_RUL").head(10)[["asset_id", "predicted_RUL", "risk_bucket"]]
    st.dataframe(top10, use_container_width=True)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered snapshot (CSV)",
        data=csv,
        file_name="predictive_maintenance_snapshot_filtered.csv",
        mime="text/csv"
    )

st.divider()

# -----------------------------
# Asset drill-down
# -----------------------------
st.subheader("Asset Drill-Down")

asset = st.selectbox("Select Asset", sorted(df["asset_id"].unique().tolist()))
row = df[df["asset_id"] == asset].sort_values("predicted_RUL").head(1)

if row.empty:
    st.warning("No snapshot row found for this asset.")
    st.stop()

# Recommended Actions (Asset)
st.subheader("Recommended Maintenance Action")
bucket = str(row["risk_bucket"].iloc[0]).upper()

if "RED" in bucket:
    st.error("🚨 Immediate action: Inspect bearing, check lubrication and alignment. Plan shutdown within 24–48 hours.")
elif ("AMBER" in bucket) or ("YELLOW" in bucket) or ("PLAN" in bucket):
    st.warning("⚠️ Planned maintenance: Increase monitoring frequency and schedule maintenance in next planned window.")
else:
    st.success("✅ Normal operation: Continue routine monitoring and preventive maintenance.")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Risk Bucket", row["risk_bucket"].iloc[0])
a2.metric("Predicted RUL (days)", float(row["predicted_RUL"].iloc[0]))
a3.metric("RMS", float(row["RMS"].iloc[0]))
a4.metric("Kurtosis", float(row["Kurtosis"].iloc[0]))

b1, b2 = st.columns([1, 2])

with b1:
    st.write("**Current Feature Values**")
    st.dataframe(
        row[["asset_id", "predicted_RUL", "risk_bucket", "RMS", "Kurtosis", "Fault_Energy"]],
        use_container_width=True
    )

with b2:
    asset_hist = history[history["asset_id"] == asset].copy()

    if asset_hist.empty:
        st.warning("No historical data found for this asset in dataset_full.csv.")
    else:
        asset_hist["day"] = pd.to_numeric(asset_hist["day"], errors="coerce")
        asset_hist = asset_hist.dropna(subset=["day"]).sort_values("day")

        show_rms = st.checkbox("Show RMS trend", value=False)
        show_kurt = st.checkbox("Show Kurtosis trend", value=False)
        smooth = st.checkbox("Show 7-day rolling average", value=True)

        st.write("**Degradation Trend**")

        fig, ax = plt.subplots()
        ax.plot(asset_hist["day"], asset_hist["Fault_Energy"], label="Fault_Energy")

        if smooth and len(asset_hist) >= 7:
            ax.plot(asset_hist["day"], asset_hist["Fault_Energy"].rolling(7).mean(), label="Fault_Energy (7-day avg)")

        if show_rms and "RMS" in asset_hist.columns:
            ax.plot(asset_hist["day"], asset_hist["RMS"], label="RMS")
        if show_kurt and "Kurtosis" in asset_hist.columns:
            ax.plot(asset_hist["day"], asset_hist["Kurtosis"], label="Kurtosis")

        if "failure_day" in asset_hist.columns:
            try:
                fd = asset_hist["failure_day"].iloc[0]
                ax.axvline(fd, linestyle="--")
                ax.text(fd, ax.get_ylim()[1] * 0.95, "failure_day", rotation=90, va="top")
            except Exception:
                pass

        ax.set_xlabel("Day")
        ax.set_ylabel("Signal / Feature Value")
        ax.legend()
        st.pyplot(fig)

# -----------------------------
# Auto-refresh
# -----------------------------
if AUTO_REFRESH:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
