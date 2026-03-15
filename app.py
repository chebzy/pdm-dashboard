import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# -----------------------------
# Config
# -----------------------------
REFRESH_SECONDS = 300
SNAPSHOT_FILE = "latest_snapshot.csv"
HISTORY_FILE = "dataset_full.csv"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning
    numeric_cols = ["predicted_RUL", "RMS", "Kurtosis", "Fault_Energy"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["asset_id", "predicted_RUL", "risk_bucket"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])

    if "risk_bucket" in df.columns:
        df["risk_bucket"] = df["risk_bucket"].astype(str)

    return df


@st.cache_data
def load_history(path: str) -> pd.DataFrame:
    hist = pd.read_csv(path)

    # Clean expected columns if present
    for col in ["day", "RMS", "Kurtosis", "Fault_Energy", "failure_day"]:
        if col in hist.columns:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    if "asset_id" in hist.columns:
        hist["asset_id"] = hist["asset_id"].astype(str)

    return hist


def get_action_text(bucket: str) -> tuple[str, str]:
    b = str(bucket).upper()

    if "RED" in b:
        return (
            "error",
            "🚨 Immediate action: Inspect bearing, verify lubrication, alignment, and vibration source. Plan shutdown within 24–48 hours."
        )
    elif ("AMBER" in b) or ("YELLOW" in b) or ("PLAN" in b):
        return (
            "warning",
            "⚠️ Planned maintenance: Increase monitoring frequency and schedule inspection during the next maintenance window."
        )
    else:
        return (
            "success",
            "✅ Normal operation: Continue routine monitoring and preventive maintenance."
        )


def safe_metric(value, digits=1):
    try:
        return round(float(value), digits)
    except Exception:
        return "N/A"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
AUTO_REFRESH = st.sidebar.toggle("Auto-refresh (every 5 min)", value=False)

# -----------------------------
# Load data
# -----------------------------
try:
    df = load_snapshot(SNAPSHOT_FILE)
except FileNotFoundError:
    st.error(f"Snapshot file not found: {SNAPSHOT_FILE}")
    st.stop()
except Exception as e:
    st.error(f"Error loading snapshot: {e}")
    st.stop()

if df.empty:
    st.warning("Snapshot file loaded, but it contains no usable rows.")
    st.stop()

# Optional debug
with st.expander("Debug: Risk bucket counts"):
    if "risk_bucket" in df.columns:
        st.dataframe(df["risk_bucket"].value_counts().reset_index().rename(
            columns={"index": "risk_bucket", "risk_bucket": "count"}
        ))
    else:
        st.warning("Column 'risk_bucket' not found.")

# -----------------------------
# Filters
# -----------------------------
all_buckets = sorted(df["risk_bucket"].dropna().unique().tolist())

bucket_filter = st.sidebar.multiselect(
    "Risk bucket",
    options=all_buckets,
    default=all_buckets
)

min_rul = float(df["predicted_RUL"].min())
max_rul = float(df["predicted_RUL"].max())

rul_range = st.sidebar.slider(
    "Predicted RUL (days)",
    min_value=min_rul,
    max_value=max_rul,
    value=(min_rul, max_rul)
)

view = df[
    (df["risk_bucket"].isin(bucket_filter)) &
    (df["predicted_RUL"] >= rul_range[0]) &
    (df["predicted_RUL"] <= rul_range[1])
].copy()

# -----------------------------
# Title + KPIs
# -----------------------------
st.title("Predictive Maintenance Dashboard")
st.caption("Fleet-level monitoring, asset ranking, and maintenance action support")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Assets Monitored", df["asset_id"].nunique())
col2.metric("High Risk (RED)", df["risk_bucket"].astype(str).str.upper().str.contains("RED").sum())
col3.metric("Avg Predicted RUL (days)", safe_metric(df["predicted_RUL"].mean()))
col4.metric("Assets in View", view["asset_id"].nunique())

st.divider()

# -----------------------------
# Fleet summary charts
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Risk Bucket Distribution")
    bucket_counts = view["risk_bucket"].value_counts()

    fig1, ax1 = plt.subplots()
    bucket_counts.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Risk Bucket")
    ax1.set_ylabel("Asset Count")
    ax1.set_title("Filtered Fleet Risk Distribution")
    st.pyplot(fig1)

with c2:
    st.subheader("Predicted RUL Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(view["predicted_RUL"], bins=15)
    ax2.set_xlabel("Predicted RUL (days)")
    ax2.set_ylabel("Count")
    ax2.set_title("Filtered Asset RUL Distribution")
    st.pyplot(fig2)

st.divider()

# -----------------------------
# Fleet ranking + Top risky
# -----------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Fleet Risk Ranking (Filtered View)")
    cols = [c for c in ["asset_id", "predicted_RUL", "risk_bucket", "RMS", "Kurtosis", "Fault_Energy"] if c in view.columns]
    st.dataframe(
        view.sort_values("predicted_RUL")[cols],
        use_container_width=True
    )

with right:
    st.subheader("Top 10 Most Urgent")
    urgent_cols = [c for c in ["asset_id", "predicted_RUL", "risk_bucket"] if c in df.columns]
    top10 = df.sort_values("predicted_RUL").head(10)[urgent_cols]
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

asset_list = sorted(df["asset_id"].astype(str).unique().tolist())
asset = st.selectbox("Select Asset", asset_list)

row = df[df["asset_id"].astype(str) == str(asset)].sort_values("predicted_RUL").head(1)

if row.empty:
    st.warning("No snapshot row found for this asset.")
    st.stop()

# -----------------------------
# Recommended Actions
# -----------------------------
st.subheader("Recommended Maintenance Action")

bucket = row["risk_bucket"].iloc[0]
action_type, action_message = get_action_text(bucket)

if action_type == "error":
    st.error(action_message)
elif action_type == "warning":
    st.warning(action_message)
else:
    st.success(action_message)

a1, a2, a3, a4 = st.columns(4)
a1.metric("Risk Bucket", row["risk_bucket"].iloc[0])
a2.metric("Predicted RUL (days)", safe_metric(row["predicted_RUL"].iloc[0]))
a3.metric("RMS", safe_metric(row["RMS"].iloc[0]) if "RMS" in row.columns else "N/A")
a4.metric("Kurtosis", safe_metric(row["Kurtosis"].iloc[0]) if "Kurtosis" in row.columns else "N/A")

b1, b2 = st.columns([1, 2])

with b1:
    st.write("**Current Feature Values**")
    display_cols = [c for c in ["asset_id", "predicted_RUL", "risk_bucket", "RMS", "Kurtosis", "Fault_Energy"] if c in row.columns]
    st.dataframe(row[display_cols], use_container_width=True)

    # Extra interpretation
    st.write("**Interpretation**")
    rul_val = float(row["predicted_RUL"].iloc[0])

    if rul_val <= 7:
        st.write("Very short remaining life. Prioritize inspection immediately.")
    elif rul_val <= 30:
        st.write("Failure risk is developing. Plan intervention soon.")
    else:
        st.write("Asset appears stable for now, but continue monitoring trend.")

with b2:
    try:
        history = load_history(HISTORY_FILE)
    except FileNotFoundError:
        st.warning(f"History file not found: {HISTORY_FILE}")
        history = pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load history: {e}")
        history = pd.DataFrame()

    if history.empty:
        st.warning("No historical data available.")
    else:
        asset_hist = history[history["asset_id"].astype(str) == str(asset)].copy()

        if asset_hist.empty:
            st.warning("No historical data found for this asset in dataset_full.csv.")
        else:
            asset_hist = asset_hist.dropna(subset=["day"]).sort_values("day")

            show_rms = st.checkbox("Show RMS trend", value=False)
            show_kurt = st.checkbox("Show Kurtosis trend", value=False)
            smooth = st.checkbox("Show 7-day rolling average", value=True)

            st.write("**Degradation Trend**")

            fig, ax = plt.subplots()

            if "Fault_Energy" in asset_hist.columns:
                ax.plot(asset_hist["day"], asset_hist["Fault_Energy"], label="Fault_Energy")

                if smooth and len(asset_hist) >= 7:
                    ax.plot(
                        asset_hist["day"],
                        asset_hist["Fault_Energy"].rolling(7).mean(),
                        label="Fault_Energy (7-day avg)"
                    )

            if show_rms and "RMS" in asset_hist.columns:
                ax.plot(asset_hist["day"], asset_hist["RMS"], label="RMS")

            if show_kurt and "Kurtosis" in asset_hist.columns:
                ax.plot(asset_hist["day"], asset_hist["Kurtosis"], label="Kurtosis")

            if "failure_day" in asset_hist.columns and asset_hist["failure_day"].notna().any():
                try:
                    fd = asset_hist["failure_day"].dropna().iloc[0]
                    ax.axvline(fd, linestyle="--")
                    ax.text(fd, ax.get_ylim()[1] * 0.95, "failure_day", rotation=90, va="top")
                except Exception:
                    pass

            ax.set_xlabel("Day")
            ax.set_ylabel("Signal / Feature Value")
            ax.legend()
            st.pyplot(fig)

            # Simple trend summary
            if "Fault_Energy" in asset_hist.columns and len(asset_hist) >= 7:
                recent_mean = asset_hist["Fault_Energy"].tail(7).mean()
                baseline_mean = asset_hist["Fault_Energy"].head(7).mean()

                st.write("**Trend Summary**")
                if recent_mean > baseline_mean * 1.2:
                    st.warning("Fault energy is increasing versus early-life baseline.")
                else:
                    st.success("Fault energy appears relatively stable versus baseline.")

# -----------------------------
# Auto-refresh
# -----------------------------
if AUTO_REFRESH:
    st.info(f"Auto-refresh is enabled. Refreshing every {REFRESH_SECONDS // 60} minutes.")
    time.sleep(REFRESH_SECONDS)
    st.rerun()
#bucket = row["risk_bucket"].iloc[0]
#rul = float(row["predicted_RUL"].iloc[0])

#if bucket.startswith("RED"):
#    st.error("🚨 Immediate action: Inspect bearing, check lubrication and alignment. Plan shutdown within 24–48 hours.")
#elif bucket.startswith("AMBER"):
#    st.warning("⚠️ Early warning: Increase monitoring frequency and schedule maintenance in next planned window.")
#else:
#    st.success("✅ Normal operation: Continue routine monitoring and preventive maintenance.")


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
    # Load historical dataset
    history = pd.read_csv("dataset_full.csv")
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

        # Main signal
        ax.plot(asset_hist["day"], asset_hist["Fault_Energy"], label="Fault_Energy")

        # Smoothing
        if smooth and len(asset_hist) >= 7:
            ax.plot(
                asset_hist["day"],
                asset_hist["Fault_Energy"].rolling(7).mean(),
                label="Fault_Energy (7-day avg)"
            )

        # Optional features
        if show_rms and "RMS" in asset_hist.columns:
            ax.plot(asset_hist["day"], asset_hist["RMS"], label="RMS")
        if show_kurt and "Kurtosis" in asset_hist.columns:
            ax.plot(asset_hist["day"], asset_hist["Kurtosis"], label="Kurtosis")

        # Failure marker (if exists)
        if "failure_day" in asset_hist.columns:
            try:
                fd = asset_hist["failure_day"].iloc[0]
                ax.axvline(fd, linestyle="--")
                ax.text(fd, ax.get_ylim()[1]*0.95, "failure_day", rotation=90, va="top")
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
