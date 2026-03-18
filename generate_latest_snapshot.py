import json
import joblib
import pandas as pd


DATA_FILE = "dataset_full.csv"
MODEL_FILE = "rul_model.pkl"
FEATURES_FILE = "model_features.json"
OUTPUT_FILE = "latest_snapshot.csv"


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["asset_id", "day"]).copy()

    df["RMS_change"] = df.groupby("asset_id")["RMS"].diff()
    df["Kurtosis_change"] = df.groupby("asset_id")["Kurtosis"].diff()
    df["FaultEnergy_change"] = df.groupby("asset_id")["Fault_Energy"].diff()

    df["RMS_roll5"] = (
        df.groupby("asset_id")["RMS"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Kurtosis_roll5"] = (
        df.groupby("asset_id")["Kurtosis"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["FaultEnergy_roll5"] = (
        df.groupby("asset_id")["Fault_Energy"]
        .rolling(5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def assign_risk_bucket(predicted_rul: float) -> str:
    if predicted_rul <= 7:
        return "RED - Immediate Action"
    if predicted_rul <= 30:
        return "AMBER - Plan Maintenance"
    return "GREEN - Healthy"


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df = add_trend_features(df)

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        features = json.load(f)

    model = joblib.load(MODEL_FILE)

    latest_rows = (
        df.sort_values(["asset_id", "day"])
        .groupby("asset_id", as_index=False)
        .tail(1)
        .copy()
    )

    latest_rows = latest_rows.dropna(subset=features).copy()

    latest_rows["predicted_RUL"] = model.predict(latest_rows[features])
    latest_rows["predicted_RUL"] = latest_rows["predicted_RUL"].clip(lower=0)
    latest_rows["risk_bucket"] = latest_rows["predicted_RUL"].apply(assign_risk_bucket)

    if "fail_prob_7d" not in latest_rows.columns:
        latest_rows["fail_prob_7d"] = pd.NA

    keep_cols = [
        "asset_id",
        "day",
        "failure_day",
        "days_to_failure",
        "is_failed",
        "RMS",
        "Peak_to_Peak",
        "Crest_Factor",
        "Kurtosis",
        "Amp_1x",
        "Fault_Energy",
        "fail_soon",
        "fail_prob_7d",
        "predicted_RUL",
        "risk_bucket",
    ]

    keep_cols = [c for c in keep_cols if c in latest_rows.columns]
    latest_snapshot = latest_rows[keep_cols].sort_values("predicted_RUL")

    latest_snapshot.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved snapshot to {OUTPUT_FILE}")
    print(latest_snapshot.head(10))


if __name__ == "__main__":
    main()