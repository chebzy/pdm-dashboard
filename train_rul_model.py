import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "dataset_full.csv"
MODEL_FILE = "rul_model.pkl"
FEATURES_FILE = "model_features.json"
PERF_FILE = "model_performance_regression.csv"


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


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    df = add_trend_features(df)

    features = [
        "RMS",
        "Peak_to_Peak",
        "Crest_Factor",
        "Kurtosis",
        "Amp_1x",
        "Fault_Energy",
        "RMS_change",
        "Kurtosis_change",
        "FaultEnergy_change",
        "RMS_roll5",
        "Kurtosis_roll5",
        "FaultEnergy_roll5",
    ]
    target = "days_to_failure"

    df_model = df[["asset_id", "day"] + features + [target]].dropna().sort_values(["asset_id", "day"])

    split_day = int(df_model["day"].quantile(0.8))
    train_df = df_model[df_model["day"] <= split_day].copy()
    test_df = df_model[df_model["day"] > split_day].copy()

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3),
            "Validation_Type": "Time-based split",
            "Train_Day_Max": int(train_df["day"].max()),
            "Test_Day_Min": int(test_df["day"].min()),
            "Features": ", ".join(features),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(PERF_FILE, index=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, MODEL_FILE)

    with open(FEATURES_FILE, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print(results_df)
    print(f"Saved best model to {MODEL_FILE}")
    print(f"Saved feature list to {FEATURES_FILE}")
    print(f"Saved performance table to {PERF_FILE}")


if __name__ == "__main__":
    main()