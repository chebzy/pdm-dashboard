import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("dataset_full.csv")

features = [
    "RMS",
    "Peak_to_Peak",
    "Crest_Factor",
    "Kurtosis",
    "Amp_1x",
    "Fault_Energy"
]

target = "days_to_failure"

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# -----------------------------
# Train test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ),
    "Gradient Boosting Regressor": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# -----------------------------
# Evaluate
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 3),
        "Features": ", ".join(features)
    })

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nRegression Model Comparison\n")
print(results_df)

# Save results for Streamlit app
results_df.to_csv("model_performance_regression.csv", index=False)
