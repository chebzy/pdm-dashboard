🚀 Predictive Maintenance System for Rotating Equipment

An end-to-end machine learning system for predicting Remaining Useful Life (RUL) of rotating equipment using simulated vibration data, feature engineering, and model-driven insights.

📌 Overview

This project demonstrates a full predictive maintenance workflow:

Simulate vibration signals for multiple assets

Extract condition monitoring features (RMS, kurtosis, fault energy, etc.)

Train machine learning models to predict days to failure

Evaluate performance using two validation strategies

Deploy results in an interactive Streamlit dashboard

🧠 Key Capabilities

📊 RUL Prediction (Regression)

⚠️ Risk Classification (Green / Amber / Red)

📉 Degradation Trend Visualization

🧪 Dual Validation Framework

🖥️ Interactive Dashboard (Streamlit)

🏗️ System Architecture
Data Simulation → Feature Engineering → Model Training → Validation → Inference → Dashboard
⚙️ Project Structure
.
├── simulate_dataset.py
├── train_rul_model.py
├── train_rul_model_asset_split.py
├── generate_latest_snapshot.py
├── run_pipeline.py
├── app.py
├── dataset_full.csv
├── latest_snapshot.csv
├── model_performance_regression.csv
├── model_performance_regression_asset_split.csv
├── rul_model.pkl
├── requirements.txt
🔄 Pipeline Execution

Run the full pipeline:

python run_pipeline.py

This will:

Simulate dataset → dataset_full.csv

Train model → rul_model.pkl

Generate predictions → latest_snapshot.csv

📊 Model Validation Strategy

This project uses two complementary validation approaches:

1️⃣ Time-Based Validation

Train: earlier time periods

Test: later time periods (same assets)

👉 Measures:
Ability to forecast future degradation of known assets

Result:

MAE ≈ 1.4 days

2️⃣ Asset-Based Validation

Train: subset of assets

Test: unseen assets

👉 Measures:
Ability to generalize to new equipment

Result:

MAE ≈ 20.5 days

🎯 Key Insight

Forecasting future degradation of known assets is significantly easier than generalizing to unseen assets.

This highlights the importance of proper validation design in predictive maintenance systems.

🤖 Models Compared
Model	MAE (Time-Based)	MAE (Asset-Based)
Random Forest	~1.4	~20.5
Gradient Boosting	~1.5–2	~21.8
Linear Regression	~40	~33
📈 Features Used

RMS

Peak-to-Peak

Crest Factor

Kurtosis

Fault Energy

1× Amplitude

Rolling averages (5-day)

First-order change (trend features)

📊 Dashboard Features

The Streamlit dashboard provides:

Fleet-level risk overview

Asset ranking by predicted RUL

Top critical assets

Asset-level drill-down

Degradation trend plots

Model performance comparison

CSV export of filtered results

🌐 Live Demo

👉 https://pdm-dashboard-rul-risk.streamlit.app/

(App may go to sleep due to Streamlit Cloud inactivity — click “Wake app” if needed.)

🛠️ Tech Stack

Python

Pandas / NumPy

Scikit-learn

Matplotlib

Streamlit

🚧 Future Improvements

Add classification model (failure within 7 days)

Introduce uncertainty estimation

Simulate multiple failure modes

Integrate real-time streaming (e.g., AWS Kinesis)

Deploy on always-on infrastructure

📌 Summary

This project demonstrates:

✔ End-to-end ML system design
✔ Feature engineering for condition monitoring
✔ Proper validation strategy (time vs generalization)
✔ Practical deployment via dashboard

👤 Author

Ajaero Chibuikem

⭐ If you found this useful

Feel free to star the repo or reach out!