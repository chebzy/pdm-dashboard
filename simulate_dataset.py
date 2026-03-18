import numpy as np
import pandas as pd

# -----------------------
# CONFIGURATION
# -----------------------
SEED = 42
np.random.seed(SEED)

N_ASSETS = 30
N_DAYS = 365

FS = 2000
DURATION_SEC = 1.0
N_SAMPLES = int(FS * DURATION_SEC)
t = np.arange(N_SAMPLES) / FS

RPM = 1800
F_1X = RPM / 60.0
F_FAULT = 120.0

A1_BASE = 1.0
NOISE_STD = 0.25
FAULT_RAMP_DAYS = 90

FAIL_DAY_MIN = 150
FAIL_DAY_MAX = N_DAYS

OUTPUT_FILE = "dataset_full.csv"


def fault_amplitude(day: int, failure_day: int, ramp_days: int) -> float:
    start = failure_day - ramp_days
    if day < start:
        return 0.0
    return min(1.0, (day - start) / ramp_days)


def simulate_signal(day: int, failure_day: int) -> np.ndarray:
    A1 = A1_BASE * (1.0 + 0.05 * np.random.randn())
    Af = fault_amplitude(day, failure_day, FAULT_RAMP_DAYS)

    A2 = 0.10 * (1.0 + 0.10 * np.random.randn())
    F_2X = 2.0 * F_1X

    signal = (
        A1 * np.sin(2 * np.pi * F_1X * t)
        + A2 * np.sin(2 * np.pi * F_2X * t)
        + Af * np.sin(2 * np.pi * F_FAULT * t)
        + NOISE_STD * np.random.randn(N_SAMPLES)
    )
    return signal.astype(np.float32)


def extract_features(signal: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak_to_peak = float(np.max(signal) - np.min(signal))
    crest_factor = float(np.max(np.abs(signal)) / rms) if rms > 0 else 0.0
    kurtosis = float(pd.Series(signal).kurt())

    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), d=1 / FS)

    idx_1x = np.argmin(np.abs(fft_freqs - F_1X))
    amp_1x = float(np.abs(fft_vals[idx_1x]))

    band_mask = (fft_freqs >= (F_FAULT - 10)) & (fft_freqs <= (F_FAULT + 10))
    fault_energy = float(np.sum(np.abs(fft_vals[band_mask]) ** 2))

    return {
        "RMS": rms,
        "Peak_to_Peak": peak_to_peak,
        "Crest_Factor": crest_factor,
        "Kurtosis": kurtosis,
        "Amp_1x": amp_1x,
        "Fault_Energy": fault_energy,
    }


def assign_fail_soon(days_to_failure: int, threshold: int = 7) -> int:
    return int(days_to_failure <= threshold)


def assign_risk_bucket(days_to_failure: int) -> str:
    if days_to_failure <= 7:
        return "RED - Immediate Action"
    elif days_to_failure <= 30:
        return "AMBER - Plan Maintenance"
    return "GREEN - Healthy"


def main() -> None:
    records = []

    for asset_id in range(N_ASSETS):
        failure_day = np.random.randint(FAIL_DAY_MIN, FAIL_DAY_MAX)
        asset_fault_scale = np.clip(1.0 + 0.25 * np.random.randn(), 0.6, 1.6)

        for day in range(N_DAYS):
            sig = simulate_signal(day, failure_day) * asset_fault_scale
            is_failed = int(day >= failure_day)
            days_to_failure = max(0, failure_day - day)

            features = extract_features(sig)

            records.append({
                "asset_id": asset_id,
                "day": day,
                "failure_day": failure_day,
                "days_to_failure": days_to_failure,
                "is_failed": is_failed,
                **features,
                "fail_soon": assign_fail_soon(days_to_failure, threshold=7),
                "fail_prob_7d": np.nan,
                "predicted_RUL": np.nan,
                "risk_bucket": assign_risk_bucket(days_to_failure),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved dataset to: {OUTPUT_FILE}")
    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()