import numpy as np
import pandas as pd
import scoringrules as sr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
    RESULTS_FILE = PROJECT_ROOT / "results.csv"
    GROUND_TRUTH_FILE = PROJECT_ROOT / "data_catalog" / "full_dataset.csv"

# For now we are ignoring the file with test_4 for the evaluation because we wont have the ground truth to do it
    FORECAST_FILES = [
        "test_1_2022.csv",
        "test_2_2023.csv",
        "test_3_2024.csv",
        # "test_4_2025.csv",
    ]

    EXPECTED_WINDOWS = {
        "test_1_2022.csv": ("2022-10-09", "2023-10-01"),
        "test_2_2023.csv": ("2023-10-08", "2024-09-29"),
        "test_3_2024.csv": ("2024-10-06", "2025-09-28"),
        # "test_4_2025.csv": ("2025-10-05", "2026-09-27"),
    }

    REQUIRED_COLS = {
        "state", "date", "pred",
        "lower_50", "lower_80", "lower_90", "lower_95",
        "upper_50", "upper_80", "upper_90", "upper_95",
    }

    ALPHAS = np.array([0.5, 0.2, 0.1, 0.05])
    INTERVALS = [50, 80, 90, 95]
    EPS = 1e-8

    # deletes the result file so it calculates all the models
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    # loads ground truth
    truth_df = pd.read_csv(GROUND_TRUTH_FILE)[["state", "date", "cases"]]
    truth_df["state"] = truth_df["state"].astype(str).str.strip().str.upper()
    truth_df = truth_df[truth_df["state"] != "ES"]
    truth_df["date"] = pd.to_datetime(truth_df["date"]).dt.normalize()

    # Checks which models to evaluate
    models = {p.name for p in SUBMISSIONS_DIR.iterdir() if p.is_dir()}

    if not models:
        print("No models found in submissions directory.")
        return

    rows = []

    # Evaluates each model
    for model in sorted(models):

        model_dir = SUBMISSIONS_DIR / model
        per_file_scores = {}
        metrics_acc = {k: [] for k in ["wis", "crps", "log_score", "mae", "mse", "rmse", "mape"]}

        # Each model should have exactly all files  to evaluate (all validation sets )
        missing = [f for f in FORECAST_FILES if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"[{model}] missing files: {missing}")

        # Evaluates each file independently
        for f in FORECAST_FILES:

            f_path = model_dir / f
            df = pd.read_csv(f_path)

            # checks for duplicated results
            if df.duplicated(subset=["state", "date"]).any():
                raise ValueError(f"[{model} | {f}] duplicated state/date in forecast")

            # Checks all models have the necessary rows
            missing_cols = REQUIRED_COLS - set(df.columns)
            if missing_cols:
                raise ValueError(f"[{model} | {f}] missing columns: {missing_cols}")

            df["state"] = df["state"].astype(str).str.strip().str.upper()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

            # This block gets the window we want to evaluate from the ground truth to then merge with the predictions
            start, end = EXPECTED_WINDOWS[f]
            mask = ((truth_df["date"] >= pd.to_datetime(start)) & (truth_df["date"] <= pd.to_datetime(end)))
            merged = truth_df.loc[mask].merge(df, on=["state", "date"], how="left")
            merged = merged.sort_values(["state", "date"]).reset_index(drop=True)

            # Will check if there's any NAS in the predictions, if there was is because there are missing predictions in that file
            pred_cols = list(REQUIRED_COLS - {"state", "date"})
            if merged[pred_cols].isna().any().any():
                raise ValueError(f"[{model} | {f}] Missing predictions")

            obs = merged["cases"].to_numpy()
            pred = merged["pred"].to_numpy()
            lower = merged[[f"lower_{i}" for i in INTERVALS]].to_numpy()
            upper = merged[[f"upper_{i}" for i in INTERVALS]].to_numpy()

            # This is dodgy but I'm assuming normality on the predictions intervals to calculate CRPS and Log score
            sigma_est = (merged["upper_95"].to_numpy() - merged["lower_95"].to_numpy()) / 3.92

            # FIX: stabilize sigma and avoid invalid values
            if np.any(~np.isfinite(sigma_est)):
                raise ValueError(f"[{model} | {f}] non-finite sigma values detected")

            sigma_est = np.clip(sigma_est, 1e-3, None)

            # metric calculation
            wis_vals = sr.weighted_interval_score(obs, pred, lower, upper, ALPHAS)
            crps_vals = sr.crps_normal(obs, pred, sigma_est)
            log_vals = sr.logs_normal(obs, pred, sigma_est)
            mae_val = mean_absolute_error(obs, pred)
            mse_val = mean_squared_error(obs, pred)
            rmse_val = np.sqrt(mse_val)
            mape_val = np.mean(np.abs(pred - obs) / np.maximum(np.abs(obs), EPS)) * 100

            file_results = {
    			"wis": round(float(np.mean(wis_vals)), 2),
    			"crps": round(float(np.mean(crps_vals)), 2),
    			"log_score": round(float(np.mean(log_vals)), 2),
    			"mae": round(float(mae_val), 2),
    			"mse": round(float(mse_val), 2),
    			"rmse": round(float(rmse_val), 2),
    			"mape": round(float(mape_val), 2)}

            per_file_scores[f.replace(".csv", "")] = file_results

            for k, v in file_results.items():
                metrics_acc[k].append(v)

        # aggregating results per model per file and then calculates overall
        flat_row = {"model": model}

        for horizon, scores in per_file_scores.items():
            for k, v in scores.items():
                flat_row[f"{horizon}_{k}"] = v

        for k, vals in metrics_acc.items():
            flat_row[f"overall_{k}"] = float(np.nanmean(vals))

        rows.append(flat_row)

    # final results
    final_df = pd.DataFrame(rows)

    # Ranks models on overall_wis
    final_df = final_df.sort_values("overall_wis").reset_index(drop=True)
    final_df.insert(0, "rank", np.arange(1, len(final_df) + 1))

    # Saves final table as csv
    final_df.to_csv(RESULTS_FILE, index=False)

    #Prints the ranking
    pd.set_option("display.float_format", "{:.2f}".format)
    print(final_df[["rank", "model"] + [c for c in final_df.columns if c.startswith("overall_")]])


if __name__ == "__main__":
    main()
