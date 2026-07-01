import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import xgboost as xgb
from data_catalog.data_factory import data_factory
from data_processing import _build_features, make_forecast_frame, _season_sincos, TO_LAG

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP_COL = "uf_code"
TIME_COL = "epiweek"

TARGET_COL = "cases_per_10000"
TARGET_IS_INCIDENCE = TARGET_COL == "cases_per_10000"
TARGET_LOG1P = True

TARGET_WINDOW = 52


def first_scored_h(max_horizon, window=TARGET_WINDOW):
    return max(1, max_horizon - window + 1)

QUANTILES = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
QIDX = {round(q, 4): i for i, q in enumerate(QUANTILES)}
INTERVALS = {50: (0.25, 0.75), 80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}


def _base_xgb(threads):
    return dict(
        objective="reg:quantileerror",
        quantile_alpha=np.array(QUANTILES),
        tree_method="hist",
        n_jobs=threads,
        random_state=0,
    )

def build_supervised(
    dataset_id,
    max_lag,
    max_horizon,
    target_col=TARGET_COL,
    horizon_step=1,
    lag_cols=TO_LAG,
    short_lags=(1, 2, 3, 4, 8, 12, 18),
    short_windows=(4, 12),):

    df, feature_cols, feat_valid = _build_features(
        dataset_id, max_lag, lag_cols, GROUP_COL, TIME_COL, short_lags, short_windows
    )
    g = df.groupby(GROUP_COL, sort=False)
    X_blocks, y_parts, key_blocks = [], [], []
    for h in range(1, max_horizon + 1, horizon_step):
        y_h = g[target_col].shift(-h)
        cases_h = g["cases"].shift(-h)
        pop_h = g["population"].shift(-h)
        date_h = g["_date"].shift(-h)
        valid = feat_valid & y_h.notna()
        if not valid.any():
            continue

        block = df.loc[valid, feature_cols].copy()
        block["h"] = h
        block["target_week_sin"], block["target_week_cos"] = _season_sincos(date_h[valid])
        X_blocks.append(block)
        y_arr = y_h[valid].to_numpy()
        if TARGET_LOG1P:
            y_arr = np.log1p(y_arr)
        y_parts.append(y_arr)
        key_blocks.append(
            pd.DataFrame(
                {
                    "uf_code": df.loc[valid, GROUP_COL].to_numpy(),
                    "origin_epiweek": df.loc[valid, TIME_COL].to_numpy(),
                    "h": h,
                    "target_date": date_h[valid].to_numpy(),
                    "population": pop_h[valid].to_numpy(),
                    "cases": cases_h[valid].to_numpy(),
                }
            )
        )

    X = pd.concat(X_blocks, ignore_index=True)
    y = np.concatenate(y_parts)
    keys = pd.concat(key_blocks, ignore_index=True)
    return X, y, keys


def _row_index(keys):
    return pd.MultiIndex.from_frame(keys[["uf_code", "origin_epiweek", "h"]])

def expanding_folds(origin_ord, n_folds, val_size, embargo):
    W = int(origin_ord.max()) + 1
    folds = []
    for k in range(n_folds):
        val_end = W - k * val_size
        val_start = val_end - val_size
        train_end = val_start - embargo
        if train_end <= 0:
            break
        tr = np.where(origin_ord < train_end)[0]
        va = np.where((origin_ord >= val_start) & (origin_ord < val_end))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return list(reversed(folds))


def backtest_folds(origin_ord, h, max_horizon, window=TARGET_WINDOW, n_folds=4):
    origin_ord = np.asarray(origin_ord)
    h = np.asarray(h)
    first = first_scored_h(max_horizon, window)
    W = int(origin_ord.max()) + 1
    target_ord = origin_ord + h
    folds = []
    for k in range(n_folds):
        O = (W - max_horizon) - k * window
        if O < window:
            break
        tr = np.where(target_ord <= O)[0]
        va = np.where((origin_ord == O) & (h >= first) & (h <= max_horizon))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return list(reversed(folds))

def _interval_score(y, lower, upper, alpha):
    return (
        (upper - lower)
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )

def wis(y, q_pred):
    y = np.asarray(y, dtype=float)
    median = q_pred[:, QIDX[0.5]]
    total = 0.5 * np.abs(y - median)
    for level, (lo_q, hi_q) in INTERVALS.items():
        alpha = 1.0 - level / 100.0
        lo = q_pred[:, QIDX[round(lo_q, 4)]]
        hi = q_pred[:, QIDX[round(hi_q, 4)]]
        total = total + (alpha / 2.0) * _interval_score(y, lo, hi, alpha)
    total = total / (len(INTERVALS) + 0.5)
    return float(total.mean())


def _postprocess(pred):

    pred = np.sort(pred, axis=1)
    if TARGET_LOG1P:
        pred = np.expm1(pred)
    return np.clip(pred, 0.0, None)


def _to_counts(pred, population):
    if TARGET_IS_INCIDENCE:
        return pred * (np.asarray(population)[:, None] / 10_000.0)
    return pred


def _eval_task(Xnp, y_train, population, y_cases, tr, va, params, xgb_threads,
               h=None, min_scored_h=1):
    model = xgb.XGBRegressor(**_base_xgb(xgb_threads), **params)
    model.fit(Xnp[tr], y_train[tr])
    pred = _to_counts(_postprocess(model.predict(Xnp[va])), population[va])
    if h is not None:
        keep = h[va] >= min_scored_h
        return wis(y_cases[va][keep], pred[keep])
    return wis(y_cases[va], pred)


def iter_grid(param_grid):
    keys = list(param_grid)
    for combo in product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def run_grid(
    dataset_id,
    max_lags,
    param_grid,
    max_horizon=67,
    n_folds=4,
    val_size=52,
    total_cores=200,
    xgb_threads=10,
    results_csv=None,
    temp_folder=None,):

    concurrency = max(1, total_cores // xgb_threads)
    tmp = temp_folder or os.path.join(HERE, ".joblib_tmp")
    os.makedirs(tmp, exist_ok=True)
    grid = list(iter_grid(param_grid))
    print(f"{len(max_lags)} max_lags x {len(grid)} configs x {n_folds} folds; "
          f"{concurrency} concurrent models x {xgb_threads} threads")

    canon = None
    rows = []
    for ml in sorted(max_lags, reverse=True):
        X, y, keys = build_supervised(dataset_id, ml, max_horizon)
        idx = _row_index(keys)
        if canon is None:
            canon = idx
        mask = idx.isin(canon)
        X, y, keys = X.loc[mask], y[mask], keys.loc[mask].reset_index(drop=True)
        assert len(X) == len(canon), f"row mismatch at max_lag={ml}: {len(X)} vs {len(canon)}"

        Xnp = X.to_numpy(dtype=np.float32)
        population = keys["population"].to_numpy()
        y_cases = keys["cases"].to_numpy()
        h_arr = keys["h"].to_numpy()
        min_h = first_scored_h(max_horizon)
        order = {w: i for i, w in enumerate(sorted(keys["origin_epiweek"].unique()))}
        origin_ord = keys["origin_epiweek"].map(order).to_numpy()
        folds = backtest_folds(origin_ord, h_arr, max_horizon, window=val_size, n_folds=n_folds)
        if not folds:
            raise ValueError(f"no backtest folds for max_lag={ml}; not enough history (extend data or lower max_lag)")

        tasks = [(pi, params, fi, tr, va)
                 for pi, params in enumerate(grid)
                 for fi, (tr, va) in enumerate(folds)]
        scores = Parallel(n_jobs=concurrency, backend="loky", temp_folder=tmp, verbose=5)(
            delayed(_eval_task)(Xnp, y, population, y_cases, tr, va, params, xgb_threads,
                                h=h_arr, min_scored_h=min_h)
            for (_, params, _, tr, va) in tasks
        )

        per_config = {pi: [] for pi in range(len(grid))}
        for (pi, _, _, _, _), s in zip(tasks, scores):
            per_config[pi].append(s)
        for pi, params in enumerate(grid):
            fold_scores = per_config[pi]
            rows.append({"max_lag": ml, **params,
                         "wis": float(np.mean(fold_scores)),
                         "wis_folds": [round(s, 3) for s in fold_scores],
                         "n_rows": len(canon), "n_features": Xnp.shape[1]})
        print(f"max_lag={ml:>3} done  best WIS so far="
              f"{min(r['wis'] for r in rows):.3f}")

        if results_csv:
            pd.DataFrame(rows).sort_values("wis").to_csv(results_csv, index=False)

    return pd.DataFrame(rows).sort_values("wis").reset_index(drop=True)



def make_submission(dataset_id, max_lag, params, max_horizon=67, model_out=None, device=None):
    X, y, _ = build_supervised(dataset_id, max_lag, max_horizon)
    base = _base_xgb(-1)
    if device:
        base["device"] = device
    model = xgb.XGBRegressor(**base, **params)
    model.fit(X, y)
    if model_out:
        model.save_model(model_out)

    F = make_forecast_frame(dataset_id, max_lag=max_lag, max_horizon=max_horizon)
    pred = _postprocess(model.predict(F[X.columns]))

    raw, _ = data_factory(dataset_id)
    raw["date"] = pd.to_datetime(raw["date"])
    ew_to_date = raw.drop_duplicates(TIME_COL).set_index(TIME_COL)["date"]
    last_pop = raw.sort_values(TIME_COL).groupby(GROUP_COL)["population"].last()

    origin_date = F["forecast_from_epiweek"].map(ew_to_date)
    target_date = origin_date + pd.to_timedelta(F["h"].to_numpy() * 7, unit="D")
    pred = _to_counts(pred, F["uf_code"].map(last_pop).to_numpy())

    sub = pd.DataFrame({"uf_code": F["uf_code"].to_numpy()})
    sub["date"] = target_date.dt.strftime("%Y-%m-%d").to_numpy()
    sub["pred"] = pred[:, QIDX[0.5]]
    for level, (lo_q, hi_q) in INTERVALS.items():
        sub[f"lower_{level}"] = pred[:, QIDX[round(lo_q, 4)]]
        sub[f"upper_{level}"] = pred[:, QIDX[round(hi_q, 4)]]

    cols = ["uf_code", "date", "pred"]
    for level in INTERVALS:
        cols += [f"lower_{level}", f"upper_{level}"]
    return sub[cols]


_SUB_Q_COL = {
    0.025: "lower_95", 0.05: "lower_90", 0.10: "lower_80", 0.25: "lower_50",
    0.50: "pred", 0.75: "upper_50", 0.90: "upper_80", 0.95: "upper_90", 0.975: "upper_95",
}


def score_against_target(dataset_id, submission):

    _, test = data_factory(dataset_id)
    test = test[["uf_code", "date", "cases"]].copy()
    test["date"] = pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d")
    if test["date"].nunique() < TARGET_WINDOW:
        return None
    m = submission.merge(test, on=["uf_code", "date"], how="inner")
    if m.empty:
        return None
    q_pred = np.column_stack([m[_SUB_Q_COL[round(q, 4)]].to_numpy() for q in QUANTILES])
    return float(wis(m["cases"].to_numpy(), q_pred))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return None


if __name__ == "__main__":
    DATASET_ID = 1
    MAX_HORIZON = 67
    N_FOLDS = 4
    VAL_SIZE = 52
    TOTAL_CORES = 200
    XGB_THREADS = 10

    MAX_LAGS = [26, 52, 78, 104, 130, 156]
    PARAM_GRID = {
        "n_estimators": [400, 800, 1500],
        "max_depth": [4, 6, 8, 10, 20, 30, 40],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 5],
    }

    run_dir = os.path.join(HERE, "results", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    results_csv = os.path.join(run_dir, "grid_results.csv")
    print(f"writing results to {run_dir}")

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "started": datetime.now().isoformat(timespec="seconds"),
            "dataset_id": DATASET_ID, "max_horizon": MAX_HORIZON,
            "n_folds": N_FOLDS, "val_size": VAL_SIZE, "embargo": MAX_HORIZON,
            "max_lags": MAX_LAGS, "param_grid": PARAM_GRID,
            "target_col": TARGET_COL, "quantiles": QUANTILES,
            "total_cores": TOTAL_CORES, "xgb_threads": XGB_THREADS,
            "git_commit": _git_commit(),
        }, f, indent=2)

    results = run_grid(DATASET_ID, MAX_LAGS, PARAM_GRID, max_horizon=MAX_HORIZON,
                       n_folds=N_FOLDS, val_size=VAL_SIZE,
                       total_cores=TOTAL_CORES, xgb_threads=XGB_THREADS,
                       results_csv=results_csv,
                       temp_folder=os.path.join(run_dir, "_joblib_tmp"))
    print("\n=== top configs by CV WIS (counts) ===")
    print(results.head(10).to_string(index=False))

    best = results.iloc[0]
    best_params = {k: best[k] for k in PARAM_GRID}
    with open(os.path.join(run_dir, "best_config.json"), "w") as f:
        json.dump({"max_lag": int(best["max_lag"]), "params": best_params,
                   "cv_wis": float(best["wis"]), "wis_folds": best["wis_folds"]}, f, indent=2)

    submission = make_submission(DATASET_ID, int(best["max_lag"]), best_params, MAX_HORIZON,
                                 model_out=os.path.join(run_dir, "best_model.json"))
    submission.to_csv(os.path.join(run_dir, "submission.csv"), index=False)
    true_wis = score_against_target(DATASET_ID, submission)
    print("\n=== submission head ===")
    print(submission.head().to_string(index=False))
    tw = "n/a" if true_wis is None else f"{true_wis:.3f}"
    print(f"\nselection CV WIS={float(best['wis']):.3f}  |  real-task WIS vs held-out target={tw}")
    print(f"\nsaved: grid_results.csv, best_config.json, best_model.json, submission.csv -> {run_dir}")
