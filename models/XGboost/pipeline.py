import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import xgboost as xgb
from data_processing import _lag_set, _window_set
from grid_search import (
    build_supervised, backtest_folds, make_submission, score_against_target,
    first_scored_h, _base_xgb, _postprocess, _to_counts, wis,
)

DATASETS = (1, 2, 3, 4)

MAX_HORIZON = 67
BUILD_MAX_LAG = 156
N_FOLDS = 4
VAL_SIZE = 52
N_CONFIGS = 150
SEED = 0
SHORT_LAGS = (1, 2, 3, 4, 8, 12, 18)
SHORT_WINDOWS = (4, 12)
N_ESTIMATORS_CAP = 1500
EARLY_STOP = 50

DEFAULT_CACHE = os.path.join(HERE, "results", "cache")
DEFAULT_RUN = os.path.join(HERE, "results", "search")

SEARCH_SPACE = {
    "max_lag":          [52, 78, 104, 130, 156],
    "max_depth":        [3, 4, 5, 6, 8, 10],
    "learning_rate":    [0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10, 20],
    "reg_lambda":       [0.0, 1.0, 3.0, 5.0, 10.0],
    "reg_alpha":        [0.0, 0.5, 1.0, 2.0],
}
PARAM_KEYS = [k for k in SEARCH_SPACE if k != "max_lag"]

BASELINE_MAX_LAG = 104
BASELINE_CONFIG = dict(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
)


def sample_configs(n=N_CONFIGS, seed=SEED):
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n):
        configs.append({k: rng.choice(v).item() for k, v in SEARCH_SPACE.items()})
    return configs


def _py(v):
    return v.item() if hasattr(v, "item") else v


_PAT = re.compile(r"_(lag|rollmean|rollstd)(\d+)$")


def keep_columns(feature_cols, max_lag):

    lags = set(_lag_set(max_lag, SHORT_LAGS))
    wins = set(_window_set(max_lag, SHORT_WINDOWS))
    mask = np.ones(len(feature_cols), dtype=bool)
    for i, c in enumerate(feature_cols):
        m = _PAT.search(c)
        if not m:
            continue
        kind, num = m.group(1), int(m.group(2))
        mask[i] = (num in lags) if kind == "lag" else (num in wins)
    return mask


def cmd_prep(args):
    os.makedirs(args.cache, exist_ok=True)
    for ds in args.datasets:
        t = time.time()
        X, y, keys = build_supervised(ds, args.max_lag, args.max_horizon)
        order = {w: i for i, w in enumerate(sorted(keys["origin_epiweek"].unique()))}
        oord = keys["origin_epiweek"].map(order).to_numpy()
        folds = backtest_folds(oord, keys["h"].to_numpy(), args.max_horizon,
                               window=VAL_SIZE, n_folds=N_FOLDS)
        if not folds:
            raise SystemExit(f"no backtest folds for ds={ds}; not enough history")

        d = os.path.join(args.cache, f"ds{ds}")
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "X.npy"), X.to_numpy(dtype=np.float32))
        np.save(os.path.join(d, "y.npy"), np.asarray(y, dtype=np.float32))
        np.save(os.path.join(d, "population.npy"), keys["population"].to_numpy(np.float64))
        np.save(os.path.join(d, "cases.npy"), keys["cases"].to_numpy(np.float64))
        np.save(os.path.join(d, "h.npy"), keys["h"].to_numpy(np.int16))
        np.savez(os.path.join(d, "folds.npz"),
                 **{f"tr{i}": tr for i, (tr, _) in enumerate(folds)},
                 **{f"va{i}": va for i, (_, va) in enumerate(folds)})
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"dataset": ds, "build_max_lag": args.max_lag,
                       "max_horizon": args.max_horizon, "n_folds": len(folds),
                       "feature_cols": list(X.columns)}, f)
        print(f"ds{ds}: X{X.shape} {len(folds)} folds  ({time.time() - t:.1f}s) -> {d}",
              flush=True)

def _fit_one_fold(X, y, keep_idx, tr, va, params, device, threads, subsample, seed):
    """Fit with early stopping on a train-holdout; score WIS on the (untouched) val rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_es = min(len(tr) - 1, max(2000, int(0.1 * len(tr))))
    es_idx, fit_idx = tr[perm[:n_es]], tr[perm[n_es:]]
    if subsample and len(fit_idx) > subsample:
        fit_idx = rng.choice(fit_idx, subsample, replace=False)

    Xfit = np.ascontiguousarray(X[np.ix_(fit_idx, keep_idx)])
    Xes = np.ascontiguousarray(X[np.ix_(es_idx, keep_idx)])
    Xva = np.ascontiguousarray(X[np.ix_(va, keep_idx)])

    base = _base_xgb(threads)
    base["device"] = device
    model = xgb.XGBRegressor(**base, n_estimators=N_ESTIMATORS_CAP,
                             early_stopping_rounds=EARLY_STOP, **params)
    model.fit(Xfit, y[fit_idx], eval_set=[(Xes, y[es_idx])], verbose=False)
    best_it = int(getattr(model, "best_iteration", N_ESTIMATORS_CAP - 1) or 0) + 1
    return model, best_it, Xva


def cmd_search(args):
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.task))
    configs = sample_configs()
    n = len(configs)
    if task >= len(args.datasets) * n:
        raise SystemExit(f"task {task} out of range for {len(args.datasets)} datasets x {n} configs")
    ds = args.datasets[task // n]
    ci = task % n
    cfg = configs[ci]
    params = {k: cfg[k] for k in PARAM_KEYS}

    d = os.path.join(args.cache, f"ds{ds}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(d, "y.npy"))
    pop = np.load(os.path.join(d, "population.npy"))
    cases = np.load(os.path.join(d, "cases.npy"))
    h_arr = np.load(os.path.join(d, "h.npy"))
    folds = np.load(os.path.join(d, "folds.npz"))
    keep_idx = np.where(keep_columns(meta["feature_cols"], cfg["max_lag"]))[0]
    min_h = first_scored_h(meta["max_horizon"])

    print(f"task {task}: ds{ds} cfg{ci} max_lag={cfg['max_lag']} "
          f"({len(keep_idx)} feats) {params}", flush=True)

    fold_wis, best_iters = [], []
    t0 = time.time()
    for i in range(meta["n_folds"]):
        tr, va = folds[f"tr{i}"], folds[f"va{i}"]
        model, best_it, Xva = _fit_one_fold(
            X, y, keep_idx, tr, va, params, args.device, args.threads,
            args.subsample, SEED * 1000 + i)
        pred = _to_counts(_postprocess(model.predict(Xva)), pop[va])
        keep = h_arr[va] >= min_h
        fold_wis.append(wis(cases[va][keep], pred[keep]))
        best_iters.append(best_it)

    out = {"dataset": ds, "config_idx": ci, "max_lag": int(cfg["max_lag"]), **params,
           "wis": float(np.mean(fold_wis)),
           "wis_folds": [round(s, 3) for s in fold_wis],
           "best_iteration": int(np.median(best_iters)),
           "fit_seconds": round(time.time() - t0, 1)}
    pdir = os.path.join(args.run, "partial")
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"ds{ds}_cfg{ci:03d}.json"), "w") as f:
        json.dump(out, f)
    print(f"  WIS={out['wis']:.3f}  folds={out['wis_folds']}  "
          f"({out['fit_seconds']}s)", flush=True)


def cmd_baseline(args):
    out_dir = os.path.join(HERE, "results", "baseline")
    os.makedirs(out_dir, exist_ok=True)
    env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    ds_list = [args.datasets[int(env_task)]] if env_task is not None else args.datasets

    params = dict(BASELINE_CONFIG)
    if args.n_estimators:
        params["n_estimators"] = args.n_estimators

    summary = []
    for ds in ds_list:
        t0 = time.time()
        X, y, keys = build_supervised(ds, args.max_lag, args.max_horizon)
        Xnp = X.to_numpy(dtype=np.float32)
        y = np.asarray(y)
        pop, cases = keys["population"].to_numpy(), keys["cases"].to_numpy()
        h_arr = keys["h"].to_numpy()
        min_h = first_scored_h(args.max_horizon)
        order = {w: i for i, w in enumerate(sorted(keys["origin_epiweek"].unique()))}
        oord = keys["origin_epiweek"].map(order).to_numpy()
        folds = backtest_folds(oord, h_arr, args.max_horizon, window=VAL_SIZE, n_folds=N_FOLDS)

        fold_wis = []
        for tr, va in folds:
            base = _base_xgb(args.threads)
            base["device"] = args.device
            m = xgb.XGBRegressor(**base, **params)
            m.fit(Xnp[tr], y[tr])
            pred = _to_counts(_postprocess(m.predict(Xnp[va])), pop[va])
            keep = h_arr[va] >= min_h
            fold_wis.append(wis(cases[va][keep], pred[keep]))
        cv = float(np.mean(fold_wis))

        sub = make_submission(ds, args.max_lag, params, args.max_horizon,
                              model_out=os.path.join(out_dir, f"ds{ds}_model.json"),
                              device=args.device)
        sub.to_csv(os.path.join(out_dir, f"ds{ds}_submission.csv"), index=False)
        true_wis = score_against_target(ds, sub)
        with open(os.path.join(out_dir, f"ds{ds}_baseline.json"), "w") as f:
            json.dump({"dataset": ds, "max_lag": args.max_lag, "params": params,
                       "cv_wis": cv, "real_task_wis": true_wis,
                       "wis_folds": [round(s, 3) for s in fold_wis]}, f, indent=2)
        tw = "n/a" if true_wis is None else f"{true_wis:.3f}"
        print(f"ds{ds}: CV WIS={cv:.3f}  real-task WIS={tw}  folds={[round(s, 3) for s in fold_wis]}  "
              f"({time.time() - t0:.0f}s) -> {out_dir}/ds{ds}_submission.csv", flush=True)
        summary.append({"dataset": ds, "cv_wis": round(cv, 3),
                        "real_task_wis": (None if true_wis is None else round(true_wis, 3)),
                        "max_lag": args.max_lag})

    if len(summary) > 1:
        print("\n=== baseline summary ===\n"
              + pd.DataFrame(summary).to_string(index=False), flush=True)



def cmd_finalize(args):
    final_dir = os.path.join(args.run, "final")
    os.makedirs(final_dir, exist_ok=True)
    summary = []
    for ds in args.datasets:
        parts = sorted(glob.glob(os.path.join(args.run, "partial", f"ds{ds}_cfg*.json")))
        if not parts:
            print(f"ds{ds}: no partial results, skipping", flush=True)
            continue
        rows = [json.load(open(p)) for p in parts]
        df = pd.DataFrame(rows).sort_values("wis").reset_index(drop=True)
        df.to_csv(os.path.join(final_dir, f"ds{ds}_search_results.csv"), index=False)
        best = df.iloc[0]
        params = {k: _py(best[k]) for k in PARAM_KEYS}
        params["n_estimators"] = int(best["best_iteration"])
        print(f"ds{ds}: best WIS={best['wis']:.3f} max_lag={int(best['max_lag'])} "
              f"n_est={params['n_estimators']} ({len(df)} configs) -> refitting on all rows",
              flush=True)

        sub = make_submission(ds, int(best["max_lag"]), params, args.max_horizon,
                              model_out=os.path.join(final_dir, f"ds{ds}_model.json"),
                              device=args.device)
        sub.to_csv(os.path.join(final_dir, f"ds{ds}_submission.csv"), index=False)
        true_wis = score_against_target(ds, sub)
        tw = "n/a" if true_wis is None else f"{true_wis:.3f}"
        print(f"ds{ds}: real-task WIS vs held-out target={tw}", flush=True)
        with open(os.path.join(final_dir, f"ds{ds}_best_config.json"), "w") as f:
            json.dump({"max_lag": int(best["max_lag"]), "params": params,
                       "cv_wis": float(best["wis"]), "real_task_wis": true_wis,
                       "wis_folds": best["wis_folds"]}, f, indent=2)
        summary.append({"dataset": ds, "cv_wis": float(best["wis"]),
                        "real_task_wis": (None if true_wis is None else round(true_wis, 3)),
                        "max_lag": int(best["max_lag"]), **params})

    if summary:
        print("\n=== best per dataset ===", flush=True)
        print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", default=DEFAULT_CACHE)
        p.add_argument("--run", default=DEFAULT_RUN)
        p.add_argument("--datasets", type=int, nargs="+", default=list(DATASETS))
        p.add_argument("--max-horizon", type=int, default=MAX_HORIZON, dest="max_horizon")

    p = sub.add_parser("prep"); common(p)
    p.add_argument("--max-lag", type=int, default=BUILD_MAX_LAG, dest="max_lag")
    p.set_defaults(func=cmd_prep)

    p = sub.add_parser("search"); common(p)
    p.add_argument("--task", type=int, default=0, help="used when SLURM_ARRAY_TASK_ID is unset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--subsample", type=int, default=0,
                   help="cap on train rows per fold during search (0 = use all rows)")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("baseline"); common(p)
    p.add_argument("--max-lag", type=int, default=BASELINE_MAX_LAG, dest="max_lag")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--n-estimators", type=int, default=0, dest="n_estimators",
                   help="override BASELINE_CONFIG n_estimators (e.g. small for a quick local test)")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("finalize"); common(p)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_finalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
