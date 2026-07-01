"""
Experiment runner driven by experiments.yaml.

Reads experiments.yaml, runs every experiment with status: pending,
appends results to experiment_ledger.csv, and marks each experiment
as done in the yaml file when it completes.

Usage
-----
    python run_experiments.py                        # run all pending
    python run_experiments.py --name longer_lookback_78  # run one by name
    python run_experiments.py --dry-run              # print plan, no execution
    python run_experiments.py --list                 # show all experiments + status
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ── repo layout ───────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_FILE = REPO_ROOT / "experiments.yaml"
LEDGER_FILE      = REPO_ROOT / "experiment_ledger.csv"
SUBMISSIONS_DIR  = REPO_ROOT / "submissions"

# Ledger columns — config params first, then WIS scores
LEDGER_COLS = [
    "timestamp", "name", "model_family",
    "lr", "sigma", "splits", "lookback",
    "epochs", "patience", "mc_samples", "seed",
    "notes",
    "overall_wis", "test_1_2022_wis", "test_2_2023_wis", "test_3_2024_wis",
    "best_model",
]

from scripts.lstm_geo_experiment import (
    BATCH_SIZE,
    WEIGHT_DECAY,
    BETA,
    BASE_SMALL_ARCH,
    ArchitectureConfig,
    with_changes,
    prepare_geo_split,
    train_model_with_config,
    recursive_forecast_geo_split,
    save_submission_for_split,
    validate_submission_file,
    audit_submission_folders,
    float_tag,
    sigma_tag,
    clear_memory,
)
from scripts.evaluator import main as run_evaluator_main

ARCH_LOOKUP = {
    "base_proj32_hidden64_drop10_state8_head32": BASE_SMALL_ARCH,
    "state4":  with_changes(BASE_SMALL_ARCH, state_emb_dim=4),
    "state8":  with_changes(BASE_SMALL_ARCH, state_emb_dim=8),
    "state16": with_changes(BASE_SMALL_ARCH, state_emb_dim=16),
    "hidden128": with_changes(BASE_SMALL_ARCH, hidden_size=128),
    "hidden256": with_changes(BASE_SMALL_ARCH, hidden_size=256),
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_experiments() -> list[dict]:
    with open(EXPERIMENTS_FILE) as f:
        data = yaml.safe_load(f)
    return data["experiments"]


def save_experiments(experiments: list[dict]) -> None:
    with open(EXPERIMENTS_FILE) as f:
        raw = yaml.safe_load(f)
    raw["experiments"] = experiments
    with open(EXPERIMENTS_FILE, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def model_name(model_family: str, lr: float, sigma: float) -> str:
    return f"{model_family}_lr_{float_tag(lr)}_{sigma_tag(sigma)}"


def submission_path(model_family: str, lr: float, sigma: float,
                    split_id: int, target_start) -> Path:
    year = pd.Timestamp(target_start).year
    return SUBMISSIONS_DIR / model_name(model_family, lr, sigma) / f"test_{split_id}_{year}.csv"


def append_to_ledger(row: dict) -> None:
    """Append one experiment result row to the ledger CSV."""
    row_df = pd.DataFrame([row])
    # ensure consistent column order, fill missing with NaN
    for col in LEDGER_COLS:
        if col not in row_df.columns:
            row_df[col] = np.nan
    row_df = row_df[LEDGER_COLS]

    if LEDGER_FILE.exists():
        row_df.to_csv(LEDGER_FILE, mode="a", header=False, index=False)
    else:
        row_df.to_csv(LEDGER_FILE, mode="w", header=True, index=False)

    print(f"\nLedger updated → {LEDGER_FILE}")


def print_ledger() -> None:
    if not LEDGER_FILE.exists():
        print("No ledger yet.")
        return
    df = pd.read_csv(LEDGER_FILE)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_colwidth", 40)
    cols = [c for c in LEDGER_COLS if c in df.columns]
    print(df[cols].to_string(index=False))

def run_ensemble_experiment(exp: dict, dry_run: bool = False) -> None:
    name      = exp["name"]
    models    = exp["models"]          # list of 2+ model folder names
    strategy  = exp.get("strategy", "simple")  # simple or regional
    splits    = exp["splits"]

    # regional: which states use models[0] vs models[1]
    model_a_states = set(exp.get("model_a_states", []))

    NUMERIC_COLS = [
        "pred",
        "lower_50", "lower_80", "lower_90", "lower_95",
        "upper_50", "upper_80", "upper_90", "upper_95",
    ]
    FORECAST_FILES = {
        1: "test_1_2022.csv",
        2: "test_2_2023.csv",
        3: "test_3_2024.csv",
    }

    print(f"\n{'='*80}")
    print(f"ENSEMBLE: {name}")
    print(f"  strategy : {strategy}")
    print(f"  models   : {models}")
    if strategy == "regional":
        print(f"  model_a_states: {sorted(model_a_states)}")
    print(f"{'='*80}")

    if dry_run:
        print("  [dry-run] skipping execution")
        return

    out_dir = SUBMISSIONS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_id in splits:
        fname = FORECAST_FILES[split_id]
        dfs = []
        for model in models:
            path = SUBMISSIONS_DIR / model / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            df = pd.read_csv(path)
            df["state"] = df["state"].str.strip().str.upper()
            df["date"]  = pd.to_datetime(df["date"]).dt.normalize()
            dfs.append(df)

        if strategy == "simple":
            merged = dfs[0].merge(dfs[1], on=["state", "date"], suffixes=("_a", "_b"))
            result = merged[["state", "date"]].copy()
            for col in NUMERIC_COLS:
                result[col] = (merged[f"{col}_a"] + merged[f"{col}_b"]) / 2.0

        elif strategy == "regional":
            a = dfs[0][dfs[0]["state"].isin(model_a_states)].copy()
            b = dfs[1][~dfs[1]["state"].isin(model_a_states)].copy()
            result = pd.concat([a, b]).sort_values(["state", "date"]).reset_index(drop=True)

        # enforce interval monotonicity after averaging
        result["lower_95"] = np.minimum(result["lower_95"], result["lower_90"])
        result["lower_90"] = np.minimum(result["lower_90"], result["lower_80"])
        result["lower_80"] = np.minimum(result["lower_80"], result["lower_50"])
        result["lower_50"] = np.minimum(result["lower_50"], result["pred"])
        result["upper_50"] = np.maximum(result["upper_50"], result["pred"])
        result["upper_80"] = np.maximum(result["upper_80"], result["upper_50"])
        result["upper_90"] = np.maximum(result["upper_90"], result["upper_80"])
        result["upper_95"] = np.maximum(result["upper_95"], result["upper_90"])
        result[NUMERIC_COLS] = result[NUMERIC_COLS].clip(lower=0.0)

        out_path = out_dir / fname
        result[["state", "date"] + NUMERIC_COLS].to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  shape={result.shape}")

# ── core runner ───────────────────────────────────────────────────────────────

def run_experiment(exp: dict, dry_run: bool = False) -> None:
    
    if exp.get("type") == "ensemble":
        run_ensemble_experiment(exp, dry_run)
        return
    
    name         = exp["name"]
    model_family = exp["model_family"]
    lr_values    = exp["lr"]
    sigma_scales = exp["sigma"]
    splits       = exp["splits"]
    lookback     = exp["lookback"]
    epochs       = exp["epochs"]
    patience     = exp["patience"]
    mc_samples   = exp["mc_samples"]
    lag_weeks    = exp.get("lag_weeks", []) or []
    use_incidence = exp.get("use_incidence", False)
    seed         = exp["seed"]
    force_rerun  = exp.get("force_rerun", False)
    notes        = exp.get("notes", "")

    n_trains = len(splits) * len(lr_values)
    n_files  = n_trains * len(sigma_scales)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"  model_family : {model_family}")
    print(f"  lr           : {lr_values}")
    print(f"  sigma        : {sigma_scales}")
    print(f"  splits       : {splits}")
    print(f"  lookback     : {lookback}  epochs: {epochs}  patience: {patience}")
    print(f"  mc_samples   : {mc_samples}  seed: {seed}")
    print(f"  lag_weeks    : {lag_weeks}")
    print(f"  notes        : {notes}")
    print(f"  → up to {n_trains} training runs, {n_files} submission files")
    print(f"{'='*80}")

    if dry_run:
        print("  [dry-run] skipping execution")
        return

    target_start_by_split: dict[int, pd.Timestamp] = {}

    for split_id in splits:
        print(f"\n{'─'*60}\nSPLIT {split_id}\n{'─'*60}")

        artifacts = prepare_geo_split(
            split_id, 
            lookback=lookback, 
            batch_size=BATCH_SIZE, 
            lag_weeks=lag_weeks,
            use_incidence=use_incidence,)
        target_start = artifacts["target_start"]
        target_start_by_split[split_id] = target_start

        for lr in lr_values:
            missing_sigmas = [
                s for s in sigma_scales
                if force_rerun or not submission_path(
                    model_family, lr, s, split_id, target_start
                ).exists()
            ]

            if not missing_sigmas:
                print(f"  lr={lr:.6g} | all {len(sigma_scales)} sigma files exist, skipping")
                continue

            print(f"  Training | split={split_id} | lr={lr:.6g} | missing sigmas: {missing_sigmas}")
            
            arch_config_name = exp.get("arch_config", "base_proj32_hidden64_drop10_state8_head32")
            arch_config = ARCH_LOOKUP[arch_config_name]
            print(f"  arch_config  : {arch_config_name}")

            model, history_df, best_val = train_model_with_config(
                artifacts   = artifacts,
                arch_name   = f"{name}_lr_{float_tag(lr)}",
                arch_config = arch_config,
                lr          = lr,
                epochs      = epochs,
                patience    = patience,
                weight_decay= WEIGHT_DECAY,
                beta        = BETA,
                seed        = seed + split_id,
            )
            print(f"  arch_config  : {exp.get('arch_config', 'base_proj32_hidden64_drop10_state8_head32')}")

            log_dir = REPO_ROOT / "experiment_logs" / name
            log_dir.mkdir(parents=True, exist_ok=True)
            history_df.to_csv(
                log_dir / f"history_split{split_id}_lr{float_tag(lr)}.csv",
                index=False,
            )

            for sigma in missing_sigmas:
                mname = model_name(model_family, lr, sigma)
                print(f"  Forecasting | split={split_id} | lr={lr:.6g} | sigma={sigma}")

                submission = recursive_forecast_geo_split(
                    model       = model,
                    artifacts   = artifacts,
                    lookback    = lookback,
                    mc_samples  = mc_samples,
                    sigma_scale = sigma,
                )

                out_path = save_submission_for_split(
                    submission   = submission,
                    split_id     = split_id,
                    target_start = target_start,
                    model_name   = mname,
                )

                validate_submission_file(out_path)

            del model
            clear_memory()

    # ── evaluate and write ledger ─────────────────────────────────────────────

    expected_files = [
        submission_path(model_family, lr, s, split_id, target_start_by_split[split_id])
        for split_id in splits
        for lr in lr_values
        for s in sigma_scales
    ]
    missing = [p for p in expected_files if not p.exists()]

    if missing:
        print(f"\nWarning: {len(missing)} submission file(s) missing — skipping evaluator.")
        return

    print("\nRunning evaluator …")
    # Call evaluator directly as a Python function (avoids subprocess path issues)
    try:
        run_evaluator_main()
    except Exception as e:
        print(f"Evaluator error: {e}")
        return

    results = pd.read_csv(REPO_ROOT / "results.csv")
    exp_models = [model_name(model_family, lr, s) for lr in lr_values for s in sigma_scales]
    subset = results[results["model"].isin(exp_models)].copy()

    if subset.empty:
        print("Warning: no results found for this experiment's models.")
        return

    # summarise — best model by overall WIS
    best_row = subset.sort_values("overall_wis").iloc[0]

    wis_cols = ["overall_wis", "test_1_2022_wis", "test_2_2023_wis", "test_3_2024_wis"]
    wis_cols = [c for c in wis_cols if c in subset.columns]

    print(f"\nResults for experiment '{name}' (sorted by overall WIS):")
    print(subset[["model"] + wis_cols].sort_values("overall_wis").to_string(index=False))
    print(f"\nBest: {best_row['model']}  overall_wis={best_row['overall_wis']:.2f}")

    ledger_row = {
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name"           : name,
        "model_family"   : model_family,
        "lr"             : str(lr_values),
        "sigma"          : str(sigma_scales),
        "splits"         : str(splits),
        "lookback"       : lookback,
        "epochs"         : epochs,
        "patience"       : patience,
        "mc_samples"     : mc_samples,
        "seed"           : seed,
        "notes"          : notes,
        "overall_wis"    : round(best_row.get("overall_wis", np.nan), 2),
        "test_1_2022_wis": round(best_row.get("test_1_2022_wis", np.nan), 2),
        "test_2_2023_wis": round(best_row.get("test_2_2023_wis", np.nan), 2),
        "test_3_2024_wis": round(best_row.get("test_3_2024_wis", np.nan), 2),
        "best_model"     : best_row["model"],
    }
    append_to_ledger(ledger_row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--name",    type=str, default=None,
                   help="Run only the experiment with this name.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without executing anything.")
    p.add_argument("--list",    action="store_true",
                   help="List all experiments with their status and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    experiments = load_experiments()

    if args.list:
        print(f"\n{'NAME':<35} {'STATUS':<10} NOTES")
        print("─" * 80)
        for exp in experiments:
            print(f"{exp['name']:<35} {exp['status']:<10} {exp.get('notes', '')[:40]}")
        print()
        print_ledger()
        return

    # filter to requested experiment or all pending
    if args.name:
        targets = [e for e in experiments if e["name"] == args.name]
        if not targets:
            raise ValueError(f"No experiment named '{args.name}' in {EXPERIMENTS_FILE}")
        # allow running a done experiment by name explicitly
        for t in targets:
            t["_force_run"] = True
    else:
        targets = [e for e in experiments if e["status"] == "pending"]

    if not targets:
        print("No pending experiments. Edit experiments.yaml to add new ones.")
        return

    print(f"\nWill run {len(targets)} experiment(s): {[e['name'] for e in targets]}")

    for exp in targets:
        run_experiment(exp, dry_run=args.dry_run)

        if not args.dry_run:
            # mark as done in yaml
            for e in experiments:
                if e["name"] == exp["name"]:
                    e["status"] = "done"
            save_experiments(experiments)
            print(f"\nMarked '{exp['name']}' as done in {EXPERIMENTS_FILE}")


if __name__ == "__main__":
    main()