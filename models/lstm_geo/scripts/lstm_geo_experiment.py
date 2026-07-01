from __future__ import annotations

import argparse
import copy
import gc
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from data_catalog.data_factory import data_factory
import scripts.lstm as lstm_mod


# Paths and device

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_DATASET = REPO_ROOT / "data_catalog" / "full_dataset.csv"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
LOG_DIR = REPO_ROOT / "experiment_logs"

#if torch.cuda.is_available():
 #   DEVICE = torch.device("cuda")
#elif torch.backends.mps.is_available():
 #   DEVICE = torch.device("mps")
#else:
  #  DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")

# Global defaults

LOOKBACK = 52
BATCH_SIZE = 256
WEIGHT_DECAY = 1e-4
BETA = 1e-4 #initial was 1e-4
PRIOR_SIGMA = 1.0
POSTERIOR_RHO_INIT = -5.0

EXPECTED_FILES = {
    "test_1_2022.csv",
    "test_2_2023.csv",
    "test_3_2024.csv",
}


# Config objects

@dataclass(frozen=True)
class ArchitectureConfig:
    input_proj_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    state_emb_dim: int
    head_hidden_dim: int


@dataclass(frozen=True)
class ExperimentConfig:
    model_family: str
    architectures: Dict[str, ArchitectureConfig]
    sigma_scales: List[float]
    lr: float = 1e-3
    splits: List[int] = None
    mc_samples: int = 500
    epochs: int = 30
    patience: int = 10
    seed: int = 42
    force_rerun: bool = False
    quarantine_incomplete: bool = True

    def __post_init__(self):
        if self.splits is None:
            object.__setattr__(self, "splits", [1, 2, 3])


# Predefined experiment configs

BASE_SMALL_ARCH = ArchitectureConfig(
    input_proj_dim=32,
    hidden_size=64,
    num_layers=1,
    dropout=0.10,
    state_emb_dim=8,
    head_hidden_dim=32,
)


def with_changes(base: ArchitectureConfig, **changes) -> ArchitectureConfig:
    values = base.__dict__.copy()
    values.update(changes)
    return ArchitectureConfig(**values)


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "small_ablation": ExperimentConfig(
        model_family="geo_lstm_v2b_small_ablation_v2",
        lr=1e-3,
        sigma_scales=[2.25],
        architectures={
            "base_proj32_hidden64_drop10_state8_head32": BASE_SMALL_ARCH,

            "proj8": with_changes(BASE_SMALL_ARCH, input_proj_dim=8),
            "proj16": with_changes(BASE_SMALL_ARCH, input_proj_dim=16),
            "proj48": with_changes(BASE_SMALL_ARCH, input_proj_dim=48),

            "hidden32": with_changes(BASE_SMALL_ARCH, hidden_size=32),
            "hidden96": with_changes(BASE_SMALL_ARCH, hidden_size=96),

            "dropout0": with_changes(BASE_SMALL_ARCH, dropout=0.00),
            "dropout20": with_changes(BASE_SMALL_ARCH, dropout=0.20),

            "state4": with_changes(BASE_SMALL_ARCH, state_emb_dim=4),
            "state16": with_changes(BASE_SMALL_ARCH, state_emb_dim=16),

            "head8": with_changes(BASE_SMALL_ARCH, head_hidden_dim=8),
            "head16": with_changes(BASE_SMALL_ARCH, head_hidden_dim=16),
            "head64": with_changes(BASE_SMALL_ARCH, head_hidden_dim=64),
        },
    ),

    "small_sigma": ExperimentConfig(
        model_family="geo_lstm_v2b_small_sigma_v2",
        lr=1e-3,
        sigma_scales=[2.0, 2.25, 2.5, 2.75, 3.0],
        architectures={
            "base_proj32_hidden64_drop10_state8_head32": BASE_SMALL_ARCH,
        },
    ),

    "small_head": ExperimentConfig(
        model_family="geo_lstm_v2b_small_head_v2",
        lr=1e-3,
        sigma_scales=[2.25],
        architectures={
            "head1": with_changes(BASE_SMALL_ARCH, head_hidden_dim=1),
            "head2": with_changes(BASE_SMALL_ARCH, head_hidden_dim=2),
            "head4": with_changes(BASE_SMALL_ARCH, head_hidden_dim=4),
            "head8": with_changes(BASE_SMALL_ARCH, head_hidden_dim=8),
            "head16": with_changes(BASE_SMALL_ARCH, head_hidden_dim=16),
            "head32": with_changes(BASE_SMALL_ARCH, head_hidden_dim=32),
        },
    ),
}


# General helpers

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def float_tag(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def sigma_tag(sigma_scale: float) -> str:
    return "sigma_" + float_tag(sigma_scale)


def model_name_for(config: ExperimentConfig, arch_name: str, sigma_scale: float) -> str:
    return f"{config.model_family}_{arch_name}_{sigma_tag(sigma_scale)}"


def expected_model_names(config: ExperimentConfig) -> set[str]:
    names = set()

    for arch_name in config.architectures:
        for sigma_scale in config.sigma_scales:
            names.add(model_name_for(config, arch_name, sigma_scale))

    return names


def submission_path(model_name: str, split_id: int, target_start) -> Path:
    year = pd.Timestamp(target_start).year
    return SUBMISSIONS_DIR / model_name / f"test_{split_id}_{year}.csv"


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Sequence creation and split preparation

def make_lstm_sequences_with_state(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int = 52,
    target_col: str = "target_cases",
    state_col: str = "state",
    state_id_col: str = "state_id",
    date_col: str = "date",
    lag_weeks: Optional[List[int]] = None,
    cases_col: str = "cases",
):
    X = []
    state_ids = []
    y = []
    meta = []
    lag_values = []        

    lag_weeks = [w for w in (lag_weeks or []) if 0 < w]
    cases_idx = feature_cols.index(cases_col) if cases_col in feature_cols else None

    df = df.sort_values([state_col, date_col]).copy()

    for state, g in df.groupby(state_col):
        g = g.sort_values(date_col).reset_index(drop=True)

        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        state_id_values = g[state_id_col].to_numpy(dtype=np.int64)
        dates = g[date_col].to_numpy()

        for t in range(lookback, len(g)):
            x_window = features[t - lookback:t].copy()   # [lookback, n_features]
            y_value = targets[t]

            if not np.isfinite(x_window).all():
                continue
            if not np.isfinite(y_value):
                continue
            # ── NEW: extract lag scalars separately ───────────────────────
            # Each lag k looks up cases at absolute position (t - k) in the
            # state's panel — NOT relative to the window.
            # t - k may be outside the window (e.g. k=52, lookback=78 is fine;
            # k=52, lookback=52 would be position 0, edge case handled below).
            if lag_weeks and cases_idx is not None:
                lag_vals = []
                for k in lag_weeks:
                    lag_idx = t - k
                    if lag_idx >= 0 and np.isfinite(features[lag_idx, cases_idx]):
                        lag_vals.append(features[lag_idx, cases_idx])
                    else:
                        lag_vals.append(0.0)   # fallback: zero ≈ training mean in scaled space
                lag_values.append(np.array(lag_vals, dtype=np.float32))
            else:
                lag_values.append(np.zeros(len(lag_weeks or []), dtype=np.float32))

            X.append(x_window)
            state_ids.append(state_id_values[t])
            y.append(np.log1p(y_value))
            meta.append({
                "state": state,
                "state_id": int(state_id_values[t]),
                "date": dates[t],
                "target_cases": float(y_value),
            })

    X          = np.asarray(X,          dtype=np.float32)   # [N, lookback, n_features]
    state_ids  = np.asarray(state_ids,  dtype=np.int64)     # [N]
    y          = np.asarray(y,          dtype=np.float32)   # [N]
    lag_values = np.asarray(lag_values, dtype=np.float32)   # [N, n_lags]  ← NEW
    meta       = pd.DataFrame(meta)

    return X, state_ids, y, meta, lag_values   # ← returns 5 values now, not 4


def prepare_geo_split(
    split_id: int,
    lookback: int = LOOKBACK,
    batch_size: int = BATCH_SIZE,
    lag_weeks: Optional[List[int]] = None,
    use_incidence: bool = False,
):
    """
    lag_weeks : list of positive integers, each < lookback.
        Adds lagged-cases features computed on the fly at window-reading time.
        E.g. [1, 2, 4, 7] adds cases from 1, 2, 4, and 7 weeks before each
        prediction step. Default None (disabled) is backward-compatible.
    """
    train_df, target_df = data_factory(split_id)

    train_df = train_df.copy()
    target_df = target_df.copy()

    train_df["date"] = pd.to_datetime(train_df["date"])
    target_df["date"] = pd.to_datetime(target_df["date"])

    full_df = pd.read_csv(FULL_DATASET)
    full_df["date"] = pd.to_datetime(full_df["date"])

    train_end = train_df["date"].max()
    target_start = target_df["date"].min()
    target_end = target_df["date"].max()

    states = sorted(train_df["state"].unique())
    state_to_id = {state: i for i, state in enumerate(states)}
    id_to_state = {i: state for state, i in state_to_id.items()}
    n_states = len(states)

    panel_df = full_df[
        (full_df["state"].isin(states))
        & (full_df["date"] <= target_end)
    ].copy()

    panel_df = panel_df.sort_values(["state", "date"]).reset_index(drop=True)
    panel_df["state_id"] = panel_df["state"].map(state_to_id).astype(int)

    week = panel_df["date"].dt.isocalendar().week.astype(int)
    panel_df["week_sin"] = np.sin(2 * np.pi * week / 52)
    panel_df["week_cos"] = np.cos(2 * np.pi * week / 52)

    panel_df["target_cases"] = panel_df["cases"].astype(float)

    base_features = [
        "cases",
        "population",
        "week_sin",
        "week_cos",
    ]

    forecast_features = [
        c for c in panel_df.columns
        if c.startswith("forecast_")
    ]

    # Observed meteo and climate index columns — time-varying and unknown
    # at real prediction time, so treated identically to forecast_* columns.
    observed_meteo_features = [
        c for c in panel_df.columns
        if c.startswith("temp_")
        or c.startswith("precip_")
        or c.startswith("pressure_")
        or c.startswith("rel_humid_")
        or c in ("thermal_range", "rainy_days", "enso", "iod", "pdo")
    ]

    geo_static_features = [
        c for c in panel_df.columns
        if c.startswith("koppen_")
        or c.startswith("biome_")
    ]

    lag_weeks = [w for w in (lag_weeks or []) if w >0]
    lag_col_names = [f"cases_lag_{w}w" for w in lag_weeks]

    feature_cols = (
        base_features
        + forecast_features
        + observed_meteo_features
        + geo_static_features
    )
    feature_cols = [c for c in feature_cols if c in panel_df.columns]

    # Lag feature names are appended AFTER panel columns so indices are
    # consistent: panel feature_cols come first, then lag cols.
    feature_cols_with_lags = feature_cols + lag_col_names

    train_model_df = panel_df[panel_df["date"] <= train_end].copy()
    panel_proc = panel_df.copy()

    # add incidence per 100k — used as training target instead of raw cases
    if use_incidence:
        panel_proc["incidence"] = (
            panel_proc["cases"] / panel_proc["population"].replace(0, np.nan) * 100_000
        )
        train_model_df["incidence"] = (
            train_model_df["cases"] / train_model_df["population"].replace(0, np.nan) * 100_000
        )

    
    future_rows = panel_proc["date"] > train_end

    # Null future cases (filled later by recursive forecasting loop).
    panel_proc.loc[future_rows, "cases"] = np.nan

    # Null future forecast_* and observed meteo/climate columns.
    leaky_cols = [
        c for c in forecast_features + observed_meteo_features
        if c in panel_proc.columns
    ]
    if leaky_cols:
        panel_proc.loc[future_rows, leaky_cols] = np.nan

    # Compute imputation medians from training data only.
    non_cases_cols = [c for c in feature_cols if c != "cases"]
    medians = train_model_df[feature_cols].median(numeric_only=True)

    train_model_df[feature_cols] = train_model_df[feature_cols].fillna(medians)
    # Fill all non-cases future rows (including the now-nulled forecast_* cols)
    # with training medians, then 0 as a final safety net.
    panel_proc[non_cases_cols] = panel_proc[non_cases_cols].fillna(medians)
    # future-row cases remain NaN; they are filled by the recursive loop.

    train_model_df[feature_cols] = train_model_df[feature_cols].fillna(0.0)
    panel_proc[non_cases_cols] = panel_proc[non_cases_cols].fillna(0.0)

    state_mean_cases = (
        panel_df[panel_df["date"] <= train_end]
        .groupby("state")["cases"]
        .mean()
        .clip(lower=1.0)
    )
    state_weight_map = 1.0 / state_mean_cases
    state_weight_map = state_weight_map.clip(lower=0.1, upper=5.0) 
    state_weight_map = state_weight_map / state_weight_map.mean()  # normalise to mean=1

    scaler = StandardScaler()
    scaler.fit(train_model_df[feature_cols])

    train_model_df[feature_cols] = scaler.transform(train_model_df[feature_cols])

    # Scale the full panel using train-fitted scaler.
    # We temporarily fill future cases with 0 just to satisfy sklearn's
    # no-NaN requirement; those values are immediately overwritten below.
    panel_proc_fill = panel_proc[feature_cols].copy()
    panel_proc_fill["cases"] = panel_proc_fill["cases"].fillna(0.0)
    panel_proc[feature_cols] = scaler.transform(panel_proc_fill)

    # Re-null future cases in scaled space so the recursive loop's NaN
    # guard (np.isfinite) can detect un-filled positions if the loop
    # ever processes them out of order.
    panel_proc.loc[future_rows, "cases"] = np.nan

    X_all, state_all, y_all, meta_all, lag_all = make_lstm_sequences_with_state(
        train_model_df,
        feature_cols=feature_cols,
        lookback=lookback,
        lag_weeks=lag_weeks,
        target_col="incidence" if use_incidence else "target_cases",
    )

    val_start = train_end - pd.Timedelta(weeks=52)

    train_mask = meta_all["date"] <= val_start
    val_mask = meta_all["date"] > val_start

    X_train = X_all[train_mask.to_numpy()]
    state_train = state_all[train_mask.to_numpy()]
    y_train = y_all[train_mask.to_numpy()]

    X_val = X_all[val_mask.to_numpy()]
    state_val = state_all[val_mask.to_numpy()]
    y_val = y_all[val_mask.to_numpy()]

    lag_train = lag_all[train_mask.to_numpy()]
    lag_val = lag_all[val_mask.to_numpy()]


    weights_all = np.array(
        [float(state_weight_map.get(meta_all.loc[i, "state"], 1.0))
        for i in range(len(meta_all))],
        dtype=np.float32,
    )
    weights_train = weights_all[train_mask.to_numpy()]
    weights_val   = weights_all[val_mask.to_numpy()]


    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(state_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(lag_train, dtype=torch.float32),
        torch.tensor(weights_train, dtype=torch.float32),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(state_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(lag_val, dtype=torch.float32),
        torch.tensor(weights_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    artifacts = {
        "split_id": split_id,
        "train_df": train_df,
        "target_df": target_df,
        "panel_df": panel_df,
        "panel_proc": panel_proc,
        "train_model_df": train_model_df,
        "feature_cols": feature_cols,           # panel columns only (no lags)
        "feature_cols_with_lags": feature_cols_with_lags,  # panel + lag col names
        "lag_weeks": lag_weeks,                 # list of lag offsets used
        "lag_col_names": lag_col_names,         # e.g. ["cases_lag_1w", ...]
        "scaler": scaler,
        "states": states,
        "state_to_id": state_to_id,
        "id_to_state": id_to_state,
        "n_states": n_states,
        "train_end": train_end,
        "target_start": target_start,
        "target_end": target_end,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_train": X_train,
        "X_val": X_val,
        "state_train": state_train,
        "state_val": state_val,
        "y_train": y_train,
        "y_val": y_val,
        "meta_all": meta_all,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "lag_train": lag_train,
        "lag_val": lag_val,
        "use_incidence": use_incidence,
        "weights_train": weights_train,
        "weights_val": weights_val,
    }

    print(f"\nPrepared split {split_id}")
    print("train:", train_df.shape, train_df["date"].min(), "→", train_end)
    print("target:", target_df.shape, target_start, "→", target_end)
    print("X_train:", X_train.shape, f"  ({X_train.shape[2]} features incl. {len(lag_weeks)} lags)")
    print("X_val:", X_val.shape)
    print(f"features (panel): {len(feature_cols)}  |  lag features: {len(lag_weeks)} {lag_col_names}")
    print(f"leakage fix: nulled {len(leaky_cols)} future cols → replaced with training-period medians")
    print(f"  forecast_*       ({len(forecast_features)}): {forecast_features}")
    print(f"  observed_meteo   ({len(observed_meteo_features)}): {observed_meteo_features}")

    return artifacts


# Model

class DeepGeoLSTMBayesianHead(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_states: int,
        input_proj_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
        state_emb_dim: int = 8,
        head_hidden_dim: int = 32,
        n_lags: int = 0, 
        prior_sigma: float = 1.0,
        posterior_rho_init: float = -5.0,
    ):
        super().__init__()

        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, input_proj_dim),
            nn.LayerNorm(input_proj_dim),
            nn.ReLU(), #GELU or ReLU
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=input_proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.state_embedding = nn.Embedding(
            num_embeddings=n_states,
            embedding_dim=state_emb_dim,
        )

        self.n_lags = n_lags

        self.state_sigma_scale = nn.Embedding(n_states, 1)
        nn.init.constant_(self.state_sigma_scale.weight, 0.5413)

        combined_dim = hidden_size + state_emb_dim + n_lags

        self.head = nn.Sequential(
            nn.Linear(combined_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.ReLU(), #GELU or ReLU
            nn.Dropout(dropout),
        )

        self.mu_head = lstm_mod.BayesianLinear(
            in_features=head_hidden_dim,
            out_features=1,
            prior_sigma=prior_sigma,
            posterior_rho_init=posterior_rho_init,
        )

        self.rho_head = lstm_mod.BayesianLinear(
            in_features=head_hidden_dim,
            out_features=1,
            prior_sigma=prior_sigma,
            posterior_rho_init=posterior_rho_init,
        )

    def forward(self, x, state_id, lag_values=None, sample: bool = True):
        x_proj = self.feature_encoder(x)
        lstm_out, _ = self.lstm(x_proj)
        h_last = lstm_out[:, -1, :]
        state_vec = self.state_embedding(state_id)

        parts = [h_last, state_vec]
        if lag_values is not None and self.n_lags > 0:
            parts.append(lag_values)
        z = torch.cat(parts, dim=1)
        
        z = self.head(z)
        mu = self.mu_head(z, sample=sample).squeeze(-1)
        rho = self.rho_head(z, sample=sample).squeeze(-1)
        sigma = F.softplus(rho) + 1e-6
        state_scale = F.softplus(self.state_sigma_scale(state_id)).squeeze(-1) + 0.5
        state_scale = torch.clamp(state_scale, 0.5, 3.0)
        sigma = sigma * state_scale

        return mu, sigma

    def kl_divergence(self):
        return (
            self.mu_head.kl_divergence()
            + self.rho_head.kl_divergence()
        )


def build_model_from_config(artifacts, arch_config: ArchitectureConfig):
    model = DeepGeoLSTMBayesianHead(
        n_features=artifacts["X_train"].shape[2],
        n_states=artifacts["n_states"],
        input_proj_dim=arch_config.input_proj_dim,
        hidden_size=arch_config.hidden_size,
        num_layers=arch_config.num_layers,
        dropout=arch_config.dropout,
        state_emb_dim=arch_config.state_emb_dim,
        head_hidden_dim=arch_config.head_hidden_dim,
        n_lags=len(artifacts["lag_weeks"] or []),
        prior_sigma=PRIOR_SIGMA,
        posterior_rho_init=POSTERIOR_RHO_INIT,
    ).to(DEVICE)

    return model


# Loss and evaluation

def geo_bayesian_lstm_loss(
    model,
    x,
    state_id,
    y,
    lag_values=None,
    state_weights=None,     # ← add
    beta: float = BETA,
    sample: bool = True,
):
    mu, sigma = model(x, state_id, lag_values=lag_values, sample=sample)

    nll_per_sample = lstm_mod.gaussian_nll(y, mu, sigma, reduction="none")
    
    if state_weights is not None:
        nll = (nll_per_sample * state_weights).mean()
    else: 
        nll = nll_per_sample.mean()

    kl = model.kl_divergence()
    total_loss = nll + beta * kl

    return {
        "total_loss": total_loss,
        "nll": nll.detach(),
        "kl": kl.detach(),
    }


def evaluate_geo_model(model, loader, beta: float = BETA):
    model.eval()

    total_loss_sum = 0.0
    nll_sum = 0.0
    kl_sum = 0.0
    n_obs = 0

    with torch.no_grad():
        for xb, sidb, yb, lagb, wb in loader:
            xb = xb.to(DEVICE)
            sidb = sidb.to(DEVICE)
            yb = yb.to(DEVICE)
            lagb = lagb.to(DEVICE)
            wb = wb.to(DEVICE)

            loss_out = geo_bayesian_lstm_loss(
                model=model,
                x=xb,
                state_id=sidb,
                y=yb,
                lag_values=lagb,
                state_weights=wb,
                beta=beta,
                sample=False,
            )

            batch_n = xb.shape[0]

            total_loss_sum += loss_out["total_loss"].item() * batch_n
            nll_sum += loss_out["nll"].item() * batch_n
            kl_sum += loss_out["kl"].item()
            n_obs += batch_n

    return {
        "loss": total_loss_sum / n_obs,
        "nll": nll_sum / n_obs,
        "kl": kl_sum / max(len(loader), 1),
    }


# Training

def train_model_with_config(
    artifacts,
    arch_name: str,
    arch_config: ArchitectureConfig,
    lr: float,
    epochs: int,
    patience: int,
    weight_decay: float = WEIGHT_DECAY,
    beta: float = BETA,
    seed: int = 42,
):
    set_seed(seed)

    model = build_model_from_config(artifacts, arch_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = artifacts["train_loader"]
    val_loader = artifacts["val_loader"]

    history = []
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_nll_sum = 0.0
        train_kl_sum = 0.0
        n_obs = 0

        for xb, sidb, yb, lagb, wb in train_loader:
            xb = xb.to(DEVICE)
            sidb = sidb.to(DEVICE)
            yb = yb.to(DEVICE)
            lagb=lagb.to(DEVICE)
            wb = wb.to(DEVICE)

            optimizer.zero_grad()

            loss_out = geo_bayesian_lstm_loss(
                model=model,
                x=xb,
                state_id=sidb,
                y=yb,
                lag_values=lagb,
                state_weights=wb,
                beta=beta,
                sample=True,
            )

            loss_out["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n = xb.shape[0]

            train_loss_sum += loss_out["total_loss"].item() * batch_n
            train_nll_sum += loss_out["nll"].item() * batch_n
            train_kl_sum += loss_out["kl"].item()
            n_obs += batch_n

        train_loss = train_loss_sum / n_obs
        train_nll = train_nll_sum / n_obs
        train_kl = train_kl_sum / max(len(train_loader), 1)

        val_metrics = evaluate_geo_model(
            model=model,
            loader=val_loader,
            beta=beta,
        )

        row = {
            "split_id": artifacts["split_id"],
            "architecture": arch_name,
            "lr": lr,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_nll": train_nll,
            "train_kl": train_kl,
            "val_loss": val_metrics["loss"],
            "val_nll": val_metrics["nll"],
            "val_kl": val_metrics["kl"],
            "trainable_params": count_trainable_params(model),
            **arch_config.__dict__,
        }

        history.append(row)

        improved = row["val_loss"] < best_val_loss

        if improved:
            best_val_loss = row["val_loss"]
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"split={artifacts['split_id']} | "
            f"arch={arch_name} | "
            f"epoch={epoch:03d} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"{'best' if improved else f'no improve {epochs_without_improvement}/{patience}'}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping | split={artifacts['split_id']} | arch={arch_name}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history_df = pd.DataFrame(history)

    print(
        f"Best val loss | split={artifacts['split_id']} | "
        f"arch={arch_name}: {best_val_loss:.4f}"
    )

    return model, history_df, best_val_loss


# Monte Carlo prediction and recursive forecasting

def mc_predict_cases_geo(
    model,
    x_np,
    state_ids_np,
    lag_values_np=None,
    mc_samples: int = 500,
    sigma_scale: float = 1.0,
):
    model.eval()

    x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
    state_ids = torch.tensor(state_ids_np, dtype=torch.long).to(DEVICE)

    lag_t = (
        torch.tensor(lag_values_np, dtype=torch.float32).to(DEVICE)
        if lag_values_np is not None and lag_values_np.ndim == 2 and lag_values_np.shape[1] > 0
        else None
    )

    samples = []

    with torch.no_grad():
        for _ in range(mc_samples):
            mu, sigma = model(x, state_ids, lag_values=lag_t, sample=True)

            sigma = sigma * sigma_scale

            y_log_sample = mu + sigma * torch.randn_like(sigma)
            y_cases = torch.expm1(y_log_sample)
            y_cases = torch.clamp(y_cases, min=0.0)

            samples.append(y_cases.cpu().numpy())

    samples = np.stack(samples, axis=0)

    if samples.ndim == 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)

    return samples


def recursive_forecast_geo_split(
    model,
    artifacts,
    lookback: int = LOOKBACK,
    mc_samples: int = 500,
    sigma_scale: float = 1.0,
):
    panel = artifacts["panel_proc"].copy()
    target_df = artifacts["target_df"].copy()
    feature_cols = artifacts["feature_cols"]           # panel columns only
    lag_weeks = artifacts.get("lag_weeks", []) or []  # e.g. [1, 2, 4, 7]
    states = artifacts["states"]
    train_end = artifacts["train_end"]
    target_end = artifacts["target_end"]
    scaler = artifacts["scaler"]
    use_incidence = artifacts.get("use_incidence", False)

    panel = panel.sort_values(["state", "date"]).reset_index(drop=True)
    target_df["date"] = pd.to_datetime(target_df["date"])

    if "cases" not in feature_cols:
        raise ValueError("'cases' must be in feature_cols for recursive forecasting.")

    cases_col_idx = feature_cols.index("cases")
    cases_mean = scaler.mean_[cases_col_idx]
    cases_scale = scaler.scale_[cases_col_idx]

    # panel_proc already has future cases nulled at prepare_geo_split time
    # (before scaling), so no second nulling is needed here.  Assert this
    # invariant so it is obvious if the calling code ever changes.
    future_mask = panel["date"] > train_end
    n_future_non_nan = panel.loc[future_mask, "cases"].notna().sum()
    if n_future_non_nan > 0:
        raise ValueError(
            f"panel_proc has {n_future_non_nan} non-null future cases. "
            "prepare_geo_split must null future cases before scaling."
        )

    state_to_indices = {
        state: panel.index[panel["state"] == state].to_list()
        for state in states
    }

    state_date_to_index = {
        (row.state, row.date): idx
        for idx, row in panel[["state", "date"]].iterrows()
    }

    target_pairs = set(
        zip(target_df["state"].to_list(), target_df["date"].to_list())
    )

    future_dates = sorted(
        panel.loc[
            (panel["date"] > train_end)
            & (panel["date"] <= target_end),
            "date",
        ].unique()
    )

    records = []

    for current_date in future_dates:
        X_batch = []
        state_id_batch = []
        lag_batch = []
        row_indices = []
        meta_batch = []

        for state in states:
            key = (state, current_date)

            if key not in state_date_to_index:
                continue

            row_idx = state_date_to_index[key]
            state_indices = state_to_indices[state]
            pos = state_indices.index(row_idx)

            if pos < lookback:
                continue

            window_indices = state_indices[pos - lookback:pos]
            x_window = panel.loc[window_indices, feature_cols].to_numpy(dtype=np.float32)

            if not np.isfinite(x_window).all():
                continue

            if lag_weeks:
                lag_vals = []
                for k in lag_weeks:
                    lag_idx = pos - k   # absolute position in state_indices list
                    if lag_idx >= 0:
                        panel_lag_row = state_indices[lag_idx]
                        val = panel.loc[panel_lag_row, "cases"]
                        lag_vals.append(float(val) if np.isfinite(float(val)) else 0.0)
                    else:
                        lag_vals.append(0.0)
                lag_batch.append(np.array(lag_vals, dtype=np.float32))
            else:
                lag_batch.append(np.zeros(0, dtype=np.float32))

            X_batch.append(x_window)
            state_id_batch.append(int(panel.loc[row_idx, "state_id"]))
            row_indices.append(row_idx)
            meta_batch.append((state, current_date))

        if not X_batch:
            continue

        X_batch        = np.asarray(X_batch,        dtype=np.float32)
        state_id_batch = np.asarray(state_id_batch, dtype=np.int64)
        lag_batch_np   = np.asarray(lag_batch,      dtype=np.float32)  # [n_states, n_lags]

        samples_cases = mc_predict_cases_geo(
            model=model,
            x_np=X_batch,
            state_ids_np=state_id_batch,
            lag_values_np=lag_batch_np,   # ← add
            mc_samples=mc_samples,
            sigma_scale=sigma_scale,
        )

        quantiles = {
            "lower_95": np.quantile(samples_cases, 0.025, axis=0),
            "lower_90": np.quantile(samples_cases, 0.05, axis=0),
            "lower_80": np.quantile(samples_cases, 0.10, axis=0),
            "lower_50": np.quantile(samples_cases, 0.25, axis=0),
            "pred": np.quantile(samples_cases, 0.50, axis=0),
            "upper_50": np.quantile(samples_cases, 0.75, axis=0),
            "upper_80": np.quantile(samples_cases, 0.90, axis=0),
            "upper_90": np.quantile(samples_cases, 0.95, axis=0),
            "upper_95": np.quantile(samples_cases, 0.975, axis=0),
        }

        pred_cases = quantiles["pred"]
        
        if use_incidence:
            for j, row_idx in enumerate(row_indices):
                state_pop = panel.loc[row_idx, "population"]  # already in panel
                for q in quantiles:
                    quantiles[q] = quantiles[q] * state_pop / 100_000


        for j, row_idx in enumerate(row_indices):
            scaled_pred_cases = (pred_cases[j] - cases_mean) / cases_scale
            panel.loc[row_idx, "cases"] = scaled_pred_cases

            state, date = meta_batch[j]

            if (state, date) in target_pairs:
                records.append({
                    "state": state,
                    "date": date,
                    "pred": float(quantiles["pred"][j]),
                    "lower_50": float(quantiles["lower_50"][j]),
                    "lower_80": float(quantiles["lower_80"][j]),
                    "lower_90": float(quantiles["lower_90"][j]),
                    "lower_95": float(quantiles["lower_95"][j]),
                    "upper_50": float(quantiles["upper_50"][j]),
                    "upper_80": float(quantiles["upper_80"][j]),
                    "upper_90": float(quantiles["upper_90"][j]),
                    "upper_95": float(quantiles["upper_95"][j]),
                })

    pred_df = pd.DataFrame(records)

    submission_cols = [
        "state",
        "date",
        "pred",
        "lower_50",
        "lower_80",
        "lower_90",
        "lower_95",
        "upper_50",
        "upper_80",
        "upper_90",
        "upper_95",
    ]

    submission = target_df[["state", "date"]].merge(
        pred_df,
        on=["state", "date"],
        how="left",
    )

    missing = submission["pred"].isna().sum()

    if missing > 0:
        missing_rows = submission[submission["pred"].isna()].head()
        raise ValueError(
            f"Missing {missing} target predictions. Examples:\n{missing_rows}"
        )

    submission = submission[submission_cols]

    numeric_cols = [c for c in submission_cols if c not in ["state", "date"]]
    submission[numeric_cols] = submission[numeric_cols].clip(lower=0.0)

    submission["lower_95"] = np.minimum(submission["lower_95"], submission["lower_90"])
    submission["lower_90"] = np.minimum(submission["lower_90"], submission["lower_80"])
    submission["lower_80"] = np.minimum(submission["lower_80"], submission["lower_50"])
    submission["lower_50"] = np.minimum(submission["lower_50"], submission["pred"])

    submission["upper_50"] = np.maximum(submission["upper_50"], submission["pred"])
    submission["upper_80"] = np.maximum(submission["upper_80"], submission["upper_50"])
    submission["upper_90"] = np.maximum(submission["upper_90"], submission["upper_80"])
    submission["upper_95"] = np.maximum(submission["upper_95"], submission["upper_90"])

    return submission


# Submission helpers

def save_submission_for_split(
    submission: pd.DataFrame,
    split_id: int,
    target_start,
    model_name: str,
) -> Path:
    out_dir = SUBMISSIONS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    file_year = pd.Timestamp(target_start).year
    out_path = out_dir / f"test_{split_id}_{file_year}.csv"

    submission.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Shape:", submission.shape)

    return out_path


def validate_submission_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required_cols = [
        "state",
        "date",
        "pred",
        "lower_50",
        "lower_80",
        "lower_90",
        "lower_95",
        "upper_50",
        "upper_80",
        "upper_90",
        "upper_95",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"{path.name}: missing columns {missing_cols}")

    if df.duplicated(subset=["state", "date"]).any():
        raise ValueError(f"{path.name}: duplicated state/date rows")

    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [c for c in required_cols if c not in ["state", "date"]]

    if df[numeric_cols].isna().any().any():
        raise ValueError(f"{path.name}: missing numeric predictions")

    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError(f"{path.name}: non-finite numeric predictions")

    if (df[numeric_cols] < 0).any().any():
        raise ValueError(f"{path.name}: negative predictions or intervals")

    checks = {
        "lower_95 <= lower_90": (df["lower_95"] <= df["lower_90"]).all(),
        "lower_90 <= lower_80": (df["lower_90"] <= df["lower_80"]).all(),
        "lower_80 <= lower_50": (df["lower_80"] <= df["lower_50"]).all(),
        "lower_50 <= pred": (df["lower_50"] <= df["pred"]).all(),
        "pred <= upper_50": (df["pred"] <= df["upper_50"]).all(),
        "upper_50 <= upper_80": (df["upper_50"] <= df["upper_80"]).all(),
        "upper_80 <= upper_90": (df["upper_80"] <= df["upper_90"]).all(),
        "upper_90 <= upper_95": (df["upper_90"] <= df["upper_95"]).all(),
    }

    failed = [name for name, ok in checks.items() if not ok]

    if failed:
        raise ValueError(f"{path.name}: interval ordering failed: {failed}")

    print(
        f"OK: {path.name} | shape={df.shape} | "
        f"dates={df['date'].min().date()} → {df['date'].max().date()}"
    )

    return df


def expected_csv_count(model_dir: Path) -> int:
    files = {p.name for p in model_dir.glob("*.csv")}
    return len(EXPECTED_FILES & files)


def quarantine_incomplete_submission_folders(config: ExperimentConfig) -> None:
    allowed_current_models = expected_model_names(config)

    quarantine_dir = LOG_DIR / "incomplete_submission_folders"
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    moved = []

    for model_dir in sorted(SUBMISSIONS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        n_files = expected_csv_count(model_dir)

        if n_files == 3:
            continue

        if model_dir.name in allowed_current_models:
            continue

        target = quarantine_dir / model_dir.name

        if target.exists():
            shutil.rmtree(target)

        shutil.move(str(model_dir), str(target))
        moved.append((model_dir.name, n_files, str(target)))

    if moved:
        print("Moved incomplete submission folders:")
        for name, n_files, target in moved:
            print(f"  {name}: {n_files}/3 files -> {target}")
    else:
        print("No unrelated incomplete submission folders found.")


def audit_submission_folders() -> pd.DataFrame:
    rows = []

    for model_dir in sorted(SUBMISSIONS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        files = {p.name for p in model_dir.glob("*.csv")}
        missing = sorted(EXPECTED_FILES - files)

        rows.append({
            "model": model_dir.name,
            "n_expected_files": len(EXPECTED_FILES & files),
            "missing": ", ".join(missing),
        })

    audit = pd.DataFrame(rows).sort_values(["n_expected_files", "model"])
    print(audit.to_string(index=False))

    return audit


# Evaluator and results

def run_official_evaluator() -> Path:
    result = subprocess.run(
        [sys.executable, "scripts/evaluator.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    print("STDOUT")
    print("=" * 80)
    print(result.stdout)

    if result.stderr:
        print("STDERR")
        print("=" * 80)
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Evaluator failed.")

    results_path = REPO_ROOT / "results.csv"

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    return results_path


def summarize_results(results_path: Path, config: ExperimentConfig) -> pd.DataFrame:
    results = pd.read_csv(results_path)

    if "rank" not in results.columns:
        results = results.sort_values("overall_wis").reset_index(drop=True)
        results.insert(0, "rank", range(1, len(results) + 1))

    experiment_results = results[
        results["model"].str.startswith(config.model_family + "_", na=False)
    ].copy()

    print("\nALL MODELS")
    print(
        results[
            [
                c for c in [
                    "rank",
                    "model",
                    "overall_wis",
                    "overall_crps",
                    "overall_mae",
                    "overall_rmse",
                    "overall_mape",
                ]
                if c in results.columns
            ]
        ]
        .sort_values("rank")
        .to_string(index=False)
    )

    print("\nEXPERIMENT MODELS")
    print(
        experiment_results[
            [
                c for c in [
                    "rank",
                    "model",
                    "overall_wis",
                    "overall_crps",
                    "overall_mae",
                    "overall_rmse",
                    "overall_mape",
                ]
                if c in experiment_results.columns
            ]
        ]
        .sort_values("overall_wis")
        .to_string(index=False)
    )

    if not experiment_results.empty:
        best = experiment_results.sort_values("overall_wis").iloc[0]
        print("\nBest experiment model")
        print(best.to_string())

    out_dir = LOG_DIR / config.model_family
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "all_results_after_evaluator.csv", index=False)
    experiment_results.to_csv(out_dir / "experiment_results.csv", index=False)

    return experiment_results


# Main experiment runner

def architecture_table(config: ExperimentConfig, artifacts) -> pd.DataFrame:
    rows = []

    for arch_name, arch_config in config.architectures.items():
        model = build_model_from_config(artifacts, arch_config)

        rows.append({
            "architecture": arch_name,
            "n_features": artifacts["X_train"].shape[2],
            "n_states": artifacts["n_states"],
            "trainable_params": count_trainable_params(model),
            **arch_config.__dict__,
        })

        del model
        clear_memory()

    return pd.DataFrame(rows).sort_values("trainable_params")


def run_experiment(config: ExperimentConfig, run_evaluator: bool = True):
    print("Device:", DEVICE)
    print("Repo root:", REPO_ROOT)
    print("Full dataset:", FULL_DATASET)
    print("Model family:", config.model_family)

    if config.quarantine_incomplete:
        quarantine_incomplete_submission_folders(config)

    records = []
    histories = []

    for split_id in config.splits:
        print("\n" + "=" * 90)
        print(f"RUNNING SPLIT {split_id}")
        print("=" * 90)

        artifacts = prepare_geo_split(
            split_id=split_id,
            lookback=LOOKBACK,
            batch_size=BATCH_SIZE,
        )

        if split_id == config.splits[0]:
            print("\nArchitecture table")
            print(architecture_table(config, artifacts).to_string(index=False))

        for arch_name, arch_config in config.architectures.items():
            missing_sigmas = []

            for sigma_scale in config.sigma_scales:
                model_name = model_name_for(config, arch_name, sigma_scale)

                path = submission_path(
                    model_name=model_name,
                    split_id=split_id,
                    target_start=artifacts["target_start"],
                )

                if path.exists() and not config.force_rerun:
                    print(f"Reusing existing submission: {path}")
                    validate_submission_file(path)

                    records.append({
                        "split_id": split_id,
                        "architecture": arch_name,
                        "sigma_scale": sigma_scale,
                        "model": model_name,
                        "path": str(path),
                        "status": "reused",
                        **arch_config.__dict__,
                    })
                else:
                    missing_sigmas.append(sigma_scale)

            if not missing_sigmas:
                continue

            print("\n" + "-" * 90)
            print(f"Training | split={split_id} | arch={arch_name}")
            print(f"Missing sigma scales: {missing_sigmas}")
            print("-" * 90)

            model, history_df, best_val_loss = train_model_with_config(
                artifacts=artifacts,
                arch_name=arch_name,
                arch_config=arch_config,
                lr=config.lr,
                epochs=config.epochs,
                patience=config.patience,
                weight_decay=WEIGHT_DECAY,
                beta=BETA,
                seed=config.seed + split_id,
            )

            history_df["model_family"] = config.model_family
            histories.append(history_df)

            for sigma_scale in missing_sigmas:
                model_name = model_name_for(config, arch_name, sigma_scale)

                print(
                    f"\nForecasting | split={split_id} | "
                    f"arch={arch_name} | sigma={sigma_scale}"
                )

                submission = recursive_forecast_geo_split(
                    model=model,
                    artifacts=artifacts,
                    lookback=LOOKBACK,
                    mc_samples=config.mc_samples,
                    sigma_scale=sigma_scale,
                )

                out_path = save_submission_for_split(
                    submission=submission,
                    split_id=split_id,
                    target_start=artifacts["target_start"],
                    model_name=model_name,
                )

                validate_submission_file(out_path)

                records.append({
                    "split_id": split_id,
                    "architecture": arch_name,
                    "sigma_scale": sigma_scale,
                    "model": model_name,
                    "best_val_loss": best_val_loss,
                    "path": str(out_path),
                    "status": "created",
                    **arch_config.__dict__,
                })

            del model
            clear_memory()

    out_dir = LOG_DIR / config.model_family
    out_dir.mkdir(parents=True, exist_ok=True)

    run_log = pd.DataFrame(records)
    run_log.to_csv(out_dir / "full_run_log.csv", index=False)

    print("\nRun log")
    print(run_log.to_string(index=False))

    if histories:
        history_all = pd.concat(histories, ignore_index=True)
        history_all.to_csv(out_dir / "training_history.csv", index=False)

    print("\nSubmission audit before evaluator")
    audit_submission_folders()

    if run_evaluator:
        results_path = run_official_evaluator()
        summarize_results(results_path, config)

    return run_log


# CLI

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        default="small_ablation",
        choices=sorted(EXPERIMENTS.keys()),
        help="Experiment config to run.",
    )

    parser.add_argument(
        "--no-evaluator",
        action="store_true",
        help="Train and save submissions, but do not run scripts/evaluator.py.",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Overwrite existing submissions for this experiment.",
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=None,
        help="Override MC samples.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )

    return parser.parse_args()


def override_config(config: ExperimentConfig, args) -> ExperimentConfig:
    values = config.__dict__.copy()

    if args.force_rerun:
        values["force_rerun"] = True

    if args.mc_samples is not None:
        values["mc_samples"] = args.mc_samples

    if args.epochs is not None:
        values["epochs"] = args.epochs

    if args.patience is not None:
        values["patience"] = args.patience

    return ExperimentConfig(**values)


def main():
    args = parse_args()
    config = override_config(EXPERIMENTS[args.experiment], args)

    run_experiment(
        config=config,
        run_evaluator=not args.no_evaluator,
    )


if __name__ == "__main__":
    main()