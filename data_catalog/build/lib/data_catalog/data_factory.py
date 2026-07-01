from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DF = pd.read_csv(ROOT / "data_catalog" / "full_dataset.csv")


def data_factory(horizon: int):
    if horizon not in {1, 2, 3, 4}: raise ValueError("horizon must be 1, 2, 3, or 4, see documentation")

    train_col = f"train_{horizon}"
    target_col = f"target_{horizon}"

    train = DF[DF[train_col] == True].copy()
    test = DF[DF[target_col] == True].copy()

    drop_cols = [c for c in DF.columns if c.startswith("train_") or c.startswith("target_")]

    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    return train, test
