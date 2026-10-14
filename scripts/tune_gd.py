"""Tune GD on the 5 folds: validation RMSE after each epoch, for a grid of configurations.

Training for n epochs is a prefix of training for more (same rng draws), so
one run of max_epochs gives the 5-fold RMSE at every number of epochs.
Run from the repository root: python3 -m scripts.tune_gd
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from cf.data import load_ratings
from cf.models.gd import GD
from cf.protocol import PROTOCOL_VERSION, kfold_splits

SEED = 0
OUT = Path("outputs/gd")


def curves(model: GD, splits) -> dict:
    """Mean over folds of the train and validation RMSE after each epoch, and the fit time."""
    train, valid, seconds = [], [], []
    for split in splits:
        start = time.perf_counter()
        f = model.fit(split.visible, np.random.default_rng(SEED), monitor=(split.users, split.items, split.targets))
        seconds.append(time.perf_counter() - start)
        train.append(f.history["train"])
        valid.append(f.history["valid"])
    valid = np.array(valid)
    return {
        "train": np.mean(train, axis=0).tolist(),
        "valid": valid.mean(axis=0).tolist(),
        "valid_std": valid.std(axis=0).tolist(),
        "seconds": float(np.mean(seconds)),
    }


def main(grid_name: str) -> None:
    grids = {
        "lr": dict(k=[10], lam=[10.0], lr=[0.002, 0.005, 0.01, 0.02, 0.05], n_epochs=[100]),
        "lam": dict(k=[10], lam=[0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0], lr=[0.02], n_epochs=[100]),
        "k": dict(k=[2, 5, 10, 20, 50], lam=[10.0], lr=[0.02], n_epochs=[100]),
        "joint": dict(k=[10], lam=[5.0, 10.0, 15.0, 20.0, 30.0], lr=[0.02], n_epochs=[100], bias_reg=[2.0, 3.0, 4.0]),
        "lam3": dict(k=[10], lam=[0.1, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0], lr=[0.02], n_epochs=[100], bias_reg=[3.0]),
        "bias": dict(k=[10], lam=[10.0], lr=[0.02], n_epochs=[100], bias_reg=[1.0, 3.0, 5.0, 10.0]),
    }
    splits = kfold_splits(load_ratings("ratings_train.npy"))
    OUT.mkdir(parents=True, exist_ok=True)
    grid = grids[grid_name]
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        c = curves(GD(**params), splits)
        best = int(np.argmin(c["valid"]))
        print(f"{params}  best epoch {best + 1}: {c['valid'][best]:.4f} ± {c['valid_std'][best]:.4f}"
              f"  last: {c['valid'][-1]:.4f}  {c['seconds']:.1f} s", flush=True)
        results.append({"params": params, **c, "protocol_version": PROTOCOL_VERSION, "seed": SEED})
    (OUT / f"grid_{grid_name}.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main(sys.argv[1])
