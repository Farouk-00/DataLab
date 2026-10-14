"""Figures of the GD slides and report section, from outputs/gd/grid_lam.json.

Run from the repository root: python3 -m scripts.plot_gd
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("report/generated/gd")
BIASES_CV = 0.8839  # Biases(reg=5), 5-fold, protocol v1, seed 0

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid = json.loads(Path("outputs/gd/grid_lam.json").read_text())
    by_lam = {r["params"]["lam"]: r for r in grid}
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(by_lam[10.0]["valid"]) + 1)
    for lam, color in [(1.0, "tab:red"), (10.0, "tab:blue")]:
        r = by_lam[lam]
        ax.plot(epochs, r["train"], color=color, ls="--", label=f"train, λ={lam:g}")
        ax.plot(epochs, r["valid"], color=color, label=f"validation, λ={lam:g}")
    ax.axhline(BIASES_CV, color="gray", lw=1, ls=":", label="biases only")
    ax.set(xlabel="epoch", ylabel="RMSE (5-fold mean)", ylim=(0.3, 1.15))
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "rmse_epochs.pdf")

    fig, ax = plt.subplots(figsize=(6, 4))
    lams = sorted(by_lam)
    best = [min(by_lam[l]["valid"]) for l in lams]
    last = [by_lam[l]["valid"][-1] for l in lams]
    ax.plot(lams, last, "o-", color="tab:blue", label="after 300 epochs")
    ax.plot(lams, best, "o--", color="tab:green", label="best epoch (early stopping)")
    ax.axhline(BIASES_CV, color="gray", lw=1, ls=":", label="biases only")
    ax.set(xscale="log", xlabel="λ (factors)", ylabel="validation RMSE (5 folds)", ylim=(0.875, 0.95))
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "rmse_lambda.pdf")
