"""Official 5-fold score, out-of-fold predictions and the one final_split evaluation of the chosen GD.

Run from the repository root: python3 -m scripts.final_gd
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cf.data import load_ratings
from cf.models.gd import GD
from cf.protocol import PROTOCOL_VERSION, cross_validate, evaluate, final_split

SEED = 0
OUT = Path("outputs/gd")
# Chosen on the 5 folds (scripts/tune_gd.py), never on the test set.
CHOSEN = GD(k=10, lam=10.0, bias_reg=3.0, lr=0.03, n_epochs=150, weight_decay=1.0)
ABLATION = GD(k=0, lam=10.0, bias_reg=3.0, lr=0.03, n_epochs=150, weight_decay=1.0)  # same model without p_u · q_i

if __name__ == "__main__":
    R_train, R_test = load_ratings("ratings_train.npy"), load_ratings("ratings_test.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"protocol_version": PROTOCOL_VERSION, "seed": SEED}
    for name, model in [("gd", CHOSEN), ("gd_k0", ABLATION)]:
        cv = cross_validate(model, R_train, SEED)
        record[name] = {"model": repr(model), "cv": cv.summary()}
        print(name, cv.summary())
    np.save(OUT / "oof_gd.npy", cross_validate(CHOSEN, R_train, SEED).oof)
    test = evaluate(CHOSEN, final_split(R_train, R_test), SEED).metrics
    record["gd"]["test"] = test
    print("test", test)
    (OUT / "final.json").write_text(json.dumps(record, indent=2))
