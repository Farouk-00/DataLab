"""The evaluation protocol shared by every experiment.

A Split is one evaluation situation: the matrix a model may see, the cells it
is scored on, and their true ratings (the targets). The validation folds and
the final test evaluation are both Splits, scored by the same evaluate().

Every result depends on this file. Changing its behaviour means bumping
PROTOCOL_VERSION, which every recorded result carries, and re-running
everything.
"""

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cf.data import RATING_VALUES

PROTOCOL_VERSION = 1

# Every rating of ratings_train is validated exactly once, and each fold's model
# sees 4/5 of train. More folds would cost proportionally more without closing
# either real gap: the platform's denser input and single-rating movies, which
# are cold in their fold whatever the number of folds.
N_FOLDS = 5

# The value is arbitrary; what matters is that it is fixed, so that every model
# is scored on the same folds and comparisons between models are paired.
FOLD_SEED = 0


class Model(Protocol):
    """Anything with this method is a model; no inheritance needed."""

    def complete(self, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Predict every cell of R.

        R is users × items with NaN for unobserved cells, and is read-only.
        Returns an array of R's shape holding only finite values. Users or items
        with no visible rating must still get a prediction (a fallback).
        """
        ...


@dataclass(frozen=True)
class Split:
    """One evaluation situation. All arrays are read-only."""

    visible: np.ndarray  # users × items: what the model sees, NaN = hidden or unobserved
    users: np.ndarray    # row index of each scored cell
    items: np.ndarray    # column index of each scored cell
    targets: np.ndarray  # true rating of each scored cell, hidden from the model


def _make_split(visible: np.ndarray, users: np.ndarray, items: np.ndarray, targets: np.ndarray) -> Split:
    # Read-only because one Split is evaluated by many models: an in-place
    # write by one of them would change what the next one sees.
    for array in (visible, users, items, targets):
        array.setflags(write=False)
    return Split(visible, users, items, targets)


def kfold_splits(R_train: np.ndarray) -> list[Split]:
    """The N_FOLDS validation splits of R_train.

    The observed cells are shuffled with FOLD_SEED and cut into N_FOLDS pieces;
    fold f hides piece f and is scored on it. Cells are split, not users or
    items, because the instructor's train/test/eval split is a split of cells.
    """
    users, items = np.nonzero(~np.isnan(R_train))
    order = np.random.default_rng(FOLD_SEED).permutation(users.size)
    splits = []
    for fold in np.array_split(order, N_FOLDS):
        u, i = users[fold], items[fold]
        visible = R_train.copy()
        visible[u, i] = np.nan
        splits.append(_make_split(visible, u, i, R_train[u, i]))
    return splits


def final_split(R_train: np.ndarray, R_test: np.ndarray) -> Split:
    """The final evaluation: all of R_train visible, scored on R_test's cells."""
    if (~np.isnan(R_train) & ~np.isnan(R_test)).any():
        raise ValueError("train and test share observed cells")
    users, items = np.nonzero(~np.isnan(R_test))
    return _make_split(R_train.copy(), users, items, R_test[users, items])


def postprocess(R_hat: np.ndarray) -> np.ndarray:
    """Clip predictions to the rating scale.

    A prediction outside the scale is always further from the target than the
    nearest bound, so clipping never increases the error. generate.py applies
    this same function, so local scores describe the submitted matrix.
    """
    return np.clip(R_hat, RATING_VALUES[0], RATING_VALUES[-1])


def to_scale(pred: np.ndarray) -> np.ndarray:
    """Round predictions to the nearest half star, ties rounding up.

    np.round would round ties to even instead. The platform's rule is not
    confirmed yet; this function is the single place to change once it is.
    """
    return np.clip(np.floor(pred * 2 + 0.5) / 2, RATING_VALUES[0], RATING_VALUES[-1])


def rmse(pred: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - targets) ** 2)))


def accuracy(pred: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of cells whose rounded prediction equals the true rating."""
    return float(np.mean(to_scale(pred) == targets))


@dataclass(frozen=True)
class Evaluation:
    """One model scored on one split."""

    metrics: dict[str, float]  # rmse, accuracy, seconds
    predictions: np.ndarray    # clipped prediction of each scored cell, aligned with split.targets


@dataclass(frozen=True)
class CrossValidation:
    """One model scored on the N_FOLDS folds of R_train."""

    folds: list[dict[str, float]]  # metrics of each fold, in fold order
    oof: np.ndarray                # users × items: each observed cell of R_train, predicted while
                                   # it was hidden; NaN elsewhere. The input of a blend.

    def summary(self) -> dict[str, float]:
        """Mean over folds of each metric, plus its spread as <metric>_std.

        The spread is not a confidence interval: the folds share most of their
        training cells, so their scores are correlated.
        """
        summary = {}
        for metric in self.folds[0]:
            values = np.array([fold[metric] for fold in self.folds])
            summary[metric] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
        return summary


def evaluate(model: Model, split: Split, seed: int) -> Evaluation:
    """Run one model on one split: RMSE, exact-rating accuracy, fit time.

    The time covers complete() only, the part the platform also measures.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    R_hat = model.complete(split.visible, rng)
    seconds = time.perf_counter() - start
    if R_hat.shape != split.visible.shape:
        raise ValueError(f"{model!r} returned shape {R_hat.shape}, expected {split.visible.shape}")
    if not np.isfinite(R_hat).all():
        raise ValueError(f"{model!r} returned NaN or infinite predictions")
    pred = postprocess(R_hat)[split.users, split.items]
    pred.setflags(write=False)
    metrics = {"rmse": rmse(pred, split.targets), "accuracy": accuracy(pred, split.targets), "seconds": seconds}
    return Evaluation(metrics, pred)


def cross_validate(model: Model, R_train: np.ndarray, seed: int) -> CrossValidation:
    """evaluate() on each of the N_FOLDS folds of R_train.

    Every observed cell is hidden in exactly one fold, so the out-of-fold
    predictions fill every observed cell of R_train once.
    """
    oof = np.full(R_train.shape, np.nan)
    folds = []
    for split in kfold_splits(R_train):
        result = evaluate(model, split, seed)
        oof[split.users, split.items] = result.predictions
        folds.append(result.metrics)
    oof.setflags(write=False)
    return CrossValidation(folds, oof)
