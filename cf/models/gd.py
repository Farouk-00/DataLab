"""Matrix factorisation with biases, trained by mini-batch gradient descent.

    r̂(u, i) = mu + b_u + b_i + p_u · q_i

    L = Σ_{(u,i) observed} (r − r̂)²  +  lam (‖P‖² + ‖Q‖²)  +  bias_reg (‖b_user‖² + ‖b_item‖²)

mu is the mean of the visible ratings and stays fixed; b, P and Q are learned.

The penalty is the plain one of the loss above, the same as in ALS, so the
values of lam are comparable between the two methods. To keep it so with
mini-batches, each rating of user u carries lam / n_u of the penalty on p_u:
over one epoch, user u receives exactly lam · p_u.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Factors:
    """The learned parameters, and the RMSE after each epoch."""

    mu: float
    b_user: np.ndarray  # (n_users,)
    b_item: np.ndarray  # (n_items,)
    P: np.ndarray       # (n_users, k)
    Q: np.ndarray       # (n_items, k)
    history: dict[str, list[float]] = field(default_factory=dict)  # "train", and "valid" if monitored

    def predict(self) -> np.ndarray:
        return self.mu + self.b_user[:, None] + self.b_item[None, :] + self.P @ self.Q.T


@dataclass(frozen=True)
class GD:
    k: int = 10
    lam: float = 10.0
    lr: float = 0.01
    n_epochs: int = 30
    bias_reg: Optional[float] = None  # None: same as lam
    batch_size: int = 256
    init_std: float = 0.1

    def complete(self, R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.fit(R, rng).predict()

    def fit(self, R: np.ndarray, rng: np.random.Generator, monitor=None) -> Factors:
        """Learn the factors on the visible cells of R.

        monitor: optional (users, items, targets) of held-out cells, whose RMSE
        is recorded after each epoch in history["valid"]. It never changes the fit.
        """
        n_users, n_items = R.shape
        users, items = np.nonzero(~np.isnan(R))
        ratings = R[users, items]
        mu = float(ratings.mean())

        # Ratings per user and per item, to spread the penalty over them.
        n_u = np.bincount(users, minlength=n_users)
        n_i = np.bincount(items, minlength=n_items)
        inv_n_u = 1.0 / np.maximum(n_u, 1)
        inv_n_i = 1.0 / np.maximum(n_i, 1)
        bias_reg = self.lam if self.bias_reg is None else self.bias_reg

        b_user = np.zeros(n_users)
        b_item = np.zeros(n_items)
        P = rng.normal(0.0, self.init_std, (n_users, self.k))
        Q = rng.normal(0.0, self.init_std, (n_items, self.k))

        history: dict[str, list[float]] = {"train": []}
        if monitor is not None:
            history["valid"] = []

        for _ in range(self.n_epochs):
            order = rng.permutation(users.size)
            for start in range(0, order.size, self.batch_size):
                batch = order[start:start + self.batch_size]
                u, i, r = users[batch], items[batch], ratings[batch]
                p, q = P[u], Q[i]

                err = r - (mu + b_user[u] + b_item[i] + np.sum(p * q, axis=1))

                # Gradient step on −L/2. np.add.at sums the contributions of a
                # user (or item) that appears several times in the batch.
                np.add.at(b_user, u, self.lr * (err - bias_reg * inv_n_u[u] * b_user[u]))
                np.add.at(b_item, i, self.lr * (err - bias_reg * inv_n_i[i] * b_item[i]))
                np.add.at(P, u, self.lr * (err[:, None] * q - self.lam * inv_n_u[u, None] * p))
                np.add.at(Q, i, self.lr * (err[:, None] * p - self.lam * inv_n_i[i, None] * q))

            pred = mu + b_user[users] + b_item[items] + np.sum(P[users] * Q[items], axis=1)
            history["train"].append(float(np.sqrt(np.mean((pred - ratings) ** 2))))
            if monitor is not None:
                vu, vi, target = monitor
                pred = mu + b_user[vu] + b_item[vi] + np.sum(P[vu] * Q[vi], axis=1)
                pred = np.clip(pred, 0.5, 5.0)
                history["valid"].append(float(np.sqrt(np.mean((pred - target) ** 2))))

        # Fallback: a user or item with no visible rating keeps b = 0, and its
        # random initial factors carry no information, so they are zeroed.
        P[n_u == 0] = 0.0
        Q[n_i == 0] = 0.0
        return Factors(mu, b_user, b_item, P, Q, history)
