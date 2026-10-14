"""Matrix factorisation with biases, trained by full-batch gradient descent with AdamW.

    r̂(u, i) = mu + b_u + b_i + p_u · q_i

    L = Σ_{(u,i) ∈ Ω} (r − r̂)²  +  lam (‖P‖² + ‖Q‖²)  +  bias_reg (‖b_user‖² + ‖b_item‖²)

Ω is the set of visible ratings. This is the loss of ALS, so lam means the same
in both methods.

mu is the mean of the visible ratings and stays fixed; b, P and Q are learned.

Each step uses the exact gradient over all visible ratings (one step = one epoch).
AdamW scales the step of each parameter by the running size of its gradient,
so a user with 1 000 ratings and a user with 1 move at comparable speeds.
AdamW's decoupled weight decay (weight_decay, on P and Q) is kept as an extra
knob, but it cannot replace lam: it shrinks a movie rated once as much as one
rated 300 times, whereas the penalty in L weighs against fewer ratings for the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


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


class _AdamW:
    """AdamW on one parameter array, updated in place (Loshchilov & Hutter, 2019)."""

    def __init__(self, param: np.ndarray, lr: float, weight_decay: float, betas=(0.9, 0.999), eps=1e-8):
        self.param, self.lr, self.weight_decay = param, lr, weight_decay
        self.beta1, self.beta2, self.eps = betas[0], betas[1], eps
        self.m = np.zeros_like(param)  # running mean of the gradient
        self.v = np.zeros_like(param)  # running mean of the squared gradient
        self.t = 0

    def step(self, grad: np.ndarray) -> None:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.param -= self.lr * self.weight_decay * self.param  # decoupled weight decay
        self.param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass(frozen=True)
class GD:
    k: int = 10
    lam: float = 10.0
    bias_reg: float = 3.0
    lr: float = 0.01
    n_epochs: int = 200
    weight_decay: float = 0.0  # AdamW's decoupled decay, on P and Q only
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

        b_user = np.zeros(n_users)
        b_item = np.zeros(n_items)
        P = rng.normal(0.0, self.init_std, (n_users, self.k))
        Q = rng.normal(0.0, self.init_std, (n_items, self.k))
        optimizers = [
            _AdamW(b_user, self.lr, 0.0),
            _AdamW(b_item, self.lr, 0.0),
            _AdamW(P, self.lr, self.weight_decay),
            _AdamW(Q, self.lr, self.weight_decay),
        ]

        # The errors as a sparse users × items matrix: its products with Q and P
        # sum the gradient contributions of each user's and each item's ratings.
        E = csr_matrix((np.zeros(ratings.size), (users, items)), shape=(n_users, n_items))
        order = np.lexsort((items, users))  # csr_matrix stores the cells in this order

        history: dict[str, list[float]] = {"train": []}
        if monitor is not None:
            history["valid"] = []

        for _ in range(self.n_epochs):
            err = ratings - (mu + b_user[users] + b_item[items] + np.sum(P[users] * Q[items], axis=1))
            E.data[:] = err[order]

            # Gradients of L, divided by 2.
            grads = [
                -np.asarray(E.sum(axis=1)).ravel() + self.bias_reg * b_user,
                -np.asarray(E.sum(axis=0)).ravel() + self.bias_reg * b_item,
                -(E @ Q) + self.lam * P,
                -(E.T @ P) + self.lam * Q,
            ]
            for optimizer, grad in zip(optimizers, grads):
                optimizer.step(grad)

            pred = mu + b_user[users] + b_item[items] + np.sum(P[users] * Q[items], axis=1)
            history["train"].append(float(np.sqrt(np.mean((pred - ratings) ** 2))))
            if monitor is not None:
                vu, vi, target = monitor
                pred = mu + b_user[vu] + b_item[vi] + np.sum(P[vu] * Q[vi], axis=1)
                pred = np.clip(pred, 0.5, 5.0)
                history["valid"].append(float(np.sqrt(np.mean((pred - target) ** 2))))

        # Fallback: a user or item with no visible rating keeps b = 0, and its
        # random initial factors carry no information, so they are zeroed.
        P[np.bincount(users, minlength=n_users) == 0] = 0.0
        Q[np.bincount(items, minlength=n_items) == 0] = 0.0
        return Factors(mu, b_user, b_item, P, Q, history)
