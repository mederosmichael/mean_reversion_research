# hmm.py
# minimal hidden markov model (gaussian emissions) with baum-welch + viterbi
# dependencies: numpy only

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)


def _row_normalize(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return x / s


def _col_normalize(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    s = x.sum(axis=0, keepdims=True)
    s = np.maximum(s, eps)
    return x / s


@dataclass
class GaussianHMM:
    # hmm with diagonal-cov gaussian emissions
    n_states: int
    n_features: int
    seed: int = 0
    min_var: float = 1e-6

    # parameters
    pi: Optional[np.ndarray] = None          # (K,)
    A: Optional[np.ndarray] = None           # (K,K)
    mu: Optional[np.ndarray] = None          # (K,D)
    var: Optional[np.ndarray] = None         # (K,D)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._init_params()

    def _init_params(self) -> None:
        K, D = self.n_states, self.n_features
        pi = self.rng.random(K) + 1e-2
        pi = pi / pi.sum()

        A = self.rng.random((K, K)) + 1e-2
        A = _row_normalize(A)

        # mu, var initialized later from data in fit if not already set
        self.pi = pi
        self.A = A
        self.mu = None
        self.var = None

    def _ensure_emissions_initialized(self, X: np.ndarray) -> None:
        if self.mu is not None and self.var is not None:
            return
        K, D = self.n_states, self.n_features
        if X.ndim == 1:
            X = X[:, None]
        assert X.shape[1] == D, "X has wrong feature dimension"
        # pick random points as initial means
        idx = self.rng.choice(X.shape[0], size=K, replace=False) if X.shape[0] >= K else self.rng.integers(0, X.shape[0], size=K)
        mu = X[idx].copy()
        # global variance as starting point
        gv = np.var(X, axis=0, ddof=0) + self.min_var
        var = np.tile(gv[None, :], (K, 1))
        self.mu = mu
        self.var = var

    def _log_gaussian(self, X: np.ndarray) -> np.ndarray:
        # returns log p(x_t | z_t=k) as (T,K)
        if X.ndim == 1:
            X = X[:, None]
        T, D = X.shape
        K = self.n_states
        mu = self.mu  # (K,D)
        var = self.var  # (K,D)
        assert mu is not None and var is not None
        var = np.maximum(var, self.min_var)

        # diagonal gaussian logpdf
        # log N(x|mu,var) = -0.5*(sum(log(2pi var)) + sum((x-mu)^2/var))
        log_det = np.sum(np.log(2.0 * np.pi * var), axis=1)  # (K,)
        # compute squared mahalanobis for each k
        # (T,1,D) - (1,K,D) -> (T,K,D)
        diff = X[:, None, :] - mu[None, :, :]
        maha = np.sum((diff * diff) / var[None, :, :], axis=2)  # (T,K)
        return -0.5 * (log_det[None, :] + maha)

    def _forward_backward(self, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        # returns (loglik, gamma(T,K), xi(T-1,K,K), logB(T,K))
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        K = self.n_states

        log_pi = np.log(np.maximum(self.pi, 1e-15))          # (K,)
        log_A = np.log(np.maximum(self.A, 1e-15))            # (K,K)
        logB = self._log_gaussian(X)                         # (T,K)

        # forward in log space
        log_alpha = np.empty((T, K), dtype=float)
        log_alpha[0] = log_pi + logB[0]
        for t in range(1, T):
            # log_alpha[t,k] = logB[t,k] + logsumexp_j(log_alpha[t-1,j] + log_A[j,k])
            log_alpha[t] = logB[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)

        loglik = float(_logsumexp(log_alpha[T - 1], axis=0))

        # backward in log space
        log_beta = np.empty((T, K), dtype=float)
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            # log_beta[t,i] = logsumexp_k( log_A[i,k] + logB[t+1,k] + log_beta[t+1,k] )
            log_beta[t] = _logsumexp(log_A + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)

        # gamma
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        # xi
        xi = np.empty((T - 1, K, K), dtype=float)
        for t in range(T - 1):
            # log_xi[i,k] ∝ log_alpha[t,i] + log_A[i,k] + logB[t+1,k] + log_beta[t+1,k]
            log_xi = (log_alpha[t][:, None] + log_A +
                      (logB[t + 1] + log_beta[t + 1])[None, :])
            log_xi = log_xi - _logsumexp(log_xi, axis=None)
            xi[t] = np.exp(log_xi)

        return loglik, gamma, xi, logB

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False
    ) -> "GaussianHMM":
        if X.ndim == 1:
            X = X[:, None]
        T, D = X.shape
        assert D == self.n_features, "X has wrong feature dimension"

        self._ensure_emissions_initialized(X)

        prev = -np.inf
        for it in range(n_iter):
            ll, gamma, xi, _ = self._forward_backward(X)

            # m-step
            self.pi = gamma[0].copy()

            # transition matrix
            A_num = xi.sum(axis=0)                    # (K,K)
            A_den = gamma[:-1].sum(axis=0)[:, None]   # (K,1)
            self.A = A_num / np.maximum(A_den, 1e-15)

            # gaussian params
            # weights: gamma[t,k]
            w = gamma.sum(axis=0)  # (K,)
            w = np.maximum(w, 1e-15)

            mu = (gamma.T @ X) / w[:, None]  # (K,D)

            # var_kd = sum_t gamma[t,k] (x_td - mu_kd)^2 / w_k
            diff = X[:, None, :] - mu[None, :, :]          # (T,K,D)
            var = (gamma[:, :, None] * (diff * diff)).sum(axis=0) / w[:, None]
            var = np.maximum(var, self.min_var)

            self.mu = mu
            self.var = var

            if verbose:
                print(f"iter={it:03d} loglik={ll:.6f}")

            if ll - prev < tol:
                break
            prev = ll

        return self

    def score(self, X: np.ndarray) -> float:
        if X.ndim == 1:
            X = X[:, None]
        self._ensure_emissions_initialized(X)
        ll, _, _, _ = self._forward_backward(X)
        return ll

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        self._ensure_emissions_initialized(X)
        _, gamma, _, _ = self._forward_backward(X)
        return gamma  # (T,K)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        self._ensure_emissions_initialized(X)
        T = X.shape[0]
        K = self.n_states

        log_pi = np.log(np.maximum(self.pi, 1e-15))
        log_A = np.log(np.maximum(self.A, 1e-15))
        logB = self._log_gaussian(X)

        dp = np.empty((T, K), dtype=float)
        ptr = np.empty((T, K), dtype=int)

        dp[0] = log_pi + logB[0]
        ptr[0] = -1

        for t in range(1, T):
            # dp[t,k] = logB[t,k] + max_j(dp[t-1,j] + log_A[j,k])
            scores = dp[t - 1][:, None] + log_A  # (K,K)
            ptr[t] = np.argmax(scores, axis=0)
            dp[t] = logB[t] + np.max(scores, axis=0)

        states = np.empty(T, dtype=int)
        states[T - 1] = int(np.argmax(dp[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = ptr[t + 1, states[t + 1]]
        return states

    def sample(self, n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        # returns (X, z)
        assert self.mu is not None and self.var is not None
        K, D = self.n_states, self.n_features
        z = np.empty(n, dtype=int)
        X = np.empty((n, D), dtype=float)

        z[0] = int(self.rng.choice(K, p=self.pi))
        X[0] = self.rng.normal(self.mu[z[0]], np.sqrt(self.var[z[0]]), size=D)

        for t in range(1, n):
            z[t] = int(self.rng.choice(K, p=self.A[z[t - 1]]))
            X[t] = self.rng.normal(self.mu[z[t]], np.sqrt(self.var[z[t]]), size=D)

        return X, z


if __name__ == "__main__":
    # quick smoke test
    rng = np.random.default_rng(1)
    true = GaussianHMM(n_states=2, n_features=1, seed=1)
    true.pi = np.array([0.7, 0.3])
    true.A = np.array([[0.95, 0.05],
                       [0.10, 0.90]])
    true.mu = np.array([[0.0],
                        [3.0]])
    true.var = np.array([[1.0],
                         [1.0]])

    X, z = true.sample(2000)

    model = GaussianHMM(n_states=2, n_features=1, seed=0)
    model.fit(X, n_iter=50, verbose=True)

    print("true mu:", true.mu.ravel(), "learned mu:", model.mu.ravel())
    print("loglik:", model.score(X))
    zhat = model.viterbi(X)
    print("viterbi states head:", zhat[:10])
