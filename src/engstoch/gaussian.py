"""Gaussian processes and random fields.

Covariance functions (exponential/Ornstein-Uhlenbeck, squared exponential, Matern 3/2 and 5/2, periodic,
Brownian motion and fractional Brownian motion); sampling by Cholesky factorisation (with the jitter a
smooth kernel needs and the conditioning numbers that explain why); exact and fast sampling of stationary
processes on a regular grid by circulant embedding (Dietrich-Newsam / Wood-Chan, with the minimal-embedding
check); fractional Brownian motion by Davies-Harte; the Hurst exponent by the aggregated-variance method; and
Gaussian conditioning (kriging / GP regression) with the posterior mean, variance and log marginal likelihood.

>>> import numpy as np
>>> from engstoch import gaussian as gp
>>> t = np.linspace(0, 1, 5)
>>> bool(np.allclose(gp.cov_matrix(gp.brownian, t), np.minimum.outer(t, t)))
True
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------------------------- kernels (of two arrays)


def exponential(s, t, ell: float = 1.0, var: float = 1.0):
    """var exp(-|s - t| / ell): the stationary OU covariance, continuous but nowhere differentiable paths."""
    return var * np.exp(-np.abs(np.asarray(s) - np.asarray(t)) / ell)


def squared_exponential(s, t, ell: float = 1.0, var: float = 1.0):
    """var exp(-(s - t)^2 / (2 ell^2)): infinitely differentiable paths."""
    return var * np.exp(-0.5 * (np.asarray(s) - np.asarray(t)) ** 2 / ell**2)


def matern32(s, t, ell: float = 1.0, var: float = 1.0):
    r = math.sqrt(3) * np.abs(np.asarray(s) - np.asarray(t)) / ell
    return var * (1 + r) * np.exp(-r)


def matern52(s, t, ell: float = 1.0, var: float = 1.0):
    r = math.sqrt(5) * np.abs(np.asarray(s) - np.asarray(t)) / ell
    return var * (1 + r + r * r / 3) * np.exp(-r)


def periodic(s, t, ell: float = 1.0, period: float = 1.0, var: float = 1.0):
    d = np.abs(np.asarray(s) - np.asarray(t))
    return var * np.exp(-2 * np.sin(math.pi * d / period) ** 2 / ell**2)


def brownian(s, t):
    """min(s, t)."""
    return np.minimum(np.asarray(s), np.asarray(t))


def fbm(s, t, hurst: float = 0.5):
    """Fractional Brownian motion: (|s|^2H + |t|^2H - |s - t|^2H) / 2."""
    s, t = np.asarray(s, float), np.asarray(t, float)
    h2 = 2 * hurst
    return 0.5 * (np.abs(s) ** h2 + np.abs(t) ** h2 - np.abs(s - t) ** h2)


def fgn_autocovariance(k, hurst: float):
    """Autocovariance of unit-step fractional Gaussian noise: (|k+1|^2H - 2|k|^2H + |k-1|^2H) / 2."""
    k = np.abs(np.asarray(k, float))
    h2 = 2 * hurst
    return 0.5 * (np.abs(k + 1) ** h2 - 2 * k**h2 + np.abs(k - 1) ** h2)


def cov_matrix(kernel, s, t=None, **params) -> np.ndarray:
    """Covariance matrix K_ij = k(s_i, t_j)."""
    s = np.asarray(s, float)
    t = s if t is None else np.asarray(t, float)
    return kernel(s[:, None], t[None, :], **params)


# --------------------------------------------------------------------------------------------- sampling


def sample_cholesky(kernel, t, n_paths: int, rng, jitter: float = 0.0, mean=None, **params) -> dict:
    """Paths by K = L L^T and X = L Z. Smooth kernels give numerically singular K; `jitter` adds a small multiple
    of the identity. Returns the paths, the jitter used and the condition number of K."""
    K = cov_matrix(kernel, t, **params)
    cond = float(np.linalg.cond(K))
    j = jitter
    while True:
        try:
            L = np.linalg.cholesky(K + j * np.eye(len(K)))
            break
        except np.linalg.LinAlgError:
            j = max(1e-12, 10 * j) if j else 1e-12 * np.trace(K) / len(K)
    X = rng.standard_normal((n_paths, len(K))) @ L.T
    if mean is not None:
        X = X + np.asarray(mean, float)
    return {"X": X, "jitter": j, "cond": cond}


def circulant_embedding(acov, n_paths: int, rng, pad: int = 0) -> dict:
    """Exact samples of a stationary Gaussian sequence with autocovariance acov[0..n-1] on n grid points: embed
    the Toeplitz covariance in a circulant matrix of size 2(n - 1 + pad), diagonalise it by FFT and colour
    complex white noise (Wood and Chan 1994). Fails (negative eigenvalues) if the embedding is not non-negative
    definite; increase `pad` (the acov must then be given for the padded lags). Each FFT gives two independent
    paths (real and imaginary parts)."""
    c = np.asarray(acov, float)
    n = len(c) - pad
    row = np.r_[c, c[-2:0:-1]]
    m = len(row)
    lam = np.fft.fft(row).real
    if np.min(lam) < -1e-10 * np.max(lam):
        raise ValueError(f"circulant embedding not non-negative definite (min eigenvalue {np.min(lam):.3g}); pad it")
    lam = np.maximum(lam, 0.0)
    n_fft = (n_paths + 1) // 2
    Z = rng.standard_normal((n_fft, m)) + 1j * rng.standard_normal((n_fft, m))
    Y = np.fft.fft(np.sqrt(lam / m) * Z, axis=1)
    X = np.vstack([Y.real[:, :n], Y.imag[:, :n]])[:n_paths]
    return {"X": X, "eigenvalues": lam, "embedding_size": m}


def fbm_davies_harte(n: int, hurst: float, T: float, n_paths: int, rng) -> np.ndarray:
    """Fractional Brownian motion on n steps of [0, T]: fractional Gaussian noise by circulant embedding
    (always valid for fGn), cumulated and scaled by (T/n)^H. Array (n_paths, n + 1)."""
    acov = fgn_autocovariance(np.arange(n), hurst)
    fgn = circulant_embedding(acov, n_paths, rng)["X"]
    return np.hstack([np.zeros((n_paths, 1)), np.cumsum(fgn, axis=1)]) * (T / n) ** hurst


def hurst_aggregated_variance(x, block_sizes=None) -> dict:
    """Estimate H from increments x: Var(block mean of size m) ~ m^(2H - 2); slope of the log-log fit."""
    x = np.asarray(x, float)
    n = len(x)
    if block_sizes is None:
        block_sizes = np.unique(np.logspace(0, math.log10(n // 10), 15).astype(int))
    v = []
    for m in block_sizes:
        k = n // m
        v.append(np.var(x[: k * m].reshape(k, m).mean(axis=1), ddof=1))
    slope = float(np.polyfit(np.log(block_sizes), np.log(v), 1)[0])
    return {"H": 1 + slope / 2, "block_sizes": np.asarray(block_sizes), "variances": np.array(v)}


# --------------------------------------------------------------------------------------------- conditioning


def condition(kernel, t_obs, y_obs, t_new, noise_var: float = 0.0, mean: float = 0.0, **params) -> dict:
    """Posterior of a zero-mean (or constant-mean) GP given noisy observations: mean k*^T (K + s^2 I)^-1 (y - m) + m,
    covariance K** - k*^T (K + s^2 I)^-1 k*, by Cholesky; also the log marginal likelihood
    -1/2 y^T A^-1 y - sum log L_ii - n/2 log 2 pi used to choose the kernel parameters."""
    t_obs, y = np.asarray(t_obs, float), np.asarray(y_obs, float) - mean
    K = cov_matrix(kernel, t_obs, **params) + (noise_var + 1e-12) * np.eye(len(t_obs))
    Ks = cov_matrix(kernel, t_obs, t_new, **params)
    Kss = cov_matrix(kernel, t_new, **params)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    v = np.linalg.solve(L, Ks)
    mu = mean + Ks.T @ alpha
    cov = Kss - v.T @ v
    lml = float(-0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * math.log(2 * math.pi))
    return {"mean": mu, "cov": cov, "std": np.sqrt(np.maximum(np.diag(cov), 0.0)), "log_marginal_likelihood": lml}


def empirical_variogram(x, max_lag: int) -> np.ndarray:
    """gamma(k) = (1/2) mean (x_{t+k} - x_t)^2 for a sequence (or the mean over several rows)."""
    x = np.atleast_2d(np.asarray(x, float))
    return np.array([0.5 * np.mean((x[:, k:] - x[:, :-k]) ** 2) if k else 0.0 for k in range(max_lag + 1)])
