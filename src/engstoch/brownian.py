"""Brownian motion and the processes built from it.

Paths by independent Gaussian increments; Levy's midpoint (Brownian-bridge) construction, which refines a
coarse path without changing it; the Brownian bridge; Donsker's scaled random walk; geometric Brownian
motion and the Ornstein-Uhlenbeck process sampled exactly on any grid; the reflection principle (the law of
the running maximum) and the first-passage time to a level (Levy's distribution; inverse Gaussian with drift);
quadratic and total variation along refining partitions; the Karhunen-Loeve expansion; and the
Brownian-bridge crossing probability that corrects discretely monitored barrier problems.

>>> import numpy as np
>>> from engstoch import brownian as bm
>>> W = bm.paths(1.0, 1000, 2000, np.random.default_rng(0))
>>> bool(abs(np.var(W[:, -1]) - 1.0) < 0.1)
True
"""

from __future__ import annotations

import math

import numpy as np

from .special import norm_cdf


def paths(T: float, n_steps: int, n_paths: int, rng, sigma: float = 1.0, drift: float = 0.0) -> np.ndarray:
    """Brownian motion with drift on a uniform grid: array (n_paths, n_steps + 1) starting at 0."""
    dt = T / n_steps
    inc = drift * dt + sigma * math.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    return np.hstack([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)])


def time_grid(T: float, n_steps: int) -> np.ndarray:
    return np.linspace(0.0, T, n_steps + 1)


def refine(W, T: float, rng) -> np.ndarray:
    """Levy's construction: insert a midpoint between each pair of grid values, W_mid = (W_a + W_b)/2 + N(0, dt/4),
    keeping the existing values. Repeating doubles the resolution of the same path."""
    W = np.atleast_2d(np.asarray(W, float))
    n = W.shape[1] - 1
    dt = T / n
    mid = 0.5 * (W[:, :-1] + W[:, 1:]) + math.sqrt(dt / 4) * rng.standard_normal((W.shape[0], n))
    out = np.empty((W.shape[0], 2 * n + 1))
    out[:, ::2] = W
    out[:, 1::2] = mid
    return out


def bridge_construction(T: float, levels: int, n_paths: int, rng) -> np.ndarray:
    """Brownian paths on 2^levels steps by successive midpoint refinement from W_0 = 0, W_T ~ N(0, T)."""
    W = np.column_stack([np.zeros(n_paths), math.sqrt(T) * rng.standard_normal(n_paths)])
    for _ in range(levels):
        W = refine(W, T, rng)
    return W


def bridge(T: float, n_steps: int, n_paths: int, rng, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Brownian bridge from a at 0 to b at T: B_t = a + W_t - (t/T)(W_T - (b - a)); covariance s(T - t)/T."""
    W = paths(T, n_steps, n_paths, rng)
    t = time_grid(T, n_steps)
    return a + W - (t / T) * (W[:, -1:] - (b - a))


def donsker(n: int, n_paths: int, rng) -> np.ndarray:
    """Scaled simple random walk S_{floor(nt)} / sqrt(n) on t in [0, 1]: converges to Brownian motion."""
    steps = rng.choice(np.array([-1.0, 1.0]), (n_paths, n))
    return np.hstack([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)]) / math.sqrt(n)


def gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, rng) -> np.ndarray:
    """Geometric Brownian motion dS = mu S dt + sigma S dW, exactly: S_t = S0 exp((mu - sigma^2/2) t + sigma W_t)."""
    W = paths(T, n_steps, n_paths, rng)
    t = time_grid(T, n_steps)
    return S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)


def ou_exact(x0: float, theta: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, rng) -> np.ndarray:
    """Ornstein-Uhlenbeck dX = theta (mu - X) dt + sigma dW sampled exactly: an AR(1) with coefficient
    e^{-theta dt} and innovation variance sigma^2 (1 - e^{-2 theta dt}) / (2 theta)."""
    dt = T / n_steps
    a = math.exp(-theta * dt)
    s = sigma * math.sqrt((1 - a * a) / (2 * theta))
    X = np.empty((n_paths, n_steps + 1))
    X[:, 0] = x0
    z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        X[:, k + 1] = mu + a * (X[:, k] - mu) + s * z[:, k]
    return X


def ou_moments(x0: float, theta: float, mu: float, sigma: float, t) -> dict:
    """Mean mu + (x0 - mu) e^{-theta t} and variance sigma^2 (1 - e^{-2 theta t}) / (2 theta) of the OU process."""
    t = np.asarray(t, float)
    return {"mean": mu + (x0 - mu) * np.exp(-theta * t), "var": sigma**2 * (1 - np.exp(-2 * theta * t)) / (2 * theta)}


def max_cdf(a, t: float):
    """Reflection principle: P(max_{s <= t} W_s >= a) = 2 P(W_t >= a) = 2 (1 - Phi(a / sqrt t)), a >= 0."""
    return 2.0 * (1.0 - norm_cdf(np.asarray(a, float) / math.sqrt(t)))


def first_passage_cdf(t, a: float, drift: float = 0.0, sigma: float = 1.0):
    """P(T_a <= t) for X_t = drift t + sigma W_t and a > 0 (inverse Gaussian when drift > 0):
    Phi((drift t - a)/(sigma sqrt t)) + exp(2 drift a / sigma^2) Phi((-drift t - a)/(sigma sqrt t))."""
    t = np.asarray(t, float)
    st = sigma * np.sqrt(t)
    return norm_cdf((drift * t - a) / st) + math.exp(2 * drift * a / sigma**2) * norm_cdf((-drift * t - a) / st)


def first_passage_pdf(t, a: float, drift: float = 0.0, sigma: float = 1.0):
    """Density a / (sigma sqrt(2 pi t^3)) exp(-(a - drift t)^2 / (2 sigma^2 t))."""
    t = np.asarray(t, float)
    return a / (sigma * np.sqrt(2 * math.pi * t**3)) * np.exp(-((a - drift * t) ** 2) / (2 * sigma**2 * t))


def first_passage_simulated(
    a: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng,
    drift: float = 0.0,
    sigma: float = 1.0,
    bridge_correction: bool = False,
) -> np.ndarray:
    """First grid time at which the simulated path reaches a (inf if not by T). With bridge_correction, a
    crossing between grid points is also detected with the Brownian-bridge probability
    exp(-2 (a - x_k)(a - x_{k+1}) / (sigma^2 dt)), which removes the discretisation bias."""
    W = paths(T, n_steps, n_paths, rng, sigma, drift)
    t = time_grid(T, n_steps)
    hit = W >= a
    if bridge_correction:
        dt = T / n_steps
        x0, x1 = W[:, :-1], W[:, 1:]
        p = np.where((x0 < a) & (x1 < a), np.exp(-2 * (a - x0) * (a - x1) / (sigma**2 * dt)), 0.0)
        hit[:, 1:] |= rng.random(p.shape) < p
    first = np.where(hit.any(axis=1), t[np.argmax(hit, axis=1)], np.inf)
    return first


def quadratic_variation(W) -> np.ndarray:
    """Sum of squared increments of each path (tends to T for Brownian motion as the mesh goes to zero)."""
    return np.sum(np.diff(np.atleast_2d(W), axis=1) ** 2, axis=1)


def total_variation(W) -> np.ndarray:
    """Sum of absolute increments (grows like sqrt(n): Brownian paths have unbounded variation)."""
    return np.sum(np.abs(np.diff(np.atleast_2d(W), axis=1)), axis=1)


def karhunen_loeve(T: float, n_terms: int, t, n_paths: int, rng) -> np.ndarray:
    """W_t = sum_k Z_k sqrt(2T) sin((k - 1/2) pi t / T) / ((k - 1/2) pi): the eigenfunction expansion of the
    covariance min(s, t); truncation leaves variance T/(pi^2 n) on average."""
    t = np.asarray(t, float)
    k = np.arange(1, n_terms + 1) - 0.5
    basis = math.sqrt(2 * T) * np.sin(np.outer(k, t) * math.pi / T) / (k[:, None] * math.pi)
    return rng.standard_normal((n_paths, n_terms)) @ basis


def arcsine_positive_time(W) -> np.ndarray:
    """Fraction of grid times at which each path is positive (arcsine-distributed in the limit)."""
    return np.mean(np.atleast_2d(W)[:, 1:] > 0, axis=1)


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """The closed-form European call price - the reference value for the SDE and Monte Carlo notebooks."""
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))


def down_and_out_call(S0: float, K: float, B: float, r: float, sigma: float, T: float) -> float:
    """Continuously monitored down-and-out call (B < K, B < S0), Merton's reflection formula."""
    lam = (r + 0.5 * sigma**2) / sigma**2
    c = black_scholes_call(S0, K, r, sigma, T)
    y = math.log(B * B / (S0 * K)) / (sigma * math.sqrt(T)) + lam * sigma * math.sqrt(T)
    cdi = S0 * (B / S0) ** (2 * lam) * norm_cdf(y) - K * math.exp(-r * T) * (B / S0) ** (2 * lam - 2) * norm_cdf(
        y - sigma * math.sqrt(T)
    )
    return c - float(cdi)
