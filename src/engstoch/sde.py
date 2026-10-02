"""Numerical solution of stochastic differential equations.

Euler-Maruyama for vector Ito SDEs dX = a(t, X) dt + b(t, X) dW with diagonal or general noise; Milstein for
scalar (and diagonal) noise; the stochastic Heun scheme, which converges to the Stratonovich solution;
convergence studies on shared Brownian paths (strong order from the error at T on coarsened increments,
weak order from moments); the Cox-Ingersoll-Ross process by full-truncation Euler; the Ito-to-Stratonovich
drift correction; and multilevel Monte Carlo (Giles 2008) with the optimal number of samples per level.

>>> import numpy as np
>>> from engstoch import sde
>>> a, b = (lambda t, x: 0.05 * x), (lambda t, x: 0.4 * x)
>>> exact = lambda t, W: np.exp((0.05 - 0.08) * t + 0.4 * W)
>>> r = sde.strong_order(a, b, exact, 1.0, 1.0, [16, 32, 64, 128, 256], 2000, np.random.default_rng(0))
>>> bool(0.35 < r["order"] < 0.7)
True
"""

from __future__ import annotations

import math

import numpy as np


def euler_maruyama(drift, diffusion, x0, T: float, n_steps: int, n_paths: int, rng, dW=None) -> dict:
    """X_{k+1} = X_k + a(t_k, X_k) dt + b(t_k, X_k) dW_k. x0 is a scalar or a vector of length d; diffusion
    returns an array of the same shape as x (diagonal noise) or (n_paths, d, m) for general noise (then dW
    has m columns). dW (n_paths, n_steps[, m]) may be supplied to reuse Brownian increments."""
    x0 = np.atleast_1d(np.asarray(x0, float))
    d = len(x0)
    dt = T / n_steps
    if dW is None:
        dW = math.sqrt(dt) * rng.standard_normal((n_paths, n_steps, d))
    dW = np.asarray(dW, float)
    if dW.ndim == 2:
        dW = dW[:, :, None]
    X = np.empty((n_paths, n_steps + 1, d))
    X[:, 0] = x0
    for k in range(n_steps):
        t = k * dt
        x = X[:, k]
        b = np.asarray(diffusion(t, x), float)
        noise = np.einsum("pdm,pm->pd", b, dW[:, k]) if b.ndim == 3 else b * dW[:, k]
        X[:, k + 1] = x + np.asarray(drift(t, x), float) * dt + noise
    return {"t": np.linspace(0, T, n_steps + 1), "X": X[..., 0] if d == 1 else X, "dW": dW}


def milstein(drift, diffusion, diffusion_dx, x0, T: float, n_steps: int, n_paths: int, rng, dW=None) -> dict:
    """Milstein's scheme for scalar (or diagonal) noise: Euler plus (1/2) b b' (dW^2 - dt), strong order 1."""
    x0 = np.atleast_1d(np.asarray(x0, float))
    dt = T / n_steps
    if dW is None:
        dW = math.sqrt(dt) * rng.standard_normal((n_paths, n_steps, len(x0)))
    dW = np.asarray(dW, float)
    if dW.ndim == 2:
        dW = dW[:, :, None]
    X = np.empty((n_paths, n_steps + 1, len(x0)))
    X[:, 0] = x0
    for k in range(n_steps):
        t = k * dt
        x = X[:, k]
        b = np.asarray(diffusion(t, x), float)
        X[:, k + 1] = (
            x
            + np.asarray(drift(t, x), float) * dt
            + b * dW[:, k]
            + 0.5 * b * np.asarray(diffusion_dx(t, x), float) * (dW[:, k] ** 2 - dt)
        )
    return {"t": np.linspace(0, T, n_steps + 1), "X": X[..., 0] if len(x0) == 1 else X, "dW": dW}


def heun_stratonovich(drift, diffusion, x0, T: float, n_steps: int, n_paths: int, rng, dW=None) -> dict:
    """Stochastic Heun (predictor-corrector with the trapezoidal rule in the noise term): converges to the
    Stratonovich solution of dX = a dt + b o dW."""
    x0 = np.atleast_1d(np.asarray(x0, float))
    dt = T / n_steps
    if dW is None:
        dW = math.sqrt(dt) * rng.standard_normal((n_paths, n_steps, len(x0)))
    dW = np.asarray(dW, float)
    if dW.ndim == 2:
        dW = dW[:, :, None]
    X = np.empty((n_paths, n_steps + 1, len(x0)))
    X[:, 0] = x0
    for k in range(n_steps):
        t = k * dt
        x = X[:, k]
        a0, b0 = np.asarray(drift(t, x), float), np.asarray(diffusion(t, x), float)
        xp = x + a0 * dt + b0 * dW[:, k]
        a1, b1 = np.asarray(drift(t + dt, xp), float), np.asarray(diffusion(t + dt, xp), float)
        X[:, k + 1] = x + 0.5 * (a0 + a1) * dt + 0.5 * (b0 + b1) * dW[:, k]
    return {"t": np.linspace(0, T, n_steps + 1), "X": X[..., 0] if len(x0) == 1 else X, "dW": dW}


def ito_to_stratonovich_drift(drift, diffusion, diffusion_dx):
    """The Stratonovich SDE with drift a - (1/2) b b' has the same solution as the Ito SDE with drift a."""
    return lambda t, x: np.asarray(drift(t, x)) - 0.5 * np.asarray(diffusion(t, x)) * np.asarray(diffusion_dx(t, x))


def _coarsen(dW, factor):
    n_paths, n, d = dW.shape
    return dW.reshape(n_paths, n // factor, factor, d).sum(axis=2)


def strong_order(
    drift, diffusion, exact, x0, T: float, steps_list, n_paths: int, rng, scheme: str = "euler", diffusion_dx=None
) -> dict:
    """Strong convergence: E|X_N - X(T)| for each step count, with all schemes driven by the same Brownian
    path (the finest increments summed), and the order as the least-squares slope in log-log. `exact(T, W_T)`
    gives the true solution from the Brownian value at T."""
    steps_list = sorted(steps_list)
    nmax = steps_list[-1]
    dt = T / nmax
    dW = math.sqrt(dt) * rng.standard_normal((n_paths, nmax, 1))
    WT = dW.sum(axis=1)[:, 0]
    Xtrue = exact(T, WT)
    errs = []
    for n in steps_list:
        dWc = _coarsen(dW, nmax // n)
        if scheme == "milstein":
            X = milstein(drift, diffusion, diffusion_dx, x0, T, n, n_paths, rng, dW=dWc)["X"]
        else:
            X = euler_maruyama(drift, diffusion, x0, T, n, n_paths, rng, dW=dWc)["X"]
        errs.append(float(np.mean(np.abs(X[:, -1] - Xtrue))))
    h = T / np.asarray(steps_list, float)
    order = float(np.polyfit(np.log(h), np.log(errs), 1)[0])
    return {"h": h, "error": np.array(errs), "order": order}


def weak_order(drift, diffusion, x0, T: float, steps_list, n_paths: int, rng, f, exact_value: float) -> dict:
    """Weak convergence: |E f(X_N) - E f(X(T))| for each step count (independent samples per step size, so the
    Monte Carlo noise must be well below the bias; the standard errors are returned)."""
    errs, ses = [], []
    for n in steps_list:
        X = euler_maruyama(drift, diffusion, x0, T, n, n_paths, rng)["X"][:, -1]
        v = np.asarray(f(X), float)
        errs.append(abs(float(v.mean()) - exact_value))
        ses.append(float(v.std(ddof=1) / math.sqrt(n_paths)))
    h = T / np.asarray(steps_list, float)
    return {
        "h": h,
        "error": np.array(errs),
        "std_error": np.array(ses),
        "order": float(np.polyfit(np.log(h), np.log(errs), 1)[0]),
    }


def cir_full_truncation(
    x0: float, kappa: float, theta: float, sigma: float, T: float, n_steps: int, n_paths: int, rng
) -> np.ndarray:
    """Cox-Ingersoll-Ross dX = kappa (theta - X) dt + sigma sqrt(X) dW by full-truncation Euler (Lord et al. 2010):
    the auxiliary process may go negative, X^+ is used in drift and diffusion and reported."""
    dt = T / n_steps
    x = np.full(n_paths, float(x0))
    out = np.empty((n_paths, n_steps + 1))
    out[:, 0] = x0
    for k in range(n_steps):
        xp = np.maximum(x, 0.0)
        x = x + kappa * (theta - xp) * dt + sigma * np.sqrt(xp * dt) * rng.standard_normal(n_paths)
        out[:, k + 1] = np.maximum(x, 0.0)
    return out


def cir_mean(x0: float, kappa: float, theta: float, t) -> np.ndarray:
    """E[X_t] = theta + (x0 - theta) e^{-kappa t} for the CIR process."""
    return theta + (x0 - theta) * np.exp(-kappa * np.asarray(t, float))


def mlmc_level(
    level: int, n_samples: int, payoff, S0: float, r: float, sigma: float, T: float, rng, M: int = 2
) -> np.ndarray:
    """Samples of P_l - P_{l-1} for GBM by Euler with M^l steps; the coarse path uses the summed fine increments
    (the coupling that makes the variance of the difference decay)."""
    nf = M**level
    hf = T / nf
    dW = math.sqrt(hf) * rng.standard_normal((n_samples, nf))
    Sf = np.full(n_samples, S0)
    for k in range(nf):
        Sf = Sf + r * Sf * hf + sigma * Sf * dW[:, k]
    Pf = math.exp(-r * T) * payoff(Sf)
    if level == 0:
        return Pf
    nc = nf // M
    hc = T / nc
    dWc = dW.reshape(n_samples, nc, M).sum(axis=2)
    Sc = np.full(n_samples, S0)
    for k in range(nc):
        Sc = Sc + r * Sc * hc + sigma * Sc * dWc[:, k]
    return Pf - math.exp(-r * T) * payoff(Sc)


def mlmc(
    payoff, S0: float, r: float, sigma: float, T: float, eps: float, rng, L: int = 6, n_pilot: int = 2000, M: int = 2
) -> dict:
    """Multilevel Monte Carlo estimate of E[e^{-rT} payoff(S_T)] to RMS accuracy ~eps (variance part eps^2/2),
    with the optimal sample sizes N_l = ceil(2 eps^-2 sqrt(V_l / C_l) sum_k sqrt(V_k C_k)) and C_l = M^l (Giles
    2008). Returns the estimate, the per-level means, variances and sample sizes and the total cost in Euler
    steps (compare: standard MC at the finest level costs ~2 Var(P_L) M^L / eps^2)."""
    V = np.zeros(L + 1)
    for lv in range(L + 1):
        V[lv] = np.var(mlmc_level(lv, n_pilot, payoff, S0, r, sigma, T, rng, M), ddof=1)
    C = M ** np.arange(L + 1, dtype=float)
    N = np.ceil(2 * eps**-2 * np.sqrt(V / C) * np.sum(np.sqrt(V * C))).astype(int)
    means = np.zeros(L + 1)
    for lv in range(L + 1):
        y = mlmc_level(lv, int(N[lv]), payoff, S0, r, sigma, T, rng, M)
        means[lv] = y.mean()
        V[lv] = y.var(ddof=1)
    return {
        "estimate": float(means.sum()),
        "means": means,
        "variances": V,
        "samples": N,
        "cost": float(np.sum(N * C)),
        "std_error": float(math.sqrt(np.sum(V / N))),
    }
