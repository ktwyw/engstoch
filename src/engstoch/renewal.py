"""Renewal processes.

Simulation of renewal epochs and counts; the renewal function m(t) = E[N(t)] by numerical solution of the
renewal equation m = F + F * m (a Riemann-Stieltjes discretisation on a uniform grid, exact in the limit) and
its closed forms for Poisson and Erlang-2 inter-renewals; the elementary renewal theorem and the
second-order asymptote m(t) ~ t/mu + (sigma^2 - mu^2)/(2 mu^2); the renewal-reward theorem with a
replacement-policy optimiser (age replacement); the inspection paradox (the length-biased interval E[X^2]/E[X]
covering a fixed time, and the equilibrium residual-life law); and alternating renewal availability.

>>> import numpy as np
>>> from engstoch import renewal as rn
>>> t, m = rn.renewal_function(lambda x: 1 - np.exp(-2 * x), 3.0, 3000)
>>> bool(abs(m[-1] - 6.0) < 1e-2)
True
"""

from __future__ import annotations

import math

import numpy as np


def simulate(sampler, t_end: float, rng) -> np.ndarray:
    """Renewal epochs S_1 < S_2 < ... <= t_end, inter-renewal times from sampler(n, rng)."""
    out, t = [], 0.0
    while True:
        x = np.asarray(sampler(64, rng), float)
        s = t + np.cumsum(x)
        out.append(s[s <= t_end])
        if s[-1] > t_end:
            break
        t = s[-1]
    return np.concatenate(out)


def count_at(sampler, t: float, n_paths: int, rng) -> np.ndarray:
    """N(t) on n_paths independent realisations."""
    return np.array([len(simulate(sampler, t, rng)) for _ in range(n_paths)])


def renewal_function(cdf, t_end: float, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Solve m(t) = F(t) + int_0^t m(t - s) dF(s) on a grid of n steps: with F_k = F(k h) and the increments
    dF_j = F_j - F_{j-1}, m_k = F_k + sum_{j=1..k} m_{k-j} (dF_j) using the midpoint value of m on each cell
    (an O(n^2) scheme, second order for smooth F)."""
    h = t_end / n
    t = np.arange(n + 1) * h
    F = np.asarray(cdf(t), float)
    dF = np.diff(F)
    m = np.zeros(n + 1)
    for k in range(1, n + 1):
        upper = m[k:0:-1]  # m_k (still 0), m_{k-1}, ..., m_1: the value at the left end of each cell of t - s
        lower = m[k - 1 :: -1]  # m_{k-1}, ..., m_0
        rest = F[k] + 0.5 * np.dot(dF[:k], upper + lower)  # the m_k term contributes 0 here ...
        m[k] = rest / (1.0 - 0.5 * dF[0])  # ... and is solved for: m_k (1 - dF_1 / 2) = F_k + the known terms
    return t, m


def renewal_function_poisson(t, rate: float):
    """Exponential inter-renewals: m(t) = rate t exactly."""
    return rate * np.asarray(t, float)


def renewal_function_erlang2(t, rate: float):
    """Erlang(2, rate) inter-renewals: m(t) = rate t / 2 - (1 - exp(-2 rate t)) / 4."""
    t = np.asarray(t, float)
    return rate * t / 2 - 0.25 * (1 - np.exp(-2 * rate * t))


def asymptote(t, mean: float, var: float):
    """Second-order renewal asymptote m(t) ~ t / mu + (sigma^2 - mu^2) / (2 mu^2)."""
    return np.asarray(t, float) / mean + (var - mean**2) / (2 * mean**2)


def reward_rate(cycle_reward_mean: float, cycle_length_mean: float) -> float:
    """Renewal-reward theorem: the long-run reward per unit time is E[R] / E[X]."""
    return cycle_reward_mean / cycle_length_mean


def age_replacement_cost(T, survival, preventive_cost: float, failure_cost: float, n: int = 4000):
    """Long-run cost rate of replacing at age T or at failure, whichever first:
    (c_p R(T) + c_f (1 - R(T))) / int_0^T R(t) dt, by the renewal-reward theorem."""

    def one(Tv):
        s = np.linspace(0, Tv, n)
        R = np.asarray(survival(s), float)
        mean_cycle = float(np.sum(0.5 * (R[1:] + R[:-1]) * np.diff(s)))
        RT = float(survival(np.array([Tv]))[0])
        return (preventive_cost * RT + failure_cost * (1 - RT)) / mean_cycle

    T = np.atleast_1d(np.asarray(T, float))
    return np.array([one(v) for v in T])


def optimal_age_replacement(survival, preventive_cost: float, failure_cost: float, T_grid) -> dict:
    """Minimise the age-replacement cost rate over a grid of ages (compare with run-to-failure, T -> inf)."""
    T_grid = np.asarray(T_grid, float)
    c = age_replacement_cost(T_grid, survival, preventive_cost, failure_cost)
    i = int(np.argmin(c))
    return {"T": float(T_grid[i]), "cost_rate": float(c[i]), "costs": c}


def length_biased_mean(mean: float, second_moment: float) -> float:
    """Mean length of the inter-renewal interval covering a fixed (large) time: E[X^2] / E[X] >= E[X]."""
    return second_moment / mean


def mean_residual_life(mean: float, second_moment: float) -> float:
    """Long-run mean residual life (time to the next renewal from a random inspection): E[X^2] / (2 E[X])."""
    return second_moment / (2 * mean)


def equilibrium_cdf(x, survival, mean: float, n: int = 4000):
    """Equilibrium (stationary residual-life) distribution F_e(x) = (1/mu) int_0^x (1 - F(s)) ds."""

    def one(xv):
        s = np.linspace(0, xv, n)
        R = np.asarray(survival(s), float)
        return float(np.sum(0.5 * (R[1:] + R[:-1]) * np.diff(s))) / mean

    return np.array([one(v) for v in np.atleast_1d(np.asarray(x, float))])


def inspect(sampler, t_inspect: float, n_paths: int, rng) -> dict:
    """At a fixed time, observe the current interval length, age and residual life on n_paths realisations."""
    L, A, Rr = [], [], []
    for _ in range(n_paths):
        t, prev = 0.0, 0.0
        while t <= t_inspect:  # renew until the first epoch after the inspection time
            x = np.asarray(sampler(64, rng), float)
            s = t + np.cumsum(x)
            k = int(np.searchsorted(s, t_inspect, side="right"))
            if k < len(s):
                prev = s[k - 1] if k > 0 else t
                t = s[k]
                break
            t = s[-1]
        L.append(t - prev)
        A.append(t_inspect - prev)
        Rr.append(t - t_inspect)
    return {"length": np.array(L), "age": np.array(A), "residual": np.array(Rr)}


def availability(mean_up: float, mean_down: float) -> float:
    """Alternating renewal: the long-run fraction of time up is E[U] / (E[U] + E[D])."""
    return mean_up / (mean_up + mean_down)


def availability_exponential(t, fail_rate: float, repair_rate: float):
    """Point availability A(t) of a unit with exponential up and down times, starting up:
    mu/(lambda + mu) + lambda/(lambda + mu) exp(-(lambda + mu) t)."""
    s = fail_rate + repair_rate
    return repair_rate / s + fail_rate / s * np.exp(-s * np.asarray(t, float))


def weibull_moments(shape: float, scale: float) -> dict:
    """Mean and variance of a Weibull(shape, scale) lifetime."""
    m1 = scale * math.gamma(1 + 1 / shape)
    m2 = scale**2 * math.gamma(1 + 2 / shape)
    return {"mean": m1, "var": m2 - m1**2, "second_moment": m2}
