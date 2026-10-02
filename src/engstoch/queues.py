"""Queueing theory: closed-form results and fast simulation.

Markovian queues (M/M/1, M/M/c with the Erlang C formula, M/M/c/K, M/M/inf, Erlang B with its stable
recursion), the M/G/1 queue by Pollaczek-Khinchine, the Kingman and Allen-Cunneen approximations for G/G/1
and G/G/c, Little's law, the waiting-time distribution of M/M/c, staffing (the smallest c meeting a service
level, and the square-root staffing rule), open Jackson networks (traffic equations and product form); and
simulation: the Lindley recursion for G/G/1, a G/G/c FCFS simulation with optional abandonment (exact, by
tracking when each server frees up), and time-averaged queue lengths from the waiting times.

>>> from engstoch import queues as qu
>>> round(qu.erlang_c(2, 1.5), 6)
0.642857
>>> round(qu.erlang_b(10, 7.0), 6)
0.078741
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def mm1(lam: float, mu: float) -> dict:
    """M/M/1: rho = lam/mu, L = rho/(1 - rho), W = 1/(mu - lam), Wq = rho/(mu - lam), P(N = n) = (1 - rho) rho^n."""
    rho = lam / mu
    if rho >= 1:
        raise ValueError("unstable: lam >= mu")
    return {
        "rho": rho,
        "L": rho / (1 - rho),
        "Lq": rho**2 / (1 - rho),
        "W": 1 / (mu - lam),
        "Wq": rho / (mu - lam),
        "P0": 1 - rho,
    }


def erlang_b(c: int, a: float) -> float:
    """Erlang B blocking probability for c servers and offered load a = lam/mu, by the stable recursion
    B(k) = a B(k-1) / (k + a B(k-1))."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in M/M/c with offered load a = lam/mu < c: C = c B / (c - a (1 - B))."""
    if a >= c:
        return 1.0
    b = erlang_b(c, a)
    return c * b / (c - a * (1 - b))


def mmc(lam: float, mu: float, c: int) -> dict:
    """M/M/c: the probability of waiting (Erlang C), Wq = C / (c mu - lam), and the other means by Little's law."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        raise ValueError("unstable: lam >= c mu")
    C = erlang_c(c, a)
    Wq = C / (c * mu - lam)
    return {"rho": rho, "P_wait": C, "Wq": Wq, "W": Wq + 1 / mu, "Lq": lam * Wq, "L": lam * (Wq + 1 / mu)}


def mmc_wait_cdf(t, lam: float, mu: float, c: int):
    """P(Wq <= t) = 1 - C exp(-(c mu - lam) t) for M/M/c FCFS."""
    C = erlang_c(c, lam / mu)
    return 1.0 - C * np.exp(-(c * mu - lam) * np.asarray(t, float))


def service_level(lam: float, mu: float, c: int, t: float) -> float:
    """Fraction of customers answered within t (the call-centre 'service level')."""
    if lam >= c * mu:
        return 0.0
    return float(mmc_wait_cdf(t, lam, mu, c))


def staffing(lam: float, mu: float, target: float, t: float, c_max: int | None = None) -> dict:
    """Smallest number of servers with service level P(Wq <= t) >= target, compared with the square-root
    staffing rule c = a + beta sqrt(a) (Halfin-Whitt) for the beta that rule needs."""
    a = lam / mu
    c_max = int(a + 20 * math.sqrt(a) + 100) if c_max is None else c_max
    c = max(1, int(math.floor(a)) + 1)
    while c < c_max and service_level(lam, mu, c, t) < target:
        c += 1
    beta = (c - a) / math.sqrt(a) if a > 0 else 0.0
    return {"c": c, "offered_load": a, "service_level": service_level(lam, mu, c, t), "beta": beta}


def mmck(lam: float, mu: float, c: int, K: int) -> dict:
    """M/M/c/K (at most K in the system): stationary law by detailed balance; blocking P_K, throughput,
    L and W for admitted customers."""
    n = np.arange(K + 1)
    rates = np.minimum(n, c) * mu
    w = np.ones(K + 1)
    for k in range(1, K + 1):
        w[k] = w[k - 1] * lam / rates[k]
    p = w / w.sum()
    L = float(np.sum(n * p))
    lam_eff = lam * (1 - p[-1])
    Lq = float(np.sum(np.maximum(n - c, 0) * p))
    return {
        "p": p,
        "P_block": float(p[-1]),
        "throughput": lam_eff,
        "L": L,
        "Lq": Lq,
        "W": L / lam_eff,
        "Wq": Lq / lam_eff,
    }


def mm_inf(lam: float, mu: float) -> dict:
    """M/M/inf (and M/G/inf): the number in system is Poisson(lam/mu)."""
    return {"L": lam / mu, "W": 1 / mu, "distribution": "Poisson"}


def mg1(lam: float, mean_s: float, second_moment_s: float) -> dict:
    """M/G/1 by Pollaczek-Khinchine: Wq = lam E[S^2] / (2 (1 - rho)); equivalently Wq = rho E[S] (1 + c_s^2) /
    (2 (1 - rho))."""
    rho = lam * mean_s
    if rho >= 1:
        raise ValueError("unstable")
    Wq = lam * second_moment_s / (2 * (1 - rho))
    return {"rho": rho, "Wq": Wq, "W": Wq + mean_s, "Lq": lam * Wq, "L": lam * (Wq + mean_s)}


def kingman(lam: float, mean_s: float, ca2: float, cs2: float) -> float:
    """Kingman's heavy-traffic approximation for G/G/1: Wq ~ (rho / (1 - rho)) ((ca^2 + cs^2) / 2) E[S]."""
    rho = lam * mean_s
    return rho / (1 - rho) * (ca2 + cs2) / 2 * mean_s


def allen_cunneen(lam: float, mean_s: float, c: int, ca2: float, cs2: float) -> float:
    """Allen-Cunneen approximation for G/G/c: the M/M/c waiting time scaled by (ca^2 + cs^2) / 2."""
    return mmc(lam, 1 / mean_s, c)["Wq"] * (ca2 + cs2) / 2


def little(L: float | None = None, lam: float | None = None, W: float | None = None) -> float:
    """Little's law L = lam W: give two of the three, get the third."""
    if L is None:
        return lam * W
    if lam is None:
        return L / W
    return L / lam


def jackson(arrivals, routing, mu, servers=None) -> dict:
    """Open Jackson network: solve the traffic equations Lambda = gamma + Lambda R, then treat each node as an
    independent M/M/c_i queue (product form). Returns node arrival rates, utilisations, L and the network's
    mean sojourn time by Little's law."""
    gamma = np.asarray(arrivals, float)
    R = np.asarray(routing, float)
    mu = np.asarray(mu, float)
    c = np.ones(len(mu), dtype=int) if servers is None else np.asarray(servers, int)
    Lam = np.linalg.solve(np.eye(len(gamma)) - R.T, gamma)
    nodes = [mmc(Lam[i], mu[i], int(c[i])) for i in range(len(mu))]
    L = np.array([nd["L"] for nd in nodes])
    return {"Lambda": Lam, "rho": Lam / (c * mu), "L": L, "W_network": float(L.sum() / gamma.sum()), "nodes": nodes}


# --------------------------------------------------------------------------------------------- simulation


def lindley(interarrival, service) -> np.ndarray:
    """Waiting times in a G/G/1 FCFS queue from the Lindley recursion W_{n+1} = max(0, W_n + S_n - A_{n+1}),
    with interarrival[k] the gap before customer k (interarrival[0] is ignored)."""
    a = np.asarray(interarrival, float)
    s = np.asarray(service, float)
    w = np.zeros(len(s))
    for k in range(1, len(s)):
        w[k] = max(0.0, w[k - 1] + s[k - 1] - a[k])
    return w


def simulate_ggc(arrival_times, service_times, c: int, patience=None) -> dict:
    """FCFS G/G/c queue with optional abandonment. Customer k arrives at arrival_times[k], needs service_times[k]
    and (if patience is given) leaves unserved if not started within patience[k]. Servers are a heap of the
    times they next become free. Returns waits, start and departure times and the abandonment flags."""
    arr = np.asarray(arrival_times, float)
    s = np.asarray(service_times, float)
    pat = None if patience is None else np.asarray(patience, float)
    free = [0.0] * c
    heapq.heapify(free)
    n = len(arr)
    wait, start, depart = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    abandoned = np.zeros(n, dtype=bool)
    # With abandonment the FCFS queue still serves customers in arrival order; a customer who would have to wait
    # longer than their patience leaves and occupies no server.
    for k in range(n):
        earliest = free[0]
        begin = max(arr[k], earliest)
        if pat is not None and begin - arr[k] > pat[k]:
            abandoned[k] = True
            wait[k] = pat[k]
            continue
        heapq.heapreplace(free, begin + s[k])
        wait[k] = begin - arr[k]
        start[k], depart[k] = begin, begin + s[k]
    return {"wait": wait, "start": start, "depart": depart, "abandoned": abandoned, "served": ~abandoned}


def number_in_system(arrivals, departures, t_grid) -> np.ndarray:
    """N(t) on a grid from arrival and departure times (departures NaN for customers who never entered)."""
    a = np.sort(np.asarray(arrivals, float))
    d = np.sort(np.asarray(departures, float)[np.isfinite(departures)])
    return np.searchsorted(a, t_grid, side="right") - np.searchsorted(d, t_grid, side="right")


def time_average(t_events, values, t_end: float) -> float:
    """Time average of a piecewise-constant process with value values[k] on [t_events[k], t_events[k+1])."""
    t = np.r_[np.asarray(t_events, float), t_end]
    return float(np.sum(np.asarray(values, float) * np.diff(t)) / (t_end - t[0]))
