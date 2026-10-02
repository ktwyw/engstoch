"""The course's example models, each returning a dict with the object and its parameters.

Discrete-time chains: two-state weather, the Ehrenfest urn, an (s, S) inventory chain, a small web graph,
snakes and ladders (an absorbing chain). Continuous time: a machine with degradation and repair, the
M/M/c/K queue as a birth-death chain. Reaction networks for Gillespie's algorithm: the stochastic SIR epidemic,
Lotka-Volterra predator-prey, and two-stage gene expression. The call centre's arrival-rate profile used by the
course data and the project.

>>> from engstoch import models
>>> models.weather()["chain"].stationary().round(4).tolist()
[0.75, 0.25]
"""

from __future__ import annotations

import math

import numpy as np

from .ctmc import CTMC, birth_death
from .markov import MarkovChain


def weather(p_dw: float = 0.2, p_wd: float = 0.6) -> dict:
    """Dry/wet chain: P(dry -> wet) = p_dw, P(wet -> dry) = p_wd."""
    P = [[1 - p_dw, p_dw], [p_wd, 1 - p_wd]]
    return {"chain": MarkovChain(P, ["dry", "wet"]), "params": {"p_dw": p_dw, "p_wd": p_wd}}


def ehrenfest(N: int = 10) -> dict:
    """Ehrenfest urn: N balls in two urns, a random ball changes urn. State = balls in urn 1; period 2;
    stationary Binomial(N, 1/2); a lazy version (stay with probability 1/2) is aperiodic."""
    P = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        if k > 0:
            P[k, k - 1] = k / N
        if k < N:
            P[k, k + 1] = (N - k) / N
    lazy = 0.5 * (np.eye(N + 1) + P)
    return {"chain": MarkovChain(P), "lazy": MarkovChain(lazy), "N": N}


def inventory_chain(s: int = 2, S: int = 6, demand_mean: float = 2.0) -> dict:
    """End-of-week stock under an (s, S) policy with Poisson demand and lost sales: if stock <= s order up to S
    (delivered before the next week), then demand D removes min(D, stock). States 0..S."""
    kmax = 60
    k = np.arange(kmax)
    pmf = np.exp(-demand_mean + k * math.log(demand_mean) - np.array([math.lgamma(j + 1.0) for j in k]))
    P = np.zeros((S + 1, S + 1))
    for x in range(S + 1):
        start = S if x <= s else x
        for d in range(kmax):
            P[x, max(start - d, 0)] += pmf[d]
    P /= P.sum(axis=1, keepdims=True)
    return {"chain": MarkovChain(P), "params": {"s": s, "S": S, "demand_mean": demand_mean}, "demand_pmf": pmf}


def web_graph() -> dict:
    """A 6-page directed link graph (page 5 is dangling) for PageRank."""
    A = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    return {"adjacency": A, "pages": ["home", "about", "blog", "docs", "shop", "pdf"]}


def snakes_and_ladders(size: int = 36, jumps=None) -> dict:
    """A 6 x 6 board (squares 0..36, 36 = finish) with a fair die and the rule that a throw past the end wastes
    the turn. Default ladders 3 -> 16, 5 -> 7, 15 -> 25, 18 -> 20, 21 -> 32; snakes 12 -> 2, 14 -> 11, 17 -> 4,
    31 -> 19, 35 -> 22."""
    if jumps is None:
        jumps = {3: 16, 5: 7, 15: 25, 18: 20, 21: 32, 12: 2, 14: 11, 17: 4, 31: 19, 35: 22}
    P = np.zeros((size + 1, size + 1))
    for x in range(size + 1):
        if x == size:
            P[x, x] = 1.0
            continue
        for d in range(1, 7):
            y = x + d if x + d <= size else x
            y = jumps.get(y, y)
            P[x, y] += 1 / 6
    return {"chain": MarkovChain(P), "jumps": jumps}


def machine(
    fail_rate: float = 0.02,
    degrade_rate: float = 0.05,
    degraded_fail_rate: float = 0.2,
    repair_rate: float = 0.5,
    maintenance_rate: float = 1.0,
) -> dict:
    """Machine with states up -> degraded -> down -> up (rates per hour): up fails directly at fail_rate or degrades;
    a degraded machine fails or is restored by maintenance; a down machine is repaired."""
    Q = np.array(
        [
            [-(fail_rate + degrade_rate), degrade_rate, fail_rate],
            [maintenance_rate, -(maintenance_rate + degraded_fail_rate), degraded_fail_rate],
            [repair_rate, 0.0, -repair_rate],
        ]
    )
    return {"chain": CTMC(Q, ["up", "degraded", "down"]), "Q": Q}


def mmck_chain(lam: float, mu: float, c: int, K: int) -> dict:
    """The M/M/c/K queue as a birth-death CTMC."""
    return birth_death([lam] * K, [min(k, c) * mu for k in range(1, K + 1)])


def sir(N: int = 200, beta: float = 0.3, gamma: float = 0.1, I0: int = 2) -> dict:
    """Stochastic SIR: infection S + I -> 2I at rate beta S I / N, recovery I -> R at rate gamma I;
    R0 = beta / gamma."""
    stoich = np.array([[-1, 1, 0], [0, -1, 1]])
    return {
        "x0": np.array([N - I0, I0, 0]),
        "stoich": stoich,
        "propensity": lambda x: np.array([beta * x[0] * x[1] / N, gamma * x[1]]),
        "R0": beta / gamma,
        "params": {"N": N, "beta": beta, "gamma": gamma, "I0": I0},
    }


def lotka_volterra(a: float = 1.0, b: float = 0.005, c: float = 0.6, x0=(50, 100)) -> dict:
    """Prey X, predator Y: X -> 2X (a X), X + Y -> 2Y (b X Y), Y -> 0 (c Y)."""
    stoich = np.array([[1, 0], [-1, 1], [0, -1]])
    return {
        "x0": np.array(x0),
        "stoich": stoich,
        "propensity": lambda x: np.array([a * x[0], b * x[0] * x[1], c * x[1]]),
        "params": {"a": a, "b": b, "c": c},
    }


def gene_expression(k_m: float = 10.0, g_m: float = 1.0, k_p: float = 5.0, g_p: float = 0.1) -> dict:
    """Two-stage gene expression: mRNA made at k_m and degraded at g_m per molecule; protein translated at k_p per
    mRNA and degraded at g_p. Stationary means k_m/g_m and k_m k_p/(g_m g_p); protein Fano factor
    1 + k_p / (g_m + g_p) (translational bursting)."""
    stoich = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    return {
        "x0": np.array([0, 0]),
        "stoich": stoich,
        "propensity": lambda x: np.array([k_m, g_m * x[0], k_p * x[0], g_p * x[1]]),
        "mean_mrna": k_m / g_m,
        "mean_protein": k_m * k_p / (g_m * g_p),
        "fano_protein": 1 + k_p / (g_m + g_p),
    }


def call_centre_rate(t_hours):
    """Arrival rate (calls per hour) of the course's call centre, open 08:00-20:00: a morning peak near 10:30
    and a smaller afternoon peak near 14:30 on a base load. Zero outside opening hours."""
    t = np.asarray(t_hours, float)
    lam = 40 + 110 * np.exp(-0.5 * ((t - 10.5) / 1.2) ** 2) + 70 * np.exp(-0.5 * ((t - 14.5) / 1.5) ** 2)
    return np.where((t >= 8) & (t < 20), lam, 0.0)


CALL_CENTRE = {
    "open": 8.0,
    "close": 20.0,
    "rate_max": 160.0,
    "monday_factor": 1.15,
    "service_lognormal": (
        math.log(4.0) - 0.5 * math.log(1 + 0.8**2),
        math.sqrt(math.log(1 + 0.8**2)),
    ),  # mean 4 min, cv 0.8
    "patience_mean_min": 3.0,
}
