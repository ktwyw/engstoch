"""Continuous-time Markov chains and stochastic simulation of reaction networks.

A ``CTMC`` holds a generator Q (non-negative off-diagonal rates, rows summing to zero) and computes the
transition function P(t) = exp(Qt) two ways - the matrix exponential by scaling and squaring with a [6/6]
Pade approximant, and uniformisation (Jensen's method) with its Poisson truncation bound - the stationary
distribution, the embedded jump chain and holding rates, expected times to absorption, simulation, and
maximum-likelihood estimation of Q from an observed path (transition counts over occupation times).
``birth_death`` builds the generator of a birth-death process and its product-form stationary law.
``gillespie`` and ``tau_leap`` simulate chemical reaction networks from stoichiometry and propensities.

>>> import numpy as np
>>> from engstoch import ctmc
>>> c = ctmc.CTMC([[-2.0, 2.0], [1.0, -1.0]])
>>> c.stationary().round(6).tolist()
[0.333333, 0.666667]
"""

from __future__ import annotations

import math

import numpy as np

from .markov import MarkovChain


def expm(A) -> np.ndarray:
    """exp(A) by scaling and squaring with the diagonal [6/6] Pade approximant (Moler and Van Loan, 'method 3')."""
    A = np.asarray(A, float)
    norm = np.linalg.norm(A, np.inf)
    s = max(0, int(math.ceil(math.log2(norm / 0.5))) + 1) if norm > 0.5 else 0
    X = A / 2.0**s
    c = 0.5
    n = len(A)
    E = np.eye(n) + c * X
    D = np.eye(n) - c * X
    Xk = X.copy()
    q = 6
    for k in range(2, q + 1):
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        Xk = X @ Xk
        E = E + c * Xk
        D = D + (c if k % 2 == 0 else -c) * Xk
    E = np.linalg.solve(D, E)
    for _ in range(s):
        E = E @ E
    return E


class CTMC:
    """Finite continuous-time Markov chain with generator Q."""

    def __init__(self, Q, states=None):
        Q = np.asarray(Q, float)
        off = Q - np.diag(np.diag(Q))
        if np.any(off < -1e-12) or not np.allclose(Q.sum(axis=1), 0.0, atol=1e-9):
            raise ValueError("Q needs non-negative off-diagonal rates and zero row sums")
        self.Q = Q
        self.n = len(Q)
        self.states = list(states) if states is not None else list(range(self.n))
        self.index = {s: i for i, s in enumerate(self.states)}

    def __repr__(self) -> str:
        return f"CTMC({self.n} states)"

    @property
    def rates(self) -> np.ndarray:
        """Total exit rate q_i = -Q_ii of each state (holding times are Exponential(q_i))."""
        return -np.diag(self.Q)

    def jump_chain(self) -> MarkovChain:
        """Embedded discrete chain: P_ij = Q_ij / q_i off the diagonal; absorbing states keep P_ii = 1."""
        q = self.rates
        P = np.zeros_like(self.Q)
        for i in range(self.n):
            if q[i] > 0:
                P[i] = self.Q[i] / q[i]
                P[i, i] = 0.0
            else:
                P[i, i] = 1.0
        return MarkovChain(P, self.states)

    def transition_matrix(self, t: float) -> np.ndarray:
        """P(t) = exp(Q t), the solution of Kolmogorov's forward and backward equations."""
        return expm(self.Q * t)

    def uniformisation(self, t: float, tol: float = 1e-12) -> dict:
        """P(t) = sum_k e^{-Lt} (Lt)^k / k! R^k with R = I + Q/L and L >= max q_i. All terms are non-negative,
        so the truncation error is the Poisson tail beyond the last term (returned as `bound`)."""
        L = float(np.max(self.rates)) or 1.0
        R = np.eye(self.n) + self.Q / L
        lt = L * t
        term_w = math.exp(-lt)
        acc = term_w
        Rk = np.eye(self.n)
        P = term_w * Rk
        k = 0
        while 1.0 - acc > tol and k < 100_000:
            k += 1
            term_w *= lt / k
            Rk = Rk @ R
            P = P + term_w * Rk
            acc += term_w
            if term_w == 0.0 and acc < 1e-300:  # underflow of e^{-Lt}: fall back to the exponential
                return {"P": self.transition_matrix(t), "terms": k, "bound": 0.0}
        return {"P": P, "terms": k + 1, "bound": max(0.0, 1.0 - acc)}

    def distribution(self, p0, t: float) -> np.ndarray:
        return np.asarray(p0, float) @ self.transition_matrix(t)

    def stationary(self) -> np.ndarray:
        """pi Q = 0, sum pi = 1 (irreducible chains)."""
        A = self.Q.T.copy()
        A[-1] = 1.0
        b = np.zeros(self.n)
        b[-1] = 1.0
        return np.linalg.solve(A, b)

    def stationary_via_jump_chain(self) -> np.ndarray:
        """pi_i proportional to nu_i / q_i, with nu the stationary law of the jump chain (time in a state is the
        visit frequency times the mean holding time)."""
        nu = self.jump_chain().stationary()
        w = nu / self.rates
        return w / w.sum()

    def absorption_times(self) -> dict:
        """Expected time to absorption from each transient state: solve Q_TT m = -1."""
        absorbing = [i for i in range(self.n) if self.rates[i] == 0]
        transient = [i for i in range(self.n) if i not in absorbing]
        QT = self.Q[np.ix_(transient, transient)]
        m = np.linalg.solve(QT, -np.ones(len(transient)))
        B = np.linalg.solve(-QT, self.Q[np.ix_(transient, absorbing)]) if absorbing else np.zeros((len(transient), 0))
        return {"transient": [self.states[i] for i in transient], "expected_time": m, "absorption_probabilities": B}

    def simulate(self, t_end: float, start, rng) -> dict:
        """Path by exponential holding times and jump-chain moves: event times (starting at 0) and states."""
        i = self.index[start]
        t = 0.0
        times, states = [0.0], [i]
        q = self.rates
        P = self.jump_chain().P
        cdf = np.cumsum(P, axis=1)
        cdf[:, -1] = 1.0
        while True:
            if q[i] == 0:
                break
            t += rng.exponential(1.0 / q[i])
            if t > t_end:
                break
            i = int(np.searchsorted(cdf[i], rng.random(), side="right"))
            times.append(t)
            states.append(i)
        return {"times": np.array(times), "states": np.array(states), "t_end": t_end}

    def occupation(self, path) -> np.ndarray:
        """Fraction of [0, t_end] spent in each state along a simulated path."""
        times = np.r_[path["times"], path["t_end"]]
        occ = np.zeros(self.n)
        np.add.at(occ, path["states"], np.diff(times))
        return occ / path["t_end"]


def estimate_generator(times, states, t_end: float, n_states: int) -> dict:
    """MLE of Q from a fully observed path: Q_ij = N_ij / T_i (transitions i -> j over total time in i), with
    standard errors sqrt(N_ij) / T_i. The last sojourn is right-censored at t_end and enters T_i only."""
    times, states = np.asarray(times, float), np.asarray(states, int)
    T = np.zeros(n_states)
    np.add.at(T, states, np.diff(np.r_[times, t_end]))
    N = np.zeros((n_states, n_states))
    np.add.at(N, (states[:-1], states[1:]), 1)
    Q = np.divide(N, T[:, None], out=np.zeros_like(N), where=T[:, None] > 0)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    se = np.divide(np.sqrt(N), T[:, None], out=np.zeros_like(N), where=T[:, None] > 0)
    return {"Q": Q, "std_error": se, "counts": N, "occupation_time": T}


def birth_death(birth, death) -> dict:
    """Birth-death generator on 0..n with birth rates lambda_0..lambda_{n-1} and death rates mu_1..mu_n, and its
    stationary law pi_k proportional to prod_{i<k} lambda_i / mu_{i+1} (detailed balance)."""
    lam, mu = np.asarray(birth, float), np.asarray(death, float)
    n = len(lam)
    Q = np.zeros((n + 1, n + 1))
    for k in range(n):
        Q[k, k + 1] = lam[k]
        Q[k + 1, k] = mu[k]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    w = np.r_[1.0, np.cumprod(lam / mu)]
    return {"chain": CTMC(Q), "Q": Q, "stationary": w / w.sum()}


# --------------------------------------------------------------------------------------------- reaction networks


def gillespie(x0, stoich, propensity, t_end: float, rng, max_events: int = 10_000_000) -> dict:
    """Gillespie's direct method (1977): exact sample paths of a reaction network. stoich has one row per
    reaction (the change in each species); propensity(x) returns the reaction rates at state x."""
    x = np.array(x0, dtype=np.int64)
    S = np.asarray(stoich, dtype=np.int64)
    t = 0.0
    ts, xs = [0.0], [x.copy()]
    for _ in range(max_events):
        a = np.asarray(propensity(x), float)
        a0 = a.sum()
        if a0 <= 0:
            break
        t += rng.exponential(1.0 / a0)
        if t > t_end:
            break
        j = int(np.searchsorted(np.cumsum(a), rng.random() * a0, side="right"))
        x = x + S[min(j, len(a) - 1)]
        ts.append(t)
        xs.append(x.copy())
    return {"t": np.array(ts), "x": np.array(xs)}


def tau_leap(x0, stoich, propensity, t_end: float, tau: float, rng) -> dict:
    """Explicit tau-leaping (Gillespie 2001): each reaction fires Poisson(a_j tau) times per step; negative
    populations are clipped at zero (the method's known failure for small counts)."""
    x = np.array(x0, dtype=np.int64)
    S = np.asarray(stoich, dtype=np.int64)
    n = int(math.ceil(t_end / tau))
    xs = np.zeros((n + 1, len(x)), dtype=np.int64)
    xs[0] = x
    clipped = 0
    for k in range(n):
        a = np.maximum(np.asarray(propensity(x), float), 0.0)
        fires = rng.poisson(a * tau)
        x = x + fires @ S
        if np.any(x < 0):
            clipped += 1
            x = np.maximum(x, 0)
        xs[k + 1] = x
    return {"t": np.arange(n + 1) * tau, "x": xs, "clipped_steps": clipped}


def sample_path_at(t_events, x_events, t_grid) -> np.ndarray:
    """Values of a piecewise-constant path (jumps at t_events) on a time grid."""
    idx = np.searchsorted(t_events, t_grid, side="right") - 1
    return np.asarray(x_events)[np.maximum(idx, 0)]
