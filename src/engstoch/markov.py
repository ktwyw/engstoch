"""Discrete-time Markov chains on a finite state space.

A ``MarkovChain`` holds a row-stochastic matrix P and state labels and computes: n-step transition matrices
and distributions; the communicating classes (Tarjan's algorithm on the transition graph), which are closed
(recurrent) or open (transient), and the period of each class; the stationary distribution (a linear solve,
and power iteration for comparison); reversibility (detailed balance); absorption by the fundamental matrix
N = (I - Q)^-1 (absorption probabilities, expected times and their variances); hitting probabilities and
expected hitting times of any set; mean first-passage times (Kemeny-Snell); total-variation mixing and the
spectral gap; simulation; and maximum-likelihood estimation of P from an observed path with standard errors.

>>> import numpy as np
>>> from engstoch import markov as mk
>>> mc = mk.MarkovChain([[0.9, 0.1], [0.5, 0.5]], ["dry", "wet"])
>>> mc.stationary().round(4).tolist()
[0.8333, 0.1667]
"""

from __future__ import annotations

import math
from functools import reduce

import numpy as np


class MarkovChain:
    """A finite, time-homogeneous Markov chain with transition matrix P (rows sum to one)."""

    def __init__(self, P, states=None, tol: float = 1e-10):
        P = np.asarray(P, float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be square")
        if np.any(P < -tol) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("P must be non-negative with rows summing to one")
        self.P = np.clip(P, 0.0, None)
        self.n = len(P)
        self.states = list(states) if states is not None else list(range(self.n))
        self.index = {s: i for i, s in enumerate(self.states)}

    def __repr__(self) -> str:
        return f"MarkovChain({self.n} states: {self.states[:6]}{' ...' if self.n > 6 else ''})"

    # ------------------------------------------------------------------ transient behaviour
    def n_step(self, k: int) -> np.ndarray:
        """P^k by repeated squaring."""
        return np.linalg.matrix_power(self.P, k)

    def distribution(self, p0, k: int) -> np.ndarray:
        """Distribution after k steps from the initial distribution p0 (a row vector)."""
        p = np.asarray(p0, float)
        for _ in range(k):
            p = p @ self.P
        return p

    # ------------------------------------------------------------------ structure
    def classes(self) -> list[list[int]]:
        """Communicating classes (strongly connected components of the graph i -> j when P_ij > 0), by Tarjan."""
        adj = [np.flatnonzero(self.P[i] > 0) for i in range(self.n)]
        index, low, on, stack, out = {}, {}, set(), [], []
        counter = [0]

        def strong(v):
            work = [(v, 0)]
            index[v] = low[v] = counter[0]
            counter[0] += 1
            stack.append(v)
            on.add(v)
            while work:
                node, i = work[-1]
                if i < len(adj[node]):
                    work[-1] = (node, i + 1)
                    w = int(adj[node][i])
                    if w not in index:
                        index[w] = low[w] = counter[0]
                        counter[0] += 1
                        stack.append(w)
                        on.add(w)
                        work.append((w, 0))
                    elif w in on:
                        low[node] = min(low[node], index[w])
                else:
                    work.pop()
                    if work:
                        low[work[-1][0]] = min(low[work[-1][0]], low[node])
                    if low[node] == index[node]:
                        comp = []
                        while True:
                            w = stack.pop()
                            on.discard(w)
                            comp.append(w)
                            if w == node:
                                break
                        out.append(sorted(comp))

        for v in range(self.n):
            if v not in index:
                strong(v)
        return sorted(out, key=lambda c: c[0])

    def classify(self) -> dict:
        """Each class with its type (closed classes are recurrent, open ones transient in a finite chain) and
        its period."""
        res = []
        for c in self.classes():
            cs = set(c)
            closed = all(set(np.flatnonzero(self.P[i] > 0)) <= cs for i in c)
            res.append({"states": [self.states[i] for i in c], "recurrent": closed, "period": self._period(c)})
        return {
            "classes": res,
            "irreducible": len(res) == 1,
            "aperiodic": all(r["period"] == 1 for r in res if r["recurrent"]),
        }

    def _period(self, c: list[int]) -> int:
        """gcd of (level(i) + 1 - level(j)) over edges i -> j inside the class, levels from a BFS."""
        cs = set(c)
        level = {c[0]: 0}
        queue = [c[0]]
        g = 0
        while queue:
            i = queue.pop(0)
            for j in np.flatnonzero(self.P[i] > 0):
                j = int(j)
                if j not in cs:
                    continue
                if j not in level:
                    level[j] = level[i] + 1
                    queue.append(j)
                else:
                    g = math.gcd(g, level[i] + 1 - level[j])
        return abs(g) if g else 1 if len(c) > 1 or self.P[c[0], c[0]] > 0 else 0

    def is_irreducible(self) -> bool:
        return len(self.classes()) == 1

    # ------------------------------------------------------------------ long run
    def stationary(self) -> np.ndarray:
        """A stationary distribution pi P = pi, sum pi = 1, from the linear system with one balance equation
        replaced by the normalisation (unique for an irreducible chain; for a reducible one, the solution
        supported on the first recurrent class)."""
        rec = [c for c in self.classify()["classes"] if c["recurrent"]]
        idx = [self.index[s] for s in rec[0]["states"]]
        Pc = self.P[np.ix_(idx, idx)]
        m = len(idx)
        A = Pc.T - np.eye(m)
        A[-1] = 1.0
        b = np.zeros(m)
        b[-1] = 1.0
        pi = np.zeros(self.n)
        pi[idx] = np.linalg.solve(A, b)
        return pi

    def stationary_power(self, tol: float = 1e-13, max_iter: int = 100_000) -> dict:
        """Power iteration p <- p P from the uniform distribution; converges at the rate of |lambda_2|."""
        p = np.full(self.n, 1.0 / self.n)
        for k in range(1, max_iter + 1):
            q = p @ self.P
            if np.max(np.abs(q - p)) < tol:
                return {"pi": q, "iterations": k}
            p = q
        return {"pi": p, "iterations": max_iter}

    def is_reversible(self, tol: float = 1e-10) -> bool:
        """Detailed balance pi_i P_ij = pi_j P_ji."""
        F = self.stationary()[:, None] * self.P
        return bool(np.max(np.abs(F - F.T)) < tol)

    def mean_return_times(self) -> np.ndarray:
        """Kac's formula: the mean return time to state i is 1 / pi_i (irreducible chains)."""
        return 1.0 / self.stationary()

    def fundamental_matrix_ergodic(self) -> np.ndarray:
        """Kemeny-Snell fundamental matrix Z = (I - P + 1 pi)^-1 of an irreducible chain."""
        pi = self.stationary()
        return np.linalg.inv(np.eye(self.n) - self.P + np.outer(np.ones(self.n), pi))

    def mean_first_passage(self) -> np.ndarray:
        """Matrix of mean first-passage times m_ij (m_ii = mean return time), m_ij = (z_jj - z_ij) / pi_j."""
        pi = self.stationary()
        Z = self.fundamental_matrix_ergodic()
        M = (np.diag(Z)[None, :] - Z) / pi[None, :]
        return M + np.diag(1.0 / pi)

    def kemeny_constant(self) -> float:
        """sum_j pi_j m_ij, the same for every starting state i: trace(Z) - 1."""
        return float(np.trace(self.fundamental_matrix_ergodic()) - 1.0)

    def spectral_gap(self) -> dict:
        """Eigenvalues by modulus; the gap 1 - |lambda_2| sets the geometric rate of convergence."""
        lam = np.linalg.eigvals(self.P)
        lam = lam[np.argsort(-np.abs(lam))]
        return {
            "eigenvalues": lam,
            "slem": float(np.abs(lam[1])),
            "gap": float(1 - np.abs(lam[1])),
            "relaxation_time": float(1 / (1 - np.abs(lam[1]))),
        }

    def tv_distance(self, k: int) -> float:
        """Worst-case total-variation distance max_i ||P^k(i, .) - pi||_TV after k steps."""
        pi = self.stationary()
        return float(0.5 * np.max(np.sum(np.abs(self.n_step(k) - pi), axis=1)))

    def mixing_time(self, eps: float = 0.25, k_max: int = 100_000) -> int:
        """Smallest k with worst-case TV distance <= eps (computed by repeated multiplication)."""
        pi = self.stationary()
        Pk = np.eye(self.n)
        for k in range(1, k_max + 1):
            Pk = Pk @ self.P
            if 0.5 * np.max(np.sum(np.abs(Pk - pi), axis=1)) <= eps:
                return k
        raise RuntimeError("not mixed within k_max steps (periodic or reducible?)")

    # ------------------------------------------------------------------ absorption and hitting
    def absorption(self) -> dict:
        """Absorbing-chain analysis with transient states T and absorbing states A (P_aa = 1): fundamental matrix
        N = (I - Q)^-1, absorption probabilities B = N R, expected steps to absorption t = N 1 and their
        variance (2N - I) t - t^2."""
        absorbing = [i for i in range(self.n) if self.P[i, i] == 1.0]
        transient = [i for i in range(self.n) if i not in absorbing]
        Q = self.P[np.ix_(transient, transient)]
        R = self.P[np.ix_(transient, absorbing)]
        N = np.linalg.inv(np.eye(len(transient)) - Q)
        t = N @ np.ones(len(transient))
        var = (2 * N - np.eye(len(transient))) @ t - t * t
        return {
            "transient": [self.states[i] for i in transient],
            "absorbing": [self.states[i] for i in absorbing],
            "N": N,
            "B": N @ R,
            "expected_steps": t,
            "variance_steps": var,
        }

    def hitting_probabilities(self, target) -> np.ndarray:
        """h_i = P(ever reach the target set from i): the minimal non-negative solution of h = P h off the target,
        h = 1 on it (states that cannot reach the target get 0)."""
        T = {self.index[s] for s in target}
        can = set(T)
        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                if i not in can and any(self.P[i, j] > 0 for j in can):
                    can.add(i)
                    changed = True
        h = np.zeros(self.n)
        h[list(T)] = 1.0
        rest = [i for i in can if i not in T]
        if rest:
            A = np.eye(len(rest)) - self.P[np.ix_(rest, rest)]
            b = self.P[np.ix_(rest, list(T))].sum(axis=1)
            h[rest] = np.linalg.solve(A, b)
        return h

    def expected_hitting_times(self, target) -> np.ndarray:
        """k_i = E[steps to reach the target set from i] (inf where the target is not reached with probability 1)."""
        T = [self.index[s] for s in target]
        h = self.hitting_probabilities(target)
        k = np.full(self.n, np.inf)
        k[T] = 0.0
        rest = [i for i in range(self.n) if i not in T and h[i] > 1 - 1e-12]
        if rest:
            A = np.eye(len(rest)) - self.P[np.ix_(rest, rest)]
            k[rest] = np.linalg.solve(A, np.ones(len(rest)))
        return k

    # ------------------------------------------------------------------ simulation and estimation
    def simulate(self, n_steps: int, start, rng) -> np.ndarray:
        """A path of n_steps + 1 state indices from `start` (a label), by inversion of each row's cdf."""
        cdf = np.cumsum(self.P, axis=1)
        cdf[:, -1] = 1.0
        x = np.empty(n_steps + 1, dtype=int)
        x[0] = self.index[start]
        u = rng.random(n_steps)
        for k in range(n_steps):
            x[k + 1] = int(np.searchsorted(cdf[x[k]], u[k], side="right"))
        return x

    def labels(self, path) -> list:
        return [self.states[i] for i in path]


def estimate(path, n_states: int | None = None, states=None) -> dict:
    """Maximum-likelihood estimate P_ij = n_ij / n_i from one observed path of state indices, with the
    binomial standard errors sqrt(P_ij (1 - P_ij) / n_i) and the transition counts."""
    path = np.asarray(path, int)
    m = n_states or int(path.max()) + 1
    counts = np.zeros((m, m))
    np.add.at(counts, (path[:-1], path[1:]), 1)
    ni = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, ni, out=np.full_like(counts, 1.0 / m), where=ni > 0)
    se = np.sqrt(np.divide(P * (1 - P), ni, out=np.zeros_like(P), where=ni > 0))
    return {"P": P, "std_error": se, "counts": counts, "chain": MarkovChain(P, states)}


def order_test(path, n_states: int) -> dict:
    """Likelihood-ratio test of a first-order chain against a second-order one (Anderson and Goodman 1957):
    G^2 = 2 sum n_ijk ln(p_ijk / p_jk), chi-square with m (m - 1)^2 degrees of freedom."""
    from .special import chi2_sf

    path = np.asarray(path, int)
    m = n_states
    n3 = np.zeros((m, m, m))
    np.add.at(n3, (path[:-2], path[1:-1], path[2:]), 1)
    n_ij = n3.sum(axis=2, keepdims=True)
    n_jk = n3.sum(axis=0, keepdims=True)
    n_j = n3.sum(axis=(0, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n3 > 0, (n3 / n_ij) / (n_jk / n_j), 1.0)
    g2 = float(2 * np.sum(n3 * np.log(ratio)))
    df = m * (m - 1) ** 2
    return {"G2": g2, "df": df, "p_value": float(chi2_sf(g2, df))}


def random_walk_on_graph(adjacency) -> MarkovChain:
    """Simple random walk on an undirected graph: P_ij = a_ij / deg(i); stationary pi_i = deg(i) / 2|E|."""
    A = np.asarray(adjacency, float)
    return MarkovChain(A / A.sum(axis=1, keepdims=True))


def pagerank(adjacency, damping: float = 0.85) -> np.ndarray:
    """PageRank as the stationary distribution of the damped random surfer on a directed graph (dangling pages
    jump uniformly)."""
    A = np.asarray(adjacency, float)
    n = len(A)
    out = A.sum(axis=1, keepdims=True)
    S = np.where(out > 0, A / np.where(out > 0, out, 1), 1.0 / n)
    G = damping * S + (1 - damping) / n
    return MarkovChain(G).stationary()


def lumpable(P, partition) -> bool:
    """Strong lumpability (Kemeny-Snell): for every block pair, P(i -> block) is the same for all i in a block."""
    P = np.asarray(P, float)
    for blk in partition:
        for other in partition:
            v = P[np.ix_(blk, other)].sum(axis=1)
            if np.ptp(v) > 1e-10:
                return False
    return True


def product(*chains: MarkovChain) -> MarkovChain:
    """Independent chains run together: the Kronecker product of their transition matrices."""
    return MarkovChain(reduce(np.kron, [c.P for c in chains]))
