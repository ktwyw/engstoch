"""Random walks and branching processes.

Simple random walks in one and several dimensions; the gambler's ruin probability and expected duration in
closed form; the reflection principle (paths that touch a level, the distribution of the maximum) and the
ballot theorem; first return to the origin and the arcsine law for the last zero; Polya's recurrence in
d = 1, 2, 3 (return probability up to a horizon); and the Galton-Watson branching process: probability
generating functions of common offspring laws, the extinction probability as the smallest fixed point of the
pgf, the mean and variance of generation sizes, simulation and the total progeny.

>>> from engstoch import walks as wk
>>> round(wk.ruin_probability(3, 10, 0.5), 6)
0.7
>>> round(wk.extinction_probability(wk.pgf_poisson(1.5))["q"], 6)
0.417188
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------------------------- random walks


def simple_walk(n_steps: int, rng, p: float = 0.5, n_paths: int = 1, start: int = 0) -> np.ndarray:
    """Paths of the simple random walk with up-probability p: array (n_paths, n_steps + 1)."""
    steps = np.where(rng.random((n_paths, n_steps)) < p, 1, -1)
    return np.hstack([np.full((n_paths, 1), start), start + np.cumsum(steps, axis=1)])


def lattice_walk(n_steps: int, d: int, rng, n_paths: int = 1) -> np.ndarray:
    """Simple symmetric walk on Z^d: each step moves one coordinate by +-1. Array (n_paths, n_steps + 1, d)."""
    axis = rng.integers(0, d, (n_paths, n_steps))
    sign = rng.choice([-1, 1], (n_paths, n_steps))
    steps = np.zeros((n_paths, n_steps, d), dtype=int)
    np.put_along_axis(steps, axis[..., None], sign[..., None], axis=2)
    return np.concatenate([np.zeros((n_paths, 1, d), dtype=int), np.cumsum(steps, axis=1)], axis=1)


def ruin_probability(i: int, N: int, p: float) -> float:
    """Probability that a gambler starting with i reaches 0 before N, winning each bet with probability p:
    ((q/p)^i - (q/p)^N) / (1 - (q/p)^N), and 1 - i/N for the fair game."""
    q = 1 - p
    if abs(p - q) < 1e-14:
        return 1 - i / N
    r = q / p
    return (r**i - r**N) / (1 - r**N)


def ruin_duration(i: int, N: int, p: float) -> float:
    """Expected number of bets until the gambler reaches 0 or N: i (N - i) for the fair game, otherwise
    i/(q - p) - N/(q - p) (1 - (q/p)^i) / (1 - (q/p)^N)."""
    q = 1 - p
    if abs(p - q) < 1e-14:
        return i * (N - i)
    r = q / p
    return i / (q - p) - N / (q - p) * (1 - r**i) / (1 - r**N)


def paths_count(n: int, k: int) -> int:
    """Number of n-step +-1 paths from 0 to k: C(n, (n + k)/2) (zero if the parity is wrong)."""
    if (n + k) % 2 or abs(k) > n:
        return 0
    return math.comb(n, (n + k) // 2)


def reflection_touching(n: int, a: int, b: int) -> int:
    """Paths of length n from 0 to b that touch level a > max(0, b): by reflection, the number of paths from 0
    to 2a - b."""
    return paths_count(n, 2 * a - b)


def max_distribution(n: int, a: int) -> float:
    """P(max_{k <= n} S_k >= a) for the symmetric walk: P(S_n >= a) + P(S_n > a) by the reflection principle."""
    tot = 2**n
    ge = sum(paths_count(n, k) for k in range(a, n + 1))
    gt = sum(paths_count(n, k) for k in range(a + 1, n + 1))
    return (ge + gt) / tot


def ballot(a: int, b: int) -> float:
    """Probability that candidate A (a votes) stays strictly ahead of B (b < a votes) throughout the count:
    (a - b) / (a + b)."""
    return (a - b) / (a + b)


def first_return_probability(n: int) -> float:
    """P(first return to 0 at step 2n) = C(2n, n) / ((2n - 1) 4^n) for the symmetric walk."""
    return math.exp(math.lgamma(2 * n + 1) - 2 * math.lgamma(n + 1) - n * math.log(4.0)) / (2 * n - 1)


def arcsine_cdf(x):
    """Limit law of the fraction of time the walk is positive (or of the last zero before time n):
    (2 / pi) arcsin(sqrt(x))."""
    return 2.0 / np.pi * np.arcsin(np.sqrt(np.asarray(x, float)))


def return_probability_by(d: int, n_steps: int, n_paths: int, rng) -> dict:
    """Fraction of walks on Z^d that revisit the origin within n_steps (Polya: tends to 1 for d <= 2 and to
    1 - 1/u_d ~ 0.3405 for d = 3, slowly)."""
    hit = np.zeros(n_paths, dtype=bool)
    chunk = 2000
    pos = np.zeros((n_paths, d), dtype=np.int64)
    for start in range(0, n_steps, chunk):
        m = min(chunk, n_steps - start)
        axis = rng.integers(0, d, (n_paths, m))
        sign = rng.choice(np.array([-1, 1]), (n_paths, m))
        steps = np.zeros((n_paths, m, d), dtype=np.int64)
        np.put_along_axis(steps, axis[..., None], sign[..., None], axis=2)
        traj = pos[:, None, :] + np.cumsum(steps, axis=1)
        hit |= np.any(np.all(traj == 0, axis=2), axis=1)
        pos = traj[:, -1, :]
    return {"fraction": float(hit.mean()), "std_error": float(math.sqrt(hit.mean() * (1 - hit.mean()) / n_paths))}


POLYA_3D = 0.340537329550999  # return probability of the simple walk on Z^3 (Watson's integral)

# --------------------------------------------------------------------------------------------- branching


def pgf_poisson(lam: float):
    """Offspring pgf of Poisson(lam): G(s) = exp(lam (s - 1)); also returns its mean and variance."""
    return {
        "G": lambda s: np.exp(lam * (np.asarray(s, float) - 1.0)),
        "mean": lam,
        "var": lam,
        "pmf": lambda k: _pois_pmf(k, lam),
    }


def _pois_pmf(k, lam):
    k = np.asarray(k)
    return np.exp(-lam + k * math.log(lam) - np.vectorize(math.lgamma)(k + 1.0))


def pgf_geometric(p: float):
    """Offspring P(k) = p (1 - p)^k, k = 0, 1, ...: G(s) = p / (1 - (1 - p) s); extinction probability
    min(1, p / (1 - p))."""
    q = 1 - p
    return {
        "G": lambda s: p / (1 - q * np.asarray(s, float)),
        "mean": q / p,
        "var": q / p**2,
        "pmf": lambda k: p * q ** np.asarray(k),
    }


def pgf_discrete(probs):
    """Offspring law with P(k) = probs[k]."""
    pr = np.asarray(probs, float)
    k = np.arange(len(pr))
    mean = float(np.sum(k * pr))
    var = float(np.sum(k * k * pr) - mean**2)
    return {
        "G": lambda s: np.polyval(pr[::-1], np.asarray(s, float)),
        "mean": mean,
        "var": var,
        "pmf": lambda j: np.where(np.asarray(j) < len(pr), pr[np.minimum(np.asarray(j), len(pr) - 1)], 0.0),
    }


def extinction_probability(offspring, tol: float = 1e-14, max_iter: int = 1_000_000) -> dict:
    """Smallest root of G(s) = s in [0, 1], by the iteration q_{n+1} = G(q_n) from q_0 = 0, which is exactly
    P(extinct by generation n). Converges linearly at rate G'(q) (slowly near criticality)."""
    G = offspring["G"]
    q = 0.0
    for n in range(1, max_iter + 1):
        q_new = float(G(q))
        if abs(q_new - q) < tol:
            return {"q": q_new, "iterations": n}
        q = q_new
    return {"q": q, "iterations": max_iter}


def extinction_by_generation(offspring, n: int) -> np.ndarray:
    """P(Z_k = 0) for k = 0..n: the iterates of the pgf at 0."""
    G = offspring["G"]
    out = [0.0]
    for _ in range(n):
        out.append(float(G(out[-1])))
    return np.array(out)


def generation_moments(offspring, n: int) -> dict:
    """E[Z_n] = m^n and Var[Z_n] = sigma^2 m^(n-1) (m^n - 1)/(m - 1) (n sigma^2 when m = 1), from Z_0 = 1."""
    m, s2 = offspring["mean"], offspring["var"]
    var = n * s2 if abs(m - 1) < 1e-14 else s2 * m ** (n - 1) * (m**n - 1) / (m - 1)
    return {"mean": m**n, "var": var}


def simulate_galton_watson(offspring_sampler, n_generations: int, rng, z0: int = 1, cap: int = 10_000) -> np.ndarray:
    """Generation sizes Z_0..Z_n; offspring_sampler(k, rng) returns the offspring counts of k individuals.
    Once the population exceeds `cap` it is frozen (extinction from there has probability q^cap, negligible for a
    supercritical process), which keeps simulation of exploding lines cheap."""
    z = [z0]
    for _ in range(n_generations):
        k = z[-1]
        if k == 0 or k > cap:
            z.append(0 if k == 0 else k)
            continue
        z.append(int(np.sum(offspring_sampler(k, rng))))
    return np.array(z)


def total_progeny_mean(offspring) -> float:
    """Expected total number of individuals ever born (including the ancestor) for a subcritical process:
    1 / (1 - m)."""
    m = offspring["mean"]
    return 1.0 / (1.0 - m) if m < 1 else math.inf
