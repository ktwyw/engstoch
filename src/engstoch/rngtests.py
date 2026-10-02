"""Statistical tests of random number generators and of samples against a distribution: chi-square
goodness of fit, Kolmogorov-Smirnov, the serial (pairs) test, the runs-up test with Knuth's covariance
matrix, the gap test, the lattice structure of linear congruential generators (the 2-D spectral test by
Gauss reduction, and the planes of RANDU), and Lag-k autocorrelation.

Every test returns a dict with the statistic and a p-value; a good generator gives p-values that are
themselves uniform over many runs, which `second_level` checks.

>>> import numpy as np
>>> from engstoch import rngtests as rt
>>> u = np.random.default_rng(0).random(10_000)
>>> bool(rt.chi_square_uniform(u)["p_value"] > 0.001)
True
"""

from __future__ import annotations

import math

import numpy as np

from .special import chi2_sf, ks_sf, norm_cdf


def chi_square(observed, expected, ddof: int = 0) -> dict:
    """Pearson's chi-square statistic sum (O - E)^2 / E with k - 1 - ddof degrees of freedom."""
    o, e = np.asarray(observed, float), np.asarray(expected, float)
    stat = float(np.sum((o - e) ** 2 / e))
    df = len(o) - 1 - ddof
    return {"statistic": stat, "df": df, "p_value": float(chi2_sf(stat, df))}


def chi_square_uniform(u, bins: int = 100) -> dict:
    """Equal-width chi-square test of uniformity on [0, 1)."""
    counts = np.bincount(np.minimum((np.asarray(u) * bins).astype(int), bins - 1), minlength=bins)
    return chi_square(counts, np.full(bins, len(u) / bins))


def ks_test(x, cdf) -> dict:
    """One-sample Kolmogorov-Smirnov test of x against the continuous distribution function `cdf`."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    F = np.asarray(cdf(x), float)
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
    return {"statistic": d, "p_value": float(ks_sf(d, n))}


def serial_test(u, d: int = 10) -> dict:
    """Non-overlapping pairs (u_2k, u_2k+1) counted on a d x d grid: chi-square with d^2 - 1 df."""
    u = np.asarray(u, float)
    m = len(u) // 2
    a = np.minimum((u[: 2 * m : 2] * d).astype(int), d - 1)
    b = np.minimum((u[1 : 2 * m : 2] * d).astype(int), d - 1)
    counts = np.bincount(a * d + b, minlength=d * d)
    return chi_square(counts, np.full(d * d, m / d**2))


_KNUTH_A = np.array(
    [
        [4529.4, 9044.9, 13568, 18091, 22615, 27892],
        [9044.9, 18097, 27139, 36187, 45234, 55789],
        [13568, 27139, 40721, 54281, 67852, 83685],
        [18091, 36187, 54281, 72414, 90470, 111580],
        [22615, 45234, 67852, 90470, 113262, 139476],
        [27892, 55789, 83685, 111580, 139476, 172860],
    ]
)
_KNUTH_B = np.array([1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840])


def runs_up_test(u) -> dict:
    """Knuth's runs-up test (TAOCP vol. 2, 3.3.2 G): counts of ascending runs of length 1..5 and >= 6, with the
    statistic V = (count - n b)' A (count - n b) / n, asymptotically chi-square with 6 df."""
    u = np.asarray(u, float)
    counts = np.zeros(6)
    run = 1
    for k in range(1, len(u)):
        if u[k] > u[k - 1]:
            run += 1
        else:
            counts[min(run, 6) - 1] += 1
            run = 1
    counts[min(run, 6) - 1] += 1
    n = len(u)
    r = counts - n * _KNUTH_B
    v = float(r @ _KNUTH_A @ r / n)
    return {"statistic": v, "df": 6, "counts": counts, "p_value": float(chi2_sf(v, 6))}


def gap_test(u, a: float = 0.0, b: float = 0.5, t: int = 10) -> dict:
    """Knuth's gap test: lengths of gaps between visits to [a, b), geometric with p = b - a, pooled at t."""
    u = np.asarray(u, float)
    hits = np.flatnonzero((u >= a) & (u < b))
    gaps = np.diff(hits) - 1
    p = b - a
    counts = np.bincount(np.minimum(gaps, t), minlength=t + 1)
    probs = np.array([p * (1 - p) ** r for r in range(t)] + [(1 - p) ** t])
    return chi_square(counts, probs * len(gaps))


def autocorrelation_test(u, lag: int = 1) -> dict:
    """Lag-k serial correlation of uniforms; under independence sqrt(n) r is approximately N(0, 1)."""
    u = np.asarray(u, float) - 0.5
    r = float(np.sum(u[:-lag] * u[lag:]) / np.sum(u * u))
    z = r * math.sqrt(len(u) - lag)
    return {"statistic": r, "z": z, "p_value": float(2 * (1 - norm_cdf(abs(z))))}


def second_level(test, generator, n_runs: int = 50, n: int = 10_000) -> dict:
    """Run a test on n_runs independent blocks and KS-test the p-values for uniformity (a 'test of the test')."""
    p = np.array([test(generator(n))["p_value"] for _ in range(n_runs)])
    return {"p_values": p, **ks_test(p, lambda x: np.clip(x, 0, 1))}


def spectral_test_2d(a: int, m: int) -> dict:
    """The 2-D spectral test of the multiplicative LCG x -> a x mod m: the pairs (x_k, x_{k+1}) lie on lines,
    the dual lattice {(s1, s2): s1 + a s2 = 0 mod m} has shortest vector nu_2 (found by Gauss-Lagrange reduction)
    and the lines are 1/nu_2 apart. Returns nu_2 and the figure of merit nu_2 / (gamma_2 m)^(1/2) in (0, 1]."""
    u, v = np.array([m, 0], dtype=object), np.array([-a, 1], dtype=object)

    def n2(w):
        return int(w[0]) ** 2 + int(w[1]) ** 2

    if n2(u) < n2(v):
        u, v = v, u
    while True:
        dot = int(u[0]) * int(v[0]) + int(u[1]) * int(v[1])
        q = round(dot / n2(v))
        u = u - q * v
        if n2(u) >= n2(v):
            break
        u, v = v, u
    nu = math.sqrt(n2(v))
    gamma2 = 2 / math.sqrt(3)
    return {"nu": nu, "line_spacing": 1 / nu, "merit": nu / math.sqrt(gamma2 * m), "vector": (int(v[0]), int(v[1]))}


def randu_planes(u) -> dict:
    """RANDU satisfies x_{k+2} = 6 x_{k+1} - 9 x_k (mod 2^31): 9u_k - 6u_{k+1} + u_{k+2} is an integer, so the
    triples lie on at most 15 parallel planes. Returns the distinct integer values found."""
    u = np.asarray(u, float)
    s = 9 * u[:-2] - 6 * u[1:-1] + u[2:]
    vals = np.unique(np.round(s).astype(int))
    return {"plane_values": vals, "n_planes": len(vals), "max_deviation": float(np.max(np.abs(s - np.round(s))))}
