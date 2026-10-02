"""Quasi-Monte Carlo: low-discrepancy point sets and their error estimates.

The van der Corput sequence and Halton points (with random-shift and permutation scrambling), Sobol points
by Gray-code construction with the Joe-Kuo direction numbers (up to 10 dimensions), rank-1 lattice rules
(Korobov generators, with a search for a good one), the L2-star discrepancy by Warnock's formula, and
randomised QMC: independent random shifts (Cranley-Patterson rotations) of one point set give an unbiased
estimate and an honest standard error.

>>> from engstoch import qmc
>>> [float(v) for v in qmc.van_der_corput(5)]
[0.0, 0.5, 0.25, 0.75, 0.125]
"""

from __future__ import annotations

import math

import numpy as np

from .montecarlo import mean_ci

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Joe and Kuo (2008), new-joe-kuo-6.21201: (degree s, coefficient a, initial direction numbers m) for d = 2..10
_JOE_KUO = [
    (1, 0, [1]),
    (2, 1, [1, 3]),
    (3, 1, [1, 3, 1]),
    (3, 2, [1, 1, 1]),
    (4, 1, [1, 1, 3, 3]),
    (4, 4, [1, 3, 5, 13]),
    (5, 2, [1, 1, 5, 5, 17]),
    (5, 4, [1, 1, 5, 5, 5]),
    (5, 7, [1, 1, 7, 11, 19]),
]


def radical_inverse(i, base: int = 2, perm=None) -> np.ndarray:
    """phi_b(i): the base-b digits of i mirrored about the radix point, optionally digit-permuted."""
    i = np.atleast_1d(np.asarray(i, dtype=np.int64)).copy()
    out = np.zeros(len(i))
    f = 1.0 / base
    while np.any(i > 0):
        d = i % base
        out += f * (d if perm is None else np.asarray(perm)[d])
        i //= base
        f /= base
    return out


def van_der_corput(n: int, base: int = 2, start: int = 0) -> np.ndarray:
    """First n points of the van der Corput sequence in the given base."""
    return radical_inverse(np.arange(start, start + n), base)


def halton(n: int, d: int, start: int = 0, rng=None, scramble: str | None = None) -> np.ndarray:
    """Halton points: the van der Corput sequences in the first d prime bases. scramble='permute' applies a
    random digit permutation per dimension (keeping 0 fixed), which breaks the correlations between high bases."""
    idx = np.arange(start, start + n)
    cols = []
    for j in range(d):
        b = PRIMES[j]
        perm = None
        if scramble == "permute":
            perm = np.r_[0, 1 + rng.permutation(b - 1)]
        cols.append(radical_inverse(idx, b, perm))
    return np.column_stack(cols)


def _sobol_directions(d: int, bits: int = 30) -> np.ndarray:
    V = np.zeros((d, bits), dtype=np.int64)
    V[0] = [1 << (bits - 1 - k) for k in range(bits)]
    for j in range(1, d):
        s, a, m = _JOE_KUO[j - 1]
        v = [m[k] << (bits - 1 - k) for k in range(s)]
        for k in range(s, bits):
            val = v[k - s] ^ (v[k - s] >> s)
            for r in range(1, s):
                if (a >> (s - 1 - r)) & 1:
                    val ^= v[k - r]
            v.append(val)
        V[j] = v[:bits]
    return V


def sobol(n: int, d: int, skip_zero: bool = False) -> np.ndarray:
    """Sobol points (Gray-code order, as SciPy's unscrambled Sobol) for d <= 10. The first point is the origin;
    use n a power of two to keep the (t, m, s)-net balance."""
    if d > len(_JOE_KUO) + 1:
        raise ValueError("direction numbers are included for up to 10 dimensions")
    bits = 30
    V = _sobol_directions(d, bits)
    X = np.zeros((n + int(skip_zero), d), dtype=np.int64)
    x = np.zeros(d, dtype=np.int64)
    for i in range(1, n + int(skip_zero)):
        c = ((i - 1) ^ i).bit_length() - 1  # index of the lowest zero bit of i - 1
        x = x ^ V[:, c]
        X[i] = x
    out = X / float(1 << bits)
    return out[1:] if skip_zero else out


def korobov(n: int, d: int, a: int) -> np.ndarray:
    """Rank-1 lattice {i z / n mod 1} with the Korobov generating vector z = (1, a, a^2, ...) mod n."""
    z = np.array([pow(a, j, n) for j in range(d)], dtype=np.int64)
    i = np.arange(n, dtype=np.int64)[:, None]
    return (i * z % n) / n


def l2_star_discrepancy(x) -> float:
    """Warnock's closed form of the L2-star discrepancy of the points x (n, d) in [0, 1]^d."""
    x = np.asarray(x, float)
    n, d = x.shape
    t1 = 3.0**-d
    t2 = 2.0 ** (1 - d) / n * np.sum(np.prod(1 - x**2, axis=1))
    mx = np.maximum(x[:, None, :], x[None, :, :])
    t3 = np.sum(np.prod(1 - mx, axis=2)) / n**2
    return float(math.sqrt(max(t1 - t2 + t3, 0.0)))


def search_korobov(n: int, d: int, alpha: int = 2) -> dict:
    """Search a in 1..n-1 for the Korobov lattice with the smallest P_alpha criterion (the worst-case error in a
    weighted Korobov space with smoothness alpha = 2, computed with the Bernoulli polynomial B_2)."""
    best = None
    for a in range(1, n):
        if math.gcd(a, n) != 1:
            continue
        x = korobov(n, d, a)
        b2 = x * x - x + 1.0 / 6.0
        p = float(np.mean(np.prod(1 + 2 * math.pi**2 * b2, axis=1)) - 1.0)
        if best is None or p < best[1]:
            best = (a, p)
    return {"a": best[0], "criterion": best[1]}


def randomised_qmc(f, points, n_shifts: int, rng) -> dict:
    """Randomised QMC by Cranley-Patterson rotations: f averaged over (points + shift) mod 1 for n_shifts
    independent uniform shifts; the spread of the n_shifts averages gives an unbiased standard error."""
    pts = np.asarray(points, float)
    means = np.array([np.mean(f(np.mod(pts + rng.random(pts.shape[1]), 1.0))) for _ in range(n_shifts)])
    r = mean_ci(means, method="t")
    return r | {"replicates": means, "n_evaluations": n_shifts * len(pts)}


def genz_product_peak(x, c=None, w=None) -> np.ndarray:
    """Genz's product-peak test integrand prod 1/(c_j^-2 + (x_j - w_j)^2); its integral is known in closed form."""
    x = np.asarray(x, float)
    d = x.shape[1]
    c = np.full(d, 2.0) if c is None else np.asarray(c, float)
    w = np.full(d, 0.5) if w is None else np.asarray(w, float)
    return np.prod(1.0 / (c**-2.0 + (x - w) ** 2), axis=1)


def genz_product_peak_integral(d: int, c=None, w=None) -> float:
    """Exact integral of `genz_product_peak` over [0, 1]^d: prod c_j (atan(c_j (1 - w_j)) + atan(c_j w_j))."""
    c = np.full(d, 2.0) if c is None else np.asarray(c, float)
    w = np.full(d, 0.5) if w is None else np.asarray(w, float)
    return float(np.prod(c * (np.arctan(c * (1 - w)) + np.arctan(c * w))))
