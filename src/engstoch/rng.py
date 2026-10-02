"""Pseudo-random number generators and the generation of random variates.

Generators: linear congruential (with the infamous RANDU and the Park-Miller minimal standard), xorshift64*
and PCG32 (the O'Neill permuted congruential generator), all as small classes with a ``random(n)`` method
returning uniforms on [0, 1). Variates: inversion (continuous and discrete), acceptance-rejection with its
acceptance count, Box-Muller and Marsaglia's polar method, ratio of uniforms, Walker/Vose alias tables,
Poisson and binomial by inversion, gamma by Marsaglia-Tsang, beta from gammas, the multivariate normal by
Cholesky, and a Gaussian copula.

Every variate generator takes a NumPy ``Generator`` (or anything with a ``random(n)`` method), so the same
algorithms can be driven by the course's own generators.

>>> from engstoch import rng
>>> g = rng.LCG.park_miller(seed=1)
>>> [int(v) for v in g.integers(3)]
[16807, 282475249, 1622650073]
"""

from __future__ import annotations

import math

import numpy as np

from .special import norm_cdf

# --------------------------------------------------------------------------------------------- generators


class LCG:
    """Linear congruential generator x_{k+1} = (a x_k + c) mod m, uniforms x_k / m."""

    def __init__(self, a: int, c: int, m: int, seed: int = 1):
        self.a, self.c, self.m, self.state = int(a), int(c), int(m), int(seed) % int(m)

    @classmethod
    def park_miller(cls, seed: int = 1) -> LCG:
        """The 'minimal standard' multiplicative generator a = 16807, m = 2^31 - 1 (Park and Miller 1988)."""
        return cls(16807, 0, 2**31 - 1, seed)

    @classmethod
    def randu(cls, seed: int = 1) -> LCG:
        """IBM's RANDU, a = 65539, m = 2^31: consecutive triples lie on 15 planes (Marsaglia 1968)."""
        return cls(65539, 0, 2**31, seed)

    def integers(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        x, a, c, m = self.state, self.a, self.c, self.m
        for i in range(n):
            x = (a * x + c) % m
            out[i] = x
        self.state = x
        return out

    def random(self, n: int | None = None):
        if n is None:
            return float(self.integers(1)[0] / self.m)
        return self.integers(n) / self.m

    def period(self, limit: int = 10**7) -> int | None:
        """Length of the cycle from the current state (None if longer than `limit`); the state is unchanged."""
        x0 = x = self.state
        for k in range(1, limit + 1):
            x = (self.a * x + self.c) % self.m
            if x == x0:
                return k
        return None


def full_period_conditions(a: int, c: int, m: int) -> bool:
    """Hull-Dobell theorem: an LCG with c != 0 has full period m iff gcd(c, m) = 1, a - 1 is divisible by every
    prime factor of m, and a - 1 is divisible by 4 when m is."""
    if c == 0 or math.gcd(c, m) != 1:
        return False
    n, p, primes = m, 2, set()
    while p * p <= n:
        while n % p == 0:
            primes.add(p)
            n //= p
        p += 1
    if n > 1:
        primes.add(n)
    return all((a - 1) % q == 0 for q in primes) and ((a - 1) % 4 == 0 if m % 4 == 0 else True)


_MASK64 = (1 << 64) - 1


class XorShift64Star:
    """Marsaglia's xorshift64 with Vigna's multiplicative output scrambler (xorshift64*); 53-bit uniforms."""

    def __init__(self, seed: int = 88172645463325252):
        self.state = (int(seed) & _MASK64) or 1

    def next64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & _MASK64
        x ^= x >> 27
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & _MASK64

    def random(self, n: int | None = None):
        if n is None:
            return (self.next64() >> 11) * 2.0**-53
        return np.array([(self.next64() >> 11) * 2.0**-53 for _ in range(n)])


class PCG32:
    """O'Neill's PCG32 (XSH RR): a 64-bit LCG state with a permuted 32-bit output."""

    MULT = 6364136223846793005

    def __init__(self, seed: int = 42, stream: int = 54):
        self.inc = ((int(stream) << 1) | 1) & _MASK64
        self.state = 0
        self.next32()
        self.state = (self.state + int(seed)) & _MASK64
        self.next32()

    def next32(self) -> int:
        old = self.state
        self.state = (old * self.MULT + self.inc) & _MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def random(self, n: int | None = None):
        if n is None:
            return self.next32() * 2.0**-32
        return np.array([self.next32() * 2.0**-32 for _ in range(n)])


def _uniforms(gen, n):
    return np.asarray(gen.random(n), float)


# --------------------------------------------------------------------------------------------- continuous variates


def inverse_transform(ppf, gen, n: int) -> np.ndarray:
    """X = F^-1(U): exact for any distribution whose quantile function can be evaluated."""
    return np.asarray(ppf(_uniforms(gen, n)), float)


def exponential(rate: float, gen, n: int) -> np.ndarray:
    """Exponential variates by inversion, -ln(1 - U) / rate (1 - U avoids log 0)."""
    return -np.log1p(-_uniforms(gen, n)) / rate


def weibull(shape: float, scale: float, gen, n: int) -> np.ndarray:
    """Weibull variates by inversion: scale (-ln(1 - U))^(1/shape)."""
    return scale * (-np.log1p(-_uniforms(gen, n))) ** (1.0 / shape)


def box_muller(gen, n: int) -> np.ndarray:
    """Standard normals by the Box-Muller transform (pairs from two uniforms)."""
    m = (n + 1) // 2
    u1, u2 = 1.0 - _uniforms(gen, m), _uniforms(gen, m)
    r = np.sqrt(-2.0 * np.log(u1))
    return np.concatenate([r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)])[:n]


def polar_normal(gen, n: int) -> dict:
    """Marsaglia's polar method: normals without trigonometric functions; returns the values and the
    acceptance rate (pi/4 in theory)."""
    out, tried, accepted = [], 0, 0
    while len(out) < n:
        m = max(8, int(1.3 * (n - len(out)) / 2))
        v = 2.0 * _uniforms(gen, 2 * m).reshape(m, 2) - 1.0
        s = np.sum(v * v, axis=1)
        ok = (s > 0) & (s < 1)
        tried += m
        accepted += int(ok.sum())
        f = np.sqrt(-2.0 * np.log(s[ok]) / s[ok])
        out.extend((v[ok] * f[:, None]).ravel())
    return {"x": np.array(out[:n]), "acceptance": accepted / tried}


def rejection(target_pdf, proposal_sampler, proposal_pdf, M: float, gen, n: int) -> dict:
    """Acceptance-rejection: draw Y from the proposal g and accept with probability f(Y) / (M g(Y)), where
    f <= M g. Returns the samples, the number of proposals and the acceptance rate (1/M for normalised f, g)."""
    out, proposed = [], 0
    while len(out) < n:
        m = max(16, int(1.2 * M * (n - len(out))))
        y = np.asarray(proposal_sampler(m), float)
        u = _uniforms(gen, m)
        ratio = np.asarray(target_pdf(y), float) / (M * np.asarray(proposal_pdf(y), float))
        if np.any(ratio > 1 + 1e-12):
            raise ValueError("the envelope M g does not dominate f")
        proposed += m
        out.extend(y[u <= ratio])
    accepted = len(out)
    return {"x": np.array(out[:n]), "proposed": proposed, "acceptance": accepted / proposed}


def ratio_of_uniforms_normal(gen, n: int) -> dict:
    """Kinderman-Monahan ratio of uniforms for N(0, 1): (U, V) uniform on [0, 1] x [-sqrt(2/e), sqrt(2/e)],
    accept X = V/U when X^2 <= -4 ln U. Acceptance rate sqrt(pi e)/4 ~ 0.731."""
    b = math.sqrt(2.0 / math.e)
    out, proposed = [], 0
    while len(out) < n:
        m = max(16, int(1.5 * (n - len(out))))
        u = 1.0 - _uniforms(gen, m)
        v = (2.0 * _uniforms(gen, m) - 1.0) * b
        x = v / u
        proposed += m
        out.extend(x[x * x <= -4.0 * np.log(u)])
    return {"x": np.array(out[:n]), "acceptance": len(out) / proposed}


def gamma(shape: float, gen_normal, gen_uniform, n: int, scale: float = 1.0) -> np.ndarray:
    """Gamma(shape, scale) by Marsaglia and Tsang (2000); shape < 1 by the boost X U^(1/shape).
    `gen_normal(m)` must return m standard normals, `gen_uniform` provides uniforms."""
    if shape < 1.0:
        g = gamma(shape + 1.0, gen_normal, gen_uniform, n)
        return scale * g * _uniforms(gen_uniform, n) ** (1.0 / shape)
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    out = []
    while len(out) < n:
        m = max(16, int(1.1 * (n - len(out))))
        z = np.asarray(gen_normal(m), float)
        v = (1.0 + c * z) ** 3
        u = _uniforms(gen_uniform, m)
        ok = (v > 0) & (
            np.log(np.where(u > 0, u, 1e-300))
            < 0.5 * z * z + d - d * np.where(v > 0, v, 1) + d * np.log(np.where(v > 0, v, 1))
        )
        out.extend((d * v)[ok])
    return scale * np.array(out[:n])


def beta(a: float, b: float, gen_normal, gen_uniform, n: int) -> np.ndarray:
    """Beta(a, b) as X / (X + Y) with X ~ Gamma(a), Y ~ Gamma(b)."""
    x = gamma(a, gen_normal, gen_uniform, n)
    y = gamma(b, gen_normal, gen_uniform, n)
    return x / (x + y)


def multivariate_normal(mean, cov, gen_normal, n: int) -> np.ndarray:
    """N(mean, cov) as mean + L Z with L the Cholesky factor of cov (rows are samples)."""
    mean = np.asarray(mean, float)
    L = np.linalg.cholesky(np.asarray(cov, float))
    z = np.asarray(gen_normal(n * len(mean)), float).reshape(n, len(mean))
    return mean + z @ L.T


def gaussian_copula(corr, marginal_ppfs, gen_normal, n: int) -> np.ndarray:
    """Dependent variates with given marginals: Z ~ N(0, corr), U = Phi(Z), X_j = F_j^-1(U_j)."""
    z = multivariate_normal(np.zeros(len(marginal_ppfs)), corr, gen_normal, n)
    u = np.clip(norm_cdf(z), 1e-16, 1 - 1e-16)
    return np.column_stack([ppf(u[:, j]) for j, ppf in enumerate(marginal_ppfs)])


# --------------------------------------------------------------------------------------------- discrete variates


def discrete_inverse(probs, gen, n: int, values=None) -> np.ndarray:
    """Discrete inversion: the smallest k with F(k) >= U, by binary search on the cumulative sums."""
    p = np.asarray(probs, float)
    cdf = np.cumsum(p / p.sum())
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, _uniforms(gen, n), side="right")
    idx = np.minimum(idx, len(p) - 1)
    return idx if values is None else np.asarray(values)[idx]


def alias_table(probs) -> tuple[np.ndarray, np.ndarray]:
    """Vose's O(n) construction of Walker's alias table: (prob, alias) with one comparison per draw."""
    p = np.asarray(probs, float)
    n = len(p)
    scaled = p / p.sum() * n
    prob, alias = np.zeros(n), np.zeros(n, dtype=int)
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s, g = small.pop(), large.pop()
        prob[s], alias[s] = scaled[s], g
        scaled[g] = scaled[g] + scaled[s] - 1.0
        (small if scaled[g] < 1.0 else large).append(g)
    for i in large + small:
        prob[i], alias[i] = 1.0, i
    return prob, alias


def alias_sample(table, gen, n: int) -> np.ndarray:
    """Draw from an alias table: a uniform column, then the column's own value or its alias."""
    prob, alias = table
    u = _uniforms(gen, n) * len(prob)
    col = np.minimum(u.astype(int), len(prob) - 1)
    frac = u - col
    return np.where(frac < prob[col], col, alias[col])


def poisson(lam: float, gen, n: int) -> np.ndarray:
    """Poisson(lam) by sequential-search inversion, starting the search at the mode for large lam."""
    u = _uniforms(gen, n)
    kmax = int(lam + 12 * math.sqrt(lam) + 25)
    k = np.arange(kmax + 1)
    logp = (
        -lam + k * math.log(lam) - np.array([math.lgamma(j + 1.0) for j in k])
        if lam > 0
        else np.where(k == 0, 0.0, -np.inf)
    )
    cdf = np.cumsum(np.exp(logp))
    cdf[-1] = max(cdf[-1], 1.0)
    return np.searchsorted(cdf, u, side="right")


def binomial(trials: int, p: float, gen, n: int) -> np.ndarray:
    """Binomial(trials, p) by inversion of the cumulative distribution (exact recursion for the pmf)."""
    if p <= 0.0 or p >= 1.0:
        return np.full(n, 0 if p <= 0.0 else trials, dtype=int)
    k = np.arange(trials + 1)
    logp = np.array([math.lgamma(trials + 1.0) - math.lgamma(j + 1.0) - math.lgamma(trials - j + 1.0) for j in k])
    logp = logp + k * math.log(p) + (trials - k) * math.log1p(-p)
    cdf = np.cumsum(np.exp(logp))
    cdf[-1] = 1.0
    return np.searchsorted(cdf, _uniforms(gen, n), side="right")


def geometric(p: float, gen, n: int) -> np.ndarray:
    """Number of trials to the first success (support 1, 2, ...) by inversion: ceil(ln(1 - U) / ln(1 - p))."""
    return np.maximum(1, np.ceil(np.log1p(-_uniforms(gen, n)) / math.log1p(-p))).astype(int)


def random_permutation(m: int, gen) -> np.ndarray:
    """Fisher-Yates shuffle of 0..m-1 driven by the generator's uniforms."""
    a = np.arange(m)
    u = _uniforms(gen, m)
    for i in range(m - 1, 0, -1):
        j = int(u[i] * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a
