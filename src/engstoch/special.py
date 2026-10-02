"""Special functions needed by the rest of the library, written out so that the core depends on NumPy only:
the normal distribution function and its inverse (Wichura's AS 241), the regularised incomplete gamma and
beta functions (series and Lentz continued fractions), and the chi-square, Student t, Poisson and Kolmogorov
distribution functions built from them.

>>> from engstoch import special as sf
>>> round(sf.norm_ppf(0.975), 10)
1.9599639845
>>> round(sf.chi2_sf(3.84145882069, 1), 6)
0.05
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-15


def norm_cdf(x):
    """Standard normal distribution function Phi(x), via erfc (accurate in both tails)."""
    x = np.asarray(x, float)
    out = 0.5 * np.vectorize(math.erfc)(-x / math.sqrt(2.0))
    return out if out.ndim else float(out)


def norm_pdf(x):
    """Standard normal density."""
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _ppf_scalar(p: float) -> float:
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must lie in [0, 1]")
    q = p - 0.5
    if abs(q) <= 0.425:  # central region, AS 241 PPND16
        r = 0.180625 - q * q
        num = (
            (
                (
                    (
                        ((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                        + 45921.953931549871457
                    )
                    * r
                    + 13731.693765509461125
                )
                * r
                + 1971.5909503065514427
            )
            * r
            + 133.14166789178437745
        ) * r + 3.387132872796366608
        den = (
            (
                (
                    (
                        ((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                        + 21213.794301586595867
                    )
                    * r
                    + 5394.1960214247511077
                )
                * r
                + 687.1870074920579083
            )
            * r
            + 42.313330701600911252
        ) * r + 1.0
        return q * num / den
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        num = (
            (
                (
                    (
                        ((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                        + 1.27045825245236838258
                    )
                    * r
                    + 3.64784832476320460504
                )
                * r
                + 5.7694972214606914055
            )
            * r
            + 4.6303378461565452959
        ) * r + 1.42343711074968357734
        den = (
            (
                (
                    (
                        ((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                        + 0.14810397642748007459
                    )
                    * r
                    + 0.68976733498510000455
                )
                * r
                + 1.6763848301838038494
            )
            * r
            + 2.05319162663775882187
        ) * r + 1.0
    else:
        r -= 5.0
        num = (
            (
                (
                    (
                        ((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                        + 0.026532189526576123093
                    )
                    * r
                    + 0.29656057182850489123
                )
                * r
                + 1.7848265399172913358
            )
            * r
            + 5.4637849111641143699
        ) * r + 6.6579046435011037772
        den = (
            (
                (
                    (
                        ((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                        + 7.868691311456132591e-4
                    )
                    * r
                    + 0.0148753612908506148525
                )
                * r
                + 0.13692988092273580531
            )
            * r
            + 0.59983220655588793769
        ) * r + 1.0
    val = num / den
    return -val if q < 0 else val


def norm_ppf(p):
    """Inverse of the standard normal distribution function (Wichura 1988, AS 241; relative error ~1e-16)."""
    p = np.asarray(p, float)
    out = np.vectorize(_ppf_scalar)(p)
    return out if out.ndim else float(out)


def _gammainc_scalar(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x < a + 1.0:  # series
        term = total = 1.0 / a
        ap = a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _EPS:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return 1.0 - _gammaincc_cf(a, x)


def _gammaincc_cf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) by the modified Lentz continued fraction (x >= a + 1)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a, x):
    """Regularised lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)."""
    out = np.vectorize(_gammainc_scalar)(np.asarray(a, float), np.asarray(x, float))
    return out if out.ndim else float(out)


def gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x), computed directly where P is close to 1."""

    def one(a_, x_):
        if x_ <= 0:
            return 1.0
        return 1.0 - _gammainc_scalar(a_, x_) if x_ < a_ + 1.0 else _gammaincc_cf(a_, x_)

    out = np.vectorize(one)(np.asarray(a, float), np.asarray(x, float))
    return out if out.ndim else float(out)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc_scalar(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def betainc(a, b, x):
    """Regularised incomplete beta function I_x(a, b) (continued fraction, Numerical Recipes 6.4)."""
    out = np.vectorize(_betainc_scalar)(np.asarray(a, float), np.asarray(b, float), np.asarray(x, float))
    return out if out.ndim else float(out)


def chi2_cdf(x, k):
    """Chi-square distribution function with k degrees of freedom."""
    return gammainc(np.asarray(k, float) / 2.0, np.asarray(x, float) / 2.0)


def chi2_sf(x, k):
    """Chi-square upper tail probability P(X > x) (accurate for small p-values)."""
    return gammaincc(np.asarray(k, float) / 2.0, np.asarray(x, float) / 2.0)


def chi2_ppf(p: float, k: float) -> float:
    """Chi-square quantile by bisection-safeguarded Newton from the Wilson-Hilferty start."""
    z = norm_ppf(p)
    x = max(k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3, 1e-8)
    lo, hi = 0.0, max(10 * k + 100.0, 2 * x)
    for _ in range(200):
        f = chi2_cdf(x, k) - p
        if abs(f) < 1e-14:
            break
        if f > 0:
            hi = x
        else:
            lo = x
        dens = math.exp((k / 2 - 1) * math.log(x) - x / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2)) if x > 0 else 0
        step = x - f / dens if dens > 0 else 0.5 * (lo + hi)
        x = step if lo < step < hi else 0.5 * (lo + hi)
    return x


def t_cdf(x, nu):
    """Student t distribution function with nu degrees of freedom."""

    def one(x_, n_):
        ib = _betainc_scalar(n_ / 2.0, 0.5, n_ / (n_ + x_ * x_))
        return 1.0 - 0.5 * ib if x_ >= 0 else 0.5 * ib

    out = np.vectorize(one)(np.asarray(x, float), np.asarray(nu, float))
    return out if out.ndim else float(out)


def t_ppf(p: float, nu: float) -> float:
    """Student t quantile, by Newton iteration on t_cdf from the normal quantile (bisection-safeguarded)."""
    if nu > 1e7:
        return norm_ppf(p)
    if p == 0.5:
        return 0.0
    x = norm_ppf(p)
    lo, hi = -1e10, 1e10
    c = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
    for _ in range(200):
        f = t_cdf(x, nu) - p
        if abs(f) < 1e-15:
            break
        if f > 0:
            hi = x
        else:
            lo = x
        dens = c * (1 + x * x / nu) ** (-(nu + 1) / 2)
        step = x - f / dens
        x = step if lo < step < hi else 0.5 * (lo + hi)
    return x


def poisson_cdf(k, lam):
    """P(N <= k) for N ~ Poisson(lam), as the upper incomplete gamma Q(k + 1, lam)."""
    k = np.floor(np.asarray(k, float))
    out = np.where(k < 0, 0.0, gammaincc(np.maximum(k, 0) + 1.0, np.asarray(lam, float)))
    return out if out.ndim else float(out)


def kolmogorov_sf(x):
    """Limiting distribution of sqrt(n) D_n: P(K > x) = 2 sum (-1)^(j-1) exp(-2 j^2 x^2)."""

    def one(x_):
        if x_ <= 0:
            return 1.0
        if x_ < 0.2:  # the alternating series converges slowly; use the theta-function form of the cdf
            s = sum(math.exp(-((2 * j - 1) ** 2) * math.pi**2 / (8 * x_ * x_)) for j in range(1, 50))
            return 1.0 - math.sqrt(2 * math.pi) / x_ * s
        return min(1.0, 2.0 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * x_ * x_) for j in range(1, 101)))

    out = np.vectorize(one)(np.asarray(x, float))
    return out if out.ndim else float(out)


def ks_sf(d: float, n: int) -> float:
    """Approximate P(D_n > d) for the one-sample Kolmogorov-Smirnov statistic (Stephens' correction)."""
    sn = math.sqrt(n)
    return kolmogorov_sf((sn + 0.12 + 0.11 / sn) * d)
