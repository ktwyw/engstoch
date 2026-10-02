"""Output analysis of simulations and Markov chain Monte Carlo.

Autocovariance by FFT; the integrated autocorrelation time with Sokal's self-consistent window and the
initial-positive-sequence estimator of Geyer; the effective sample size; batch means (non-overlapping) and
the asymptotic variance they estimate; confidence intervals from independent replications; the initial
transient: Welch's moving-average plot data and the MSER-5 truncation rule; the regenerative method's ratio
estimator with its delta-method interval; and Gelman-Rubin's potential scale reduction for several chains.

>>> import numpy as np
>>> from engstoch import output as oa
>>> x = np.random.default_rng(0).normal(size=20_000)
>>> bool(abs(oa.iat(x)["tau"] - 1.0) < 0.2)
True
"""

from __future__ import annotations

import math

import numpy as np

from .montecarlo import mean_ci
from .special import norm_ppf, t_ppf


def autocovariance(x, max_lag: int | None = None) -> np.ndarray:
    """Biased sample autocovariance gamma(k) = (1/n) sum (x_t - xbar)(x_{t+k} - xbar), by FFT."""
    x = np.asarray(x, float)
    n = len(x)
    y = x - x.mean()
    m = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfft(y, m)
    acov = np.fft.irfft(f * np.conj(f), m)[:n] / n
    return acov if max_lag is None else acov[: max_lag + 1]


def autocorrelation(x, max_lag: int | None = None) -> np.ndarray:
    a = autocovariance(x, max_lag)
    return a / a[0]


def iat(x, c: float = 5.0) -> dict:
    """Integrated autocorrelation time tau = 1 + 2 sum_k rho(k) with Sokal's automatic window: the smallest M
    with M >= c tau(M). Var(xbar) ~ tau sigma^2 / n."""
    rho = autocorrelation(x)
    tau = 1.0
    M = len(rho) - 1
    for m in range(1, len(rho)):
        tau = 1.0 + 2.0 * np.sum(rho[1 : m + 1])
        if m >= c * tau:
            M = m
            break
    n = len(rho)
    return {
        "tau": float(tau),
        "window": M,
        "ess": float(n / tau),
        "std_error_tau": float(tau * math.sqrt(2 * (2 * M + 1) / n)),
    }


def iat_geyer(x) -> dict:
    """Geyer's initial positive sequence: sum pairs Gamma_m = rho(2m) + rho(2m+1) while positive."""
    rho = autocorrelation(x)
    s = 0.0
    for m in range(0, len(rho) // 2 - 1):
        g = rho[2 * m] + rho[2 * m + 1]
        if g <= 0:
            break
        s += g
    tau = 2 * s - 1
    return {"tau": float(tau), "ess": float(len(rho) / tau)}


def ess(x) -> float:
    """Effective sample size n / tau."""
    return iat(x)["ess"]


def batch_means(x, n_batches: int = 20, level: float = 0.95) -> dict:
    """Non-overlapping batch means: the batch averages are nearly independent when batches are long compared
    with the correlation time; the t interval uses n_batches - 1 degrees of freedom. Also returns the estimated
    asymptotic variance sigma^2_inf = b Var(batch means) and the lag-1 correlation of the batch means (should
    be near 0)."""
    x = np.asarray(x, float)
    b = len(x) // n_batches
    means = x[: b * n_batches].reshape(n_batches, b).mean(axis=1)
    m = float(means.mean())
    s = float(means.std(ddof=1))
    q = t_ppf(0.5 + level / 2, n_batches - 1)
    se = s / math.sqrt(n_batches)
    r1 = float(np.corrcoef(means[:-1], means[1:])[0, 1]) if n_batches > 3 else math.nan
    return {
        "estimate": m,
        "std_error": se,
        "ci": (m - q * se, m + q * se),
        "batch_size": b,
        "asymptotic_var": b * s * s,
        "lag1_corr": r1,
    }


def replications(values, level: float = 0.95) -> dict:
    """Independent replications: the t interval of the replication averages."""
    return mean_ci(values, level, method="t")


def welch_moving_average(runs, window: int) -> np.ndarray:
    """Welch's procedure: average several replications pointwise, then smooth with a centred moving average of
    half-width `window`; the warm-up ends where the curve flattens."""
    avg = np.mean(np.asarray(runs, float), axis=0)
    n = len(avg)
    out = np.empty(n - window)
    for i in range(n - window):
        lo = max(0, i - window) if i >= window else 0
        hi = i + window if i >= window else 2 * i
        out[i] = avg[lo : hi + 1].mean()
    return out


def mser(x, batch: int = 5, max_fraction: float = 0.5) -> dict:
    """MSER-5 (White 1997): batch the output in groups of 5 and choose the truncation d minimising
    Var(remaining) / (n - d)^2, i.e. the half-width of the remaining mean. Searches d over the first half."""
    x = np.asarray(x, float)
    m = len(x) // batch
    z = x[: m * batch].reshape(m, batch).mean(axis=1)
    best_d, best = 0, math.inf
    for d in range(int(max_fraction * m)):
        rest = z[d:]
        val = np.var(rest) / len(rest)
        if val < best:
            best, best_d = val, d
    return {"truncate": best_d * batch, "criterion": best}


def regenerative(cycle_rewards, cycle_lengths, level: float = 0.95) -> dict:
    """Regenerative ratio estimator r = sum Y_i / sum tau_i with the delta-method variance
    (S_YY - 2 r S_Ytau + r^2 S_tautau) / (n taubar^2)."""
    y = np.asarray(cycle_rewards, float)
    t = np.asarray(cycle_lengths, float)
    n = len(y)
    r = y.sum() / t.sum()
    v = np.var(y - r * t, ddof=1)
    se = math.sqrt(v / n) / t.mean()
    z = norm_ppf(0.5 + level / 2)
    return {"estimate": float(r), "std_error": se, "ci": (r - z * se, r + z * se), "cycles": n}


def gelman_rubin(chains) -> dict:
    """Potential scale reduction R-hat from m chains of length n (second halves recommended): compares the
    between-chain and within-chain variances; values near 1 (< 1.01 by current practice) indicate mixing."""
    c = np.asarray(chains, float)
    m, n = c.shape
    means = c.mean(axis=1)
    W = float(np.mean(c.var(axis=1, ddof=1)))
    B = float(n * means.var(ddof=1))
    var_plus = (n - 1) / n * W + B / n
    return {"R_hat": math.sqrt(var_plus / W), "W": W, "B": B}


def coverage(lowers, uppers, truth: float) -> dict:
    """Fraction of intervals that contain the true value, with its binomial standard error."""
    lo, hi = np.asarray(lowers, float), np.asarray(uppers, float)
    hit = (lo <= truth) & (truth <= hi)
    p = float(hit.mean())
    return {"coverage": p, "std_error": math.sqrt(p * (1 - p) / len(hit))}
