"""Monte Carlo estimation and variance reduction.

Plain Monte Carlo means with standard errors and confidence intervals (normal or t), the sample size needed
for a target precision, a two-stage sequential procedure, integration over boxes; and the classical
variance-reduction methods: antithetic variates, control variates (one or several, with the estimated
optimal coefficients and the regression form of the estimator), importance sampling (plain and
self-normalised, with the effective sample size and exponential tilting), stratified sampling (proportional
and Neyman allocation), conditional Monte Carlo and common random numbers for comparisons.

Every estimator returns a dict with at least ``estimate`` and ``std_error``; ``variance_reduction`` compares
an estimator with plain Monte Carlo at the same number of function evaluations.

>>> import numpy as np
>>> from engstoch import montecarlo as mc
>>> r = mc.mean_ci(np.random.default_rng(1).random(40_000))
>>> bool(abs(r["estimate"] - 0.5) < 4 * r["std_error"])
True
"""

from __future__ import annotations

import math

import numpy as np

from .special import norm_ppf, t_ppf


def mean_ci(x, level: float = 0.95, method: str = "normal") -> dict:
    """Sample mean with its standard error s/sqrt(n) and a confidence interval (normal or Student t)."""
    x = np.asarray(x, float)
    n = len(x)
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    q = t_ppf(0.5 + level / 2, n - 1) if method == "t" else norm_ppf(0.5 + level / 2)
    return {"estimate": m, "std_error": se, "std_dev": s, "n": n, "ci": (m - q * se, m + q * se), "level": level}


def proportion_ci(successes: int, n: int, level: float = 0.95) -> dict:
    """Probability estimate with the Wilson score interval (valid for rare events where the Wald interval fails)."""
    p = successes / n
    z = norm_ppf(0.5 + level / 2)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"estimate": p, "std_error": math.sqrt(max(p * (1 - p), 1e-300) / n), "ci": (centre - half, centre + half)}


def relative_error(result: dict) -> float:
    """Relative standard error se / |estimate| - the quantity that explodes for rare-event estimators."""
    return result["std_error"] / abs(result["estimate"]) if result["estimate"] != 0 else math.inf


def sample_size(std_dev: float, half_width: float, level: float = 0.95) -> int:
    """Number of samples for a confidence half-width h: n = (z sigma / h)^2."""
    z = norm_ppf(0.5 + level / 2)
    return int(math.ceil((z * std_dev / half_width) ** 2))


def two_stage(sampler, half_width: float, n0: int = 1000, level: float = 0.95, max_n: int = 10**7) -> dict:
    """Pilot run of n0 samples to estimate sigma, then enough further samples for the requested half-width
    (Stein's two-stage idea with a normal quantile). `sampler(n)` returns n i.i.d. outputs."""
    pilot = np.asarray(sampler(n0), float)
    n = min(max_n, max(n0, sample_size(float(np.std(pilot, ddof=1)), half_width, level)))
    x = np.r_[pilot, np.asarray(sampler(n - n0), float)] if n > n0 else pilot
    return mean_ci(x, level) | {"pilot_std": float(np.std(pilot, ddof=1))}


def integrate(f, lower, upper, n: int, rng) -> dict:
    """Monte Carlo integral of f over the box [lower, upper]: volume times the mean of f at uniform points.
    f receives an (n, d) array and returns n values."""
    lo, hi = np.atleast_1d(np.asarray(lower, float)), np.atleast_1d(np.asarray(upper, float))
    vol = float(np.prod(hi - lo))
    x = lo + (hi - lo) * rng.random((n, len(lo)))
    r = mean_ci(np.asarray(f(x), float))
    return r | {
        "estimate": vol * r["estimate"],
        "std_error": vol * r["std_error"],
        "ci": (vol * r["ci"][0], vol * r["ci"][1]),
    }


def antithetic(h, n_pairs: int, rng, dim: int = 1) -> dict:
    """Antithetic variates: average h(U) and h(1 - U) over n_pairs pairs; effective when h is monotone.
    Returns the estimate, its standard error, the pair correlation and the variance ratio against plain MC
    with the same 2 n_pairs evaluations."""
    u = rng.random((n_pairs, dim)) if dim > 1 else rng.random(n_pairs)
    a, b = np.asarray(h(u), float), np.asarray(h(1.0 - u), float)
    y = 0.5 * (a + b)
    r = mean_ci(y)
    rho = float(np.corrcoef(a, b)[0, 1])
    plain_var = 0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1)) / (2 * n_pairs)
    vr = float(plain_var / r["std_error"] ** 2) if r["std_error"] > 0 else math.inf  # exact cancellation for linear h
    return r | {"correlation": rho, "variance_reduction": vr}


def control_variate(y, c, c_mean, level: float = 0.95) -> dict:
    """Control-variate estimator Y - beta' (C - E[C]) with beta = Cov(C)^-1 Cov(C, Y) estimated from the same
    sample (the least-squares regression of Y on C). One control: c is shape (n,); several: (n, k).
    The variance reduction is 1 / (1 - R^2)."""
    y = np.asarray(y, float)
    C = np.asarray(c, float).reshape(len(y), -1)
    mu = np.atleast_1d(np.asarray(c_mean, float))
    Cc = C - C.mean(axis=0)
    beta = np.linalg.lstsq(Cc, y - y.mean(), rcond=None)[0]
    adj = y - (C - mu) @ beta
    n, k = len(y), C.shape[1]
    resid = y - y.mean() - Cc @ beta
    se = math.sqrt(np.sum(resid**2) / (n - k - 1) / n)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    est = float(np.mean(adj))
    z = norm_ppf(0.5 + level / 2)
    return {
        "estimate": est,
        "std_error": se,
        "beta": beta if k > 1 else float(beta[0]),
        "r_squared": float(r2),
        "variance_reduction": float(1 / (1 - r2)) if r2 < 1 else math.inf,
        "ci": (est - z * se, est + z * se),
    }


def importance_sampling(
    h, target_logpdf, proposal_sampler, proposal_logpdf, n: int, self_normalised: bool = False
) -> dict:
    """Importance sampling: E_f[h(X)] = E_g[h(X) f(X)/g(X)] with X drawn from g. With self_normalised=True the
    weights are normalised (f, g need only be known up to constants) and the delta-method standard error is
    reported. The effective sample size (sum w)^2 / sum w^2 diagnoses weight degeneracy."""
    x = proposal_sampler(n)
    logw = np.asarray(target_logpdf(x), float) - np.asarray(proposal_logpdf(x), float)
    hx = np.asarray(h(x), float)
    if self_normalised:
        w = np.exp(logw - np.max(logw))
        wn = w / w.sum()
        est = float(np.sum(wn * hx))
        se = float(math.sqrt(np.sum(wn**2 * (hx - est) ** 2)))
    else:
        w = np.exp(logw)
        r = mean_ci(hx * w)
        est, se = r["estimate"], r["std_error"]
    ess = float(w.sum() ** 2 / np.sum(w**2))
    return {
        "estimate": est,
        "std_error": se,
        "ess": ess,
        "weights": w,
        "max_weight_share": float(np.max(w) / np.sum(w)),
    }


def exponential_tilting_normal(theta: float):
    """For X ~ N(0, 1) the exponentially tilted density is N(theta, 1), and the likelihood ratio f/g is
    exp(-theta x + theta^2/2); returns its logarithm as a function. For P(X > a) the best theta is about a."""

    def log_lr(x):
        return -theta * np.asarray(x) + 0.5 * theta * theta

    return log_lr


def stratified(h, n: int, strata: int, rng, allocation: str = "proportional", pilot: int = 200) -> dict:
    """Stratified sampling of E[h(U)], U ~ U(0, 1), on `strata` equal intervals. Proportional allocation never
    increases the variance; Neyman allocation n_j proportional to sigma_j (from a pilot) minimises it."""
    edges = np.linspace(0, 1, strata + 1)
    p = np.diff(edges)
    if allocation == "neyman":
        sig = np.array([np.std(h(edges[j] + p[j] * rng.random(pilot)), ddof=1) for j in range(strata)])
        weights = p * sig
        nj = (
            np.maximum(2, np.round(n * weights / weights.sum()).astype(int))
            if weights.sum() > 0
            else np.full(strata, n // strata)
        )
    else:
        nj = np.maximum(2, np.round(n * p).astype(int))
    means, varis = np.zeros(strata), np.zeros(strata)
    for j in range(strata):
        y = np.asarray(h(edges[j] + p[j] * rng.random(nj[j])), float)
        means[j], varis[j] = y.mean(), y.var(ddof=1)
    est = float(np.sum(p * means))
    se = float(math.sqrt(np.sum(p**2 * varis / nj)))
    return {"estimate": est, "std_error": se, "allocation": nj, "stratum_std": np.sqrt(varis)}


def variance_reduction(result: dict, plain_std: float, n_evaluations: int) -> float:
    """Variance of plain MC with the same number of evaluations divided by the estimator's variance."""
    return (plain_std**2 / n_evaluations) / result["std_error"] ** 2


def common_random_numbers(sim_a, sim_b, n: int, seed: int = 0) -> dict:
    """Estimate E[A] - E[B] twice: with common random numbers (both simulations driven by the same seeds) and
    with independent streams. `sim(rng)` returns one output. CRN helps when the outputs respond alike."""
    crn = np.array(
        [sim_a(np.random.default_rng([seed, i])) - sim_b(np.random.default_rng([seed, i])) for i in range(n)]
    )
    ind = np.array(
        [sim_a(np.random.default_rng([seed, i, 1])) - sim_b(np.random.default_rng([seed, i, 2])) for i in range(n)]
    )
    rc, ri = mean_ci(crn), mean_ci(ind)
    return {"crn": rc, "independent": ri, "variance_reduction": (ri["std_error"] / rc["std_error"]) ** 2}
