"""Poisson processes: homogeneous, non-homogeneous, compound, marked and spatial.

Homogeneous arrivals two ways (cumulative exponential gaps, and a Poisson count with sorted uniforms - the
order-statistics property); non-homogeneous arrivals by Lewis-Shedler thinning and by inversion of the
cumulative intensity; compound Poisson sums and their moments; splitting (thinning by marks) and
superposition; spatial Poisson processes on rectangles and discs; and inference: the rate MLE with its exact
(Garwood) confidence interval, the index of dispersion test of counts, the conditional-uniformity KS test of
arrival times, and a piecewise-constant intensity estimate from many days of arrivals.

>>> import numpy as np
>>> from engstoch import poisson as pp
>>> t = pp.homogeneous(2.0, 1000.0, np.random.default_rng(0))
>>> bool(abs(len(t) / 1000.0 - 2.0) < 0.15)
True
"""

from __future__ import annotations

import math

import numpy as np

from .rngtests import ks_test
from .special import chi2_ppf, chi2_sf


def homogeneous(rate: float, t_end: float, rng) -> np.ndarray:
    """Arrival times on [0, t_end] as cumulative sums of Exponential(rate) gaps."""
    out = []
    t = 0.0
    batch = max(16, int(rate * t_end * 1.1) + 10)
    while True:
        gaps = rng.exponential(1.0 / rate, batch)
        times = t + np.cumsum(gaps)
        out.append(times[times <= t_end])
        if times[-1] > t_end:
            break
        t = times[-1]
    return np.concatenate(out)


def homogeneous_order_statistics(rate: float, t_end: float, rng) -> np.ndarray:
    """N ~ Poisson(rate t_end) points placed as sorted uniforms on [0, t_end] (the same law as `homogeneous`)."""
    n = rng.poisson(rate * t_end)
    return np.sort(rng.random(n) * t_end)


def thinning(intensity, lam_max: float, t_end: float, rng) -> dict:
    """Lewis-Shedler thinning: candidates from a rate-lam_max process kept with probability lambda(t)/lam_max.
    Returns the arrivals and the acceptance fraction (the integral of lambda over lam_max t_end)."""
    cand = homogeneous(lam_max, t_end, rng)
    lam = np.asarray(intensity(cand), float)
    if np.any(lam > lam_max * (1 + 1e-12)):
        raise ValueError("lam_max must bound the intensity")
    keep = rng.random(len(cand)) < lam / lam_max
    return {"times": cand[keep], "candidates": len(cand), "acceptance": float(keep.mean()) if len(cand) else 0.0}


def time_change(cumulative_intensity_inverse, total: float, rng) -> np.ndarray:
    """Inversion: if E_1 < E_2 < ... are unit-rate Poisson arrivals then Lambda^-1(E_k) has intensity lambda.
    `total` is Lambda(t_end); the inverse cumulative intensity is applied to the unit-rate arrivals below it."""
    e = homogeneous(1.0, total, rng)
    return np.asarray(cumulative_intensity_inverse(e), float)


def cumulative_intensity(intensity, t, n: int = 2001) -> float:
    """Lambda(t) = integral_0^t lambda(s) ds by the trapezoidal rule."""
    s = np.linspace(0.0, t, n)
    y = np.asarray(intensity(s), float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(s)))


def counts_in_windows(times, edges) -> np.ndarray:
    """Number of arrivals in each window [edges_k, edges_k+1)."""
    return np.histogram(np.asarray(times), bins=np.asarray(edges))[0]


def compound(rate: float, t_end: float, jump_sampler, rng) -> dict:
    """Compound Poisson S(t_end) = sum of N(t_end) i.i.d. jumps; returns the arrival times, jumps and the path."""
    t = homogeneous(rate, t_end, rng)
    y = np.asarray(jump_sampler(len(t), rng), float) if len(t) else np.zeros(0)
    return {"times": t, "jumps": y, "total": float(y.sum()), "path": np.cumsum(y)}


def compound_moments(rate: float, t: float, jump_mean: float, jump_second_moment: float) -> dict:
    """E[S] = lambda t E[Y] and Var[S] = lambda t E[Y^2] (Wald and the law of total variance)."""
    return {"mean": rate * t * jump_mean, "var": rate * t * jump_second_moment}


def split(times, probs, rng) -> list[np.ndarray]:
    """Mark each arrival independently with type k with probability probs[k]: the typed streams are independent
    Poisson processes of rates lambda p_k (Poisson splitting)."""
    marks = rng.choice(len(probs), size=len(times), p=np.asarray(probs, float) / np.sum(probs))
    return [np.asarray(times)[marks == k] for k in range(len(probs))]


def superpose(*streams) -> np.ndarray:
    """Union of independent arrival streams (a Poisson process of the summed rate if they are Poisson)."""
    return np.sort(np.concatenate([np.asarray(s, float) for s in streams]))


def spatial_rectangle(rate: float, width: float, height: float, rng) -> np.ndarray:
    """Homogeneous spatial Poisson process on [0, w] x [0, h]: Poisson count, uniform locations."""
    n = rng.poisson(rate * width * height)
    return rng.random((n, 2)) * [width, height]


def spatial_disc(rate: float, radius: float, rng) -> np.ndarray:
    """Spatial Poisson process on a disc: uniform points by r = R sqrt(U), theta = 2 pi V."""
    n = rng.poisson(rate * math.pi * radius**2)
    r = radius * np.sqrt(rng.random(n))
    th = 2 * math.pi * rng.random(n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


def nearest_neighbour_cdf(r, rate: float):
    """Distance from a typical point (or any fixed location) to the nearest point of a planar Poisson process:
    P(D <= r) = 1 - exp(-lambda pi r^2)."""
    return 1.0 - np.exp(-rate * math.pi * np.asarray(r, float) ** 2)


def rate_mle(n_events: int, exposure: float, level: float = 0.95) -> dict:
    """Rate estimate N / T with the exact Garwood interval [chi2_{a/2}(2N) / 2T, chi2_{1-a/2}(2N + 2) / 2T]."""
    a = 1 - level
    lo = 0.0 if n_events == 0 else chi2_ppf(a / 2, 2 * n_events) / (2 * exposure)
    hi = chi2_ppf(1 - a / 2, 2 * n_events + 2) / (2 * exposure)
    return {"rate": n_events / exposure, "ci": (lo, hi), "std_error": math.sqrt(max(n_events, 1)) / exposure}


def dispersion_test(counts) -> dict:
    """Index of dispersion D = (k - 1) s^2 / mean of counts in equal windows: chi-square with k - 1 df under the
    Poisson hypothesis; D/(k-1) well above 1 signals overdispersion (clustering or a varying rate)."""
    c = np.asarray(counts, float)
    k = len(c)
    m = c.mean()
    d = float((k - 1) * c.var(ddof=1) / m)
    p_upper = float(chi2_sf(d, k - 1))
    return {"index": d / (k - 1), "statistic": d, "df": k - 1, "p_value": float(2 * min(p_upper, 1 - p_upper))}


def uniformity_test(times, t_end: float) -> dict:
    """Given N(t_end) = n, the arrival times of a homogeneous Poisson process are n sorted uniforms: KS test of
    times / t_end against U(0, 1)."""
    return ks_test(np.asarray(times) / t_end, lambda x: np.clip(x, 0, 1))


def intensity_estimate(days_of_times, edges) -> dict:
    """Piecewise-constant intensity from arrivals observed on several days (each a list of times within the day):
    the mean count per bin divided by the bin width, with Poisson standard errors."""
    edges = np.asarray(edges, float)
    counts = np.array([counts_in_windows(d, edges) for d in days_of_times])
    width = np.diff(edges)
    k = len(days_of_times)
    lam = counts.mean(axis=0) / width
    return {"edges": edges, "rate": lam, "std_error": np.sqrt(counts.sum(axis=0)) / (k * width), "counts": counts}


def interarrival_gaps(times) -> np.ndarray:
    """Gaps between consecutive arrivals (the first measured from 0)."""
    return np.diff(np.r_[0.0, np.asarray(times, float)])
