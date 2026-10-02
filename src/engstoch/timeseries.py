"""Stationary time series: ARMA models, their second-order properties, estimation and spectra.

Simulation of ARMA(p, q) with a burn-in; causality and invertibility from the roots of the polynomials; the
MA(infinity) weights and the theoretical autocovariance; sample ACF with Bartlett bands and the PACF from the
Durbin-Levinson recursion; Yule-Walker and Burg-free least-squares AR estimation with AIC order selection;
the Ljung-Box portmanteau test; the periodogram, Welch's averaged periodogram and the ARMA spectral density.

>>> import numpy as np
>>> from engstoch import timeseries as ts
>>> ts.arma_acf([0.5], [], 3).round(6).tolist()
[1.0, 0.5, 0.25, 0.125]
"""

from __future__ import annotations

import math

import numpy as np

from .special import chi2_sf, norm_ppf


def simulate_arma(phi, theta, n: int, rng, sigma: float = 1.0, burn: int = 500, mean: float = 0.0) -> np.ndarray:
    """X_t = sum phi_i X_{t-i} + e_t + sum theta_j e_{t-j}, e ~ N(0, sigma^2); the first `burn` values are dropped."""
    phi, theta = np.asarray(phi, float), np.asarray(theta, float)
    p, q = len(phi), len(theta)
    m = n + burn
    e = sigma * rng.standard_normal(m + q)
    x = np.zeros(m + q)
    for t in range(q, m + q):
        ar = sum(phi[i] * x[t - 1 - i] for i in range(p) if t - 1 - i >= 0)
        ma = sum(theta[j] * e[t - 1 - j] for j in range(q))
        x[t] = ar + e[t] + ma
    return mean + x[q + burn :]


def is_causal(phi) -> bool:
    """AR polynomial 1 - phi_1 z - ... - phi_p z^p has all roots outside the unit circle."""
    phi = np.asarray(phi, float)
    if len(phi) == 0:
        return True
    return bool(np.all(np.abs(np.roots(np.r_[-phi[::-1], 1.0])) > 1.0))


def is_invertible(theta) -> bool:
    """MA polynomial 1 + theta_1 z + ... has all roots outside the unit circle."""
    theta = np.asarray(theta, float)
    if len(theta) == 0:
        return True
    return bool(np.all(np.abs(np.roots(np.r_[theta[::-1], 1.0])) > 1.0))


def psi_weights(phi, theta, n: int) -> np.ndarray:
    """MA(infinity) weights: psi_0 = 1, psi_j = theta_j + sum_i phi_i psi_{j-i}."""
    phi, theta = np.asarray(phi, float), np.asarray(theta, float)
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = (theta[j - 1] if j <= len(theta) else 0.0) + sum(
            phi[i] * psi[j - 1 - i] for i in range(len(phi)) if j - 1 - i >= 0
        )
    return psi


def arma_acov(phi, theta, max_lag: int, sigma: float = 1.0, n_psi: int = 5000) -> np.ndarray:
    """Theoretical autocovariance gamma(k) = sigma^2 sum_j psi_j psi_{j+k} (truncated MA(infinity) sum)."""
    psi = psi_weights(phi, theta, n_psi + max_lag + 1)
    return sigma**2 * np.array([np.dot(psi[:n_psi], psi[k : k + n_psi]) for k in range(max_lag + 1)])


def arma_acf(phi, theta, max_lag: int) -> np.ndarray:
    g = arma_acov(phi, theta, max_lag)
    return g / g[0]


def sample_acf(x, max_lag: int) -> np.ndarray:
    """Sample autocorrelation with the biased (1/n) autocovariance (guaranteed non-negative definite)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    g = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(max_lag + 1)])
    return g / g[0]


def acf_bands(n: int, level: float = 0.95) -> float:
    """White-noise bands +- z / sqrt(n) for the sample ACF."""
    return norm_ppf(0.5 + level / 2) / math.sqrt(n)


def durbin_levinson(acov) -> dict:
    """Durbin-Levinson recursion on gamma(0..p): AR coefficients of every order, the PACF phi_kk and the
    one-step prediction variances v_k."""
    g = np.asarray(acov, float)
    p = len(g) - 1
    phi = np.zeros((p + 1, p + 1))
    v = np.zeros(p + 1)
    v[0] = g[0]
    pacf = np.zeros(p + 1)
    pacf[0] = 1.0
    for k in range(1, p + 1):
        a = (g[k] - np.dot(phi[k - 1, 1:k], g[k - 1 : 0 : -1])) / v[k - 1]
        phi[k, k] = a
        phi[k, 1:k] = phi[k - 1, 1:k] - a * phi[k - 1, k - 1 : 0 : -1]
        v[k] = v[k - 1] * (1 - a * a)
        pacf[k] = a
    return {"phi": phi, "pacf": pacf, "variance": v}


def sample_pacf(x, max_lag: int) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    g = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(max_lag + 1)])
    return durbin_levinson(g)["pacf"]


def yule_walker(x, p: int) -> dict:
    """AR(p) by the Yule-Walker equations (solved by Durbin-Levinson), with the asymptotic standard errors
    sqrt(diag(sigma^2 Gamma_p^-1) / n)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    g = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(p + 1)])
    dl = durbin_levinson(g)
    phi = dl["phi"][p, 1:]
    s2 = dl["variance"][p]
    G = np.array([[g[abs(i - j)] for j in range(p)] for i in range(p)])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(G)) / n)
    return {"phi": phi, "sigma2": float(s2), "std_error": se}


def ar_least_squares(x, p: int) -> dict:
    """Conditional least squares for AR(p) with intercept; AIC = n log sigma^2 + 2 (p + 1)."""
    x = np.asarray(x, float)
    n = len(x)
    X = np.column_stack([np.ones(n - p)] + [x[p - i - 1 : n - i - 1] for i in range(p)])
    y = x[p:]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    s2 = float(r @ r / len(y))
    return {
        "intercept": float(beta[0]),
        "phi": beta[1:],
        "sigma2": s2,
        "aic": len(y) * math.log(s2) + 2 * (p + 1),
        "residuals": r,
    }


def select_ar_order(x, p_max: int) -> dict:
    """AIC over p = 1..p_max on a common sample (the first p_max values used only as lags)."""
    x = np.asarray(x, float)
    aic = []
    for p in range(1, p_max + 1):
        aic.append(ar_least_squares(x[p_max - p :], p)["aic"])
    return {"p": int(np.argmin(aic)) + 1, "aic": np.array(aic)}


def ljung_box(x, lags: int, fitted_params: int = 0) -> dict:
    """Q = n (n + 2) sum_k r_k^2 / (n - k), chi-square with lags - fitted_params df under whiteness."""
    x = np.asarray(x, float)
    n = len(x)
    r = sample_acf(x, lags)[1:]
    Q = float(n * (n + 2) * np.sum(r**2 / (n - np.arange(1, lags + 1))))
    df = lags - fitted_params
    return {"statistic": Q, "df": df, "p_value": float(chi2_sf(Q, df))}


def periodogram(x, dt: float = 1.0) -> dict:
    """One-sided periodogram I(f) = (dt / n) |sum x_t e^{-2 pi i f t dt}|^2 (doubled except at 0 and Nyquist),
    an unbiased but inconsistent estimate of the spectral density in units of variance per unit frequency."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x)
    P = dt / n * np.abs(X) ** 2
    P[1:] *= 2
    if n % 2 == 0:
        P[-1] /= 2
    return {"f": np.fft.rfftfreq(n, dt), "psd": P}


def welch(x, segment: int, dt: float = 1.0, overlap: float = 0.5) -> dict:
    """Welch's method: Hann-windowed periodograms of overlapping segments averaged; variance falls roughly in
    proportion to the number of segments, at the price of resolution 1 / (segment dt)."""
    x = np.asarray(x, float)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(segment) / segment)  # periodic Hann window
    step = max(1, int(segment * (1 - overlap)))
    starts = range(0, len(x) - segment + 1, step)
    U = np.sum(w**2)
    P = []
    for s in starts:
        seg = x[s : s + segment]
        seg = seg - seg.mean()
        Xf = np.fft.rfft(w * seg)
        p = dt / U * np.abs(Xf) ** 2
        p[1:] *= 2
        if segment % 2 == 0:
            p[-1] /= 2
        P.append(p)
    return {"f": np.fft.rfftfreq(segment, dt), "psd": np.mean(P, axis=0), "segments": len(P)}


def arma_spectrum(phi, theta, f, sigma: float = 1.0, dt: float = 1.0):
    """One-sided spectral density 2 sigma^2 dt |theta(e^{-iw})|^2 / |phi(e^{-iw})|^2 with w = 2 pi f dt
    (integrates to the variance over 0..1/(2 dt))."""
    w = 2 * np.pi * np.asarray(f, float) * dt
    z = np.exp(-1j * w)
    num = np.abs(1 + sum(t * z ** (j + 1) for j, t in enumerate(theta))) ** 2
    den = np.abs(1 - sum(p * z ** (i + 1) for i, p in enumerate(phi))) ** 2
    return 2 * sigma**2 * dt * num / den
