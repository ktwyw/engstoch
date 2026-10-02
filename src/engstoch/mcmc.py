"""Markov chain Monte Carlo.

Random-walk Metropolis (with optional adaptation of the proposal scale towards a target acceptance rate during
burn-in), the independence sampler, Metropolis-within-Gibbs and a Gibbs sampler driven by user-supplied full
conditionals, Hamiltonian Monte Carlo with the leapfrog integrator, the Ising model by single-spin Metropolis
and heat-bath updates (with the exact Onsager magnetisation for comparison), and simulated annealing. Chain
diagnostics live in ``output`` (autocorrelation time, effective sample size, R-hat).

>>> import numpy as np
>>> from engstoch import mcmc
>>> r = mcmc.metropolis(lambda x: -0.5 * np.sum(x**2), [0.0], 20_000, 2.4, np.random.default_rng(0))
>>> bool(abs(np.var(r["chain"][2000:, 0]) - 1.0) < 0.15)
True
"""

from __future__ import annotations

import math

import numpy as np


def metropolis(logpdf, x0, n: int, scale, rng, adapt_until: int = 0, target_acceptance: float = 0.234) -> dict:
    """Random-walk Metropolis with Gaussian proposals x' = x + scale * Z (scale a number or a vector). With
    adapt_until > 0 the log scale is adapted by a Robbins-Monro step towards the target acceptance rate during
    the first adapt_until iterations (which should then be discarded)."""
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    d = len(x)
    scale = np.full(d, float(scale)) if np.ndim(scale) == 0 else np.asarray(scale, float).copy()
    lp = float(logpdf(x))
    chain = np.empty((n, d))
    accepted = 0
    log_s = 0.0
    for k in range(n):
        s = scale * math.exp(log_s)
        y = x + s * rng.standard_normal(d)
        ly = float(logpdf(y))
        a = 1.0 if ly >= lp else math.exp(ly - lp)
        if rng.random() < a:
            x, lp = y, ly
            if k >= adapt_until:
                accepted += 1
        if k < adapt_until:
            log_s += (a - target_acceptance) / math.sqrt(k + 1)
        chain[k] = x
    return {"chain": chain, "acceptance": accepted / max(1, n - adapt_until), "final_scale": scale * math.exp(log_s)}


def independence_sampler(logpdf, proposal_sampler, proposal_logpdf, x0, n: int, rng) -> dict:
    """Metropolis-Hastings with proposals independent of the current state: accept with
    min(1, w(y)/w(x)), w = pi/q. Excellent when q is close to pi and has heavier tails, disastrous otherwise."""
    x = np.atleast_1d(np.asarray(x0, float))
    lw = float(logpdf(x) - proposal_logpdf(x))
    chain = np.empty((n, len(x)))
    acc = 0
    for k in range(n):
        y = np.atleast_1d(proposal_sampler(rng))
        lwy = float(logpdf(y) - proposal_logpdf(y))
        if math.log(rng.random() + 1e-300) < lwy - lw:
            x, lw = y, lwy
            acc += 1
        chain[k] = x
    return {"chain": chain, "acceptance": acc / n}


def gibbs(conditionals, x0, n: int, rng) -> np.ndarray:
    """Systematic-scan Gibbs sampler: conditionals[i](x, rng) draws coordinate i from its full conditional."""
    x = np.asarray(x0, float).copy()
    chain = np.empty((n, len(x)))
    for k in range(n):
        for i, draw in enumerate(conditionals):
            x[i] = draw(x, rng)
        chain[k] = x
    return chain


def gibbs_bivariate_normal(rho: float, n: int, rng, x0=(0.0, 0.0)) -> np.ndarray:
    """Gibbs for a standard bivariate normal with correlation rho: x | y ~ N(rho y, 1 - rho^2). The chain of x has
    lag-1 autocorrelation rho^2, so it mixes badly when |rho| is near 1."""
    s = math.sqrt(1 - rho * rho)
    return gibbs(
        [lambda x, r: rho * x[1] + s * r.standard_normal(), lambda x, r: rho * x[0] + s * r.standard_normal()],
        x0,
        n,
        rng,
    )


def hmc(logpdf, grad_logpdf, x0, n: int, step: float, n_leapfrog: int, rng, mass=None) -> dict:
    """Hamiltonian Monte Carlo (Duane et al. 1987; Neal 2011): momentum p ~ N(0, M), n_leapfrog leapfrog steps
    of size `step` in the potential -log pi, accept with min(1, exp(-dH)). Returns the chain, the acceptance rate
    and the energy errors dH (which grow like step^2 for the second-order leapfrog)."""
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    d = len(x)
    m = np.ones(d) if mass is None else np.asarray(mass, float)
    chain = np.empty((n, d))
    dHs = np.empty(n)
    acc = 0
    lp, g = float(logpdf(x)), np.asarray(grad_logpdf(x), float)
    for k in range(n):
        p = rng.standard_normal(d) * np.sqrt(m)
        H0 = -lp + 0.5 * np.sum(p * p / m)
        xn, pn, gn = x.copy(), p + 0.5 * step * g, g
        for i in range(n_leapfrog):
            xn = xn + step * pn / m
            gn = np.asarray(grad_logpdf(xn), float)
            if i < n_leapfrog - 1:
                pn = pn + step * gn
        pn = pn + 0.5 * step * gn
        lpn = float(logpdf(xn))
        H1 = -lpn + 0.5 * np.sum(pn * pn / m)
        dH = H1 - H0
        dHs[k] = dH
        if math.log(rng.random() + 1e-300) < -dH:
            x, lp, g = xn, lpn, gn
            acc += 1
        chain[k] = x
    return {"chain": chain, "acceptance": acc / n, "energy_error": dHs}


def leapfrog_energy_error(grad_logpdf, logpdf, x0, p0, step: float, n_steps: int) -> float:
    """Change in the Hamiltonian over one leapfrog trajectory (unit mass)."""
    x, p = np.atleast_1d(np.asarray(x0, float)), np.atleast_1d(np.asarray(p0, float))
    H0 = -logpdf(x) + 0.5 * np.sum(p * p)
    p = p + 0.5 * step * grad_logpdf(x)
    for i in range(n_steps):
        x = x + step * p
        if i < n_steps - 1:
            p = p + step * grad_logpdf(x)
    p = p + 0.5 * step * grad_logpdf(x)
    return float(-logpdf(x) + 0.5 * np.sum(p * p) - H0)


# --------------------------------------------------------------------------------------------- Ising model


def ising(
    L: int, beta: float, n_sweeps: int, rng, method: str = "metropolis", start: str = "cold", h: float = 0.0
) -> dict:
    """2-D Ising model on an L x L torus, energy -sum_<ij> s_i s_j - h sum s_i. Each sweep visits the sites of the
    two checkerboard sublattices in turn and updates them simultaneously (they do not interact), by Metropolis
    (flip with min(1, e^{-beta dE})) or heat bath (s = +1 with probability 1/(1 + e^{-2 beta local field})).
    Returns the magnetisation per site and energy per site after every sweep."""
    s = np.ones((L, L), dtype=int) if start == "cold" else rng.choice(np.array([-1, 1]), (L, L))
    ii, jj = np.indices((L, L))
    masks = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]
    mags, energies = np.empty(n_sweeps), np.empty(n_sweeps)
    for k in range(n_sweeps):
        for mask in masks:
            nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            field = nb + h
            u = rng.random((L, L))
            if method == "heatbath":
                new = np.where(u < 1.0 / (1.0 + np.exp(-2 * beta * field)), 1, -1)
                s = np.where(mask, new, s)
            else:
                dE = 2 * s * field
                flip = mask & (u < np.exp(-beta * np.maximum(dE, 0)))
                s = np.where(flip, -s, s)
        nb = np.roll(s, 1, 0) + np.roll(s, 1, 1)
        mags[k] = s.mean()
        energies[k] = -(np.sum(s * nb) + h * s.sum()) / (L * L)
    return {"magnetisation": mags, "energy": energies, "spins": s}


BETA_C = 0.5 * math.log(1 + math.sqrt(2))  # critical inverse temperature of the square-lattice Ising model


def onsager_magnetisation(beta: float) -> float:
    """Spontaneous magnetisation (1 - sinh(2 beta)^-4)^(1/8) for beta > beta_c (Onsager, Yang), 0 below."""
    if beta <= BETA_C:
        return 0.0
    return (1 - math.sinh(2 * beta) ** -4) ** 0.125


def onsager_energy(beta: float) -> float:
    """Exact energy per site of the infinite square-lattice Ising model (Onsager 1944):
    -coth(2b) (1 + (2/pi)(2 tanh(2b)^2 - 1) K(k)), k = 2 sinh(2b)/cosh(2b)^2, with the complete elliptic integral
    K(k) = pi / (2 AGM(1, sqrt(1 - k^2))). At beta_c the logarithmic singularity of K is cancelled by its zero
    prefactor and the energy is -sqrt(2)."""
    c = 2 * math.tanh(2 * beta) ** 2 - 1
    coth = 1 / math.tanh(2 * beta)
    k = 2 * math.sinh(2 * beta) / math.cosh(2 * beta) ** 2
    a, b = 1.0, math.sqrt(max(1 - k * k, 0.0))
    if abs(c) < 1e-12 or b == 0.0:  # at (or within rounding of) beta_c: c K(k) -> 0
        return -coth
    for _ in range(60):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    K = math.pi / (2 * a)
    return -coth * (1 + 2 / math.pi * c * K)


def simulated_annealing(cost, x0, neighbour, n: int, T0: float, rng, cooling: float = 0.999) -> dict:
    """Minimise cost by Metropolis moves at a decreasing temperature T_k = T0 cooling^k; neighbour(x, rng) proposes
    a move. Returns the best state found and the cost history."""
    x = x0
    c = cost(x)
    best, best_c = x, c
    hist = np.empty(n)
    T = T0
    for k in range(n):
        y = neighbour(x, rng)
        cy = cost(y)
        if cy <= c or rng.random() < math.exp(-(cy - c) / T):
            x, c = y, cy
            if c < best_c:
                best, best_c = x, c
        hist[k] = c
        T *= cooling
    return {"x": best, "cost": best_c, "history": hist}
