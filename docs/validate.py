"""Validation of engstoch against SciPy, closed-form results of probability theory, published reference values
and the theory itself (convergence orders, coverage, limit theorems). Run:  python docs/validate.py
(writes docs/VALIDATION.md). Statistical checks use fixed seeds and tolerances of several standard errors.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import signal as ssg
from scipy import special as ssp
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import qmc as sqmc

from engstoch import brownian as bm
from engstoch import ctmc, datasets, des, mcmc, models, qmc, queues, renewal, sde
from engstoch import gaussian as gp
from engstoch import markov as mk
from engstoch import montecarlo as mc
from engstoch import output as oa
from engstoch import poisson as pp
from engstoch import rng as rg
from engstoch import rngtests as rt
from engstoch import special as sf
from engstoch import timeseries as ts
from engstoch import walks as wk

RESULTS: list[tuple] = []
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # NumPy 2 renamed trapz


_T0 = [__import__("time").perf_counter(), None]


def check(section, name, reference, value, expected, rtol=0.0, atol=0.0):
    if section != _T0[1]:
        now = __import__("time").perf_counter()
        if _T0[1] is not None and "--timing" in __import__("sys").argv:
            print(f"  {_T0[1]}: {now - _T0[0]:.1f} s", flush=True)
        _T0[:] = [now, section]
    ok = bool(np.isclose(float(value), float(expected), rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))
    if not ok:
        print(f"FAIL: {section} | {name}: computed {value!r}, expected {expected!r}")


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def R(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ 1. special functions
S = "Special functions"
p = np.array([1e-12, 1e-6, 0.01, 0.2, 0.5, 0.8, 0.975, 1 - 1e-9])
check(S, "normal quantile (AS 241) at 8 probabilities from 1e-12 to 1 - 1e-9", "scipy.stats.norm.ppf", maxerr(sf.norm_ppf(p), stats.norm.ppf(p)), 0, atol=1e-12)
x = np.linspace(-8, 8, 33)
check(S, "normal distribution function on [-8, 8]", "scipy.stats.norm.cdf", maxerr(sf.norm_cdf(x), stats.norm.cdf(x)), 0, atol=1e-15)
check(S, "Phi(-8) relative accuracy in the far tail", "scipy", float(sf.norm_cdf(-8.0) / stats.norm.cdf(-8.0)), 1.0, rtol=1e-13)
aa, xx = np.array([0.3, 1, 2.5, 10, 50, 200]), np.array([0.1, 1, 4, 8, 60, 180])
check(S, "regularised lower incomplete gamma P(a, x), 6 points", "scipy.special.gammainc", maxerr(sf.gammainc(aa, xx), ssp.gammainc(aa, xx)), 0, atol=1e-13)
check(S, "upper incomplete gamma Q(a, x) in the tail, relative", "scipy.special.gammaincc", float(sf.gammaincc(5.0, 80.0) / ssp.gammaincc(5.0, 80.0)), 1.0, rtol=1e-11)
ab = [(0.5, 0.5, 0.3), (2, 3, 0.4), (10, 20, 0.35), (50, 5, 0.9), (1, 1, 0.77)]
check(S, "regularised incomplete beta I_x(a, b), 5 points", "scipy.special.betainc", max(abs(sf.betainc(a, b, z) - ssp.betainc(a, b, z)) for a, b, z in ab), 0, atol=1e-13)
check(S, "chi-square upper tail P(X > 100), 10 df", "scipy.stats.chi2.sf", float(sf.chi2_sf(100, 10) / stats.chi2.sf(100, 10)), 1, rtol=1e-12)
check(S, "chi-square quantiles (0.025, 0.5, 0.95; 1, 7, 40 df)", "scipy.stats.chi2.ppf", max(abs(sf.chi2_ppf(q, k) - stats.chi2.ppf(q, k)) for q in (0.025, 0.5, 0.95) for k in (1, 7, 40)), 0, atol=1e-9)
check(S, "Student t quantiles (0.005, 0.975; 1, 2.5, 4, 30 df)", "scipy.stats.t.ppf", max(abs(sf.t_ppf(q, v) - stats.t.ppf(q, v)) for q in (0.005, 0.975) for v in (1, 2.5, 4, 30)), 0, atol=1e-9)
check(S, "Student t distribution function", "scipy.stats.t.cdf", maxerr([sf.t_cdf(z, 3.0) for z in (-4, -1, 0.3, 2.5)], stats.t.cdf([-4, -1, 0.3, 2.5], 3.0)), 0, atol=1e-13)
check(S, "Poisson distribution function P(N <= 5), lambda = 3.2", "scipy.stats.poisson.cdf", float(sf.poisson_cdf(5, 3.2)), float(stats.poisson.cdf(5, 3.2)), atol=1e-14)
kx = np.array([0.1, 0.3, 0.5, 1.0, 1.36, 2.0])
check(S, "Kolmogorov limiting distribution P(K > x)", "scipy.stats.kstwobign.sf", maxerr(sf.kolmogorov_sf(kx), stats.kstwobign.sf(kx)), 0, atol=1e-10)
check(S, "KS p-value with Stephens' correction, n = 50, D = 0.1", "scipy.stats.kstwo.sf (exact)", float(sf.ks_sf(0.1, 50)), float(stats.kstwo.sf(0.1, 50)), atol=0.02)

# ------------------------------------------------------------------ 2. generators
S = "Pseudo-random generators"
pm = rg.LCG.park_miller(1)
pm.integers(9999)
check(S, "Park-Miller minimal standard: x_10000 from seed 1 is 1043618065", "Park & Miller (1988)", int(pm.integers(1)[0]), 1043618065)
check(S, "PCG32 (seed 42, stream 54): first output 0xa15c02b7", "O'Neill's pcg32-demo", rg.PCG32(42, 54).next32(), 0xA15C02B7)
pc = rg.PCG32(42, 54)
pc.next32()
check(S, "... second and third outputs 0x7b47f409, 0xba1d3330", "O'Neill's pcg32-demo", pc.next32() * 2**32 + pc.next32(), 0x7B47F409 * 2**32 + 0xBA1D3330)
check(S, "Hull-Dobell: a = 1103515245, c = 12345, m = 2^31 has full period", "theorem", float(rg.full_period_conditions(1103515245, 12345, 2**31)), 1)
check(S, "... the period of x -> 5x + 3 mod 16 is 16", "Hull-Dobell", rg.LCG(5, 3, 16).period(), 16)
check(S, "... while x -> 4x + 3 mod 16 (a - 1 odd) fails the conditions", "Hull-Dobell", float(rg.full_period_conditions(4, 3, 16)), 0)
check(S, "Park-Miller multiplier 16807 is a primitive root: period 2^31 - 2", "Park & Miller", float(pow(16807, (2**31 - 2) // 2, 2**31 - 1) != 1 and pow(16807, (2**31 - 2) // 3, 2**31 - 1) != 1), 1)
check(S, "RANDU triples lie on 15 planes: 9u_k - 6u_k+1 + u_k+2 is an integer", "Marsaglia (1968)", rt.randu_planes(rg.LCG.randu(1).random(30000))["n_planes"], 15)
check(S, "... to rounding", "exact", rt.randu_planes(rg.LCG.randu(7).random(3000))["max_deviation"], 0, atol=1e-6)
sp2 = rt.spectral_test_2d(16807, 2**31 - 1)
check(S, "2-D spectral test of the minimal standard: nu_2^2 = 282475250", "Knuth TAOCP 3.3.4", sp2["nu"] ** 2, 282475250, rtol=1e-12)
check(S, "xorshift64* passes the runs-up test (p > 0.001)", "Knuth's runs test", float(rt.runs_up_test(rg.XorShift64Star(12345).random(20000))["p_value"] > 0.001), 1)
check(S, "PCG32 passes the serial pairs test (p > 0.001)", "chi-square, 99 df", float(rt.serial_test(rg.PCG32(1, 1).random(40000))["p_value"] > 0.001), 1)
check(S, "a full-period LCG mod 16 fails the runs-up test (p < 1e-10)", "Knuth's runs test", float(rt.runs_up_test(rg.LCG(5, 3, 16).random(10000))["p_value"] < 1e-10), 1)
ul = rg.LCG.randu(1).random(30000)
check(S, "... yet RANDU passes the 1-D chi-square and the serial pairs test (p > 0.001): the flaw is 3-D", "chi-square", float(rt.chi_square_uniform(ul)["p_value"] > 0.001 and rt.serial_test(ul)["p_value"] > 0.001), 1)

# ------------------------------------------------------------------ 3. variates and tests
S = "Random variates and goodness of fit"
g = R(1)
check(S, "Box-Muller normals: KS p-value > 0.01 (n = 20000)", "scipy.stats.kstest", float(stats.kstest(rg.box_muller(g, 20000), "norm").pvalue > 0.01), 1)
pol = rg.polar_normal(R(2), 40000)
check(S, "polar method acceptance rate pi/4", "geometry", pol["acceptance"], math.pi / 4, atol=0.01)
check(S, "... normals pass KS", "scipy.stats.kstest", float(stats.kstest(pol["x"], "norm").pvalue > 0.01), 1)
rou = rg.ratio_of_uniforms_normal(R(3), 40000)
check(S, "ratio-of-uniforms acceptance sqrt(pi e)/4", "Kinderman & Monahan (1977)", rou["acceptance"], math.sqrt(math.pi * math.e) / 4, atol=0.01)
rej = rg.rejection(lambda x: 6 * x * (1 - x), lambda m: R(4).random(m), lambda x: np.ones_like(x), 1.5, R(5), 20000)
check(S, "rejection from Beta(2, 2) with M = 1.5: acceptance 1/M", "theory", rej["acceptance"], 1 / 1.5, atol=0.01)
g6 = R(6)
x = rg.gamma(2.5, g6.standard_normal, g6, 40000)
check(S, "Marsaglia-Tsang Gamma(2.5): KS p-value > 0.01", "scipy.stats.gamma", float(stats.kstest(x, "gamma", args=(2.5,)).pvalue > 0.01), 1)
x = rg.gamma(0.4, g6.standard_normal, g6, 40000)
check(S, "Gamma(0.4) by the U^(1/a) boost: KS p-value > 0.01", "scipy.stats.gamma", float(stats.kstest(x, "gamma", args=(0.4,)).pvalue > 0.01), 1)
check(S, "Beta(2, 5) from two gammas: KS p-value > 0.01", "scipy.stats.beta", float(stats.kstest(rg.beta(2, 5, g6.standard_normal, g6, 20000), "beta", args=(2, 5)).pvalue > 0.01), 1)
probs = np.array([0.1, 0.2, 0.3, 0.4])
tab = rg.alias_table(probs)
recon = [(tab[0][j] + sum(1 - tab[0][i] for i in range(4) if tab[1][i] == j and i != j)) / 4 for j in range(4)]
check(S, "alias table: columns and aliases reproduce the pmf exactly", "Vose (1991)", maxerr(recon, probs), 0, atol=1e-14)
cnt = np.bincount(rg.alias_sample(rg.alias_table(probs), R(7), 200000), minlength=4)
check(S, "alias sampling: chi-square p-value > 0.01", "chi-square, 3 df", float(rt.chi_square(cnt, probs * 200000)["p_value"] > 0.01), 1)
pv = rg.poisson(3.5, R(8), 100000)
check(S, "Poisson(3.5) by inversion: chi-square p-value > 0.01 (counts 0..9, 10+)", "scipy.stats.poisson", float(rt.chi_square(np.bincount(np.minimum(pv, 10), minlength=11), 100000 * np.r_[stats.poisson.pmf(np.arange(10), 3.5), stats.poisson.sf(9, 3.5)])["p_value"] > 0.01), 1)
check(S, "Poisson(200): variance = mean", "theory", float(np.var(rg.poisson(200, R(9), 100000)) / 200), 1, atol=0.02)
check(S, "Binomial(20, 0.3) mean", "np", float(np.mean(rg.binomial(20, 0.3, R(10), 100000))), 6.0, atol=0.03)
check(S, "geometric(0.25) mean 1/p", "theory", float(np.mean(rg.geometric(0.25, R(11), 100000))), 4.0, atol=0.04)
g12 = R(12)
mv = rg.multivariate_normal([1, -1], [[2, 0.8], [0.8, 1]], g12.standard_normal, 100000)
check(S, "multivariate normal by Cholesky: sample covariance", "target", maxerr(np.cov(mv.T), [[2, 0.8], [0.8, 1]]), 0, atol=0.03)
cop = rg.gaussian_copula([[1, 0.7], [0.7, 1]], [lambda u: stats.expon.ppf(u), lambda u: stats.uniform.ppf(u)], g12.standard_normal, 50000)
check(S, "Gaussian copula: Spearman rho = (6/pi) asin(r/2)", "copula theory", float(stats.spearmanr(cop[:, 0], cop[:, 1])[0]), 6 / math.pi * math.asin(0.35), atol=0.01)
xk = R(13).normal(size=400)
check(S, "KS statistic", "scipy.stats.kstest", rt.ks_test(xk, stats.norm.cdf)["statistic"], float(stats.kstest(xk, "norm").statistic), atol=1e-14)
obs, ex = np.array([18, 22, 30, 30]), np.array([25, 25, 25, 25])
check(S, "Pearson chi-square p-value", "scipy.stats.chisquare", rt.chi_square(obs, ex)["p_value"], float(stats.chisquare(obs, ex).pvalue), atol=1e-12)
pvals = rt.second_level(rt.chi_square_uniform, R(14).random, 60, 5000)["p_values"]
check(S, "second-level test: chi-square p-values of a good generator are uniform", "KS on 60 p-values", float(stats.kstest(pvals, "uniform").pvalue > 0.01), 1)
check(S, "Fisher-Yates permutation is a permutation", "construction", float(np.array_equal(np.sort(rg.random_permutation(50, R(15))), np.arange(50))), 1)

# ------------------------------------------------------------------ 4. Monte Carlo and variance reduction
S = "Monte Carlo and variance reduction"
e1 = math.e - 1
cover = [mc.mean_ci(np.exp(R(100 + k).random(200)))["ci"] for k in range(1000)]
check(S, "95 % normal intervals for E[e^U], n = 200: coverage over 1000 runs", "nominal", oa.coverage([c[0] for c in cover], [c[1] for c in cover], e1)["coverage"], 0.95, atol=0.02)
cover = [mc.mean_ci(R(2000 + k).exponential(size=10), method="t")["ci"] for k in range(2000)]
check(S, "t intervals for a skewed mean with n = 10 undercover (Exp(1): about 0.90)", "Edgeworth / known", oa.coverage([c[0] for c in cover], [c[1] for c in cover], 1.0)["coverage"], 0.90, atol=0.02)
anti = mc.antithetic(np.exp, 50000, R(16))
check(S, "antithetic estimate of int e^u du", "e - 1", anti["estimate"], e1, atol=4 * anti["std_error"])
v_plain = (0.5 * (math.e**2 - 1) - e1**2) / 2
v_anti = (0.5 * (0.5 * (math.e**2 - 1)) + 0.5 * math.e - e1**2) / 1
check(S, "... variance reduction factor = Var(e^U)/2 / Var((e^U + e^(1-U))/2) = 30.6", "closed form", anti["variance_reduction"], v_plain / v_anti, rtol=0.05)
u = R(17).random(100000)
cv = mc.control_variate(np.exp(u), u, 0.5)
check(S, "control variate U for e^U: estimate", "e - 1", cv["estimate"], e1, atol=4 * cv["std_error"])
beta_star = 12 * (1 - e1 / 2)
check(S, "... beta* = Cov(e^U, U)/Var(U) = 12 (1 - (e-1)/2) = 1.690", "closed form", cv["beta"], beta_star, atol=0.01)
rho2 = (beta_star**2 / 12) / (0.5 * (math.e**2 - 1) - e1**2)
check(S, "... variance reduction 1/(1 - rho^2) = 62", "closed form", cv["variance_reduction"], 1 / (1 - rho2), rtol=0.05)
r4 = 1 - stats.norm.cdf(4)
isr = mc.importance_sampling(lambda z: (z > 4).astype(float), lambda z: -0.5 * z**2, lambda n: R(18).normal(4, 1, n), lambda z: -0.5 * (z - 4) ** 2, 100000)
check(S, "importance sampling with the shifted normal: P(Z > 4)", "1 - Phi(4) = 3.17e-5", isr["estimate"], r4, rtol=0.03)
check(S, "... relative error below 1 % with 10^5 samples (plain MC: 56 %)", "theory", float(mc.relative_error(isr) < 0.01), 1)
sn = mc.importance_sampling(lambda z: z**2, lambda z: -0.5 * z**2, lambda n: R(19).standard_t(5, n), lambda z: stats.t.logpdf(z, 5), 100000, self_normalised=True)
check(S, "self-normalised IS with unnormalised target: E[Z^2] = 1 from t5 proposals", "theory", sn["estimate"], 1.0, atol=4 * sn["std_error"])
st = mc.stratified(np.exp, 10000, 20, R(20))
check(S, "stratified sampling (20 strata) estimate", "e - 1", st["estimate"], e1, atol=4 * st["std_error"])
check(S, "... variance below plain MC by about the factor of strata^2 for smooth h (> 100)", "theory", float(((0.5 * (math.e**2 - 1) - e1**2) / 10000) / st["std_error"] ** 2 > 100), 1)
ny, pr = mc.stratified(lambda v: (v > 0.9) * v * 10, 4000, 10, R(21), "neyman"), mc.stratified(lambda v: (v > 0.9) * v * 10, 4000, 10, R(21))
check(S, "Neyman allocation beats proportional when the variance sits in one stratum", "theory", float(ny["std_error"] < 0.6 * pr["std_error"]), 1)
mci = mc.integrate(lambda z: np.sum(z**2, axis=1), [0, 0, 0], [1, 1, 1], 200000, R(22))
check(S, "MC integral of |x|^2 over the unit cube", "1", mci["estimate"], 1.0, atol=4 * mci["std_error"])
check(S, "sample size for half-width 0.01 with sigma = 1 at 95 %", "(1.96/0.01)^2", mc.sample_size(1, 0.01), 38415)
wil = mc.proportion_ci(0, 1000)
check(S, "Wilson interval upper limit with 0 successes in 1000", "z^2/(n + z^2)", wil["ci"][1], 1.959963984540054**2 / (1000 + 1.959963984540054**2), rtol=1e-12)
crn = mc.common_random_numbers(lambda r: np.mean(queues.lindley(r.exponential(1 / 0.8, 500), r.exponential(1.0, 500))), lambda r: np.mean(queues.lindley(r.exponential(1 / 0.8, 500), r.exponential(0.95, 500))), 300)
check(S, "common random numbers reduce the variance of a difference of queues (> 5x)", "Glasserman ch. 4", float(crn["variance_reduction"] > 5), 1)

# ------------------------------------------------------------------ 5. quasi-Monte Carlo
S = "Quasi-Monte Carlo"
check(S, "Sobol points, 1024 x 10 (Joe-Kuo directions)", "scipy.stats.qmc.Sobol(scramble=False)", maxerr(qmc.sobol(1024, 10), sqmc.Sobol(10, scramble=False).random(1024)), 0)
check(S, "Halton points, 500 x 6", "scipy.stats.qmc.Halton(scramble=False)", maxerr(qmc.halton(500, 6), sqmc.Halton(6, scramble=False).random(500)), 0, atol=1e-15)
check(S, "van der Corput base 3: 7 = 21_3 maps to 0.12_3 = 5/9", "radical inverse", float(qmc.van_der_corput(8, 3)[7]), 5 / 9, atol=1e-15)
xr = R(23).random((300, 4))
check(S, "L2-star discrepancy by Warnock's formula", "scipy.stats.qmc.discrepancy(method='L2-star')", qmc.l2_star_discrepancy(xr), float(sqmc.discrepancy(xr, method="L2-star")), rtol=1e-10)
check(S, "Sobol's first 2^m points form a (0, m, 2)-net: every 1/16 x 1/16 elementary box of area 1/256 holds one point", "net property", float(np.all(np.bincount((qmc.sobol(256, 2)[:, 0] * 16).astype(int) * 16 + (qmc.sobol(256, 2)[:, 1] * 16).astype(int), minlength=256) == 1)), 1)
dq, dm = qmc.l2_star_discrepancy(qmc.sobol(1024, 3)), np.mean([qmc.l2_star_discrepancy(R(30 + k).random((1024, 3))) for k in range(5)])
check(S, "Sobol L2-star discrepancy about a tenth of random points' at n = 1024 (ratio < 0.2)", "QMC theory", float(dq / dm < 0.2), 1)
check(S, "rank-1 lattice: every 1-D projection of a Korobov lattice is the full grid {i/n}", "construction", float(all(np.array_equal(np.sort(qmc.korobov(101, 3, 12)[:, j]), np.arange(101) / 101) for j in range(3))), 1)
truth = qmc.genz_product_peak_integral(5)
rq = qmc.randomised_qmc(qmc.genz_product_peak, qmc.sobol(1024, 5), 20, R(24))
check(S, "randomised Sobol estimate of Genz's product-peak integral, d = 5", "closed form", rq["estimate"], truth, atol=4 * rq["std_error"])
pm_err = np.std(qmc.genz_product_peak(R(25).random((20480, 5)))) / math.sqrt(20480)
check(S, "... with a standard error 10x smaller than plain MC at equal cost", "QMC theory", float(rq["std_error"] < pm_err / 10), 1)
errs = [abs(np.mean(np.prod(np.mod(qmc.sobol(2**m, 2) + R(26).random(2), 1) ** 2 * 3, axis=1)) - 1) for m in range(6, 15)]
check(S, "RQMC error of a smooth 2-D integrand decays faster than n^-1/2 (fitted slope < -0.8)", "Owen; theory", float(np.polyfit(np.arange(6, 15) * math.log(2), np.log(np.maximum(errs, 1e-16)), 1)[0] < -0.8), 1)

# ------------------------------------------------------------------ 6. discrete-time chains
S = "Discrete-time Markov chains"
w = mk.MarkovChain([[0.9, 0.1], [0.5, 0.5]])
check(S, "two-state stationary law pi_1 = a/(a + b)", "closed form", float(w.stationary()[1]), 0.1 / 0.6, atol=1e-14)
check(S, "... second eigenvalue 1 - a - b", "closed form", w.spectral_gap()["slem"], 0.4, atol=1e-14)
check(S, "... mean first passage dry -> wet = 1/a", "geometric", float(w.mean_first_passage()[0, 1]), 10.0, atol=1e-12)
check(S, "... Kac: mean return time to wet = 1/pi_wet", "Kac's lemma", float(w.mean_first_passage()[1, 1]), 6.0, atol=1e-12)
P3 = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.1, 0.4, 0.5]])
c3 = mk.MarkovChain(P3)
ev, vec = sla.eig(P3.T)
pi_ref = np.real(vec[:, np.argmin(np.abs(ev - 1))])
check(S, "stationary law of a 3-state chain", "left Perron eigenvector (scipy.linalg.eig)", maxerr(c3.stationary(), pi_ref / pi_ref.sum()), 0, atol=1e-14)
check(S, "... equals the power-iteration limit", "power method", maxerr(c3.stationary(), c3.stationary_power()["pi"]), 0, atol=1e-12)
mfp = c3.mean_first_passage()
check(S, "Kemeny's constant sum_j pi_j m_ij is the same for every start", "Kemeny & Snell", float(np.ptp([np.sum(c3.stationary() * (mfp[i] - np.diag(mfp) * (np.arange(3) == i))) for i in range(3)])), 0, atol=1e-12)
check(S, "... and equals trace(Z) - 1", "Kemeny & Snell", float(np.sum(c3.stationary() * mfp[0]) - c3.stationary()[0] * mfp[0, 0]), c3.kemeny_constant(), atol=1e-12)
eh = models.ehrenfest(10)
check(S, "Ehrenfest urn: stationary law Binomial(10, 1/2)", "detailed balance", maxerr(eh["chain"].stationary(), stats.binom.pmf(np.arange(11), 10, 0.5)), 0, atol=1e-14)
check(S, "... period 2", "parity", eh["chain"].classify()["classes"][0]["period"], 2)
check(S, "... reversible", "detailed balance", float(eh["chain"].is_reversible()), 1)
check(S, "... lazy version: spectral gap 1/N", "Diaconis", eh["lazy"].spectral_gap()["gap"], 0.1, atol=1e-12)
G = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
check(S, "random walk on a graph: pi_i = deg(i) / 2|E|", "theory", maxerr(mk.random_walk_on_graph(G).stationary(), G.sum(axis=1) / G.sum()), 0, atol=1e-14)
red = mk.MarkovChain([[0.5, 0.5, 0, 0], [0.3, 0.7, 0, 0], [0.2, 0, 0.3, 0.5], [0, 0, 0, 1]])
cl = red.classify()
check(S, "classification: 3 classes, {0, 1} and {3} closed, {2} transient", "Tarjan + closure", float([c["recurrent"] for c in cl["classes"]] == [True, False, True] and len(cl["classes"]) == 3), 1)
gr = np.zeros((7, 7))
gr[0, 0] = gr[6, 6] = 1
for i in range(1, 6):
    gr[i, i + 1], gr[i, i - 1] = 0.45, 0.55
ab = mk.MarkovChain(gr).absorption()
check(S, "gambler's ruin by the fundamental matrix: P(reach 6 from 3), p = 0.45", "closed form", float(ab["B"][2, 1]), 1 - wk.ruin_probability(3, 6, 0.45), atol=1e-14)
check(S, "... expected duration from 3", "closed form", float(ab["expected_steps"][2]), wk.ruin_duration(3, 6, 0.45), atol=1e-12)
check(S, "hitting probabilities agree with the absorption matrix", "two methods", maxerr(mk.MarkovChain(gr).hitting_probabilities([6])[1:6], ab["B"][:, 1]), 0, atol=1e-14)
fair = np.zeros((11, 11))
fair[0, 0] = fair[10, 10] = 1
for i in range(1, 10):
    fair[i, i + 1] = fair[i, i - 1] = 0.5
abf = mk.MarkovChain(fair).absorption()
check(S, "fair ruin: variance of the duration from i = 5, N = 10", "closed form i(N-i)((N-i)^2 + i^2 - 2)/3", float(abf["variance_steps"][4]), 5 * 5 * (25 + 25 - 2) / 3, atol=1e-9)
sl = models.snakes_and_ladders()["chain"]
sims = []
gs = R(27)
for _ in range(20000):
    path = sl.simulate(400, 0, gs)
    sims.append(int(np.argmax(path == 36)))
check(S, "snakes and ladders: expected throws by N 1 vs 20000 simulated games", "simulation", float(np.mean(sims)), float(sl.absorption()["expected_steps"][0]), rtol=0.02)
est = mk.estimate(w.simulate(200000, 0, R(28)), 2)
check(S, "MLE of P from a 200000-step path", "true P", maxerr(est["P"], w.P), 0, atol=4 * float(est["std_error"].max()))
check(S, "order test does not reject a first-order chain (p > 0.01)", "Anderson & Goodman", float(mk.order_test(w.simulate(50000, 0, R(29)), 2)["p_value"] > 0.01), 1)
check(S, "... and rejects a second-order one (p < 1e-6)", "Anderson & Goodman", float(mk.order_test(np.array([0, 0, 1, 1] * 5000) ^ (R(30).random(20000) < 0.1), 2)["p_value"] < 1e-6), 1)
check(S, "TV distance to stationarity of the 2-state chain decays as |1 - a - b|^k", "closed form", w.tv_distance(6) / w.tv_distance(5), 0.4, atol=1e-10)
web = models.web_graph()["adjacency"]
Pg = np.where(web.sum(1, keepdims=True) > 0, web / np.maximum(web.sum(1, keepdims=True), 1), 1 / 6)
Gm = 0.85 * Pg + 0.15 / 6
check(S, "PageRank equals the power-iteration fixed point of the Google matrix", "Brin & Page", maxerr(mk.pagerank(web), np.linalg.matrix_power(Gm.T, 200) @ np.full(6, 1 / 6)), 0, atol=1e-12)
inv = models.inventory_chain()
check(S, "(s, S) inventory chain rows are stochastic and its stationary law sums to 1", "construction", float(np.allclose(inv["chain"].P.sum(1), 1) and abs(inv["chain"].stationary().sum() - 1) < 1e-14), 1)
check(S, "Ehrenfest chain is lumpable by the symmetry k <-> N - k but not into {0..4}, {5..10}", "Kemeny & Snell", float(mk.lumpable(eh["chain"].P, [[k, 10 - k] for k in range(5)] + [[5]]) and not mk.lumpable(eh["chain"].P, [list(range(5)), list(range(5, 11))])), 1)
check(S, "product chain of two independent chains has the product stationary law", "Kronecker", maxerr(mk.product(w, c3).stationary(), np.kron(w.stationary(), c3.stationary())), 0, atol=1e-14)

# ------------------------------------------------------------------ 7. random walks and branching
S = "Random walks and branching"
check(S, "fair gambler's ruin P(ruin from 3 of 10)", "1 - i/N", wk.ruin_probability(3, 10, 0.5), 0.7, atol=1e-15)
check(S, "fair game expected duration i (N - i)", "closed form", wk.ruin_duration(3, 10, 0.5), 21, atol=1e-12)
wsim = wk.simple_walk(400, R(31), 0.45, 20000, start=3)
hit0 = np.argmax(wsim <= 0, axis=1)
hitN = np.argmax(wsim >= 10, axis=1)
done = (wsim <= 0).any(1) | (wsim >= 10).any(1)
ruin_sim = np.mean(np.where((wsim <= 0).any(1) & (~(wsim >= 10).any(1) | (hit0 < hitN)), 1, 0)[done])
check(S, "biased ruin probability vs 20000 simulated walks, p = 0.45", "closed form", ruin_sim, wk.ruin_probability(3, 10, 0.45), atol=0.01)
check(S, "reflection principle: P(max_{k<=20} S_k >= 4) exact enumeration", "brute force over 2^20 paths", wk.max_distribution(20, 4), float(np.mean(np.max(np.cumsum(2 * ((np.arange(2**20)[:, None] >> np.arange(20)) & 1) - 1, axis=1), axis=1) >= 4)), atol=1e-14)
check(S, "paths from 0 to 2 in 10 steps touching 4 = paths to 6", "reflection", wk.reflection_touching(10, 4, 2), math.comb(10, 8))
bal = np.mean([np.all(np.cumsum(np.where(R(32 + k).permutation(np.r_[np.ones(6), -np.ones(4)]) > 0, 1, -1)) > 0) for k in range(20000)])
check(S, "ballot theorem: A = 6, B = 4 stays ahead with probability 1/5", "Bertrand (1887)", bal, wk.ballot(6, 4), atol=0.01)
check(S, "first return at step 2: probability 1/2", "closed form", wk.first_return_probability(1), 0.5, atol=1e-15)
check(S, "first-return probabilities sum towards 1 (recurrence), slowly: 1 - sum_{n<=N} ~ 1/sqrt(pi N)", "Polya, d = 1", 1 - sum(wk.first_return_probability(n) for n in range(1, 5001)), 1 / math.sqrt(math.pi * 5000), rtol=0.01)
ws = wk.simple_walk(1000, R(33), n_paths=20000)
check(S, "arcsine law: P(fraction of time positive <= 0.1)", "(2/pi) asin(sqrt(0.1)) = 0.205", float(np.mean(np.mean(ws[:, 1:] > 0, axis=1) <= 0.1)), float(wk.arcsine_cdf(0.1)), atol=0.015)
r3 = wk.return_probability_by(3, 4000, 4000, R(34))
check(S, "Polya: the walk on Z^3 returns with probability 0.3405 (4000 steps: slightly less)", "Watson's integral", r3["fraction"], wk.POLYA_3D, atol=0.03)
check(S, "Galton-Watson Poisson(1.5): extinction probability solves q = e^{1.5(q - 1)}", "fixed point", wk.extinction_probability(wk.pgf_poisson(1.5))["q"], float(-ssp.lambertw(-1.5 * math.exp(-1.5)).real / 1.5), atol=1e-12)
check(S, "geometric offspring p = 0.4: extinction p/(1 - p)", "closed form", wk.extinction_probability(wk.pgf_geometric(0.4))["q"], 2 / 3, atol=1e-12)
check(S, "subcritical offspring: extinction certain", "m < 1", wk.extinction_probability(wk.pgf_poisson(0.8))["q"], 1.0, atol=1e-12)
ext = np.mean([wk.simulate_galton_watson(lambda k, r: r.poisson(1.5, k), 40, R(35 + i))[-1] == 0 for i in range(4000)])
check(S, "simulated extinction frequency for Poisson(1.5)", "pgf fixed point", ext, wk.extinction_probability(wk.pgf_poisson(1.5))["q"], atol=0.025)
gm = wk.generation_moments(wk.pgf_poisson(1.2), 6)
z6 = np.array([wk.simulate_galton_watson(lambda k, r: r.poisson(1.2, k), 6, R(5000 + i))[-1] for i in range(20000)])
check(S, "E[Z_6] = m^6", "theory", float(z6.mean()), gm["mean"], rtol=0.04)
check(S, "Var[Z_6] = sigma^2 m^5 (m^6 - 1)/(m - 1)", "theory", float(z6.var()), gm["var"], rtol=0.07)
check(S, "P(Z_n = 0) iterates converge to q", "pgf iteration", wk.extinction_by_generation(wk.pgf_poisson(1.5), 200)[-1], wk.extinction_probability(wk.pgf_poisson(1.5))["q"], atol=1e-12)

# ------------------------------------------------------------------ 8. Poisson processes
S = "Poisson processes"
gaps = pp.interarrival_gaps(pp.homogeneous(2.0, 20000.0, R(36)))
check(S, "interarrival gaps of a rate-2 process are Exp(2): KS p > 0.01", "scipy.stats.expon", float(stats.kstest(gaps, "expon", args=(0, 0.5)).pvalue > 0.01), 1)
cts = [len(pp.homogeneous_order_statistics(3.0, 2.0, R(1000 + k))) for k in range(20000)]
check(S, "order-statistics construction: N(2) ~ Poisson(6), mean", "theory", float(np.mean(cts)), 6.0, atol=0.06)
check(S, "... and variance", "theory", float(np.var(cts)), 6.0, atol=0.2)
lam = lambda t: 5 + 4 * np.sin(t)  # noqa: E731
th = [len(pp.thinning(lam, 9.0, 10.0, R(2000 + k))["times"]) for k in range(4000)]
check(S, "thinning: E N(10) = int_0^10 (5 + 4 sin t) dt", "50 + 4(1 - cos 10)", float(np.mean(th)), 50 + 4 * (1 - math.cos(10)), rtol=0.01)
check(S, "... and Var N(10) equals the mean (Poisson counts)", "theory", float(np.var(th) / np.mean(th)), 1.0, atol=0.06)
tcs = [len(pp.time_change(lambda e: np.sqrt(e), 100.0, R(3000 + k))) for k in range(4000)]
check(S, "time change with Lambda(t) = t^2: E N = Lambda(10) = 100", "theory", float(np.mean(tcs)), 100.0, rtol=0.01)
check(S, "cumulative intensity by the trapezoidal rule", "50 + 4(1 - cos 10)", pp.cumulative_intensity(lam, 10.0, 20001), 50 + 4 * (1 - math.cos(10)), atol=1e-6)
cp = [pp.compound(2.0, 5.0, lambda n, r: r.exponential(3.0, n), R(4000 + k))["total"] for k in range(20000)]
cmom = pp.compound_moments(2.0, 5.0, 3.0, 18.0)
check(S, "compound Poisson mean lambda t E[Y]", "Wald", float(np.mean(cp)), cmom["mean"], rtol=0.02)
check(S, "compound Poisson variance lambda t E[Y^2]", "theory", float(np.var(cp)), cmom["var"], rtol=0.04)
spl = [pp.split(pp.homogeneous(5.0, 4.0, R(5000 + k)), [0.3, 0.7], R(6000 + k)) for k in range(5000)]
n1, n2 = np.array([len(s[0]) for s in spl]), np.array([len(s[1]) for s in spl])
check(S, "splitting: typed counts uncorrelated (independent Poisson streams)", "Poisson splitting", float(np.corrcoef(n1, n2)[0, 1]), 0.0, atol=0.04)
check(S, "... with mean lambda p t", "theory", float(n1.mean()), 6.0, rtol=0.02)
dsc = [pp.spatial_disc(2.0, 3.0, R(7000 + k)) for k in range(3000)]
check(S, "spatial Poisson on a disc: E N = lambda pi r^2", "theory", float(np.mean([len(d) for d in dsc])), 2 * math.pi * 9, rtol=0.01)
dist = np.array([np.min(np.hypot(*d.T)) for d in dsc if len(d)])
check(S, "nearest point to the centre: P(D <= 0.3) = 1 - exp(-lambda pi 0.09)", "void probability", float(np.mean(dist <= 0.3)), float(pp.nearest_neighbour_cdf(0.3, 2.0)), atol=0.02)
gw = pp.rate_mle(10, 2.0)
check(S, "Garwood 95 % interval for 10 events in T = 2: lower limit", "chi2(0.025, 20)/4 = 2.398", gw["ci"][0], stats.chi2.ppf(0.025, 20) / 4, atol=1e-9)
check(S, "... upper limit", "chi2(0.975, 22)/4 = 9.195", gw["ci"][1], stats.chi2.ppf(0.975, 22) / 4, atol=1e-9)
cvg = np.mean([(lambda r: r["ci"][0] <= 3.0 <= r["ci"][1])(pp.rate_mle(int(R(8000 + k).poisson(6.0)), 2.0)) for k in range(5000)])
check(S, "Garwood intervals are conservative: coverage >= 95 % at lambda T = 6", "exact interval", float(cvg >= 0.95), 1)
check(S, "dispersion test accepts Poisson counts (p > 0.01)", "chi-square", float(pp.dispersion_test(R(37).poisson(4, 200))["p_value"] > 0.01), 1)
check(S, "... and rejects negative-binomial counts with the same mean (p < 1e-6)", "chi-square", float(pp.dispersion_test(R(38).negative_binomial(2, 1 / 3, 200))["p_value"] < 1e-6), 1)
check(S, "conditional uniformity of homogeneous arrivals (KS p > 0.01)", "order statistics", float(pp.uniformity_test(pp.homogeneous(1.0, 2000.0, R(39)), 2000.0)["p_value"] > 0.01), 1)

# ------------------------------------------------------------------ 9. renewal processes
S = "Renewal processes"
t, m = renewal.renewal_function(lambda x: 1 - np.exp(-2 * x), 3.0, 3000)
check(S, "renewal equation for exponential inter-renewals: m(t) = lambda t", "closed form", maxerr(m, renewal.renewal_function_poisson(t, 2.0)), 0, atol=1e-5)
t, m = renewal.renewal_function(lambda x: 1 - np.exp(-2 * x) * (1 + 2 * x), 5.0, 2000)
check(S, "Erlang(2, 2) inter-renewals: m(t) = t - (1 - e^{-4t})/4", "closed form", maxerr(m, renewal.renewal_function_erlang2(t, 2.0)), 0, atol=1e-5)
t2, m2 = renewal.renewal_function(lambda x: 1 - np.exp(-2 * x) * (1 + 2 * x), 5.0, 1000)
check(S, "... the discretisation is second order (error ratio ~4 on halving h)", "convergence", maxerr(m2, renewal.renewal_function_erlang2(t2, 2.0)) / maxerr(m, renewal.renewal_function_erlang2(t, 2.0)), 4.0, rtol=0.1)
wm = renewal.weibull_moments(2.0, 1.0)
tw, mw = renewal.renewal_function(lambda x: 1 - np.exp(-(x**2)), 20.0, 4000)
check(S, "Weibull(2) renewal function approaches the second-order asymptote", "Smith's theorem", float(mw[-1]), float(renewal.asymptote(20.0, wm["mean"], wm["var"])), atol=1e-3)
nsim = renewal.count_at(lambda n, r: r.weibull(2.0, n), 10.0, 4000, R(40))
check(S, "... and agrees with simulation at t = 10", "4000 paths", float(nsim.mean()), float(np.interp(10.0, tw, mw)), atol=0.06)
ins = renewal.inspect(lambda n, r: r.exponential(1.0, n), 50.0, 6000, R(41))
check(S, "inspection paradox: the interval covering t has mean E[X^2]/E[X] = 2 for Exp(1)", "length bias", float(ins["length"].mean()), renewal.length_biased_mean(1, 2), rtol=0.04)
insw = renewal.inspect(lambda n, r: r.weibull(2.0, n), 30.0, 6000, R(42))
check(S, "mean residual life E[X^2]/(2E[X]) for Weibull(2)", "equilibrium law", float(insw["residual"].mean()), renewal.mean_residual_life(wm["mean"], wm["second_moment"]), rtol=0.03)
check(S, "equilibrium cdf of Exp(1) is Exp(1)", "memorylessness", maxerr(renewal.equilibrium_cdf([0.5, 1, 2], lambda s: np.exp(-s), 1.0), 1 - np.exp(-np.array([0.5, 1, 2]))), 0, atol=1e-6)
check(S, "age replacement never pays for exponential lifetimes (cost decreasing in T)", "memorylessness", float(np.all(np.diff(renewal.age_replacement_cost(np.linspace(0.5, 10, 20), lambda s: np.exp(-s), 1.0, 5.0)) < 0)), 1)
oar = renewal.optimal_age_replacement(lambda s: np.exp(-((s / 10) ** 3)), 1.0, 10.0, np.linspace(1, 20, 400))
Tg = np.linspace(1, 20, 400)
check(S, "optimal age replacement for Weibull(3, 10) with c_f/c_p = 10", "brute-force grid of the cost formula", oar["T"], float(Tg[np.argmin([(1 * math.exp(-((T / 10) ** 3)) + 10 * (1 - math.exp(-((T / 10) ** 3)))) / (10 / 3 * math.gamma(1 / 3) * ssp.gammainc(1 / 3, (T / 10) ** 3)) for T in Tg])]), atol=0.05)
check(S, "point availability tends to mu/(lambda + mu)", "alternating renewal", float(renewal.availability_exponential(100.0, 0.1, 0.9)), renewal.availability(10.0, 1 / 0.9), atol=1e-12)
check(S, "renewal-reward: long-run rate E[R]/E[X]", "theorem", renewal.reward_rate(3.0, 1.5), 2.0)

# ------------------------------------------------------------------ 10. continuous-time chains
S = "Continuous-time Markov chains"
gA = R(43).normal(size=(8, 8)) * 2
check(S, "Pade [6/6] scaling-and-squaring exponential, random 8 x 8 of norm ~10", "scipy.linalg.expm", maxerr(ctmc.expm(gA), sla.expm(gA)) / np.max(np.abs(sla.expm(gA))), 0, atol=1e-12)
Q = np.array([[-3.0, 2, 1], [1, -1, 0], [2, 2, -4]])
c = ctmc.CTMC(Q)
check(S, "P(t) = exp(Qt) at t = 1.3", "scipy.linalg.expm", maxerr(c.transition_matrix(1.3), sla.expm(1.3 * Q)), 0, atol=1e-13)
un = c.uniformisation(1.3, 1e-13)
check(S, "uniformisation agrees with the exponential", "scipy.linalg.expm", maxerr(un["P"], sla.expm(1.3 * Q)), 0, atol=1e-12)
check(S, "... within its own a-priori error bound", "Poisson tail", float(maxerr(un["P"], sla.expm(1.3 * Q)) <= un["bound"] + 1e-14), 1)
check(S, "P(t) rows sum to one and entries are non-negative", "stochastic semigroup", float(np.allclose(c.transition_matrix(0.7).sum(1), 1) and c.transition_matrix(0.7).min() >= 0), 1)
check(S, "Chapman-Kolmogorov P(s + t) = P(s) P(t)", "semigroup", maxerr(c.transition_matrix(1.0), c.transition_matrix(0.4) @ c.transition_matrix(0.6)), 0, atol=1e-14)
check(S, "stationary law pi Q = 0 vs the limit of P(t)", "ergodic theorem", maxerr(c.stationary(), c.transition_matrix(50.0)[0]), 0, atol=1e-12)
check(S, "... equals the jump-chain law weighted by mean holding times", "nu_i / q_i", maxerr(c.stationary(), c.stationary_via_jump_chain()), 0, atol=1e-14)
pth = c.simulate(50000.0, 0, R(44))
check(S, "occupation fractions of a 50000-time-unit path", "stationary law", maxerr(c.occupation(pth), c.stationary()), 0, atol=0.01)
eg = ctmc.estimate_generator(pth["times"], pth["states"], 50000.0, 3)
check(S, "MLE of Q (transition counts / occupation times)", "true Q", maxerr(eg["Q"], Q), 0, atol=4 * float(eg["std_error"].max()))
bd = ctmc.birth_death([1.0] * 6, [2.0] * 6)
check(S, "birth-death product form = M/M/1/6 stationary law (1 - rho) rho^n / (1 - rho^7)", "closed form", maxerr(bd["stationary"], 0.5 ** np.arange(7) * 0.5 / (1 - 0.5**7)), 0, atol=1e-14)
check(S, "... equals the generator's null vector", "linear solve", maxerr(bd["stationary"], bd["chain"].stationary()), 0, atol=1e-14)
abm = ctmc.CTMC([[-1.0, 1.0, 0], [0.5, -1.0, 0.5], [0, 0, 0]]).absorption_times()
check(S, "expected absorption time from state 0 (first-step analysis: m0 = 1 + m1, m1 = 1 + m0/2)", "4", float(abm["expected_time"][0]), 4.0, atol=1e-12)
sir = models.sir(N=1000, beta=0.3, gamma=0.1, I0=10)
fin = [ctmc.gillespie(sir["x0"], sir["stoich"], sir["propensity"], 1000.0, R(9000 + k))["x"][-1, 2] for k in range(300)]
r_inf = float(brentq(lambda z: 1 - z - 0.99 * math.exp(-3 * z), 0.5, 1.0))
check(S, "stochastic SIR (N = 1000, R0 = 3): final size of major outbreaks vs 1 - r = 0.99 e^{-R0 r}", "Kermack-McKendrick", float(np.mean([f for f in fin if f > 100])) / 1000, r_inf, atol=0.01)
ge = models.gene_expression()
gpath = ctmc.gillespie(ge["x0"], ge["stoich"], ge["propensity"], 3000.0, R(45))
prot = ctmc.sample_path_at(gpath["t"], gpath["x"][:, 1], np.linspace(200, 3000, 5000))
check(S, "gene expression: stationary mean protein k_m k_p / (g_m g_p) = 500", "moment equations", float(prot.mean()), ge["mean_protein"], rtol=0.06)
check(S, "... Fano factor 1 + k_p/(g_m + g_p) = 5.5 (bursting)", "moment equations (Thattai & van Oudenaarden 2001)", float(prot.var() / prot.mean()), ge["fano_protein"], rtol=0.25)
bdq = ctmc.birth_death([2.0] * 40, [1.0 * k for k in range(1, 41)])
check(S, "M/M/inf as a birth-death chain: Poisson(2) stationary law", "theory", maxerr(bdq["stationary"][:15], stats.poisson.pmf(np.arange(15), 2.0)), 0, atol=1e-12)
tl = ctmc.tau_leap([0], [[1], [-1]], lambda x: np.array([10.0, 1.0 * x[0]]), 200.0, 0.01, R(46))
check(S, "tau-leaping immigration-death: stationary mean 10", "Poisson(10)", float(tl["x"][2000:, 0].mean()), 10.0, rtol=0.05)

# ------------------------------------------------------------------ 11. queues
S = "Queues"
check(S, "Erlang C(2, a = 1.5) = 9/14", "closed form", queues.erlang_c(2, 1.5), 9 / 14, atol=1e-15)
check(S, "Erlang B recursion against the direct formula, c = 30, a = 25", "a^c/c! / sum a^k/k!", queues.erlang_b(30, 25.0), float(stats.poisson.pmf(30, 25.0) / stats.poisson.cdf(30, 25.0)), rtol=1e-12)
check(S, "Erlang B for c = 400, a = 380 (no overflow in the recursion)", "Poisson ratio", queues.erlang_b(400, 380.0), float(stats.poisson.pmf(400, 380.0) / stats.poisson.cdf(400, 380.0)), rtol=1e-10)
check(S, "M/M/1 with rho = 0.8: Wq = 4", "closed form", queues.mm1(0.8, 1.0)["Wq"], 4.0, atol=1e-12)
check(S, "M/M/c as the c = 1 case of Erlang C", "consistency", queues.mmc(0.8, 1.0, 1)["Wq"], 4.0, atol=1e-12)
mk_ = queues.mmck(4.0, 1.0, 6, 12)
bdm = models.mmck_chain(4.0, 1.0, 6, 12)
check(S, "M/M/c/K stationary law vs the birth-death generator", "two methods", maxerr(mk_["p"], bdm["chain"].stationary()), 0, atol=1e-14)
check(S, "Little's law in M/M/c/K: L = lambda_eff W", "Little (1961)", mk_["L"], mk_["throughput"] * mk_["W"], atol=1e-12)
ng = 400000
w1 = queues.lindley(R(47).exponential(1 / 0.8, ng), R(48).exponential(1.0, ng))
check(S, "Lindley recursion, M/M/1 rho = 0.8: mean wait (batch-means interval)", "Wq = 4", oa.batch_means(w1[20000:], 20)["estimate"], 4.0, atol=3 * oa.batch_means(w1[20000:], 20)["std_error"])
wd = queues.lindley(R(49).exponential(1 / 0.8, ng), np.ones(ng))
check(S, "M/D/1: Pollaczek-Khinchine Wq = rho/(2 mu (1 - rho)) = 2", "P-K formula", oa.batch_means(wd[20000:], 20)["estimate"], queues.mg1(0.8, 1.0, 1.0)["Wq"], atol=3 * oa.batch_means(wd[20000:], 20)["std_error"])
s_ln = R(50).lognormal(-0.5 * math.log(1 + 2.25), math.sqrt(math.log(1 + 2.25)), ng)
wl = queues.lindley(R(51).exponential(1 / 0.7, ng), s_ln)
check(S, "M/G/1 with lognormal service (cv 1.5): P-K", "P-K formula", oa.batch_means(wl[20000:], 20)["estimate"], queues.mg1(0.7, 1.0, 3.25)["Wq"], atol=3 * oa.batch_means(wl[20000:], 20)["std_error"])
check(S, "Kingman's formula is exact for M/G/1 (c_a^2 = 1)", "P-K", queues.kingman(0.7, 1.0, 1.0, 2.25), queues.mg1(0.7, 1.0, 3.25)["Wq"], atol=1e-12)
arr = np.cumsum(R(52).exponential(1 / 4.0, ng))
ggc = queues.simulate_ggc(arr, R(53).exponential(1.0, ng), 6)
check(S, "G/G/c simulation of M/M/6 at rho = 2/3: mean wait", "Erlang C Wq = 0.1424", oa.batch_means(ggc["wait"][20000:], 20)["estimate"], queues.mmc(4.0, 1.0, 6)["Wq"], atol=3 * oa.batch_means(ggc["wait"][20000:], 20)["std_error"])
check(S, "... P(wait <= 0.2)", "1 - C e^{-(c mu - lambda) t}", float(np.mean(ggc["wait"][20000:] <= 0.2)), queues.service_level(4.0, 1.0, 6, 0.2), atol=0.01)
check(S, "... the c = 1 case reproduces Lindley exactly", "same recursion", maxerr(queues.simulate_ggc(np.cumsum(R(54).exponential(1.25, 2000)), R(55).exponential(1.0, 2000), 1)["wait"], queues.lindley(np.r_[0, np.diff(np.cumsum(R(54).exponential(1.25, 2000)))], R(55).exponential(1.0, 2000))), 0, atol=1e-10)
ab_ = queues.simulate_ggc(arr[:200000], R(56).exponential(1.0, 200000), 4, patience=R(57).exponential(2.0, 200000))
lam_, mu_, th_, c_ = 4.0, 1.0, 0.5, 4
ea = ctmc.birth_death([lam_] * 80, [min(k, c_) * mu_ + max(k - c_, 0) * th_ for k in range(1, 81)])["stationary"]
p_ab = float(np.sum(ea * np.maximum(np.arange(81) - c_, 0) * th_) / lam_)
check(S, "Erlang-A (M/M/4+M): abandonment fraction vs the birth-death chain", "Palm / Garnett et al. (2002)", float(ab_["abandoned"][10000:].mean()), p_ab, atol=0.01)
st_ = queues.staffing(100.0, 1 / 3, 0.8, 1 / 3)
check(S, "staffing: smallest c with 80 % answered within 20 s ... meets the target", "definition", float(st_["service_level"] >= 0.8 and queues.service_level(100.0, 1 / 3, st_["c"] - 1, 1 / 3) < 0.8), 1)
check(S, "square-root staffing: with the answer time scaled like 1/sqrt(load), beta converges (loads 300 -> 30000)", "Halfin & Whitt (1981)", queues.staffing(10000.0, 1 / 3, 0.8, 1 / (3 * 100.0))["beta"], queues.staffing(100.0, 1 / 3, 0.8, 1 / (3 * 10.0))["beta"], atol=0.03)
check(S, "... while with a fixed answer time beta falls towards 0 (economies of scale)", "Erlang C", float(queues.staffing(10000.0, 1 / 3, 0.8, 1 / 3)["beta"] < 0.1 < st_["beta"]), 1)
jk = queues.jackson([1.0, 0.0], [[0, 0.5], [0.2, 0]], [3.0, 2.0])
check(S, "Jackson traffic equations Lambda = gamma + Lambda R", "linear algebra", maxerr(jk["Lambda"], [1 / 0.9, 0.5 / 0.9]), 0, atol=1e-14)
check(S, "... sojourn time by Little's law matches the product-form L", "Jackson (1957)", jk["W_network"], float(np.sum(jk["Lambda"] / ([3.0, 2.0] - jk["Lambda"]))), atol=1e-12)
tmax = arr[50000]
grid = np.linspace(arr[20000], tmax, 200001)
check(S, "time-average number in system from the event record = lambda W (Little)", "Little's law", float(np.mean(queues.number_in_system(arr[:60000], ggc["depart"][:60000], grid))), 4.0 * (queues.mmc(4.0, 1.0, 6)["W"]), rtol=0.05)

# ------------------------------------------------------------------ 12. discrete-event simulation
S = "Discrete-event simulation"
cc = des.call_centre(lambda t: 4.0, 4.0, lambda r: r.exponential(1.0), 6, 40000.0, R(58), answer_within=0.2, warmup=200.0)
check(S, "event-driven M/M/6: mean wait", "Erlang C", float(np.nanmean(cc["wait"])), queues.mmc(4.0, 1.0, 6)["Wq"], rtol=0.06)
check(S, "... time-average busy servers = offered load", "lambda/mu = 4", cc["mean_busy"], 4.0, rtol=0.02)
check(S, "... time-average queue length = Lq", "Little's law", cc["mean_queue"], queues.mmc(4.0, 1.0, 6)["Lq"], rtol=0.1)
cca = des.call_centre(lambda t: 4.0, 4.0, lambda r: r.exponential(1.0), 4, 40000.0, R(59), patience_sampler=lambda r: r.exponential(2.0), warmup=200.0)
check(S, "event-driven Erlang-A abandonment fraction", "birth-death chain", cca["abandon_rate"], p_ab, atol=0.01)
mr = des.machine_repair(5, 2, 0.1, lambda r: r.exponential(1.0), 200000.0, R(60))
bdr = ctmc.birth_death([0.1 * (5 - k) for k in range(5)], [min(k, 2) * 1.0 for k in range(1, 6)])
check(S, "machine repair (5 machines, 2 repairers): mean number down", "finite-source birth-death chain", mr["mean_down"], float(np.dot(np.arange(6), bdr["stationary"])), rtol=0.03)
mrd = des.machine_repair(5, 5, 0.1, lambda r: 1.0, 200000.0, R(61))
mre = des.machine_repair(5, 1, 0.1, lambda r: r.exponential(1.0), 200000.0, R(62))
bd1 = ctmc.birth_death([0.1 * (5 - k) for k in range(5)], [1.0] * 5)
check(S, "one repairer per machine: availability E[U]/(E[U] + E[D]) = 10/11 even with deterministic repairs", "alternating renewal (insensitivity)", mrd["availability"], 10 / 11, atol=0.004)
check(S, "... exponential repairs match the chain", "birth-death", mre["mean_down"], float(np.dot(np.arange(6), bd1["stationary"])), rtol=0.03)
log = []
simx = des.Simulator()
for tt in (3.0, 1.0, 2.0, 1.0):
    simx.schedule(tt, log.append, tt)
simx.run()
check(S, "event list processes events in time order, ties first-in first-out", "heap with sequence numbers", float(log == [1.0, 1.0, 2.0, 3.0]), 1)
inv1 = [des.inventory_sS(10, 40, 2.0, lambda r: 1, lambda r: 0.0, 5000.0, R(63 + k))["mean_on_hand"] for k in range(5)]
check(S, "(s, S) with unit demand and zero lead time: on-hand stock uniform on s+1..S, mean (s + S + 1)/2", "renewal argument", float(np.mean(inv1)), 25.5, rtol=0.01)

# ------------------------------------------------------------------ 13. output analysis
S = "Output analysis"
phi = 0.9
e = R(64).normal(size=400000)
ar = ssg.lfilter([1.0], [1.0, -phi], e)
check(S, "autocovariance by FFT vs the direct sum", "definition", maxerr(oa.autocovariance(ar[:5000], 10), [np.sum((ar[:5000][: 5000 - k] - ar[:5000].mean()) * (ar[:5000][k:] - ar[:5000].mean())) / 5000 for k in range(11)]), 0, atol=1e-10)
check(S, "integrated autocorrelation time of AR(1), phi = 0.9 (Sokal window)", "(1 + phi)/(1 - phi) = 19", oa.iat(ar)["tau"], 19.0, rtol=0.06)
check(S, "... Geyer's initial positive sequence", "19", oa.iat_geyer(ar)["tau"], 19.0, rtol=0.06)
check(S, "... white noise has tau = 1", "theory", oa.iat(e)["tau"], 1.0, atol=0.05)
bm_cov = [oa.batch_means(ssg.lfilter([1.0], [1.0, -0.9], R(7000 + k).normal(size=20000)), 20)["ci"] for k in range(500)]
check(S, "batch means (20 batches of 1000) for an AR(1) mean: 95 % coverage", "nominal", oa.coverage([b[0] for b in bm_cov], [b[1] for b in bm_cov], 0.0)["coverage"], 0.95, atol=0.03)
naive = [mc.mean_ci(ssg.lfilter([1.0], [1.0, -0.9], R(7000 + k).normal(size=20000)))["ci"] for k in range(500)]
check(S, "... while the naive i.i.d. interval covers only ~ 2 Phi(1.96/sqrt(19)) - 1 = 35 %", "theory", oa.coverage([b[0] for b in naive], [b[1] for b in naive], 0.0)["coverage"], 2 * stats.norm.cdf(1.96 / math.sqrt(19)) - 1, atol=0.05)
check(S, "batch means estimate the asymptotic variance sigma^2 tau = 1/(1 - phi)^2", "100", oa.batch_means(ar, 40)["asymptotic_var"], 100.0, rtol=0.25)
trans = np.r_[np.linspace(10, 0, 300), R(65).normal(size=3000)]
check(S, "MSER-5 truncation point for a linear transient of 300 observations", "the transient length", oa.mser(trans)["truncate"], 300, atol=40)
cyc = [queues.lindley(R(8000 + k).exponential(1 / 0.5, 3000), R(9000 + k).exponential(1.0, 3000)) for k in range(1)][0]
starts = np.flatnonzero(cyc == 0)
ry = np.add.reduceat(cyc, starts)[:-1]
rl = np.diff(starts)
rg_ = oa.regenerative(ry, rl)
check(S, "regenerative estimator (cycles start at empty-queue arrivals), M/M/1 rho = 0.5", "Wq = 1", rg_["estimate"], 1.0, atol=4 * rg_["std_error"])
grh = oa.gelman_rubin(R(66).normal(size=(4, 2000)))
check(S, "Gelman-Rubin R-hat for four independent chains", "1", grh["R_hat"], 1.0, atol=0.01)
check(S, "... and > 1.5 for chains stuck in different places", "theory", float(oa.gelman_rubin(R(67).normal(size=(4, 2000)) + np.arange(4)[:, None] * 2)["R_hat"] > 1.5), 1)
reps = oa.replications([np.mean(queues.lindley(R(100 + k).exponential(1 / 0.5, 2000), R(200 + k).exponential(1.0, 2000))[500:]) for k in range(40)])
check(S, "independent replications with deletion, M/M/1 rho = 0.5", "Wq = 1", reps["estimate"], 1.0, atol=4 * reps["std_error"])

# ------------------------------------------------------------------ 14. Brownian motion
S = "Brownian motion"
W = bm.paths(1.0, 512, 20000, R(68))
check(S, "Var W_1 = 1", "definition", float(W[:, -1].var()), 1.0, atol=0.03)
check(S, "Cov(W_0.25, W_0.75) = 0.25", "min(s, t)", float(np.cov(W[:, 128], W[:, 384])[0, 1]), 0.25, atol=0.015)
Wl = bm.bridge_construction(1.0, 9, 20000, R(69))
check(S, "Levy midpoint construction: Cov(W_0.25, W_0.75) = 0.25", "min(s, t)", float(np.cov(Wl[:, 128], Wl[:, 384])[0, 1]), 0.25, atol=0.015)
check(S, "... refinement keeps the coarse values", "construction", maxerr(bm.refine(Wl[:3], 1.0, R(70))[:, ::2], Wl[:3]), 0, atol=0)
B = bm.bridge(2.0, 200, 30000, R(71))
check(S, "Brownian bridge on [0, 2]: Var B_0.5 = s(T - s)/T = 0.375", "theory", float(B[:, 50].var()), 0.375, atol=0.015)
check(S, "... and B_T = 0", "pinned", maxerr(B[:, -1], 0), 0, atol=1e-12)
check(S, "reflection principle: P(max W >= 1 on [0, 1]) vs 20000 paths of 512 steps (biased low by the grid)", "2(1 - Phi(1)) = 0.317", float(np.mean(W.max(axis=1) >= 1)), float(bm.max_cdf(1.0, 1.0)), atol=0.02)
fp = bm.first_passage_simulated(1.0, 2.0, 200, 30000, R(72), drift=0.5, bridge_correction=True)
check(S, "first passage with drift and Brownian-bridge correction: P(T_1 <= 2)", "inverse Gaussian cdf", float(np.mean(fp <= 2.0)), float(bm.first_passage_cdf(2.0, 1.0, 0.5)), atol=0.01)
fpn = bm.first_passage_simulated(1.0, 2.0, 200, 30000, R(72), drift=0.5)
check(S, "... the uncorrected grid estimate is biased low by > 1.5 %", "discrete monitoring", float(bm.first_passage_cdf(2.0, 1.0, 0.5) - np.mean(fpn <= 2.0) > 0.015), 1)
check(S, "first-passage cdf with drift", "scipy.stats.invgauss (mu = a/(nu a), scale = a^2)", float(bm.first_passage_cdf(3.0, 1.0, 0.5)), float(stats.invgauss.cdf(3.0, mu=2.0, scale=1.0)), atol=1e-12)
tt = np.linspace(0.01, 20, 20000)
check(S, "first-passage density integrates to the cdf", "quadrature", float(_trapz(bm.first_passage_pdf(tt, 1.0, 0.3), tt)), float(bm.first_passage_cdf(20.0, 1.0, 0.3) - bm.first_passage_cdf(0.01, 1.0, 0.3)), atol=1e-4)
S0, mu_, sg = 100.0, 0.08, 0.3
gb = bm.gbm(S0, mu_, sg, 2.0, 10, 200000, R(73))
check(S, "GBM exact: E[S_2] = S0 e^{2 mu}", "lognormal", float(gb[:, -1].mean()), S0 * math.exp(2 * mu_), rtol=0.005)
check(S, "... log S_2 ~ N(log S0 + (mu - s^2/2) 2, 2 s^2): KS p > 0.01", "scipy.stats.norm", float(stats.kstest(np.log(gb[:, -1]), "norm", args=(math.log(S0) + (mu_ - sg**2 / 2) * 2, sg * math.sqrt(2))).pvalue > 0.01), 1)
ou = bm.ou_exact(2.0, 1.5, 0.5, 0.8, 2.0, 20, 100000, R(74))
om = bm.ou_moments(2.0, 1.5, 0.5, 0.8, 2.0)
check(S, "OU exact sampling: mean at t = 2", "closed form", float(ou[:, -1].mean()), float(om["mean"]), atol=0.005)
check(S, "... variance at t = 2", "closed form", float(ou[:, -1].var()), float(om["var"]), rtol=0.02)
Wf = bm.paths(1.0, 2**16, 1, R(75))
check(S, "quadratic variation over [0, 1] with 2^16 steps", "[W]_1 = 1", float(bm.quadratic_variation(Wf)[0]), 1.0, atol=0.02)
check(S, "total variation grows like sqrt(2n/pi)", "E|dW| sum", float(bm.total_variation(Wf)[0] / math.sqrt(2 * 2**16 / math.pi)), 1.0, atol=0.02)
kl = bm.karhunen_loeve(1.0, 400, np.array([0.3, 0.8]), 50000, R(76))
check(S, "Karhunen-Loeve expansion (400 terms): Cov(W_0.3, W_0.8) = 0.3", "min(s, t)", float(np.cov(kl.T)[0, 1]), 0.3, atol=0.01)
dn = bm.donsker(1000, 20000, R(77))
check(S, "Donsker: scaled walk's maximum on [0, 1] vs P(max >= 1)", "reflection", float(np.mean(dn.max(axis=1) >= 1)), float(bm.max_cdf(1.0, 1.0)), atol=0.02)
check(S, "Black-Scholes call S = K = 100, r = 5 %, sigma = 20 %, T = 1", "Hull's textbook value 10.4506", bm.black_scholes_call(100, 100, 0.05, 0.2, 1), 10.4506, atol=1e-4)
gq = bm.gbm(100, 0.05, 0.2, 1.0, 1, 400000, R(78))[:, -1]
check(S, "... by risk-neutral Monte Carlo", "closed form", float(math.exp(-0.05) * np.mean(np.maximum(gq - 100, 0))), bm.black_scholes_call(100, 100, 0.05, 0.2, 1), rtol=0.01)
gpb = bm.gbm(100, 0.05, 0.2, 1.0, 250, 40000, R(79))
lw = np.log(gpb)
dt_ = 1 / 250
pcross = np.exp(-2 * (lw[:, :-1] - math.log(90)) * (lw[:, 1:] - math.log(90)) / (0.04 * dt_))
alive = np.all(gpb > 90, axis=1) & np.all(R(80).random(pcross.shape) >= pcross, axis=1)
check(S, "down-and-out call (B = 90): MC with bridge-corrected daily monitoring vs Merton's formula", "Merton (1973)", float(math.exp(-0.05) * np.mean(np.maximum(gpb[:, -1] - 100, 0) * alive)), bm.down_and_out_call(100, 100, 90, 0.05, 0.2, 1), rtol=0.03)

# ------------------------------------------------------------------ 15. SDEs
S = "Stochastic differential equations"
a = lambda t, x: 0.05 * x  # noqa: E731
b = lambda t, x: 0.4 * x  # noqa: E731
bx = lambda t, x: 0.4 + 0 * x  # noqa: E731
exact = lambda t, Wv: np.exp((0.05 - 0.08) * t + 0.4 * Wv)  # noqa: E731
so_e = sde.strong_order(a, b, exact, 1.0, 1.0, [16, 32, 64, 128, 256, 512], 4000, R(81))
check(S, "Euler-Maruyama strong order on GBM", "1/2 (Kloeden & Platen)", so_e["order"], 0.5, atol=0.08)
so_m = sde.strong_order(a, b, exact, 1.0, 1.0, [16, 32, 64, 128, 256, 512], 4000, R(81), "milstein", bx)
check(S, "Milstein strong order on GBM", "1", so_m["order"], 1.0, atol=0.1)
check(S, "Euler and Milstein coincide for additive noise (OU)", "b' = 0", maxerr(sde.euler_maruyama(lambda t, x: -x, lambda t, x: 0.5 + 0 * x, 1.0, 1.0, 50, 5, R(82))["X"], sde.milstein(lambda t, x: -x, lambda t, x: 0.5 + 0 * x, lambda t, x: 0 * x, 1.0, 1.0, 50, 5, R(82))["X"]), 0, atol=1e-14)
wo = sde.weak_order(lambda t, x: 1.0 * x, lambda t, x: 0.2 * x, 1.0, 1.0, [8, 16, 32, 64], 400000, R(83), lambda x: x, math.e)
check(S, "Euler weak order for E[X_1], dX = X dt + 0.2 X dW", "1 (Talay-Tubaro); exact error e - (1 + h)^(1/h)", wo["order"], 1.0, atol=0.1)
check(S, "... the weak error equals the deterministic Euler error e - (1 + h)^N", "E[X_N] = (1 + h)^N exactly", float(wo["error"][0]), math.e - (1 + 1 / 8) ** 8, atol=4 * float(wo["std_error"][0]))
hs = sde.heun_stratonovich(lambda t, x: 0 * x, lambda t, x: 0.4 * x, 1.0, 1.0, 400, 4000, R(84))
WT = hs["dW"].sum(axis=1)[:, 0]
check(S, "stochastic Heun converges to the Stratonovich solution exp(0.4 W) of dX = 0.4 X o dW", "Stratonovich calculus", float(np.mean(np.abs(hs["X"][:, -1] - np.exp(0.4 * WT)))), 0, atol=2e-3)
em_ito = sde.euler_maruyama(lambda t, x: 0 * x, lambda t, x: 0.4 * x, 1.0, 1.0, 400, 4000, R(84), dW=hs["dW"])
check(S, "Euler on the same increments solves the Ito equation instead: exp(0.4 W - 0.08)", "Ito calculus", float(np.mean(np.abs(em_ito["X"][:, -1] - np.exp(0.4 * WT - 0.08)))), 0, atol=0.01)
hs_c = sde.heun_stratonovich(sde.ito_to_stratonovich_drift(lambda t, x: 0 * x, lambda t, x: 0.4 * x, bx), lambda t, x: 0.4 * x, 1.0, 1.0, 400, 4000, R(84), dW=hs["dW"])
check(S, "Heun with the corrected drift a - b b'/2 recovers the Ito solution", "Ito-Stratonovich conversion", float(np.mean(np.abs(hs_c["X"][:, -1] - np.exp(0.4 * WT - 0.08)))), 0, atol=2e-3)
cir = sde.cir_full_truncation(0.04, 2.0, 0.05, 0.3, 1.0, 400, 40000, R(85))
check(S, "CIR full-truncation Euler: E[X_1]", "theta + (x0 - theta) e^{-kappa}", float(cir[:, -1].mean()), float(sde.cir_mean(0.04, 2.0, 0.05, 1.0)), rtol=0.01)
cv_ = 0.3**2 * 0.04 * (math.exp(-2) - math.exp(-4)) / 2 + 0.05 * 0.09 / 4 * (1 - math.exp(-2)) ** 2
check(S, "... Var[X_1] (non-central chi-square law)", "closed form", float(cir[:, -1].var()), cv_, rtol=0.05)
check(S, "... reported values never negative (the Feller condition 2 kappa theta = 0.2 > sigma^2 = 0.09 holds here)", "full truncation", float(cir.min() >= 0), 1)
ml = sde.mlmc(lambda Sx: np.maximum(Sx - 100, 0), 100.0, 0.05, 0.2, 1.0, 0.03, R(86), L=6)
check(S, "multilevel Monte Carlo price of a European call (L = 6, eps = 0.03)", "Black-Scholes 10.4506 (Euler bias at 64 steps ~0.01)", ml["estimate"], bm.black_scholes_call(100, 100, 0.05, 0.2, 1), atol=0.08)
check(S, "... level variances decay like h (V_l / V_l+1 ~ 2) for the Lipschitz payoff", "Giles (2008)", float(np.mean(ml["variances"][2:-1] / ml["variances"][3:])), 2.0, rtol=0.2)
em2 = sde.euler_maruyama(lambda t, x: -x, lambda t, x: np.stack([np.stack([0.3 + 0 * x[:, 0], 0.1 + 0 * x[:, 0]], -1), np.stack([0 * x[:, 0], 0.2 + 0 * x[:, 0]], -1)], 1), [0.0, 0.0], 3.0, 600, 20000, R(87), dW=math.sqrt(3 / 600) * R(88).standard_normal((20000, 600, 2)))
Sg = np.array([[0.3, 0.1], [0.0, 0.2]])
check(S, "2-D OU with general noise: stationary covariance solves A P + P A' + S S' = 0", "scipy.linalg.solve_continuous_lyapunov", maxerr(np.cov(em2["X"][:, -1].T), sla.solve_continuous_lyapunov(-np.eye(2), -Sg @ Sg.T)), 0, atol=0.003)

# ------------------------------------------------------------------ 16. Gaussian processes
S = "Gaussian processes"
t = np.arange(256) * 0.05
ce = gp.circulant_embedding(gp.exponential(0, t, ell=0.5), 40000, R(89))
check(S, "circulant embedding of the exponential covariance: sample covariance at lags 0, 10, 40", "exp(-|h|/ell)", maxerr(np.cov(ce["X"][:, [0, 10, 40]].T)[0], gp.exponential(0, t[[0, 10, 40]], ell=0.5)), 0, atol=0.02)
check(S, "... embedding eigenvalues non-negative (Dietrich-Newsam: exponential always embeds)", "theory", float(ce["eigenvalues"].min() >= 0), 1)
check(S, "... eigenvalues equal the FFT of the first row", "numpy.fft", maxerr(ce["eigenvalues"], np.fft.fft(np.r_[gp.exponential(0, t, ell=0.5), gp.exponential(0, t, ell=0.5)[-2:0:-1]]).real), 0, atol=1e-12)
try:
    gp.circulant_embedding(gp.squared_exponential(0, np.linspace(0, 5, 100), ell=1.0), 2, R(90))
    failed = 0
except ValueError:
    failed = 1
check(S, "minimal embedding of a smooth (squared-exponential) covariance on a short window fails", "Wood & Chan (1994)", failed, 1)
fb = gp.fbm_davies_harte(512, 0.8, 1.0, 20000, R(91))
check(S, "fBM (H = 0.8) by Davies-Harte: Var B_1 = 1", "|t|^2H", float(fb[:, -1].var()), 1.0, atol=0.03)
check(S, "... Var B_0.5 = 0.5^1.6", "|t|^2H", float(fb[:, 256].var()), 0.5**1.6, atol=0.015)
check(S, "... Cov(B_0.5, B_1) = (1 + 0.5^1.6 - 0.5^1.6)/2 = 0.5", "fBM covariance", float(np.cov(fb[:, 256], fb[:, -1])[0, 1]), float(gp.fbm(0.5, 1.0, 0.8)), atol=0.02)
check(S, "fGn autocovariance sums telescope: sum_{|k|<n} (n - |k|) gamma(k) = n^2H", "fBM variance", float(sum((64 - abs(k)) * gp.fgn_autocovariance(k, 0.7) for k in range(-63, 64))), 64**1.4, rtol=1e-12)
check(S, "Hurst exponent by aggregated variance, H = 0.75, n = 2^15", "0.75", gp.hurst_aggregated_variance(np.diff(gp.fbm_davies_harte(2**15, 0.75, 1.0, 1, R(92))[0]))["H"], 0.75, atol=0.05)
ch = gp.sample_cholesky(gp.matern32, np.linspace(0, 3, 60), 40000, R(93), ell=0.8)
check(S, "Cholesky sampling of a Matern 3/2 process: covariance at lag 1.0", "(1 + sqrt3 r) e^{-sqrt3 r}", float(np.cov(ch["X"][:, 0], ch["X"][:, 20])[0, 1]), float(gp.matern32(0, np.linspace(0, 3, 60)[20], ell=0.8)), atol=0.015)
check(S, "squared-exponential covariance on 100 points of [0, 1] (ell = 0.3) is numerically singular (cond > 1e15)", "spectral decay", float(gp.sample_cholesky(gp.squared_exponential, np.linspace(0, 1, 100), 1, R(94), ell=0.3)["cond"] > 1e15), 1)
xo, yo, xn = np.array([0.0, 1.0, 2.0, 3.5]), np.array([0.2, 1.0, -0.3, 0.4]), np.array([0.5, 1.7, 3.0])
post = gp.condition(gp.squared_exponential, xo, yo, xn, noise_var=0.01, ell=0.7)
K = gp.cov_matrix(gp.squared_exponential, xo, ell=0.7) + (0.01 + 1e-12) * np.eye(4)
Ks = gp.cov_matrix(gp.squared_exponential, xo, xn, ell=0.7)
check(S, "GP posterior mean equals the kriging formula k*' K^-1 y", "direct solve", maxerr(post["mean"], Ks.T @ np.linalg.solve(K, yo)), 0, atol=1e-12)
check(S, "... log marginal likelihood", "scipy.stats.multivariate_normal.logpdf", post["log_marginal_likelihood"], float(stats.multivariate_normal(np.zeros(4), K).logpdf(yo)), atol=1e-10)
check(S, "noise-free interpolation reproduces the data with zero variance", "conditioning", maxerr(gp.condition(gp.matern52, xo, yo, xo, ell=0.7)["mean"], yo), 0, atol=1e-5)
check(S, "Brownian motion as a GP: min(s, t) kernel", "definition", maxerr(gp.cov_matrix(gp.brownian, [0.2, 0.5]), [[0.2, 0.2], [0.2, 0.5]]), 0)
vg = gp.empirical_variogram(gp.circulant_embedding(gp.exponential(0, t, ell=0.5), 2000, R(95))["X"], 20)
check(S, "variogram gamma(h) = C(0) - C(h) for a stationary process", "geostatistics", float(vg[10]), float(1 - gp.exponential(0, 0.5, ell=0.5)), atol=0.02)

# ------------------------------------------------------------------ 17. time series
S = "Time series"
ph, thq = [0.6, -0.3], [0.4]
check(S, "ARMA(2, 1) autocorrelations by the psi weights", "statsmodels-free: arma_acovf recursion via scipy.signal.lfilter impulse response", maxerr(ts.arma_acf(ph, thq, 5), (lambda psi: np.array([np.dot(psi[:4000], psi[k : k + 4000]) for k in range(6)]) / np.dot(psi[:4000], psi[:4000]))(ssg.lfilter([1, 0.4], [1, -0.6, 0.3], np.r_[1.0, np.zeros(4010)]))), 0, atol=1e-12)
check(S, "AR(1) autocorrelation phi^k", "closed form", maxerr(ts.arma_acf([0.7], [], 6), 0.7 ** np.arange(7)), 0, atol=1e-12)
check(S, "MA(1) variance (1 + theta^2) sigma^2 and zero ACF beyond lag 1", "closed form", float(abs(ts.arma_acov([], [0.5], 3)[0] - 1.25) + abs(ts.arma_acov([], [0.5], 3)[2])), 0, atol=1e-14)
xa = ts.simulate_arma([0.6, -0.3], [], 40000, R(96))
yw = ts.yule_walker(xa, 2)
check(S, "Yule-Walker estimates of AR(2)", "true (0.6, -0.3)", maxerr(yw["phi"], [0.6, -0.3]), 0, atol=4 * float(yw["std_error"].max()))
check(S, "... asymptotic standard error sqrt((1 - phi_2^2)/n) for AR(2)", "Brockwell & Davis", float(yw["std_error"][0]), math.sqrt((1 - 0.09) / 40000), rtol=0.03)
dl = ts.durbin_levinson(ts.arma_acov([0.6, -0.3], [], 5))
check(S, "Durbin-Levinson: theoretical PACF of AR(2) vanishes beyond lag 2", "theory", float(np.max(np.abs(dl["pacf"][3:]))), 0, atol=1e-12)
check(S, "... phi_22 = phi_2", "theory", float(dl["pacf"][2]), -0.3, atol=1e-12)
gam = ts.arma_acov([0.6, -0.3], [], 2)
check(S, "... order-2 coefficients solve the Toeplitz system", "scipy.linalg.solve_toeplitz", maxerr(dl["phi"][2, 1:3], sla.solve_toeplitz(gam[:2], gam[1:3])), 0, atol=1e-12)
check(S, "AIC selects p = 2 for AR(2) data", "Akaike", ts.select_ar_order(xa[:5000], 8)["p"], 2)
lb_white = [ts.ljung_box(R(300 + k).normal(size=300), 10)["p_value"] for k in range(400)]
check(S, "Ljung-Box size: rejection rate at 5 % for white noise", "nominal (slightly conservative)", float(np.mean(np.array(lb_white) < 0.05)), 0.05, atol=0.025)
check(S, "Ljung-Box statistic", "hand formula n(n+2) sum r_k^2/(n-k)", ts.ljung_box(xa[:500], 5)["statistic"], float(500 * 502 * np.sum(ts.sample_acf(xa[:500], 5)[1:] ** 2 / (500 - np.arange(1, 6)))), rtol=1e-12)
check(S, "causality of AR(2) (0.6, -0.3) and non-causality of AR(1) phi = 1.1", "roots outside the unit circle", float(ts.is_causal([0.6, -0.3]) and not ts.is_causal([1.1])), 1)
xm = ts.simulate_arma([0.6, -0.3], [0.4], 20000, R(97))
check(S, "periodogram", "scipy.signal.periodogram", maxerr(ts.periodogram(xm)["psd"], ssg.periodogram(xm)[1]) / float(ssg.periodogram(xm)[1].max()), 0, atol=1e-12)
check(S, "Welch averaged periodogram (Hann, 50 % overlap)", "scipy.signal.welch", maxerr(ts.welch(xm, 256)["psd"], ssg.welch(xm, nperseg=256)[1]) / float(ssg.welch(xm, nperseg=256)[1].max()), 0, atol=1e-12)
ff = np.linspace(0, 0.5, 4001)
check(S, "ARMA spectral density integrates to the variance (one-sided)", "Wiener-Khinchin", float(_trapz(ts.arma_spectrum(ph, thq, ff), ff)), float(ts.arma_acov(ph, thq, 0)[0]), rtol=1e-6)
wl_ = ts.welch(xm, 256)
check(S, "Welch estimate tracks the ARMA spectrum (median ratio)", "consistency", float(np.median(wl_["psd"][3:] / ts.arma_spectrum(ph, thq, wl_["f"][3:]))), 1.0, atol=0.05)

# ------------------------------------------------------------------ 18. MCMC
S = "Markov chain Monte Carlo"
rw = mcmc.metropolis(lambda z: -0.5 * np.sum(z**2), np.zeros(20), 40000, 0.5, R(98), adapt_until=10000)
check(S, "adaptive random-walk Metropolis in d = 20 reaches the target acceptance 0.234", "Roberts, Gelman & Gilks (1997)", rw["acceptance"], 0.234, atol=0.03)
check(S, "... with proposal scale ~ 2.38/sqrt(d)", "optimal scaling", float(rw["final_scale"][0]), 2.38 / math.sqrt(20), rtol=0.15)
check(S, "... and the right marginal variance", "N(0, I)", float(rw["chain"][10000:].var(axis=0).mean()), 1.0, atol=0.12)
gb2 = mcmc.gibbs_bivariate_normal(0.95, 100000, R(99))
check(S, "Gibbs for a bivariate normal: lag-1 autocorrelation of x is rho^2", "theory", float(oa.autocorrelation(gb2[:, 0], 1)[1]), 0.9025, atol=0.01)
check(S, "... integrated autocorrelation time (1 + rho^2)/(1 - rho^2) = 19.5", "AR(1) in rho^2", oa.iat(gb2[:, 0])["tau"], 19.5, rtol=0.1)
check(S, "... sample correlation", "0.95", float(np.corrcoef(gb2.T)[0, 1]), 0.95, atol=0.01)
ind = mcmc.independence_sampler(lambda z: float(stats.gamma.logpdf(z[0], 3.0)) if z[0] > 0 else -np.inf, lambda r: r.exponential(3.0), lambda z: float(stats.expon.logpdf(z[0], scale=3.0)), [1.0], 40000, R(100))
check(S, "independence sampler for Gamma(3) with Exp(mean 3) proposals: mean", "3", float(ind["chain"][2000:, 0].mean()), 3.0, rtol=0.03)
h_ = mcmc.hmc(lambda z: -0.5 * np.sum(z**2 / np.array([1.0, 100.0])), lambda z: -z / np.array([1.0, 100.0]), np.zeros(2), 5000, 0.5, 20, R(101))
check(S, "HMC on an ill-scaled Gaussian: variances 1 and 100", "target", float(h_["chain"][500:, 1].var() / 100), 1.0, atol=0.15)
errs_lf = [abs(mcmc.leapfrog_energy_error(lambda z: -z, lambda z: -0.5 * float(np.sum(z**2)), [1.0], [0.5], hh, int(round(1 / hh)))) for hh in (0.1, 0.05, 0.025)]
check(S, "leapfrog energy error is second order in the step", "symplectic integrator", float(np.log2(errs_lf[0] / errs_lf[1])), 2.0, atol=0.15)
check(S, "Onsager energy per site at beta_c: -sqrt(2)", "exact", mcmc.onsager_energy(mcmc.BETA_C), -math.sqrt(2), atol=1e-12)
check(S, "... and is continuous there (beta_c + 1e-6)", "exact", mcmc.onsager_energy(mcmc.BETA_C + 1e-6), -math.sqrt(2), atol=1e-4)
I1 = mcmc.ising(32, 0.6, 2000, R(102), "heatbath")
check(S, "Ising 32 x 32 at beta = 0.6 (ordered): |m| vs Onsager-Yang", "0.9736", float(np.abs(I1["magnetisation"][300:]).mean()), mcmc.onsager_magnetisation(0.6), atol=0.01)
check(S, "... energy per site vs Onsager", "-1.9091", float(I1["energy"][300:].mean()), mcmc.onsager_energy(0.6), atol=0.01)
I2 = mcmc.ising(32, 0.3, 2000, R(103), "metropolis", start="hot")
check(S, "Ising at beta = 0.3 (disordered): energy per site vs Onsager, Metropolis", "-0.7045", float(I2["energy"][300:].mean()), mcmc.onsager_energy(0.3), atol=0.01)
check(S, "... magnetisation near 0 (finite-size |m| ~ L^-1 sqrt(chi))", "Onsager", float(np.abs(I2["magnetisation"][300:]).mean()), 0.0, atol=0.08)
cities = R(104).random((12, 2))
dm = np.hypot(*(cities[:, None, :] - cities[None, :, :]).transpose(2, 0, 1))


def tour_len(p):
    return float(np.sum(dm[p, np.roll(p, -1)]))


def two_opt(p, r):
    i, j = sorted(r.choice(12, 2, replace=False))
    q = p.copy()
    q[i : j + 1] = q[i : j + 1][::-1]
    return q


sa = mcmc.simulated_annealing(tour_len, np.arange(12), two_opt, 60000, 1.0, R(105), 0.9998)
# exact optimum by Held-Karp dynamic programming
n_c = 12
dp = {(1, 0): 0.0}
for size in range(2, n_c + 1):
    for subset in itertools.combinations(range(1, n_c), size - 1):
        mask = 1 | sum(1 << k for k in subset)
        for j in subset:
            prev = mask & ~(1 << j)
            dp[(mask, j)] = min(dp[(prev, k)] + dm[k, j] for k in ([0] if prev == 1 else [k for k in subset if k != j]) if (prev, k) in dp)
full = (1 << n_c) - 1
held_karp = min(dp[(full, j)] + dm[j, 0] for j in range(1, n_c))
check(S, "simulated annealing finds the optimal 12-city tour", "Held-Karp dynamic programming", sa["cost"], held_karp, rtol=1e-9)

# ------------------------------------------------------------------ 19. course data
S = "Course data"
ccd = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
check(S, "call centre: 20 days of calls", "generator", ccd.day.nunique(), 20)
lam_int = pp.cumulative_intensity(lambda tt: models.call_centre_rate(tt + 8.0), 12.0, 24001)
tue_thu = ccd[ccd.weekday != "Mon"].groupby("day").size().drop(8)
check(S, "... mean calls per non-Monday day vs the integrated rate profile", "int lambda(t) dt over 08-20", float(tue_thu.mean()), lam_int, rtol=0.02)
mon = ccd[ccd.weekday == "Mon"].groupby("day").size()
check(S, "... Mondays carry 15 % more calls", "generator", float(mon.mean() / tue_thu.mean()), 1.15, atol=0.05)
d8 = ccd[(ccd.day == 8) & (ccd.arrival_h >= 13) & (ccd.arrival_h < 14.5)]
check(S, "... the outage: no calls on day 8 between 13:00 and 14:30", "documented fault", len(d8), 0)
hm = ccd.handle_min[ccd.handle_min > 0]
check(S, "... handle times: mean 4 min", "lognormal", float(hm.mean()), 4.0, rtol=0.02)
check(S, "... coefficient of variation 0.8 (not exponential)", "lognormal", float(hm.std() / hm.mean()), 0.8, atol=0.04)
check(S, "... about 0.4 % dropped calls (handle time 0)", "documented fault", float((ccd.handle_min == 0).mean()), 0.004, atol=0.0015)
wd_ = pd.read_csv(datasets.path("weather.csv"), comment="#")
pw = mk.estimate(wd_.wet.to_numpy(), 2)["P"]
check(S, "weather: P(dry -> wet) averaged over the seasons ~ 0.20", "generator mean", float(pw[0, 1]), 0.20, atol=0.03)
win = wd_[(wd_.day_of_year < 60) | (wd_.day_of_year > 330)].wet.to_numpy()
smr = wd_[(wd_.day_of_year > 150) & (wd_.day_of_year < 240)].wet.to_numpy()
check(S, "... winter is wetter than summer (seasonal non-homogeneity)", "generator", float(win.mean() > smr.mean() + 0.1), 1)
mlog = pd.read_csv(datasets.path("machine_log.csv"), comment="#")
code = {"up": 0, "degraded": 1, "down": 2}
egm = ctmc.estimate_generator(mlog.time_h.to_numpy(), mlog.state.map(code).to_numpy(), 17520.0, 3)
check(S, "machine log: MLE of the generator recovers the model within 4 standard errors", "models.machine()", float(np.all(np.abs(egm["Q"] - models.machine()["Q"])[egm["std_error"] > 0] <= 4 * egm["std_error"][egm["std_error"] > 0] + 1e-12)), 1)
check(S, "... the log ends in the up state", "documented censoring", float(mlog.state.iloc[-1] == "up"), 1)
pr_ = pd.read_csv(datasets.path("prices.csv"), comment="#").close.to_numpy()
lr = np.diff(np.log(pr_))
check(S, "prices: annualised volatility of the first 1000 days", "18 %", float(lr[:1000].std(ddof=1) * math.sqrt(252)), 0.18, atol=0.012)
check(S, "... and of the last 500 days (regime change)", "35 %", float(lr[1000:].std(ddof=1) * math.sqrt(252)), 0.35, atol=0.03)

# ------------------------------------------------------------------ report
passed = sum(r[5] for r in RESULTS)
lines = [
    "# Validation report",
    "",
    "Generated by `python docs/validate.py`; rerun in CI on every push. Checks compare engstoch with SciPy,",
    "closed-form results of probability theory, published reference values, and the theory itself (convergence",
    "orders, interval coverage, limit theorems). Statistical checks use fixed seeds and tolerances of several",
    "standard errors.",
    "",
    f"**{passed} / {len(RESULTS)} checks pass.**",
    "",
]
for section in dict.fromkeys(r[0] for r in RESULTS):
    lines += ["", f"## {section}", "", "| Check | Reference | engstoch | Expected | Result |", "|---|---|---|---|:-:|"]
    for s_, name, ref_, v, expected, ok in RESULTS:
        if s_ == section:
            lines.append(f"| {name} | {ref_} | {v:.10g} | {expected:.10g} | {'pass' if ok else 'FAIL'} |")
Path(__file__).with_name("VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{passed}/{len(RESULTS)} checks pass")
