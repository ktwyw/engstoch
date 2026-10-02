"""Generate the course notebooks (single source of truth) and execute them.

python notebooks/build_notebooks.py            # write and execute all notebooks
python notebooks/build_notebooks.py 03 15      # only notebooks whose names start with 03 or 15
python notebooks/build_notebooks.py --no-run   # write without executing
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
NOTEBOOKS: dict[str, list] = {}
EXTRAS: dict[str, dict] = {}

COLAB = """try:                      # on Google Colab (or anywhere engstoch is missing): install it from GitHub
    import engstoch
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/ktwyw/engstoch"], check=True)"""

STYLE = '''plt.rcParams.update({"figure.figsize": (7, 4), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

import inspect
from IPython.display import Code

def show_source(obj):
    """Display the source code of a library function or class."""
    return Code(inspect.getsource(obj), language="python")

R = np.random.default_rng      # R(seed): a fresh, reproducible generator'''


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text):
    return new_code_cell(text.strip("\n"))


def setup_cell(extra_imports=()):
    lines = ["import math", "import matplotlib.pyplot as plt", "import numpy as np", "import pandas as pd"]
    for imp in extra_imports:
        if imp not in lines:
            lines.append(imp)
    return code(COLAB + "\n" + "\n".join(lines) + "\n" + STYLE)


# =====================================================================================================

NOTEBOOKS["00_randomness_and_generators"] = [
    md(r"""
# 00 · Randomness and pseudo-random generators

Every simulation in this course starts from a stream of numbers that behave as if they were independent and
uniform on [0, 1). Computers produce them by deterministic recurrences - pseudo-random generators - and the
quality of everything downstream depends on how well the illusion holds. This notebook builds the classical
generators, shows a famous failure (IBM's RANDU, whose triples lie on 15 planes), tests generators
statistically, and ends with the practical rules for seeding and parallel streams in NumPy.
"""),
    setup_cell(["from engstoch import rng as rg, rngtests as rt"]),
    md("## Linear congruential generators"),
    code(r"""g = rg.LCG(5, 3, 16, seed=7)
print("x -> 5x + 3 mod 16 from 7:", g.integers(20).tolist())
print("period:", rg.LCG(5, 3, 16).period(), "| Hull-Dobell conditions:", rg.full_period_conditions(5, 3, 16))
print("x -> 4x + 3 mod 16:", rg.LCG(4, 3, 16, seed=7).integers(12).tolist(), "- stuck at a fixed point after one step")
pm = rg.LCG.park_miller(seed=1)
print("\nPark-Miller minimal standard, first values:", pm.integers(3).tolist())
pm = rg.LCG.park_miller(seed=1); pm.integers(9999)
print("x_10000 =", int(pm.integers(1)[0]), "(the published check value is 1043618065)")"""),
    md(r"""
A linear congruential generator x_{k+1} = (a x_k + c) mod m is a finite-state machine: it must cycle, and the
cycle length is at most m. The Hull-Dobell theorem says when the full period m is reached (c coprime to m,
a − 1 divisible by every prime factor of m, and by 4 if m is). The multiplicative Park-Miller generator
(c = 0, m = 2³¹ − 1 prime) cycles through all 2³¹ − 2 non-zero states because 16807 is a primitive root, and it
comes with a published check value - the way generator implementations are tested.

## RANDU and the lattice structure
"""),
    code(r"""u = rg.LCG.randu(seed=1).random(30_000)
pl = rt.randu_planes(u)
print(f"9 u_k - 6 u_k+1 + u_k+2 takes only the integer values {pl['plane_values'].tolist()} -> {pl['n_planes']} planes")
print("1-D and 2-D tests do not notice:", f"chi-square p = {rt.chi_square_uniform(u)['p_value']:.3f}, serial pairs p = {rt.serial_test(u)['p_value']:.3f}")
t3 = u[: 3 * 10_000].reshape(-1, 3)
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 2, 1, projection="3d"); ax.scatter(*t3[:3000].T, s=1); ax.view_init(elev=10, azim=-58); ax.set_title("RANDU triples (edge-on view)")
ax = fig.add_subplot(1, 2, 2); proj = t3 @ np.array([9, -6, 1]) / np.sqrt(118); ax.hist(proj, bins=300); ax.set_title("projection on (9, -6, 1): 15 spikes"); plt.show()"""),
    md(r"""
RANDU (a = 65539 = 2¹⁶ + 3, m = 2³¹) was the standard generator on IBM mainframes in the 1960s and 70s. Because
a² = 6a − 9 (mod 2³¹), every triple satisfies x_{k+2} − 6x_{k+1} + 9x_k ≡ 0: the points (u_k, u_{k+1}, u_{k+2})
fill only 15 parallel planes of the unit cube, 1/√118 ≈ 0.09 apart. Every LCG has such a lattice (Marsaglia's
theorem); good multipliers make the planes many and close together, which the spectral test measures.

## The spectral test and statistical tests
"""),
    code(r"""for name, a, m in (("Park-Miller", 16807, 2**31 - 1), ("Park-Miller alternative 48271", 48271, 2**31 - 1), ("RANDU", 65539, 2**31), ("a = 3", 3, 2**31 - 1)):
    s2 = rt.spectral_test_2d(a, m)
    print(f"{name:<30}: 2-D lines {s2['line_spacing']:.2e} apart (best possible {1 / np.sqrt(2 / np.sqrt(3) * m):.2e}), figure of merit {s2['merit']:.3f}")
gens = {"PCG32": rg.PCG32(1, 1).random(50_000), "xorshift64*": rg.XorShift64Star(5).random(50_000), "Park-Miller": rg.LCG.park_miller(1).random(50_000),
        "RANDU": rg.LCG.randu(1).random(50_000), "LCG mod 16": rg.LCG(5, 3, 16).random(50_000)}
rows = []
for name, x in gens.items():
    rows.append((name, rt.chi_square_uniform(x)["p_value"], rt.serial_test(x)["p_value"], rt.runs_up_test(x)["p_value"], rt.gap_test(x)["p_value"], rt.autocorrelation_test(x)["p_value"]))
print(pd.DataFrame(rows, columns=["generator", "chi-square", "serial", "runs up", "gap", "lag-1 corr"]).to_string(index=False, float_format="%.3g"))"""),
    md(r"""
Statistical tests can only find evidence *against* a generator: a p-value below a threshold flags a defect,
and a large one proves nothing (RANDU passes all five tests here - its defect is in three dimensions). The
tiny LCG mod 16 fails everything; Park-Miller's lines in 2-D are 1/16807 apart, a figure of merit of 0.34 that
is mediocre by modern standards. Batteries such as TestU01's BigCrush apply hundreds of such tests; PCG and
xorshift-family generators pass them, and so does NumPy's default PCG64.

## Seeding and streams in practice
"""),
    code(r"""a = np.random.default_rng(2026).random(3); b = np.random.default_rng(2026).random(3)
print("same seed, same numbers:", np.array_equal(a, b))
ss = np.random.SeedSequence(2026)
streams = [np.random.default_rng(s) for s in ss.spawn(4)]
x = np.array([s.random(100_000) for s in streams])
print("four spawned streams, max |correlation| between them:", np.round(np.max(np.abs(np.corrcoef(x) - np.eye(4))), 4))
print("PCG32 (O'Neill's reference) first outputs:", [hex(rg.PCG32(42, 54).next32())])"""),
    md(r"""
Reproducibility comes from seeding, independence across replications or parallel workers from *spawning*:
`SeedSequence.spawn` derives child seeds by hashing, so the streams are statistically independent. Two habits
avoid most simulation bugs: never reuse a generator's state across experiments that should be independent,
and always record the seed. The course writes `R(seed)` for a fresh generator.
"""),
]

NOTEBOOKS["01_generating_random_variates"] = [
    md(r"""
# 01 · Generating random variates

Uniform numbers are raw material; simulations need exponential service times, normal noise, Poisson counts,
gamma waiting times and dependent vectors. This notebook covers the general methods - inversion,
acceptance-rejection, transformations - and the specialised algorithms that make them fast: Box-Muller and
Marsaglia's polar method for normals, Walker's alias table for discrete laws, Marsaglia-Tsang for the gamma,
Cholesky factors for multivariate normals and copulas for dependence with arbitrary marginals.
"""),
    setup_cell(["from scipy import stats", "from engstoch import rng as rg, rngtests as rt, special as sf"]),
    md("## Inversion"),
    code(r"""g = R(1)
x = rg.exponential(2.0, g, 50_000)
w = rg.weibull(1.5, 3.0, g, 50_000)
print(f"Exp(rate 2) by -ln(1-U)/2: mean {x.mean():.4f} (0.5), KS p = {rt.ks_test(x, lambda v: 1 - np.exp(-2 * v))['p_value']:.3f}")
print(f"Weibull(1.5, 3) by inversion: KS p = {rt.ks_test(w, lambda v: 1 - np.exp(-(v / 3) ** 1.5))['p_value']:.3f}")
z = rg.inverse_transform(sf.norm_ppf, g, 50_000)
print(f"normal by inverting Phi (AS 241): KS p = {rt.ks_test(z, sf.norm_cdf)['p_value']:.3f}")
d = rg.discrete_inverse([0.1, 0.2, 0.3, 0.4], g, 100_000, values=["a", "b", "c", "d"])
print("discrete inversion frequencies:", {k: round(float(np.mean(d == k)), 3) for k in "abcd"})"""),
    md(r"""
If U is uniform then F⁻¹(U) has distribution F - the most general method, exact whenever the quantile function
can be evaluated, and monotone in U (which variance reduction exploits in notebook 03). For discrete laws the
inverse is a search through the cumulative probabilities: binary search costs O(log n) per draw.

## Acceptance-rejection
"""),
    code(r"""target = lambda v: stats.beta.pdf(v, 2.7, 6.3)
M = target(np.linspace(0, 1, 10_001)).max() * 1.001
r = rg.rejection(target, lambda m: R(2).random(m), lambda v: np.ones_like(v), M, R(3), 20_000)
print(f"Beta(2.7, 6.3) from uniform proposals: M = {M:.3f}, acceptance {r['acceptance']:.3f} (1/M = {1 / M:.3f}), KS p = {stats.kstest(r['x'], 'beta', args=(2.7, 6.3)).pvalue:.3f}")
vv = np.linspace(0, 1, 400)
plt.fill_between(vv, M * np.ones_like(vv), alpha=0.15, label="envelope M g"); plt.plot(vv, target(vv), "k", label="target f")
plt.hist(r["x"], bins=60, density=True, alpha=0.5, label="accepted samples"); plt.legend(); plt.title("acceptance-rejection"); plt.show()"""),
    md(r"""
Draw Y from a proposal g, accept it with probability f(Y)/(M g(Y)): accepted values have density f, and the
number of proposals per acceptance is geometric with mean M. Everything rests on a good envelope - a flat one
over a peaked density wastes most proposals.

## Normals: Box-Muller, polar, ratio of uniforms
"""),
    code(r"""rows = []
for name, fn in (("Box-Muller", lambda: (rg.box_muller(R(4), 100_000), 1.0)), ("polar (Marsaglia)", lambda: (lambda o: (o["x"], o["acceptance"]))(rg.polar_normal(R(5), 100_000))),
                 ("ratio of uniforms", lambda: (lambda o: (o["x"], o["acceptance"]))(rg.ratio_of_uniforms_normal(R(6), 100_000)))):
    x, acc = fn()
    rows.append((name, acc, x.mean(), x.var(), stats.kurtosis(x), stats.kstest(x, "norm").pvalue))
print(pd.DataFrame(rows, columns=["method", "acceptance", "mean", "variance", "excess kurtosis", "KS p"]).to_string(index=False, float_format="%.4f"))
print(f"theory: polar acceptance pi/4 = {np.pi / 4:.4f}, ratio of uniforms sqrt(pi e)/4 = {np.sqrt(np.pi * np.e) / 4:.4f}")"""),
    md(r"""
Box-Muller turns two uniforms into two independent normals through polar coordinates (R² = −2 ln U₁ is
exponential, the angle 2πU₂ uniform). The polar method avoids the sine and cosine by rejecting points outside
the unit disc (acceptance π/4); the ratio-of-uniforms method accepts (U, V) inside a region whose shape is set
by the density. All three are exact.

## Discrete laws: the alias method
"""),
    code(r"""p = R(7).dirichlet(np.ones(50))
prob, alias = rg.alias_table(p)
x = rg.alias_sample((prob, alias), R(8), 500_000)
obs = np.bincount(x, minlength=50)
print(f"50-point law: chi-square p = {rt.chi_square(obs, 500_000 * p)['p_value']:.3f}; one uniform and one comparison per draw")
import time
for n_out in (10, 1000, 100_000):
    q = R(9).dirichlet(np.ones(n_out)); tab = rg.alias_table(q)
    t0 = time.perf_counter(); rg.alias_sample(tab, R(10), 1_000_000); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); rg.discrete_inverse(q, R(10), 1_000_000); ti = time.perf_counter() - t0
    print(f"{n_out:>6} outcomes: alias {ta * 1000:.0f} ms, binary-search inversion {ti * 1000:.0f} ms per million draws")"""),
    md(r"""
Walker's alias table splits the n outcomes into n equal columns, each holding at most two outcomes: draw a
column uniformly, then choose between its own outcome and its alias. Vose's construction builds the table in
O(n); sampling is O(1) whatever n, while inversion grows like log n.

## Gamma, beta and dependence
"""),
    code(r"""g = R(11)
for a in (0.3, 1.0, 2.5, 20.0):
    x = rg.gamma(a, g.standard_normal, g, 50_000)
    print(f"Gamma({a:>4}) by Marsaglia-Tsang: mean {x.mean():7.3f}, var {x.var():7.3f} (both {a}); KS p = {stats.kstest(x, 'gamma', args=(a,)).pvalue:.3f}")
corr = np.array([[1.0, 0.7], [0.7, 1.0]])
xy = rg.gaussian_copula(corr, [lambda u: stats.expon.ppf(u, scale=4.0), lambda u: stats.lognorm.ppf(u, 0.8)], g.standard_normal, 20_000)
print(f"\nGaussian copula with exponential and lognormal marginals: Spearman rho {stats.spearmanr(*xy.T)[0]:.3f} (theory {6 / np.pi * np.arcsin(0.35):.3f}), Pearson {np.corrcoef(xy.T)[0, 1]:.3f}")
plt.scatter(*xy[:3000].T, s=2); plt.xscale("log"); plt.yscale("log"); plt.title("dependent variates with non-normal marginals"); plt.show()"""),
    md(r"""
Marsaglia and Tsang's gamma generator transforms a normal by a cubic and accepts with a squeeze that almost
never needs the logarithm - acceptance above 95 % for every shape. A Gaussian copula imposes the dependence of
a correlated normal vector on any marginals (via U = Φ(Z)); rank correlation is preserved, Pearson correlation
is not - a reminder that "correlation 0.7" means different things on different scales.
"""),
]

NOTEBOOKS["02_monte_carlo_estimation"] = [
    md(r"""
# 02 · Monte Carlo estimation and its error

A Monte Carlo estimate is a sample mean, so its error is governed by the law of large numbers and the central
limit theorem: it shrinks like σ/√n whatever the dimension. This notebook makes that statement operational -
standard errors, confidence intervals and their actual coverage, the sample size for a required precision,
integration in many dimensions where grids fail - and shows where it breaks: rare events, heavy tails and
skewed outputs.
"""),
    setup_cell(["from scipy import stats", "from engstoch import montecarlo as mc, output as oa, special as sf"]),
    md("## The estimator and its standard error"),
    code(r"""h = lambda u: np.exp(u)                      # E[e^U] = e - 1
truth = np.e - 1
ns = np.logspace(2, 6, 9).astype(int)
errs = [abs(mc.mean_ci(h(R(int(n)).random(n)))["estimate"] - truth) for n in ns]
ses = [mc.mean_ci(h(R(int(n)).random(n)))["std_error"] for n in ns]
plt.loglog(ns, errs, "o", label="actual error"); plt.loglog(ns, ses, "-", label="estimated standard error"); plt.loglog(ns, 0.49 / np.sqrt(ns), "k--", label="sigma / sqrt(n)")
plt.xlabel("n"); plt.legend(); plt.title("the n^(-1/2) law"); plt.show()
r = mc.mean_ci(h(R(0).random(10_000)))
print(f"n = 10000: estimate {r['estimate']:.5f} +- {r['std_error']:.5f}, 95 % CI ({r['ci'][0]:.5f}, {r['ci'][1]:.5f}), truth {truth:.5f}")"""),
    md(r"""
The estimator's standard deviation is σ/√n, and σ is estimated from the same sample. Gaining one decimal digit
costs a hundredfold more samples - the price of Monte Carlo's generality, which notebooks 03 and 04 try to
lower.

## Do the intervals cover?
"""),
    code(r"""rows = []
for name, sampler, truth_ in (("uniform, n = 20", lambda r: r.random(20), 0.5), ("exponential, n = 10", lambda r: r.exponential(size=10), 1.0),
                              ("exponential, n = 100", lambda r: r.exponential(size=100), 1.0), ("lognormal(0, 1.5), n = 100", lambda r: r.lognormal(0, 1.5, 100), np.exp(1.125)),
                              ("lognormal(0, 1.5), n = 10000", lambda r: r.lognormal(0, 1.5, 10_000), np.exp(1.125))):
    cis = [mc.mean_ci(sampler(R(100 + k)), method="t")["ci"] for k in range(2000)]
    c = oa.coverage([a for a, b in cis], [b for a, b in cis], truth_)
    rows.append((name, c["coverage"], c["std_error"]))
print(pd.DataFrame(rows, columns=["output", "coverage of nominal 95 % t intervals", "+-"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
The interval is only asymptotically right. For symmetric outputs it is accurate at n = 20; for skewed outputs
it undercovers, and a heavy right tail (a lognormal with σ = 1.5, much like insurance losses or repair costs)
drags coverage down to about 85 % at n = 100 and keeps it slightly below nominal even at n = 10⁴: the sample
rarely contains the large values that carry the mean, so both the mean and its standard error are
underestimated together. Checking coverage by simulation, as here,
is the honest test of any interval procedure.

## How many samples?
"""),
    code(r"""pilot = h(R(1).random(500))
sigma = pilot.std(ddof=1)
for hw in (1e-2, 1e-3, 1e-4):
    print(f"half-width {hw:g} at 95 %: n = {mc.sample_size(sigma, hw):,}")
r = mc.two_stage(lambda n: h(R(2).random(n)), 1e-3)
print(f"\ntwo-stage procedure: pilot sigma {r['pilot_std']:.4f} -> n = {r['n']:,}, achieved half-width {(r['ci'][1] - r['ci'][0]) / 2:.2e}")"""),
    md(r"""
With a pilot estimate of σ, n = (zσ/h)² gives a target half-width h; the two-stage procedure runs the pilot,
computes n and finishes the job. The pilot's own error in σ makes the final half-width random - a few percent
off here.

## Rare events
"""),
    code(r"""rows = []
for a in (1, 2, 3, 4, 5):
    p = 1 - sf.norm_cdf(a)
    n = 100_000
    k = int(np.sum(R(a).standard_normal(n) > a))
    r = mc.proportion_ci(k, n)
    rows.append((a, p, r["estimate"], mc.relative_error(r) if k else np.inf, r["ci"][1], np.sqrt((1 - p) / (n * p))))
print(pd.DataFrame(rows, columns=["a", "P(Z > a)", "estimate (n = 1e5)", "relative error", "Wilson upper limit", "theory sqrt((1-p)/(np))"]).to_string(index=False, float_format="%.3g"))"""),
    md(r"""
For a probability p the relative error of crude Monte Carlo is √((1 − p)/(np)): to get 10 % accuracy one needs
about 100/p samples - 10⁸ for p = 10⁻⁶. Beyond a = 4 the estimate is zero; the Wilson interval still gives an
honest upper limit (the Wald interval would claim certainty). Importance sampling (notebook 03) fixes this.

## Integration where grids fail
"""),
    code(r"""rows = []
for d in (1, 2, 5, 10, 20):
    f = lambda x: np.exp(-np.sum(x, axis=1) / d)
    exact = (d * (1 - np.exp(-1 / d))) ** d
    r = mc.integrate(f, np.zeros(d), np.ones(d), 100_000, R(d))
    m = int(round(100_000 ** (1 / d)))
    g1 = (np.arange(m) + 0.5) / m
    grid_est = (np.mean(np.exp(-g1 / d))) ** d                         # midpoint product rule with m^d points (separable f)
    rows.append((d, exact, abs(r["estimate"] - exact), r["std_error"], m, abs(grid_est - exact)))
print(pd.DataFrame(rows, columns=["d", "exact", "MC error (1e5 points)", "MC std error", "grid points per axis", "midpoint-grid error (same budget)"]).to_string(index=False, float_format="%.2e"))"""),
    md(r"""
With a budget of 10⁵ points a product grid has 10⁵⁄ᵈ points per axis - three per axis in ten dimensions, two in
twenty - and its O(m⁻²) accuracy per axis is gone: from d = 10 on it is no better than Monte Carlo, whose error
is the same σ/√n in every dimension. (This integrand is separable and very smooth, which flatters the grid; a
general grid rule would lose much earlier.) Dimension-independence is why Monte
Carlo dominates in finance, physics and reliability.
"""),
]

NOTEBOOKS["03_variance_reduction"] = [
    md(r"""
# 03 · Variance reduction

Since the error is σ/√n, halving σ is worth quadrupling n. Variance reduction exploits what we know about the
problem: symmetry (antithetic variates), a correlated quantity with a known mean (control variates), where the
important outcomes lie (importance sampling), how the variance is spread over the input space (stratified
sampling), and - when comparing systems - the fact that both see the same randomness (common random numbers).
"""),
    setup_cell(["from scipy import stats", "from engstoch import montecarlo as mc, queues, special as sf, brownian as bm"]),
    md("## Antithetic variates"),
    code(r"""rows = []
for name, h in (("e^u (monotone)", np.exp), ("u^2 (monotone on [0,1])", lambda u: u**2), ("sin(2 pi u)^2 (symmetric)", lambda u: np.sin(2 * np.pi * u) ** 2), ("(u - 0.5)^2 (even about 1/2)", lambda u: (u - 0.5) ** 2)):
    r = mc.antithetic(h, 50_000, R(1))
    rows.append((name, r["correlation"], r["variance_reduction"]))
print(pd.DataFrame(rows, columns=["integrand", "corr(h(U), h(1-U))", "variance reduction"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
Pairing U with 1 − U costs nothing and helps exactly when h(U) and h(1 − U) are negatively correlated - always
for monotone h. For a function symmetric about ½ the pair is perfectly positively correlated and antithetics
*double* the variance per evaluation: the method must match the problem.

## Control variates
"""),
    code(r"""S0, K, r_, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
ST = bm.gbm(S0, r_, sig, T, 1, 100_000, R(2))[:, -1]
payoff = np.exp(-r_ * T) * np.maximum(ST - K, 0)
plain = mc.mean_ci(payoff)
cv = mc.control_variate(payoff, ST, S0 * np.exp(r_ * T))
print(f"European call: plain MC {plain['estimate']:.4f} +- {plain['std_error']:.4f}; control variate S_T: {cv['estimate']:.4f} +- {cv['std_error']:.4f}")
print(f"  beta* = {cv['beta']:.3f}, R^2 = {cv['r_squared']:.3f}, variance reduction {cv['variance_reduction']:.1f}x; Black-Scholes {bm.black_scholes_call(S0, K, r_, sig, T):.4f}")
two = mc.control_variate(payoff, np.column_stack([ST, np.maximum(ST - 110, 0)]), [S0 * np.exp(r_ * T), np.exp(r_ * T) * bm.black_scholes_call(S0, 110, r_, sig, T)])
print(f"  two controls (S_T and the K = 110 call): {two['estimate']:.4f} +- {two['std_error']:.5f}, variance reduction {two['variance_reduction']:.0f}x")"""),
    md(r"""
A control variate C with known mean is subtracted with the coefficient β* = Cov(Y, C)/Var(C) - estimated by a
regression of Y on C - and the variance falls by the factor 1/(1 − R²). The terminal stock price explains
85 % of the call's variance; a second, similar option with a known price explains almost 98 % of it. The
best controls are simplified versions of the problem with closed-form answers.

## Importance sampling
"""),
    code(r"""a = 4.5
p = 1 - sf.norm_cdf(a)
rows = []
for theta in (0.0, 2.0, 3.5, 4.5, 5.5, 8.0):
    r = mc.importance_sampling(lambda z: (z > a).astype(float), lambda z: -0.5 * z**2, lambda n, t=theta: R(3).normal(t, 1, n), lambda z, t=theta: -0.5 * (z - t) ** 2, 100_000)
    rows.append((theta, r["estimate"], mc.relative_error(r) if r["estimate"] > 0 else np.inf))
print(f"P(Z > {a}) = {p:.4e}")
print(pd.DataFrame(rows, columns=["proposal mean theta", "estimate", "relative error"]).to_string(index=False, float_format="%.3g"))"""),
    md(r"""
Sampling from N(θ, 1) and reweighting by the likelihood ratio e^{−θz + θ²/2} keeps the estimator unbiased
for every θ, but its variance depends dramatically on θ: θ = 0 is crude Monte Carlo (no hits at all), θ ≈ a
puts half the samples in the rare region and gives 1 % accuracy with 10⁵ samples, and θ too large wastes them
far in the tail. Exponential tilting to the "dominating point" is the standard choice; a poor proposal can give
infinite variance while *looking* fine.

## Stratified sampling
"""),
    code(r"""h = lambda u: np.where(u > 0.9, 10 * u, 0.1 * u)
plain_sd = np.std(h(R(4).random(1_000_000)))
for strata in (2, 10, 50):
    for alloc in ("proportional", "neyman"):
        r = mc.stratified(h, 10_000, strata, R(5), alloc)
        print(f"{strata:>3} strata, {alloc:<12}: estimate {r['estimate']:.5f} +- {r['std_error']:.5f}, variance reduction {mc.variance_reduction(r, plain_sd, 10_000):6.1f}x")
print("exact:", 0.1 * 0.81 / 2 + 10 * 0.19 / 2)"""),
    md(r"""
Stratification removes the between-strata part of the variance. Proportional allocation can never hurt;
Neyman allocation (n_j ∝ p_j σ_j) puts samples where the variance is - here in the last tenth of the interval -
and multiplies the gain.

## Comparing systems: common random numbers
"""),
    code(r"""def mean_wait(service_scale):
    return lambda r: np.mean(queues.lindley(r.exponential(1 / 0.8, 2000), r.exponential(service_scale, 2000))[200:])
res = mc.common_random_numbers(mean_wait(1.0), mean_wait(0.95), 200)
print(f"mean wait reduction from a 5 % faster server: CRN {res['crn']['estimate']:.3f} +- {res['crn']['std_error']:.3f}, independent streams {res['independent']['estimate']:.3f} +- {res['independent']['std_error']:.3f}")
print(f"variance reduction {res['variance_reduction']:.0f}x - the same 200 replications, only the seeds differ")"""),
    md(r"""
To compare two configurations, drive both with the same random numbers: the difference then reflects the
configurations rather than the luck of the draws. For a queue - whose waits are monotone in the service times
- the variance of the difference falls by an order of magnitude, and the independent-streams interval is too
wide even to be sure of the effect's size. CRN needs synchronisation (the same uniform used for the same
purpose in both runs), which is why simulations dedicate one stream per source of randomness.
"""),
]

NOTEBOOKS["04_quasi_monte_carlo"] = [
    md(r"""
# 04 · Quasi-Monte Carlo

Random points clump and leave gaps; for integration we would rather have points that fill the cube evenly.
Low-discrepancy sequences - van der Corput, Halton, Sobol, lattice rules - do exactly that, and the
Koksma-Hlawka inequality turns evenness (discrepancy) into an error bound of order (log n)ᵈ/n. Randomising
them restores what QMC loses - an error estimate - and keeps most of the gain.
"""),
    setup_cell(["from scipy.stats import qmc as sqmc", "from engstoch import qmc, montecarlo as mc"]),
    md("## Low-discrepancy points"),
    code(r"""fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
sets = {"random": R(1).random((256, 2)), "Halton (2, 3)": qmc.halton(256, 2), "Sobol": qmc.sobol(256, 2), "Korobov lattice a = 76": qmc.korobov(256, 2, 76)}
for ax, (name, x) in zip(axes, sets.items()):
    ax.scatter(*x.T, s=4); ax.set_title(f"{name}\nL2* = {qmc.l2_star_discrepancy(x):.4f}", fontsize=9); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
plt.show()
print("van der Corput base 2:", qmc.van_der_corput(8).tolist())
print("Sobol and Halton agree with SciPy:", np.array_equal(qmc.sobol(512, 6), sqmc.Sobol(6, scramble=False).random(512)), np.allclose(qmc.halton(300, 4), sqmc.Halton(4, scramble=False).random(300)))"""),
    md(r"""
The van der Corput sequence mirrors the digits of 0, 1, 2, … about the radix point, so each new point falls
into the largest remaining gap. Halton uses a different prime base per coordinate; Sobol uses base 2 with
"direction numbers" chosen so that the first 2ᵐ points put exactly one point in every elementary box of a
fixed shape (a (t, m, s)-net); a rank-1 lattice is {i z/n mod 1}. The L2-star discrepancy, computed exactly by
Warnock's formula, quantifies the evenness.

## Convergence: n⁻¹ instead of n⁻¹ᐟ²
"""),
    code(r"""d = 5
f = lambda x: qmc.genz_product_peak(x)
exact = qmc.genz_product_peak_integral(d)
ms = np.arange(6, 16)
err_mc = [np.mean([abs(np.mean(f(R(100 * m + k).random((2**m, d)))) - exact) for k in range(10)]) for m in ms]
err_sob = [np.mean([abs(np.mean(f(np.mod(qmc.sobol(2**m, d) + R(m * 10 + k).random(d), 1))) - exact) for k in range(10)]) for m in ms]
err_hal = [abs(np.mean(f(qmc.halton(2**m, d, start=1))) - exact) for m in ms]
n = 2.0**ms
plt.loglog(n, err_mc, "o-", label="Monte Carlo"); plt.loglog(n, err_sob, "s-", label="randomly shifted Sobol"); plt.loglog(n, err_hal, "^-", label="Halton")
plt.loglog(n, err_mc[0] * (n / n[0]) ** -0.5, "k:", label="n^-1/2"); plt.loglog(n, err_sob[0] * (n / n[0]) ** -1, "k--", label="n^-1"); plt.legend(); plt.xlabel("n"); plt.title(f"Genz product-peak integral, d = {d}"); plt.show()
print(f"slopes: MC {np.polyfit(np.log(n), np.log(err_mc), 1)[0]:.2f}, Sobol {np.polyfit(np.log(n), np.log(err_sob), 1)[0]:.2f}, Halton {np.polyfit(np.log(n), np.log(err_hal), 1)[0]:.2f}")"""),
    md(r"""
For a smooth integrand in five dimensions QMC's error falls nearly like 1/n - at 32 000 points two orders of
magnitude below Monte Carlo's. The advantage shrinks for discontinuous integrands and in high nominal
dimension unless the function has low "effective dimension" (most variance in a few coordinates or in
low-order interactions) - which is often true in practice and is why QMC works in finance with hundreds of
dimensions.

## Randomised QMC: an error estimate again
"""),
    code(r"""rows = []
for m in (8, 10, 12):
    pts = qmc.sobol(2**m, d)
    r = qmc.randomised_qmc(f, pts, 16, R(m))
    plain = mc.mean_ci(f(R(m + 50).random((16 * 2**m, d))))
    rows.append((16 * 2**m, r["estimate"] - exact, r["std_error"], plain["std_error"], (plain["std_error"] / r["std_error"]) ** 2))
print(pd.DataFrame(rows, columns=["evaluations", "RQMC error", "RQMC std error (16 shifts)", "MC std error, same budget", "variance reduction"]).to_string(index=False, float_format="%.3g"))"""),
    md(r"""
A deterministic point set gives no error estimate. Shifting the whole set by a uniform random vector modulo 1
(a Cranley-Patterson rotation) makes each average unbiased; a handful of independent shifts gives a t
interval. Some convergence order is traded for the estimate, but the variance reduction over plain Monte
Carlo still runs into the hundreds or thousands.

## Pitfalls: Halton in high bases, lattice generators
"""),
    code(r"""x = qmc.halton(500, 16)
xs = qmc.halton(500, 16, rng=R(3), scramble="permute")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
axes[0].scatter(x[:, 14], x[:, 15], s=4); axes[0].set_title("Halton, bases 47 and 53")
axes[1].scatter(xs[:, 14], xs[:, 15], s=4); axes[1].set_title("with random digit permutations"); plt.show()
best = qmc.search_korobov(1021, 4)
for a in (2, 76, best["a"]):
    print(f"Korobov n = 1021, d = 4, a = {a:>3}: L2* discrepancy {qmc.l2_star_discrepancy(qmc.korobov(1021, 4, a)):.5f}")"""),
    md(r"""
In high prime bases the first few hundred Halton points march along diagonal lines (the two radical inverses
grow in step); random digit permutations break the correlation. A lattice rule is only as good as its
generating vector: a = 2 is disastrous. Practical rules come from a computer search over a for the smallest
worst-case error in a chosen function space - here a smoothness-2 Korobov space, whose winner is not the one with
the smallest L2-star discrepancy: different criteria rank generators differently, but all good ones beat a bad
one by an order of magnitude.
"""),
]

NOTEBOOKS["05_discrete_time_markov_chains"] = [
    md(r"""
# 05 · Discrete-time Markov chains

A Markov chain forgets everything but its present state: P(X_{n+1} = j | X_n = i, past) = P_ij. Weather, an
inventory, a queue observed at departures, a web surfer, a board game - all are chains, and a few matrix
computations answer most questions about them. This notebook covers the transition matrix and its powers, the
structure of the state space (communicating classes, recurrence, periodicity), and estimating a chain from
data - on ten years of daily rain records.
"""),
    setup_cell(["from engstoch import markov as mk, models, datasets"]),
    md("## Transition matrices and n-step behaviour"),
    code(r"""w = models.weather()
chain = w["chain"]
print(chain, "\nP =\n", chain.P)
for k in (1, 2, 5, 20):
    print(f"P^{k:<2} =", np.round(chain.n_step(k), 4).tolist())
p0 = np.array([0.0, 1.0])
print("from a wet day, P(wet) after 1, 2, 3, 7 days:", [round(float(chain.distribution(p0, k)[1]), 4) for k in (1, 2, 3, 7)])
print("stationary distribution:", chain.stationary().round(4), "| eigenvalues of P:", np.round(chain.spectral_gap()["eigenvalues"].real, 4))"""),
    md(r"""
The rows of Pⁿ converge to the same vector - the chain forgets where it started - geometrically at the rate of
the second eigenvalue, here 1 − a − b = 0.2: the influence of today's weather shrinks fivefold with every day. The limit is the stationary distribution π = πP.

## Structure: classes, recurrence and period
"""),
    code(r"""P = np.array([[0.5, 0.5, 0, 0, 0, 0], [0.3, 0.7, 0, 0, 0, 0], [0.2, 0, 0.3, 0.5, 0, 0], [0, 0, 0.4, 0, 0.6, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0]])
c = mk.MarkovChain(P, list("ABCDEF"))
for cl in c.classify()["classes"]:
    print(f"class {cl['states']}: {'recurrent (closed)' if cl['recurrent'] else 'transient'}, period {cl['period']}")
print("\nP^50 rows from C and from E:", np.round(c.n_step(50)[2], 3).tolist(), np.round(c.n_step(50)[4], 3).tolist())
print("P^51 row from E:", np.round(c.n_step(51)[4], 3).tolist(), " <- E and F alternate for ever")
print("Ehrenfest urn, 10 balls:", models.ehrenfest(10)["chain"].classify())"""),
    md(r"""
The transition graph's strongly connected components are the communicating classes. In a finite chain a class
is recurrent exactly when no probability leaves it; the others are transient and are eventually left for good.
A recurrent class can be periodic: {E, F} alternates, so Pⁿ does not converge, although the long-run fraction
of time in each state does. The Ehrenfest urn - molecules diffusing between two boxes - is irreducible with
period 2, because the count changes parity at every step.

## Simulation and the law of large numbers for chains
"""),
    code(r"""path = chain.simulate(20_000, "dry", R(1))
frac = np.cumsum(path == 1) / np.arange(1, len(path) + 1)
plt.semilogx(np.arange(1, len(path) + 1), frac); plt.axhline(chain.stationary()[1], color="k", ls="--"); plt.xlabel("days"); plt.ylabel("fraction wet"); plt.title("ergodic averages converge to pi"); plt.show()
runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, path, -1]) != 0))
wet_runs = [len(r) for r in np.split(path, np.flatnonzero(np.diff(path)) + 1) if r[0] == 1]
print(f"mean wet spell {np.mean(wet_runs):.3f} days; geometric with parameter P(wet -> dry) = 0.6 gives {1 / 0.6:.3f}")"""),
    md(r"""
The fraction of time in each state converges to π (the ergodic theorem for chains). Sojourns in a state are
geometric - a direct consequence of the Markov property, and a testable one: real wet spells are often longer
than geometric, which is evidence of memory beyond one day.

## Estimating a chain from data
"""),
    code(r"""wd = pd.read_csv(datasets.path("weather.csv"), comment="#")
x = wd.wet.to_numpy()
est = mk.estimate(x, 2, ["dry", "wet"])
print("MLE of P (ten years):\n", np.round(est["P"], 4), "\nstandard errors:\n", np.round(est["std_error"], 4))
print(f"\nfraction of wet days {x.mean():.4f}; the fitted chain's stationary P(wet) {est['chain'].stationary()[1]:.4f}")
print("first-order vs second-order test:", {k: round(v, 3) for k, v in mk.order_test(x, 2).items()})
season = (wd.day_of_year - 1) // 91
rows = []
for s_, name in enumerate(["Jan-Mar", "Apr-Jun", "Jul-Sep", "Oct-Dec"]):
    idx = np.flatnonzero(season.to_numpy()[:-1] == s_)
    n_dw = np.sum((x[idx] == 0) & (x[idx + 1] == 1)); n_d = np.sum(x[idx] == 0)
    n_wd = np.sum((x[idx] == 1) & (x[idx + 1] == 0)); n_w = np.sum(x[idx] == 1)
    rows.append((name, n_dw / n_d, np.sqrt((n_dw / n_d) * (1 - n_dw / n_d) / n_d), n_wd / n_w))
print(pd.DataFrame(rows, columns=["quarter", "P(dry->wet)", "s.e.", "P(wet->dry)"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
The maximum-likelihood estimate of P_ij is the fraction of departures from i that went to j, with binomial
standard errors. The Anderson-Goodman test rejects the first-order chain (p ≈ 0.006) - but the cause is not a
two-day memory: splitting the data by season shows P(dry → wet) differing between quarters by many standard
errors, and mixing seasons with different persistence mimics higher-order dependence. A homogeneous chain fitted to the whole
record averages over the seasons - it reproduces the annual fraction of wet days and misdescribes every
season. Testing homogeneity is as important as testing the order.
"""),
]

NOTEBOOKS["06_long_run_behaviour_and_mixing"] = [
    md(r"""
# 06 · Long-run behaviour and mixing

Stationary distributions answer "where does the chain spend its time?", mean return and first-passage times
answer "how long until?", and mixing times answer "how soon does the starting point stop mattering?" - the
question every Markov chain Monte Carlo user must ask. This notebook computes all three, meets reversibility
and detailed balance, and closes with PageRank, the most widely used stationary distribution.
"""),
    setup_cell(["from engstoch import markov as mk, models"]),
    md("## Stationary distributions three ways"),
    code(r"""inv = models.inventory_chain(s=2, S=6, demand_mean=2.0)
c = inv["chain"]
pi = c.stationary()
pw = c.stationary_power()
print("end-of-week stock, stationary law:", pi.round(4).tolist())
print(f"power iteration: {pw['iterations']} iterations, max difference {np.max(np.abs(pw['pi'] - pi)):.1e}; |lambda_2| = {c.spectral_gap()['slem']:.3f}")
path = c.simulate(200_000, 6, R(1))
print("long-run fractions of a simulated path:", (np.bincount(path, minlength=7) / len(path)).round(4).tolist())
print(f"expected stock {np.dot(np.arange(7), pi):.3f}; probability of ending a week empty {pi[0]:.4f}")"""),
    md(r"""
The linear solve πP = π with Σπ = 1 is exact; power iteration converges at the rate |λ₂|; a long simulation
converges at the Monte Carlo rate. The stationary law turns the policy into numbers: average stock, stock-out
frequency and, with costs attached, the long-run cost per week - the basis for choosing s and S.

## Return times, first passages and Kemeny's constant
"""),
    code(r"""M = c.mean_first_passage()
print("mean return times (diagonal) vs 1/pi:", np.round(np.diag(M), 3).tolist(), np.round(1 / pi, 3).tolist())
print("mean weeks to first stock-out, from each starting stock:", np.round(M[:, 0], 2).tolist())
print(f"Kemeny's constant: {c.kemeny_constant():.4f};  sum_j pi_j m_ij for each i (j != i):", np.round([np.sum(pi * M[i]) - pi[i] * M[i, i] for i in range(7)], 4).tolist())"""),
    md(r"""
Kac's formula says the mean return time to a state is 1/π_i - frequent states are revisited quickly. Mean
first-passage times come from the fundamental matrix Z = (I − P + 1π)⁻¹. A curiosity with uses in network
analysis: the expected time to reach a π-random target is the same from every starting state (Kemeny's
constant).

## Mixing
"""),
    code(r"""eh = models.ehrenfest(20)
lazy = eh["lazy"]
ks = np.arange(1, 200)
pi_e = lazy.stationary()
tv = [0.5 * np.sum(np.abs(lazy.n_step(k)[0] - pi_e)) for k in ks]
tv_mid = [0.5 * np.sum(np.abs(lazy.n_step(k)[10] - pi_e)) for k in ks]
plt.semilogy(ks, tv, label="start with all 20 balls in one urn"); plt.semilogy(ks, tv_mid, label="start balanced (10 / 10)")
gap = lazy.spectral_gap()
plt.semilogy(ks, (1 - gap["gap"]) ** ks, "k--", label="(1 - gap)^k"); plt.axhline(0.25, color="0.6"); plt.legend(); plt.xlabel("steps"); plt.ylabel("TV distance"); plt.show()
print(f"lazy Ehrenfest, N = 20: spectral gap {gap['gap']:.4f} (= 1/N), relaxation time {gap['relaxation_time']:.0f}, mixing time t_mix(1/4) = {lazy.mixing_time(0.25)}")
for N in (10, 20, 40, 80):
    print(f"  N = {N:3d}: t_mix = {models.ehrenfest(N)['lazy'].mixing_time(0.25):4d}  (N log N / 2 = {N * np.log(N) / 2:.0f})")"""),
    md(r"""
Total-variation distance from π falls geometrically at the rate of the spectral gap, but the time to reach a
small distance also depends on the start: from the extreme state the walk needs about ½N log N steps (a
"cutoff": the distance stays near 1 and then drops abruptly), from the balanced state much less. The
relaxation time 1/gap = N understates the mixing time by the log factor. This is the arithmetic behind MCMC
burn-in.

## Reversibility and detailed balance
"""),
    code(r"""rows = []
for name, chain in (("Ehrenfest", eh["chain"]), ("random walk on a graph", mk.random_walk_on_graph(np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]))),
                    ("inventory (s, S)", c), ("cyclic drift 0 -> 1 -> 2", mk.MarkovChain([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]))):
    lam = chain.spectral_gap()["eigenvalues"]
    rows.append((name, chain.is_reversible(), bool(np.allclose(lam.imag, 0))))
print(pd.DataFrame(rows, columns=["chain", "reversible (detailed balance)", "real spectrum"]).to_string(index=False))"""),
    md(r"""
A chain is reversible when π_i P_ij = π_j P_ji: in equilibrium the flow i → j equals the flow j → i, and a film
of the chain looks the same run backwards. Reversible chains have real spectra (P is similar to a symmetric
matrix); the cyclic chain circulates probability and has complex eigenvalues. Detailed balance is how MCMC
algorithms are *designed* (notebook 18): choose P so that it holds for the target π.

## PageRank
"""),
    code(r"""web = models.web_graph()
for d in (0.5, 0.85, 0.99):
    pr = mk.pagerank(web["adjacency"], d)
    print(f"damping {d}: " + ", ".join(f"{p} {v:.3f}" for p, v in sorted(zip(web["pages"], pr), key=lambda t: -t[1])))"""),
    md(r"""
PageRank is the stationary distribution of a surfer who follows a random link with probability d and jumps to
a random page otherwise; the jump makes the chain irreducible and aperiodic (and fixes the dangling PDF page).
As d → 1 the ranking concentrates on the pages the link structure traps the surfer in; the spectral gap is at
least 1 − d, which is why Google's power iteration converged in a few dozen steps on billions of pages.
"""),
]

NOTEBOOKS["07_absorption_hitting_and_random_walks"] = [
    md(r"""
# 07 · Absorption, hitting times and random walks

How likely is a gambler to be ruined, how long does a board game last, will a random walker come home? These
are absorption and hitting questions, answered by first-step analysis - a linear system - and, for random
walks, by combinatorics: the reflection principle, the ballot theorem, the arcsine law and Pólya's theorem
on recurrence in one, two and three dimensions.
"""),
    setup_cell(["from engstoch import markov as mk, walks as wk, models"]),
    md("## The gambler's ruin by the fundamental matrix"),
    code(r"""N, p = 10, 0.47
P = np.zeros((N + 1, N + 1)); P[0, 0] = P[N, N] = 1
for i in range(1, N):
    P[i, i + 1], P[i, i - 1] = p, 1 - p
ab = mk.MarkovChain(P).absorption()
rows = [(i, ab["B"][i - 1, 0], wk.ruin_probability(i, N, p), ab["expected_steps"][i - 1], wk.ruin_duration(i, N, p), np.sqrt(ab["variance_steps"][i - 1])) for i in (1, 3, 5, 7, 9)]
print(pd.DataFrame(rows, columns=["capital i", "P(ruin) N R", "closed form", "E[bets] N 1", "closed form", "sd of bets"]).to_string(index=False, float_format="%.4f"))
for N_ in (10, 100, 1000):
    print(f"start with half of N = {N_}: P(ruin) at p = 0.47 is {wk.ruin_probability(N_ // 2, N_, 0.47):.4f}, at p = 0.5 is {wk.ruin_probability(N_ // 2, N_, 0.5):.2f}")"""),
    md(r"""
With transient states T and absorbing states A, the fundamental matrix N = (I − Q)⁻¹ counts expected visits;
B = NR gives absorption probabilities and N1 the expected time. A 3 % disadvantage per bet is invisible over a
few bets and decisive over many: with stakes small relative to the capital, ruin becomes almost certain - the
casino's business model in one line.

## Snakes and ladders
"""),
    code(r"""sl = models.snakes_and_ladders()
ab = sl["chain"].absorption()
print(f"expected throws to finish: {ab['expected_steps'][0]:.3f} (sd {np.sqrt(ab['variance_steps'][0]):.2f})")
games = [int(np.argmax(sl["chain"].simulate(500, 0, R(k)) == 36)) for k in range(5000)]
print(f"5000 simulated games: mean {np.mean(games):.3f} +- {np.std(games) / np.sqrt(5000):.3f}")
plt.hist(games, bins=range(0, 100, 2), density=True); plt.xlabel("throws"); plt.title("game length: a long right tail"); plt.show()
vis = ab["N"][0]
top = np.argsort(-vis)[:5]
print("most visited squares (expected visits per game):", {int(ab['transient'][k]): round(float(vis[k]), 2) for k in top})"""),
    md(r"""
The board is an absorbing chain with 37 states; one linear solve gives the mean and variance of the game's
length and the expected number of visits to every square - which squares are worth a ladder, which snakes hurt
most - without playing a single game.

## The reflection principle and the ballot theorem
"""),
    code(r"""n = 20
for a in (2, 4, 6):
    print(f"P(max of a 20-step walk >= {a}) = {wk.max_distribution(n, a):.5f};  2 P(S_20 >= {a}) - P(S_20 = {a}) = {(2 * sum(wk.paths_count(n, k) for k in range(a, n + 1)) - wk.paths_count(n, a)) / 2**n:.5f}")
walks = wk.simple_walk(n, R(2), n_paths=200_000)
print("simulated P(max >= 4):", np.mean(walks.max(axis=1) >= 4).round(4))
for A, B in ((6, 4), (60, 40), (51, 49)):
    print(f"ballot: A = {A}, B = {B} -> A strictly ahead throughout the count with probability {wk.ballot(A, B):.3f}")"""),
    md(r"""
Reflecting a path at its first visit to level a maps paths that touch a and end below it one-to-one onto paths
that end above it: P(max ≥ a) = P(S_n ≥ a) + P(S_n > a). The same trick proves the ballot theorem - the winner
of a 51 : 49 vote leads all the way through the count only 2 % of the time - and reappears for Brownian motion
in notebook 15.

## Returns to the origin and the arcsine law
"""),
    code(r"""w = wk.simple_walk(1000, R(3), n_paths=20_000)
frac_pos = np.mean(w[:, 1:] > 0, axis=1)
x = np.linspace(0, 1, 201)
plt.hist(frac_pos, bins=40, density=True, alpha=0.6, label="fraction of time ahead, 1000 steps"); plt.plot(x[1:-1], 1 / (np.pi * np.sqrt(x[1:-1] * (1 - x[1:-1]))), "k", label="arcsine density")
plt.legend(); plt.show()
print(f"P(ahead less than 10 % of the time) = {np.mean(frac_pos <= 0.1):.3f} (arcsine law {wk.arcsine_cdf(0.1):.3f}); P(between 45 % and 55 %) = {np.mean((frac_pos > 0.45) & (frac_pos < 0.55)):.3f}")
print(f"P(first return at step 2n), n = 1, 2, 5: {[round(wk.first_return_probability(k), 4) for k in (1, 2, 5)]}; they sum to 1 but with infinite mean")"""),
    md(r"""
In a fair game the leader rarely changes: the fraction of time spent ahead follows the U-shaped arcsine law, so
a 10 % / 90 % split is *more* likely than 50 / 50. The walk returns to 0 with probability one, but the return
time has infinite mean (its tail decays like n^{−3/2}).

## Pólya: recurrence depends on the dimension
"""),
    code(r"""rows = []
for d in (1, 2, 3):
    for steps in (100, 1000, 10_000):
        r = wk.return_probability_by(d, steps, 2000, R(10 * d + steps % 7))
        rows.append((d, steps, r["fraction"], r["std_error"]))
print(pd.DataFrame(rows, columns=["dimension", "steps", "fraction returned", "+-"]).to_string(index=False, float_format="%.3f"))
print(f"limits: 1, 1 (slowly, like 1 - pi/log n), and {wk.POLYA_3D:.4f} in three dimensions")"""),
    md(r"""
"A drunk man will find his way home, but a drunk bird may get lost forever" (Kakutani): the simple walk is
recurrent in one and two dimensions and transient in three or more. In two dimensions the approach to 1 is
logarithmically slow - after 10⁴ steps more than a quarter of the walkers have still not returned - which no finite
simulation could distinguish from transience without the theory.
"""),
]

NOTEBOOKS["08_branching_processes"] = [
    md(r"""
# 08 · Branching processes

Each individual of a generation leaves a random number of offspring, independently: the Galton-Watson process
models family names, epidemics in their early phase, neutron chain reactions, cell lineages and the spread of a
viral post. Its fate is decided by the mean number of offspring m and computed with one tool - the probability
generating function, whose iterates give the extinction probabilities generation by generation.
"""),
    setup_cell(["from engstoch import walks as wk"]),
    md("## Generation sizes"),
    code(r"""off = wk.pgf_poisson(1.3)
paths = np.array([wk.simulate_galton_watson(lambda k, r: r.poisson(1.3, k), 15, R(i)) for i in range(2000)])
mom = [wk.generation_moments(off, n) for n in range(16)]
print("E[Z_n] simulated vs m^n:", [(round(float(paths[:, n].mean()), 2), round(mom[n]["mean"], 2)) for n in (5, 10, 15)])
print("Var[Z_n] simulated vs formula:", [(round(float(paths[:, n].var()), 1), round(mom[n]["var"], 1)) for n in (5, 10)])
for k in range(40):
    plt.semilogy(np.maximum(paths[k], 0.8), color="C0" if paths[k, -1] else "C3", alpha=0.6, lw=1)
plt.xlabel("generation"); plt.ylabel("Z_n (extinct lines in red)"); plt.title("Poisson(1.3) offspring"); plt.show()"""),
    md(r"""
The mean grows like mⁿ, but the variance grows faster and individual lines either die early or grow
geometrically: the conditional law of Z_n/mⁿ given survival converges to a non-degenerate limit (the
Kesten-Stigum theorem). A population that has survived a few generations is very likely to survive for ever.

## Extinction: the smallest fixed point of the pgf
"""),
    code(r"""s = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(5.5, 5))
for m in (0.8, 1.0, 1.5, 2.5):
    G = wk.pgf_poisson(m)["G"]
    q = wk.extinction_probability(wk.pgf_poisson(m))["q"]
    ax.plot(s, G(s), label=f"m = {m}: q = {q:.4f}")
ax.plot(s, s, "k--"); ax.legend(); ax.set_xlabel("s"); ax.set_title("G(s) and the diagonal"); plt.show()
q = wk.extinction_probability(wk.pgf_poisson(1.5))
qn = wk.extinction_by_generation(wk.pgf_poisson(1.5), 12)
print(f"Poisson(1.5): q = {q['q']:.6f} after {q['iterations']} iterations; P(extinct by generation n):", qn.round(4).tolist())
sim = np.mean([wk.simulate_galton_watson(lambda k, r: r.poisson(1.5, k), 12, R(1000 + i))[-1] == 0 for i in range(5000)])
print(f"simulated P(extinct by generation 12): {sim:.4f}")"""),
    md(r"""
Extinction by generation n has probability G∘G∘…∘G(0), and the limit q is the smallest root of G(s) = s in
[0, 1]. Since G is convex with G(1) = 1 and G′(1) = m, the root is 1 when m ≤ 1 and below 1 when m > 1 - the
threshold theorem. With Poisson offspring q = e^{m(q−1)}, the same equation as the final size of an epidemic
(notebook 11).

## Criticality and the variance of the offspring law
"""),
    code(r"""rows = []
for name, off in (("Poisson(1.2)", wk.pgf_poisson(1.2)), ("geometric, mean 1.2", wk.pgf_geometric(1 / 2.2)), ("0 or 2 children, mean 1.2", wk.pgf_discrete([0.4, 0.0, 0.6])),
                  ("1 or 2 children, mean 1.2", wk.pgf_discrete([0.0, 0.8, 0.2])), ("Poisson(1.0) - critical", wk.pgf_poisson(1.0))):
    r = wk.extinction_probability(off, tol=1e-12)
    rows.append((name, off["mean"], off["var"], r["q"], r["iterations"]))
print(pd.DataFrame(rows, columns=["offspring law", "mean", "variance", "extinction prob.", "iterations to 1e-12"]).to_string(index=False, float_format="%.4f"))
print(f"\nsubcritical Poisson(0.8): expected total progeny 1/(1 - m) = {wk.total_progeny_mean(wk.pgf_poisson(0.8)):.1f}")"""),
    md(r"""
The same mean gives very different fates: the more variable the offspring law, the likelier extinction
(five geometric lines in six die out at m = 1.2, two in three Poisson lines, and none of those in which every
individual has at least one child). At criticality
extinction is certain but slow - the fixed-point iteration converges sublinearly and the expected total
progeny is infinite. Early-epidemic control works by pushing m below one, and superspreading (high variance)
helps outbreaks fizzle.
"""),
]

NOTEBOOKS["09_the_poisson_process"] = [
    md(r"""
# 09 · The Poisson process

Arrivals of calls, customers, failures, photons or insurance claims are, to a first approximation, a Poisson
process: independent counts in disjoint intervals, exponential gaps, uniformly scattered times given their
number. This notebook builds the process three ways, makes it non-homogeneous (thinning and time change),
compounds and splits it, scatters it in the plane - and tests it on 20 days of a call centre's arrival log.
"""),
    setup_cell(["from scipy import stats", "from engstoch import poisson as pp, models, datasets, rngtests as rt"]),
    md("## Three constructions of one process"),
    code(r"""rate, T = 3.0, 2.0
a = [len(pp.homogeneous(rate, T, R(k))) for k in range(20_000)]
b = [len(pp.homogeneous_order_statistics(rate, T, R(k))) for k in range(20_000)]
print(f"N(2) by exponential gaps: mean {np.mean(a):.3f}, var {np.var(a):.3f}; by Poisson count + sorted uniforms: mean {np.mean(b):.3f}, var {np.var(b):.3f}; theory 6, 6")
t = pp.homogeneous(rate, 5000.0, R(1))
gaps = pp.interarrival_gaps(t)
print(f"gaps: mean {gaps.mean():.4f} (1/3), cv {gaps.std() / gaps.mean():.3f} (1), KS p vs Exp(3) = {rt.ks_test(gaps, lambda x: 1 - np.exp(-3 * x))['p_value']:.3f}")
print(f"memorylessness: P(gap > 0.5 | gap > 0.3) = {np.mean(gaps[gaps > 0.3] > 0.5):.4f} vs P(gap > 0.2) = {np.mean(gaps > 0.2):.4f}")"""),
    md(r"""
Exponential gaps, a Poisson number of uniform points, and (implicitly) independent Poisson counts in small
intervals are three equivalent definitions. The exponential's memorylessness is the defining property: having
waited does not bring the next arrival closer.

## Non-homogeneous arrivals: thinning and time change
"""),
    code(r"""lam = models.call_centre_rate
th = pp.thinning(lam, 160.0, 24.0, R(2))
print(f"one simulated day: {len(th['times'])} calls; expected {pp.cumulative_intensity(lam, 24.0, 20001):.1f}; thinning acceptance {th['acceptance']:.3f}")
grid = np.linspace(8, 20, 2001)
cum = np.r_[0, np.cumsum(0.5 * (lam(grid[1:]) + lam(grid[:-1])) * np.diff(grid))]
tc = pp.time_change(lambda e: np.interp(e, cum, grid), cum[-1], R(3))
print(f"by inverting the cumulative intensity: {len(tc)} calls")
days = [pp.thinning(lam, 160.0, 24.0, R(10 + k))["times"] for k in range(30)]
ie = pp.intensity_estimate(days, np.arange(8, 20.5, 0.5))
plt.step(ie["edges"][:-1], ie["rate"], where="post", label="estimate from 30 simulated days"); plt.plot(grid, lam(grid), "k", label="true lambda(t)")
plt.fill_between(ie["edges"][:-1], ie["rate"] - 2 * ie["std_error"], ie["rate"] + 2 * ie["std_error"], step="post", alpha=0.2); plt.legend(); plt.xlabel("hour"); plt.ylabel("calls / hour"); plt.show()"""),
    md(r"""
Thinning (Lewis and Shedler) generates candidates at the peak rate and keeps each with probability λ(t)/λ_max
- simple, exact, and wasteful when the peak is much higher than the average (here, over a 24-hour day with
the centre closed half of it, only 27 % are kept). The time
change maps a unit-rate process through the inverse cumulative intensity Λ⁻¹ and wastes nothing.

## The call-centre log
"""),
    code(r"""cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
per_day = cc.groupby(["day", "weekday"]).size().reset_index(name="calls")
print(per_day.groupby("weekday").calls.agg(["mean", "std", "count"]).round(1))
print("\ndispersion test, all 20 daily totals:", {k: round(v, 4) for k, v in pp.dispersion_test(per_day.calls).items()})
non_mon = per_day[(per_day.weekday != "Mon") & (per_day.day != 8)].calls
print("dispersion test, Tuesday-Friday except day 8:", {k: round(v, 4) for k, v in pp.dispersion_test(non_mon).items()})
d8 = cc[cc.day == 8].arrival_h
plt.hist(d8, bins=np.arange(8, 20.25, 0.25)); plt.title("day 8: a 90-minute hole in the log"); plt.xlabel("hour"); plt.show()"""),
    md(r"""
Daily totals are overdispersed - the variance exceeds the mean - which rejects a single Poisson process with a
fixed daily profile. Two documented causes explain it: Mondays carry about 15 % more calls, and day 8 lost an
afternoon to a telephony outage. Excluding both, the remaining days are consistent with Poisson counts. Real
arrival data almost always show this extra variability (a "doubly stochastic" or Cox process); staffing models
that ignore it under-provision.

## Within-hour behaviour: conditional uniformity
"""),
    code(r"""rows = []
for day in (2, 3, 4):
    for h0 in (9, 13, 17):
        t = cc[(cc.day == day) & (cc.arrival_h >= h0) & (cc.arrival_h < h0 + 0.5)].arrival_h.to_numpy() - h0
        rows.append((day, f"{h0}:00-{h0}:30", len(t), pp.uniformity_test(t, 0.5)["p_value"], rt.ks_test(np.diff(t), lambda x: 1 - np.exp(-len(t) / 0.5 * x))["p_value"]))
print(pd.DataFrame(rows, columns=["day", "window", "calls", "uniformity KS p", "exponential gaps KS p"]).to_string(index=False, float_format="%.3f"))
r = pp.rate_mle(int(((cc.day == 3) & (cc.arrival_h >= 10) & (cc.arrival_h < 11)).sum()), 1.0)
print(f"\nday 3, 10:00-11:00: rate {r['rate']:.0f} calls/h, exact 95 % interval ({r['ci'][0]:.0f}, {r['ci'][1]:.0f}); model value {models.call_centre_rate(10.5):.0f}")"""),
    md(r"""
Over half an hour the rate is roughly constant, and given their number the arrival times should be uniform
(the order-statistics property) with exponential gaps - both tests pass. The Garwood interval, built from
chi-square quantiles, is exact for a Poisson count.

## Compound, split and spatial processes
"""),
    code(r"""tot = [pp.compound(5.0, 10.0, lambda n, r: r.lognormal(1.0, 0.8, n), R(k))["total"] for k in range(20_000)]
EY, EY2 = np.exp(1.0 + 0.32), np.exp(2.0 + 2 * 0.64)
mom = pp.compound_moments(5.0, 10.0, EY, EY2)
print(f"compound Poisson claims: mean {np.mean(tot):.1f} (theory {mom['mean']:.1f}), sd {np.std(tot):.1f} (theory {np.sqrt(mom['var']):.1f}), 99.5 % quantile {np.quantile(tot, 0.995):.0f}")
types = pp.split(pp.homogeneous(10.0, 1000.0, R(4)), [0.2, 0.8], R(5))
c1, c2 = pp.counts_in_windows(types[0], np.arange(0, 1001, 1)), pp.counts_in_windows(types[1], np.arange(0, 1001, 1))
print(f"split stream: rates {len(types[0]) / 1000:.2f} and {len(types[1]) / 1000:.2f}, correlation of counts per unit time {np.corrcoef(c1, c2)[0, 1]:.3f}")
pts = pp.spatial_rectangle(50.0, 2.0, 1.0, R(6))
grid = np.stack(np.meshgrid(np.linspace(0, 2, 60), np.linspace(0, 1, 30)), -1).reshape(-1, 2)
dist = np.min(np.hypot(grid[:, None, 0] - pts[None, :, 0], grid[:, None, 1] - pts[None, :, 1]), axis=1)
print(f"spatial process with 50 points per unit area: {len(pts)} points; P(nearest point within 0.1) = {np.mean(dist <= 0.1):.3f} vs 1 - exp(-50 pi 0.01) = {pp.nearest_neighbour_cdf(0.1, 50.0):.3f} (edge effects lower it)")"""),
    md(r"""
A compound Poisson sum has mean λtE[Y] and variance λtE[Y²] - the variance depends on the second moment of the
claim size, so heavy-tailed claims dominate the risk. Randomly typing the arrivals of a Poisson process
produces *independent* Poisson streams - a remarkable property that makes routing calls to skills or sorting
parts by defect type tractable. In the plane, counts in regions are Poisson and the distance to the nearest
point has the law 1 − e^{−λπr²}, the starting point of spatial statistics.
"""),
]

NOTEBOOKS["10_renewal_processes"] = [
    md(r"""
# 10 · Renewal processes

Replace the Poisson process's exponential gaps by any positive i.i.d. lifetimes and you have a renewal process:
component replacements, bus arrivals, machine restarts. Memory returns - the age of the current component
matters - and with it the renewal function, the renewal-reward theorem that prices maintenance policies, and
the inspection paradox that makes the bus you wait for later than average.
"""),
    setup_cell(["from engstoch import renewal as rn"]),
    md("## The renewal function"),
    code(r"""wm = rn.weibull_moments(2.0, 1.0)
cdf = lambda x: 1 - np.exp(-(x**2))
t, m = rn.renewal_function(cdf, 8.0, 3000)
sim_t = np.arange(0.5, 8.01, 0.5)
sims = np.array([[np.searchsorted(rn.simulate(lambda n, r: r.weibull(2.0, n), 9.0, R(1000 * k + i)), tt, side="right") for tt in sim_t] for k, i in zip(range(3000), range(3000))])
plt.plot(t, m, label="renewal equation"); plt.plot(sim_t, sims.mean(axis=0), "o", label="3000 simulated paths")
plt.plot(t, t / wm["mean"], "k:", label="t / mu"); plt.plot(t, rn.asymptote(t, wm["mean"], wm["var"]), "k--", label="t / mu + (sigma^2 - mu^2)/(2 mu^2)")
plt.legend(); plt.xlabel("t"); plt.ylabel("m(t) = E N(t)"); plt.title("Weibull(2) lifetimes"); plt.show()
for x in (1.0, 4.0, 8.0):
    print(f"m({x}) = {np.interp(x, t, m):.4f};  asymptote {rn.asymptote(x, wm['mean'], wm['var']):.4f}")"""),
    md(r"""
The renewal function solves m(t) = F(t) + ∫₀ᵗ m(t − s) dF(s) (condition on the first renewal); a trapezoidal
discretisation solves it to five digits. The elementary renewal theorem gives m(t)/t → 1/μ, and Smith's key
renewal theorem adds the constant (σ² − μ²)/(2μ²): lifetimes less variable than the exponential (Weibull with
shape 2 has cv 0.52) give fewer renewals than t/μ early on.

## The inspection paradox
"""),
    code(r"""rows = []
for name, sampler, mean, m2 in (("exponential, mean 1", lambda n, r: r.exponential(1.0, n), 1.0, 2.0), ("Weibull(2, 1)", lambda n, r: r.weibull(2.0, n), wm["mean"], wm["second_moment"]),
                               ("uniform(0.5, 1.5)", lambda n, r: r.uniform(0.5, 1.5, n), 1.0, 1 + 1 / 12), ("lognormal, cv 2", lambda n, r: r.lognormal(-0.5 * np.log(5), np.sqrt(np.log(5)), n), 1.0, 5.0)):
    ins = rn.inspect(sampler, 40.0, 4000, R(len(name)))
    rows.append((name, mean, ins["length"].mean(), rn.length_biased_mean(mean, m2), ins["residual"].mean(), rn.mean_residual_life(mean, m2)))
print(pd.DataFrame(rows, columns=["inter-renewal law", "E[X]", "mean interval seen", "E[X^2]/E[X]", "mean wait", "E[X^2]/(2E[X])"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
An observer arriving at a fixed time lands in a long interval more often than a short one - length-biased
sampling - and sees a mean interval E[X²]/E[X] ≥ E[X]. With exponential gaps the interval seen is twice the
average and the wait for the next renewal is a full mean gap (memorylessness); with nearly regular gaps the
wait approaches half a gap; with highly variable gaps it is worse than a full mean gap (the lognormal estimate
is noisy - its length-biased law has a heavy tail). Passengers at a bus stop, class sizes
reported by students, and hospital-stay surveys all suffer this bias.

## Renewal-reward: when should a part be replaced?
"""),
    code(r"""shape, scale = 3.0, 10.0
surv = lambda x: np.exp(-((np.asarray(x) / scale) ** shape))
T = np.linspace(1, 25, 300)
for cf in (2.0, 5.0, 20.0):
    o = rn.optimal_age_replacement(surv, 1.0, cf, T)
    run_to_failure = cf / rn.weibull_moments(shape, scale)["mean"]
    plt.plot(T, o["costs"], label=f"failure cost {cf:g}: replace at {o['T']:.1f}, saving {100 * (1 - o['cost_rate'] / run_to_failure):.0f} %")
plt.ylim(0, 1.2); plt.xlabel("planned replacement age T"); plt.ylabel("long-run cost per unit time"); plt.legend(fontsize=8); plt.title("age replacement, Weibull(3, 10) lifetimes"); plt.show()
print("exponential lifetimes: cost always decreasing in T ->", bool(np.all(np.diff(rn.age_replacement_cost(T, lambda x: np.exp(-np.asarray(x) / 10), 1.0, 5.0)) < 0)))"""),
    md(r"""
Each replacement starts a new cycle, so the long-run cost rate is E[cycle cost]/E[cycle length] (the
renewal-reward theorem) - here (c_p R(T) + c_f (1 − R(T))) / ∫₀ᵀ R(t) dt. With wear-out (increasing hazard)
and expensive failures, preventive replacement saves up to two-thirds of the cost; with exponential lifetimes
it never pays, because an old part is as good as new.

## Availability: the alternating renewal process
"""),
    code(r"""t = np.linspace(0, 30, 300)
lam, mu = 0.1, 0.9
up_down = []
for k in range(3000):
    r = R(5000 + k); s, up, times, states = 0.0, True, [0.0], [1]
    while s < 30:
        s += r.exponential(1 / lam) if up else r.exponential(1 / mu)
        up = not up; times.append(s); states.append(int(up))
    up_down.append(np.array(states)[np.searchsorted(times, t, side="right") - 1])
plt.plot(t, np.mean(up_down, axis=0), label="simulated P(up at t)"); plt.plot(t, rn.availability_exponential(t, lam, mu), "k--", label="mu/(lambda+mu) + lambda/(lambda+mu) e^{-(lambda+mu)t}")
plt.axhline(rn.availability(1 / lam, 1 / mu), color="0.6"); plt.legend(); plt.xlabel("t"); plt.show()
print(f"long-run availability E[U]/(E[U] + E[D]) = {rn.availability(10.0, 1 / 0.9):.4f} - the same for any up and down distributions with these means")"""),
    md(r"""
A unit alternating between up and down periods is available a fraction E[U]/(E[U] + E[D]) of the time in the
long run, whatever the distributions - an *insensitivity* that makes the formula robust. The transient
approach depends on the distributions; for exponential periods it is a single exponential at rate λ + μ.
"""),
]

NOTEBOOKS["11_continuous_time_markov_chains"] = [
    md(r"""
# 11 · Continuous-time Markov chains

In continuous time a Markov chain waits an exponential time in each state and then jumps; it is specified by a
generator Q of rates. This notebook computes the transition function P(t) = e^{Qt} (and why uniformisation is a
safer way), stationary laws and absorption times, estimates a generator from a machine's two-year event log,
and simulates chemical reaction networks and epidemics with Gillespie's algorithm.
"""),
    setup_cell(["from scipy import linalg as sla", "from engstoch import ctmc, models, datasets"]),
    md("## Generators and the transition function"),
    code(r"""m = models.machine()
c, Q = m["chain"], m["Q"]
print("generator Q (per hour), states", c.states, "\n", Q)
print("holding rates q_i:", c.rates, " -> mean sojourns (h):", np.round(1 / c.rates, 2))
print("jump chain:\n", np.round(c.jump_chain().P, 3))
for t in (1.0, 10.0, 100.0):
    P = c.transition_matrix(t); U = c.uniformisation(t)
    print(f"P({t:5.0f} h) from 'up': {np.round(P[0], 4).tolist()}  | uniformisation: {U['terms']} terms, error bound {U['bound']:.0e}, diff vs scipy {np.max(np.abs(P - sla.expm(Q * t))):.0e}")
print("stationary law:", c.stationary().round(4).tolist(), "= jump-chain law weighted by mean holding times:", c.stationary_via_jump_chain().round(4).tolist())"""),
    md(r"""
P(t) solves Kolmogorov's equations P′ = QP = PQ, so P(t) = e^{Qt}. The matrix exponential is computed by
scaling and squaring with a Padé approximant; uniformisation writes P(t) as a Poisson mixture of powers of the
stochastic matrix I + Q/Λ - all terms non-negative, so there is no cancellation and the truncation error is
known in advance. The stationary law weights the jump chain's visit frequencies by the mean holding times.

## Estimating Q from an event log
"""),
    code(r"""log = pd.read_csv(datasets.path("machine_log.csv"), comment="#")
code_ = {s: i for i, s in enumerate(c.states)}
est = ctmc.estimate_generator(log.time_h.to_numpy(), log.state.map(code_).to_numpy(), 17520.0, 3)
print(f"{len(log)} events over two years; occupation hours:", est["occupation_time"].round(0).tolist())
print("estimated Q:\n", np.round(est["Q"], 4), "\nstandard errors:\n", np.round(est["std_error"], 4), "\ntrue Q:\n", Q)
soj = np.diff(np.r_[log.time_h.to_numpy(), 17520.0])
for s_ in c.states:
    x = soj[log.state.to_numpy() == s_][:-1] if s_ == "up" else soj[log.state.to_numpy() == s_]
    print(f"{s_:<9}: {len(x)} sojourns, mean {x.mean():6.2f} h, cv {x.std() / x.mean():.3f} (exponential: 1)")
print(f"\navailability: fraction of time not down {1 - est['occupation_time'][2] / 17520:.4f}; model {1 - c.stationary()[2]:.4f}")"""),
    md(r"""
With a complete path the likelihood factorises and the MLE of each rate is the number of i → j transitions
divided by the total time spent in i - including the final, censored sojourn, which contributes time but no
transition. Dropping it, or treating it as complete, biases the up-state rates (slightly, here, because it is
one of a thousand). Sojourns with a coefficient of variation near 1 support the exponential assumption.

## Birth-death chains and absorption
"""),
    code(r"""K, lam, mu = 30, 0.8, 1.0
bd = ctmc.birth_death([lam] * K, [mu] * K)
rho = lam / mu
print("M/M/1/30 stationary law vs (1 - rho) rho^n / (1 - rho^31):", np.max(np.abs(bd["stationary"] - (1 - rho) * rho ** np.arange(K + 1) / (1 - rho ** (K + 1)))))
pop = ctmc.CTMC(np.array([[0, 0, 0, 0, 0], [1.0, -2.5, 1.5, 0, 0], [0, 2.0, -5.0, 3.0, 0], [0, 0, 3.0, -7.5, 4.5], [0, 0, 0, 4.0, -4.0]]))
ab = pop.absorption_times()
print("population with births 1.5 n and deaths n (cap 4): expected time to extinction from n = 1..4:", ab["expected_time"].round(3).tolist())
sims = []
for k in range(3000):
    p = pop.simulate(1e6, 1, R(k)); sims.append(p["times"][-1])
print(f"simulated from n = 1: {np.mean(sims):.3f} +- {np.std(sims) / np.sqrt(3000):.3f}")"""),
    md(r"""
Birth-death chains satisfy detailed balance, so the stationary law is a product of rate ratios; the M/M/c/K
queues of notebook 12 are the special case. Expected absorption times solve Q_TT m = −1, the continuous-time
counterpart of the fundamental matrix.

## Gillespie's algorithm: epidemics and gene expression
"""),
    code(r"""s = models.sir(N=500, beta=0.3, gamma=0.1, I0=3)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
finals = []
for k in range(200):
    r = ctmc.gillespie(s["x0"], s["stoich"], s["propensity"], 300.0, R(k))
    finals.append(r["x"][-1, 2])
    if k < 25:
        ax[0].step(r["t"], r["x"][:, 1], where="post", lw=0.8, alpha=0.7)
ax[0].set_title("infected, 25 stochastic SIR runs (R0 = 3)"); ax[0].set_xlabel("days")
ax[1].hist(np.array(finals) / 500, bins=40); ax[1].set_title("final size: minor outbreaks or a major one"); plt.show()
q = (1 / 3) ** 3
print(f"fraction of runs with a minor outbreak (< 10 % infected): {np.mean(np.array(finals) < 50):.3f}; branching approximation (1/R0)^I0 = {q:.3f}")"""),
    md(r"""
Gillespie's direct method simulates a reaction network exactly: the time to the next event is exponential with
the total propensity, and the event is chosen in proportion to the individual propensities. The stochastic SIR
epidemic shows what the differential equations cannot: with three initial cases and R₀ = 3 about 4 % of
outbreaks die out early - the extinction probability of the branching process of notebook 8 - and the rest
infect ~94 % of the population.
"""),
    code(r"""ge = models.gene_expression()
r = ctmc.gillespie(ge["x0"], ge["stoich"], ge["propensity"], 1500.0, R(9))
grid = np.linspace(100, 1500, 5000)
prot = ctmc.sample_path_at(r["t"], r["x"][:, 1], grid)
tl = ctmc.tau_leap(ge["x0"], ge["stoich"], ge["propensity"], 1500.0, 0.05, R(10))
print(f"exact SSA: {len(r['t'])} events; protein mean {prot.mean():.0f} (theory {ge['mean_protein']:.0f}), Fano factor {prot.var() / prot.mean():.2f} (theory {ge['fano_protein']:.2f})")
print(f"tau-leaping (tau = 0.05): {len(tl['t'])} steps; protein mean {tl['x'][2000:, 1].mean():.0f}, Fano {tl['x'][2000:, 1].var() / tl['x'][2000:, 1].mean():.2f}")
plt.plot(grid, prot, lw=0.7); plt.title("protein copy number: bursts from each mRNA"); plt.xlabel("time"); plt.show()"""),
    md(r"""
Two-stage gene expression produces proteins in bursts (each short-lived mRNA is translated several times), so
the protein count is far noisier than Poisson: its Fano factor is 1 + k_p/(γ_m + γ_p) ≈ 5.5 (one path of 1400 time units spans only about 140 protein
lifetimes, so the estimated variance is uncertain by ±15 %). Tau-leaping fires
Poisson numbers of each reaction over fixed steps and reproduces the statistics with a fraction of the
events, at the price of a small bias and possible negative counts when populations are small.
"""),
]

NOTEBOOKS["12_queueing_theory"] = [
    md(r"""
# 12 · Queueing theory

Waiting lines are the purest application of the processes so far: Poisson arrivals, exponential or general
services, birth-death chains. This notebook derives the classical results - M/M/1, Erlang's B and C formulas,
M/M/c/K, the Pollaczek-Khinchine formula for general service times, Little's law - checks them by simulation,
and turns them into decisions: how many agents a call centre needs, and why variability costs as much as load.
"""),
    setup_cell(["from engstoch import queues as qu, output as oa, models, datasets"]),
    md("## M/M/1 and the price of high utilisation"),
    code(r"""rho = np.linspace(0.05, 0.97, 200)
plt.plot(rho, [qu.mm1(r, 1.0)["Wq"] for r in rho], label="M/M/1: rho/(1 - rho)")
plt.plot(rho, [qu.mg1(r, 1.0, 1.0)["Wq"] for r in rho], label="M/D/1: half of it")
plt.plot(rho, [qu.mg1(r, 1.0, 1 + 4.0)["Wq"] for r in rho], label="M/G/1, service cv 2: 2.5 x")
plt.ylim(0, 30); plt.xlabel("utilisation rho"); plt.ylabel("mean wait (in mean service times)"); plt.legend(); plt.title("Pollaczek-Khinchine: load and variability"); plt.show()
n = 300_000
w = qu.lindley(R(1).exponential(1 / 0.9, n), R(2).exponential(1.0, n))
bm_ = oa.batch_means(w[20_000:], 20)
print(f"M/M/1 at rho = 0.9 by the Lindley recursion: Wq = {bm_['estimate']:.2f} +- {bm_['std_error']:.2f} (theory {qu.mm1(0.9, 1.0)['Wq']:.0f})")"""),
    md(r"""
The mean wait in M/G/1 is ρ E[S](1 + c_s²) / (2(1 − ρ)): it explodes as the utilisation approaches one, and it
scales with (1 + c_s²)/2, so halving the variability of service times is worth as much as a big drop in load.
Even 300 000 simulated customers give only about ±7 % accuracy at ρ = 0.9 - waits are strongly
autocorrelated (notebook 14).

## Many servers: Erlang B and C
"""),
    code(r"""a = 20.0
rows = []
for c in (21, 22, 24, 26, 30):
    m = qu.mmc(a, 1.0, c)
    rows.append((c, a / c, qu.erlang_b(c, a), m["P_wait"], m["Wq"], qu.service_level(a, 1.0, c, 0.1)))
print("offered load 20 erlangs")
print(pd.DataFrame(rows, columns=["servers", "utilisation", "Erlang B (blocked)", "Erlang C P(wait)", "Wq", "P(wait <= 0.1)"]).to_string(index=False, float_format="%.4f"))
arr = np.cumsum(R(3).exponential(1 / 20.0, 400_000))
sim = qu.simulate_ggc(arr, R(4).exponential(1.0, 400_000), 22)
print(f"\nsimulated M/M/22: Wq {sim['wait'][20000:].mean():.4f}, P(wait > 0) {np.mean(sim['wait'][20000:] > 0):.4f}")"""),
    md(r"""
Erlang B gives the fraction of calls lost when there is no waiting room (telephone trunks, hospital beds
without a queue); Erlang C the probability of waiting when there is one. The computation uses the stable
recursion B(k) = aB(k − 1)/(k + aB(k − 1)) - factorials of hundreds of servers never appear.

## Economies of scale and square-root staffing
"""),
    code(r"""rows = []
for lam in (10, 100, 1000, 10_000):
    s = qu.staffing(lam, 1 / 4, 0.8, 20 / 60)                         # 4-min calls, 80 % answered within 20 s
    rows.append((lam, s["offered_load"], s["c"], s["offered_load"] / s["c"], s["beta"]))
print(pd.DataFrame(rows, columns=["calls / min", "offered load", "agents needed", "utilisation", "beta = (c - a)/sqrt(a)"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
Pooling is powerful: a small centre needs agents idle a third of the time to meet the service level, a large
one runs at 99 % utilisation. The safety staffing grows like √a (Halfin-Whitt) or slower - for a fixed answer
time the β needed even falls with the load. Splitting a large pool into independent small ones throws this
away.

## Little's law and finite capacity
"""),
    code(r"""m = qu.mmck(12.0, 1.0, 10, 15)
print(f"M/M/10/15 at 12 arrivals per unit time: blocking {m['P_block']:.4f}, throughput {m['throughput']:.3f}, L = {m['L']:.3f}, W = {m['W']:.3f}; L / (lambda_eff W) = {m['L'] / (m['throughput'] * m['W']):.6f}")
bd = models.mmck_chain(12.0, 1.0, 10, 15)
print("same law from the birth-death generator:", np.max(np.abs(bd["chain"].stationary() - m["p"])))
grid = np.linspace(arr[20_000], arr[200_000], 100_001)
Lsim = qu.number_in_system(arr[:220_000], sim["depart"][:220_000], grid).mean()
inside = (arr >= grid[0]) & (arr <= grid[-1])
lam_obs, W_obs = inside.sum() / (grid[-1] - grid[0]), np.mean((sim["depart"] - arr)[inside])
print(f"simulated M/M/22: time-average L = {Lsim:.3f}; observed lambda x observed W = {lam_obs * W_obs:.3f}")"""),
    md(r"""
L = λW holds for any stable system in which customers are conserved, with no distributional assumptions; with
blocking, λ must be the rate of *admitted* customers. It converts a time average (L) into a customer average
(W) and is the most useful consistency check of any queueing simulation.

## Real service times: the call centre's handle times
"""),
    code(r"""cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
h = cc.handle_min[cc.handle_min > 0].to_numpy()
cs2 = h.var() / h.mean() ** 2
print(f"handle times: mean {h.mean():.2f} min, cv {np.sqrt(cs2):.2f}, {np.sum(cc.handle_min == 0)} zero-length records removed")
lam = 150 / 60
for c in (11, 12, 13):
    wq_m = qu.mmc(lam, 1 / h.mean(), c)["Wq"] * 60
    wq_ac = qu.allen_cunneen(lam, h.mean(), c, 1.0, cs2) * 60
    arr = np.cumsum(R(5).exponential(1 / lam, 200_000))
    sim = qu.simulate_ggc(arr, R(6).choice(h, 200_000), c)["wait"][10_000:].mean() * 60
    print(f"{c} agents at 150 calls/h: Wq exponential model {wq_m:5.1f} s, Allen-Cunneen {wq_ac:5.1f} s, simulation with the measured handle times {sim:5.1f} s")"""),
    md(r"""
The measured handle times are lognormal with cv ≈ 0.8, less variable than the exponential that Erlang C
assumes, so Erlang C overstates the waits - here by about 15-20 %. The Allen-Cunneen correction (multiply by
(c_a² + c_s²)/2) is close; resampling the real handle times in a simulation is the reference. Which model to
trust is settled in the project, where the arrival rate also varies through the day.
"""),
]

NOTEBOOKS["13_discrete_event_simulation"] = [
    md(r"""
# 13 · Discrete-event simulation

When closed forms run out - time-varying arrivals, abandonment, priorities, finite populations, inventory
policies - we simulate. A discrete-event simulation jumps from event to event, keeps a future-event list, and
updates statistics as it goes. This notebook builds models on a 60-line engine: a multi-server queue with
impatient customers, a machine-repair shop, and an (s, S) inventory system whose policy we optimise.
"""),
    setup_cell(["from engstoch import des, queues as qu, ctmc, montecarlo as mc"]),
    md("## The engine: clock, event list, statistics"),
    code(r"""show_source(des.Simulator)"""),
    code(r"""sim = des.Simulator()
trace = []
def customer(k):
    trace.append((round(sim.now, 3), f"customer {k} arrives"))
    sim.after(1.5, lambda: trace.append((round(sim.now, 3), f"customer {k} leaves")))
for k, t in enumerate([0.4, 1.0, 1.2]):
    sim.schedule(t, customer, k)
sim.run()
for row in trace:
    print(row)"""),
    md(r"""
The event list is a binary heap ordered by time (with an insertion counter so that simultaneous events are
processed first-in, first-out - otherwise results would depend on the heap's internals). An event is any
callable; executing it may schedule further events. Time-weighted statistics (queue length, busy servers)
integrate a piecewise-constant value between events; tallies average per-customer observations.

## A queue with abandonment
"""),
    code(r"""lam, mu, theta = 4.0, 1.0, 0.5
rows = []
for c in (3, 4, 5, 6):
    r = des.call_centre(lambda t: lam, lam, lambda g: g.exponential(1 / mu), c, 20_000.0, R(c), patience_sampler=lambda g: g.exponential(1 / theta), answer_within=0.25, warmup=200.0)
    pi = ctmc.birth_death([lam] * 80, [min(k, c) * mu + max(k - c, 0) * theta for k in range(1, 81)])["stationary"]
    exact = float(np.sum(pi * np.maximum(np.arange(81) - c, 0) * theta) / lam)
    rows.append((c, r["abandon_rate"], exact, r["service_level"], r["mean_busy"] / c))
print(pd.DataFrame(rows, columns=["agents", "abandonment (simulated)", "abandonment (Erlang-A chain)", "answered within 0.25", "occupancy"]).to_string(index=False, float_format="%.4f"))"""),
    md(r"""
Customers who wait longer than an exponential patience hang up. With exponential everything the system is a
birth-death chain whose death rate in state n is min(n, c)μ + (n − c)⁺θ (the Erlang-A model, standard in call
centres), and the event simulation reproduces it. Abandonment stabilises an overloaded system - with 3 agents
and 4 erlangs of load the queue does not explode, a third of the callers simply leave.

## Machine repair: a finite source
"""),
    code(r"""rows = []
for repairers in (1, 2, 3):
    for dist, sampler in (("exponential", lambda g: g.exponential(2.0)), ("deterministic", lambda g: 2.0), ("lognormal cv 2", lambda g: g.lognormal(np.log(2.0) - 0.5 * np.log(5), np.sqrt(np.log(5))))):
        r = des.machine_repair(10, repairers, 0.05, sampler, 100_000.0, R(repairers * 10 + len(dist)))
        rows.append((repairers, dist, r["mean_down"], r["availability"]))
bd = ctmc.birth_death([0.05 * (10 - k) for k in range(10)], [min(k, 1) * 0.5 for k in range(1, 11)])
print(pd.DataFrame(rows, columns=["repairers", "repair time", "mean machines down", "availability"]).to_string(index=False, float_format="%.4f"))
print(f"exponential, 1 repairer, from the birth-death chain: mean down {np.dot(np.arange(11), bd['stationary']):.4f}")"""),
    md(r"""
Ten machines share the repair crew: when many are down, fewer can fail, so the arrival rate to the queue
depends on its length. With a single repairer the distribution of repair times matters (variable repairs
cause long backlogs); as the crew grows, availability approaches the insensitive alternating-renewal value
E[U]/(E[U] + E[D]) = 0.909.

## An inventory policy, optimised by simulation
"""),
    code(r"""def cost(s, S, seed):
    return des.inventory_sS(s, S, 2.0, lambda g: 1 + g.poisson(1.0), lambda g: g.uniform(1.0, 3.0), 2000.0, R(seed), holding=1.0, shortage=10.0, order_cost=50.0)["cost_rate"]
grid = [(s, S) for s in range(2, 16, 2) for S in range(s + 10, s + 50, 5)]
est = {k: np.mean([cost(*k, seed) for seed in range(5)]) for k in grid}
best = min(est, key=est.get)
print(f"best (s, S) on the grid with 5 replications each: {best}, cost {est[best]:.2f}")
diff = [cost(*best, 100 + k) - cost(12, 25, 100 + k) for k in range(20)]
r = mc.mean_ci(diff, method="t")
print(f"vs (12, 25) with common random numbers: difference {r['estimate']:.2f}, 95 % CI ({r['ci'][0]:.2f}, {r['ci'][1]:.2f})")
S_ = np.array(sorted({k[1] - k[0] for k in grid})); s_ = np.array(sorted({k[0] for k in grid}))
Z = np.array([[est.get((s, s + d), np.nan) for d in S_] for s in s_])
plt.contourf(S_, s_, Z, 20); plt.colorbar(label="cost per unit time"); plt.xlabel("S - s (order size)"); plt.ylabel("reorder point s"); plt.plot(best[1] - best[0], best[0], "w*", ms=14); plt.show()"""),
    md(r"""
The reorder point s buys protection against demand during the lead time; the order quantity S − s trades
ordering cost against holding cost (the economic order quantity in disguise). Simulation optimisation needs
care: the minimum over a noisy grid is biased low (the "winner's curse"), so the chosen policy is re-evaluated
with fresh, common random numbers against a competitor before any claim is made.
"""),
]

NOTEBOOKS["14_output_analysis"] = [
    md(r"""
# 14 · Output analysis

A simulation is a statistical experiment, and its outputs - waits of successive customers, daily costs, MCMC
draws - are usually autocorrelated and start in an unrepresentative state. Treating them as i.i.d. gives
intervals that are far too narrow. This notebook quantifies autocorrelation (the integrated autocorrelation
time), builds honest intervals (batch means, replications, the regenerative method), and deals with the
initial transient (Welch's plot, MSER).
"""),
    setup_cell(["from engstoch import output as oa, queues as qu, montecarlo as mc"]),
    md("## Autocorrelated output"),
    code(r"""n = 200_000
w = qu.lindley(R(1).exponential(1 / 0.8, n), R(2).exponential(1.0, n))
acf = oa.autocorrelation(w[10_000:], 400)
plt.plot(acf); plt.xlabel("lag (customers)"); plt.title("waiting times in M/M/1 at rho = 0.8: autocorrelation"); plt.show()
it = oa.iat(w[10_000:])
naive = mc.mean_ci(w[10_000:])
print(f"integrated autocorrelation time tau = {it['tau']:.0f} (window {it['window']}): {n - 10_000} waits carry the information of {it['ess']:.0f} independent ones")
print(f"naive i.i.d. standard error {naive['std_error']:.4f}; corrected sqrt(tau) x = {naive['std_error'] * np.sqrt(it['tau']):.4f}")"""),
    md(r"""
Successive waits share most of their history, so they are strongly correlated: the variance of their mean is
τσ²/n with τ = 1 + 2Σρ(k), here about 90. The naive interval is ten times too narrow. Sokal's automatic
window - sum the autocorrelations up to the smallest M with M ≥ 5τ(M) - balances the bias of truncation against
the noise of distant lags.

## Batch means
"""),
    code(r"""rows = []
for nb in (5, 10, 20, 40, 100, 400):
    b = oa.batch_means(w[10_000:], nb)
    rows.append((nb, b["batch_size"], b["estimate"], b["std_error"], b["lag1_corr"]))
print(pd.DataFrame(rows, columns=["batches", "batch size", "mean wait", "std error", "lag-1 corr of batch means"]).to_string(index=False, float_format="%.4f"))
print("truth Wq = 4.0")
cov = []
for k in range(300):
    x = qu.lindley(R(1000 + k).exponential(1 / 0.8, 20_000), R(2000 + k).exponential(1.0, 20_000))[2000:]
    cov.append((oa.batch_means(x, 20)["ci"], mc.mean_ci(x)["ci"]))
print(f"coverage of 95 % intervals over 300 runs: batch means {oa.coverage([c[0][0] for c in cov], [c[0][1] for c in cov], 4.0)['coverage']:.3f} | naive {oa.coverage([c[1][0] for c in cov], [c[1][1] for c in cov], 4.0)['coverage']:.3f}")"""),
    md(r"""
Batch means of long enough batches are nearly independent; their t interval is valid. Too many short batches
leave correlation between them (visible in the lag-1 correlation) and understate the error. Even batch means
undercover at short run lengths in heavy traffic - 18 000 customers at ρ = 0.8 is not much - but they are
vastly better than the naive interval.

## The initial transient
"""),
    code(r"""reps = np.array([qu.lindley(R(3000 + k).exponential(1 / 0.9, 3000), R(4000 + k).exponential(1.0, 3000) * 1.0) for k in range(200)])
wma = oa.welch_moving_average(reps, 50)
plt.plot(reps.mean(axis=0), alpha=0.3, label="average over 200 replications"); plt.plot(wma, label="Welch moving average (window 50)"); plt.axhline(9.0, color="k", ls="--", label="steady state Wq = 9")
cut = oa.mser(reps.mean(axis=0))["truncate"]
plt.axvline(cut, color="C3", label=f"MSER-5 truncation at {cut}"); plt.legend(); plt.xlabel("customer number"); plt.show()
print(f"mean of the first 500 waits {reps[:, :500].mean():.2f}, of the rest {reps[:, 500:].mean():.2f}")"""),
    md(r"""
Starting empty, the queue needs many customers to reach steady state - more as ρ grows (the relaxation time
of M/M/1 scales like (1 − ρ)⁻²). Welch's procedure averages replications and smooths them so the eye can pick
the warm-up; MSER-5 automates the choice by minimising the estimated standard error of what remains. Deleting
too little biases the estimate; deleting too much wastes data.

## Regeneration and replications
"""),
    code(r"""x = qu.lindley(R(5).exponential(1 / 0.8, 200_000), R(6).exponential(1.0, 200_000))
starts = np.flatnonzero(x == 0)
cyc_sum = np.add.reduceat(x, starts)[:-1]
cyc_len = np.diff(starts)
rg_ = oa.regenerative(cyc_sum, cyc_len)
print(f"regenerative method: {rg_['cycles']} cycles (customers finding the queue empty), Wq = {rg_['estimate']:.3f} +- {rg_['std_error']:.3f}")
reps = oa.replications([qu.lindley(R(7000 + k).exponential(1 / 0.8, 10_000), R(8000 + k).exponential(1.0, 10_000))[1000:].mean() for k in range(20)])
print(f"20 replications with deletion: Wq = {reps['estimate']:.3f} +- {reps['std_error']:.3f}")"""),
    md(r"""
Each time a customer finds the system empty the future is independent of the past: the cycles between such
regeneration points are i.i.d., and the long-run average is the ratio of the expected cycle sum to the
expected cycle length, with a delta-method interval - no warm-up and no batch size to choose. Independent
replications with deletion are the most transparent method and parallelise trivially; they pay for the
warm-up once per replication.
"""),
]

NOTEBOOKS["15_brownian_motion"] = [
    md(r"""
# 15 · Brownian motion

Brownian motion is the continuous limit of the random walk and the building block of continuous-time
stochastic models: diffusion, stock prices, measurement noise, the Ornstein-Uhlenbeck process. This notebook
constructs it (increments, Lévy's midpoint refinement, Donsker's scaled walk), studies its strange paths
(continuous, nowhere differentiable, with quadratic variation t), uses the reflection principle for maxima,
first passages and barrier options, and fits geometric Brownian motion to six years of stock prices.
"""),
    setup_cell(["from scipy import stats", "from engstoch import brownian as bm, walks as wk, datasets"]),
    md("## Constructions"),
    code(r"""fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
t = bm.time_grid(1.0, 1000)
W = bm.paths(1.0, 1000, 5, R(1))
for w in W: ax[0].plot(t, w, lw=0.8)
ax[0].set_title("independent N(0, dt) increments")
coarse = bm.bridge_construction(1.0, 2, 1, R(2))
for lev in range(7):
    ax[1].plot(np.linspace(0, 1, coarse.shape[1]), coarse[0], lw=0.6 + 0.2 * lev, alpha=0.4 + 0.08 * lev)
    coarse = bm.refine(coarse, 1.0, R(3 + lev))
ax[1].set_title("Levy: refining one path by midpoints")
d = bm.donsker(1000, 3, R(4))
for w in d: ax[2].plot(np.linspace(0, 1, 1001), w, lw=0.8)
ax[2].set_title("Donsker: S_[nt] / sqrt(n)"); plt.show()
Wm = bm.paths(1.0, 512, 20_000, R(5))
print("Cov(W_s, W_t) at s, t = 0.25, 0.5, 1:\n", np.round(np.cov(Wm[:, [128, 256, 512]].T), 3))"""),
    md(r"""
Gaussian increments on a grid are exact at the grid points. Lévy's construction fills in midpoints from the
conditional law of a Brownian bridge, (W_a + W_b)/2 + N(0, Δt/4), so a path can be refined where needed
without changing what was already simulated - the basis of the Brownian-bridge technique in quasi-Monte Carlo
and adaptive SDE solvers. Donsker's theorem says the rescaled random walk converges in law, as a whole path,
to Brownian motion; the covariance is min(s, t).

## Rough paths: quadratic variation
"""),
    code(r"""Wf = bm.paths(1.0, 2**18, 1, R(6))[0]
rows = []
for k in (4, 6, 8, 10, 12, 14, 16, 18):
    sub = Wf[:: 2 ** (18 - k)]
    rows.append((2**k, bm.quadratic_variation(sub)[0], bm.total_variation(sub)[0], np.sqrt(2 * 2**k / np.pi)))
print(pd.DataFrame(rows, columns=["intervals", "sum of squared increments", "sum of |increments|", "sqrt(2n/pi)"]).to_string(index=False, float_format="%.3f"))"""),
    md(r"""
Refining the partition, the sum of squared increments converges to t = 1 while the sum of absolute increments
grows without bound like √n: Brownian paths have infinite length and finite quadratic variation. This is why
ordinary calculus fails for them and Itô's (dW)² = dt rule (notebook 16) is needed.

## The reflection principle: maxima and first passages
"""),
    code(r"""T, a = 1.0, 1.0
W = bm.paths(T, 2000, 20_000, R(7))
print(f"P(max_[0,1] W >= 1): grid of 2000 steps {np.mean(W.max(axis=1) >= a):.4f}, reflection principle 2(1 - Phi(1)) = {bm.max_cdf(a, T):.4f}")
tt = np.linspace(0.02, 6, 300)
for mu in (0.0, 0.5):
    fp = bm.first_passage_simulated(1.0, 6.0, 600, 20_000, R(8), drift=mu, bridge_correction=True)
    plt.plot(tt, [np.mean(fp <= x) for x in tt], lw=2, alpha=0.5, label=f"simulated, drift {mu}")
    plt.plot(tt, bm.first_passage_cdf(tt, 1.0, mu), "k--", lw=1)
plt.legend(); plt.xlabel("t"); plt.ylabel("P(T_1 <= t)"); plt.title("first passage to level 1 (dashed: closed form)"); plt.show()
fpn = bm.first_passage_simulated(1.0, 2.0, 50, 40_000, R(9), drift=0.5)
fpc = bm.first_passage_simulated(1.0, 2.0, 50, 40_000, R(9), drift=0.5, bridge_correction=True)
print(f"P(T_1 <= 2) with 50 steps: discrete monitoring {np.mean(fpn <= 2):.4f}, with the bridge correction {np.mean(fpc <= 2):.4f}, exact {bm.first_passage_cdf(2.0, 1.0, 0.5):.4f}")"""),
    md(r"""
Reflecting the path after its first hit of a shows P(max_{s≤t} W_s ≥ a) = 2P(W_t ≥ a): the maximum has the law
of |W_t|. With drift the first-passage time is inverse Gaussian; without drift it is finite with probability
one yet has infinite mean. Checking the level only at grid points misses crossings between them and biases
the probability low; the Brownian-bridge correction (the probability that the bridge between two grid values
crosses a is exp(−2(a − x_k)(a − x_{k+1})/(σ²Δt))) removes the bias almost completely.

## Geometric Brownian motion on real prices
"""),
    code(r"""pr = pd.read_csv(datasets.path("prices.csv"), comment="#")
lr = np.diff(np.log(pr.close.to_numpy()))
mu_hat, sig_hat = lr.mean() * 252 + 0.5 * lr.var(ddof=1) * 252, lr.std(ddof=1) * np.sqrt(252)
print(f"whole record: volatility {sig_hat:.3f}, drift {mu_hat:.3f} +- {sig_hat / np.sqrt(len(lr) / 252):.3f} per year")
roll = pd.Series(lr).rolling(63).std() * np.sqrt(252)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(pr.close); ax[0].set_title("closing price"); ax[1].plot(roll); ax[1].set_title("rolling 3-month volatility"); plt.show()
for name, x in (("days 0-999", lr[:1000]), ("days 1000-1499", lr[1000:])):
    print(f"{name}: volatility {x.std(ddof=1) * np.sqrt(252):.3f}, normality of log-returns (KS p) {stats.kstest((x - x.mean()) / x.std(), 'norm').pvalue:.3f}")
print(f"whole record pooled: excess kurtosis {stats.kurtosis(lr):.2f} (0 for one GBM) -> a mixture of two volatilities looks heavy-tailed")"""),
    md(r"""
Under GBM log-returns are i.i.d. normal, so the volatility is estimated precisely from daily data while the
drift is barely identifiable (its standard error over six years is about 10 % per year - the "mean blur" of
finance). The rolling volatility reveals a regime change after day 1000; each regime is consistent with GBM,
but the pooled returns are fat-tailed. Heavy tails in real returns are partly this: volatility that moves.

## Pricing with Brownian motion
"""),
    code(r"""S0, K, r_, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
ST = bm.gbm(S0, r_, sig, T, 1, 400_000, R(10))[:, -1]
c_mc = np.exp(-r_ * T) * np.maximum(ST - K, 0)
print(f"European call: Monte Carlo {c_mc.mean():.4f} +- {c_mc.std() / np.sqrt(len(c_mc)):.4f}, Black-Scholes {bm.black_scholes_call(S0, K, r_, sig, T):.4f}")
B = 90.0
paths = bm.gbm(S0, r_, sig, T, 50, 100_000, R(11))
lw = np.log(paths); dt = T / 50
hit_grid = np.any(paths <= B, axis=1)
p_cross = np.exp(-2 * (lw[:, :-1] - np.log(B)) * (lw[:, 1:] - np.log(B)) / (sig**2 * dt))
hit_bridge = hit_grid | np.any(R(12).random(p_cross.shape) < np.where((paths[:, :-1] > B) & (paths[:, 1:] > B), p_cross, 0), axis=1)
pay = np.exp(-r_ * T) * np.maximum(paths[:, -1] - K, 0)
print(f"down-and-out call (B = 90), 50 monitoring dates: grid only {np.mean(pay * ~hit_grid):.4f}, bridge-corrected {np.mean(pay * ~hit_bridge):.4f}, continuous formula {bm.down_and_out_call(S0, K, B, r_, sig, T):.4f}")"""),
    md(r"""
Risk-neutral pricing is an expectation over GBM paths. For a barrier option the path matters: monitoring only
at 50 dates misses crossings and overprices the knock-out option; the bridge correction recovers the
continuously monitored price of Merton's reflection formula. The same correction appears in every barrier,
default-time and first-passage computation.
"""),
]

NOTEBOOKS["16_stochastic_differential_equations"] = [
    md(r"""
# 16 · Stochastic differential equations

An SDE dX = a(X) dt + b(X) dW adds Brownian noise to an ODE. Because Brownian paths are rough, the calculus is
new (Itô's lemma, the Itô-Stratonovich distinction) and so are the numerics: the Euler-Maruyama scheme
converges with strong order ½ (pathwise) and weak order 1 (in distribution), Milstein's correction restores
strong order 1, and multilevel Monte Carlo exploits both to cut the cost of expectations by orders of
magnitude.
"""),
    setup_cell(["from engstoch import sde, brownian as bm, montecarlo as mc"]),
    md("## Euler-Maruyama and an exact solution"),
    code(r"""mu, sig = 0.1, 0.5
a = lambda t, x: mu * x
b = lambda t, x: sig * x
fine = sde.euler_maruyama(a, b, 1.0, 1.0, 1024, 1, R(1))
WT = np.cumsum(fine["dW"][0, :, 0])
exact = np.exp((mu - sig**2 / 2) * fine["t"][1:] + sig * WT)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(fine["t"][1:], exact, "k", lw=1.5, label="exact exp((mu - s^2/2) t + s W_t)")
for n in (8, 32, 128):
    dWc = fine["dW"].reshape(1, n, 1024 // n, 1).sum(axis=2)
    e = sde.euler_maruyama(a, b, 1.0, 1.0, n, 1, R(0), dW=dWc)
    ax.plot(e["t"], e["X"][0], "o-", ms=3, lw=0.8, label=f"Euler, {n} steps")
ax.legend(fontsize=8); ax.set_title("the same Brownian path at three resolutions"); plt.show()"""),
    md(r"""
Itô's lemma solves geometric Brownian motion exactly: X_t = exp((μ − σ²/2)t + σW_t). The −σ²/2 is the Itô
correction - d(log X) picks up −½σ² dt from (dW)² = dt. To compare a scheme with the exact solution *pathwise*
the scheme must use the same Brownian path: the coarse increments are sums of the fine ones.

## Strong and weak convergence
"""),
    code(r"""steps = [8, 16, 32, 64, 128, 256, 512]
ex = lambda t, W: np.exp((mu - sig**2 / 2) * t + sig * W)
se = sde.strong_order(a, b, ex, 1.0, 1.0, steps, 4000, R(2))
sm = sde.strong_order(a, b, ex, 1.0, 1.0, steps, 4000, R(2), "milstein", lambda t, x: sig + 0 * x)
wo = sde.weak_order(a, b, 1.0, 1.0, [4, 8, 16, 32], 400_000, R(3), lambda x: x**2, np.exp((2 * mu + sig**2) * 1.0))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].loglog(se["h"], se["error"], "o-", label=f"Euler: slope {se['order']:.2f}"); ax[0].loglog(sm["h"], sm["error"], "s-", label=f"Milstein: slope {sm['order']:.2f}")
ax[0].set_xlabel("h"); ax[0].set_title("strong error E|X_N - X(T)|"); ax[0].legend()
ax[1].errorbar(wo["h"], wo["error"], 2 * wo["std_error"], fmt="o-", label=f"Euler weak error of E[X^2]: slope {wo['order']:.2f}"); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(); ax[1].set_xlabel("h"); plt.show()"""),
    md(r"""
Strong error measures how close individual paths are; weak error how close expectations are. Euler-Maruyama
has strong order ½ (it ignores the second-order term ½bb′((ΔW)² − Δt), of size Δt per step but random), and
weak order 1 (that term has mean zero). Milstein adds it and reaches strong order 1. For expectations - option
prices, mean queue lengths, failure probabilities - weak order is what matters, and the Monte Carlo error must
be well below the bias to see it, hence 400 000 paths.

## Itô or Stratonovich?
"""),
    code(r"""n, paths = 400, 4000
h = sde.heun_stratonovich(lambda t, x: 0 * x, lambda t, x: 0.4 * x, 1.0, 1.0, n, paths, R(4))
WT = h["dW"].sum(axis=1)[:, 0]
e = sde.euler_maruyama(lambda t, x: 0 * x, lambda t, x: 0.4 * x, 1.0, 1.0, n, paths, R(4), dW=h["dW"])
print(f"dX = 0.4 X dW integrated by Heun: mean |X - exp(0.4 W)| = {np.mean(np.abs(h['X'][:, -1] - np.exp(0.4 * WT))):.1e}  (Stratonovich solution)")
print(f"                         by Euler: mean |X - exp(0.4 W - 0.08)| = {np.mean(np.abs(e['X'][:, -1] - np.exp(0.4 * WT - 0.08))):.1e}  (Ito solution)")
print(f"E[X_1]: Heun {h['X'][:, -1].mean():.4f} (e^0.08 = {np.exp(0.08):.4f}), Euler {e['X'][:, -1].mean():.4f} (1)")"""),
    md(r"""
The same equation means two different things. Read as Itô (integrand evaluated at the left end of each step,
as Euler does), the solution exp(0.4W − 0.08) is a martingale; read as Stratonovich (midpoint, as Heun's
trapezoidal rule does), it is exp(0.4W) and grows on average. Stratonovich obeys the ordinary chain rule and
arises as the limit of smooth noise (physics, engineering); Itô is non-anticipating and natural in finance and
biology. The drift correction ½bb′ converts one into the other.

## Mean reversion: Ornstein-Uhlenbeck and CIR
"""),
    code(r"""fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ou = bm.ou_exact(0.08, 2.0, 0.03, 0.02, 5.0, 500, 2000, R(5))
ax[0].plot(np.linspace(0, 5, 501), ou[:5].T, lw=0.8); ax[0].axhline(0, color="k", lw=0.5); ax[0].set_title(f"Vasicek (OU) rates: P(r_5 < 0) = {np.mean(ou[:, -1] < 0):.3f}")
cir = sde.cir_full_truncation(0.08, 2.0, 0.03, 0.15, 5.0, 500, 2000, R(6))
ax[1].plot(np.linspace(0, 5, 501), cir[:5].T, lw=0.8); ax[1].set_title("CIR rates: never negative"); plt.show()
m = bm.ou_moments(0.08, 2.0, 0.03, 0.02, 5.0)
print(f"OU exact sampling at t = 5: mean {ou[:, -1].mean():.4f} (theory {float(m['mean']):.4f}), sd {ou[:, -1].std():.4f} (theory {np.sqrt(float(m['var'])):.4f})")
print(f"CIR at t = 5: mean {cir[:, -1].mean():.4f} (theory {float(sde.cir_mean(0.08, 2.0, 0.03, 5.0)):.4f}); Feller condition 2 kappa theta >= sigma^2: {2 * 2.0 * 0.03:.3f} vs {0.15**2:.4f}")"""),
    md(r"""
The OU process is Gaussian and can be sampled exactly on any grid (an AR(1) recursion); as an interest-rate
model (Vasicek) it allows negative rates. CIR replaces the constant noise by σ√X, which vanishes at zero and
keeps the process non-negative - but naive Euler can step below zero and then take a square root of a
negative number; the full-truncation scheme uses X⁺ in the coefficients and is the standard fix.

## Multilevel Monte Carlo
"""),
    code(r"""payoff = lambda S: np.maximum(S - 100.0, 0.0)
bs = bm.black_scholes_call(100.0, 100.0, 0.05, 0.2, 1.0)
rows = []
for eps in (0.1, 0.05, 0.02):
    r = sde.mlmc(payoff, 100.0, 0.05, 0.2, 1.0, eps, R(int(1 / eps)), L=6)
    std_cost = 2 * r["variances"][0] / eps**2 * 2**6
    rows.append((eps, r["estimate"], r["estimate"] - bs, r["std_error"], r["cost"], std_cost, std_cost / r["cost"]))
print(pd.DataFrame(rows, columns=["target eps", "MLMC price", "error vs Black-Scholes", "std error", "MLMC cost (steps)", "standard MC cost", "saving"]).to_string(index=False, float_format="%.4g"))
print("samples per level at eps = 0.02:", r["samples"].tolist(), "\nvariance of the level corrections:", np.round(r["variances"], 4).tolist())"""),
    md(r"""
Giles's idea: write E[P_L] as E[P_0] + Σ E[P_l − P_{l−1}] and estimate each term separately, with coarse and
fine paths driven by the same Brownian increments. The corrections have tiny variance (it decays like the
step size), so few expensive fine-level samples are needed; most samples are cheap coarse ones. The cost
falls from O(ε⁻³) to O(ε⁻²(log ε)²) - a factor of about fifteen here with the number of levels fixed at six;
when the finest level is chosen to match ε (as it should be, to keep the bias below ε), the saving grows as ε
shrinks.
"""),
]

NOTEBOOKS["17_gaussian_processes_and_time_series"] = [
    md(r"""
# 17 · Gaussian processes and stationary time series

A Gaussian process is specified by its mean and covariance function; the covariance decides how smooth the
paths are, how far correlation reaches, and how fast we can simulate. This notebook samples processes by
Cholesky factorisation and by circulant embedding with the FFT, simulates fractional Brownian motion with
long memory, conditions a GP on data (kriging), and then turns to discrete time: ARMA models, their
autocorrelations and spectra, and how to identify and check them from data.
"""),
    setup_cell(["import time", "from engstoch import gaussian as gp, timeseries as ts"]),
    md("## Covariance functions and sample paths"),
    code(r"""t = np.linspace(0, 5, 400)
fig, ax = plt.subplots(1, 4, figsize=(14, 3))
for axi, (name, k, kw) in zip(ax, (("exponential (OU)", gp.exponential, {"ell": 1.0}), ("Matern 3/2", gp.matern32, {"ell": 1.0}), ("squared exponential", gp.squared_exponential, {"ell": 1.0}), ("periodic", gp.periodic, {"ell": 1.0, "period": 2.0}))):
    s = gp.sample_cholesky(k, t, 3, R(1), **kw)
    axi.plot(t, s["X"].T, lw=1); axi.set_title(f"{name}\ncond(K) = {s['cond']:.1e}, jitter {s['jitter']:.0e}", fontsize=9)
plt.show()"""),
    md(r"""
The behaviour of the covariance at the origin sets the roughness: the exponential kernel's kink gives
continuous, nowhere differentiable paths (it is the OU process); Matérn 3/2 is once differentiable; the
squared exponential is infinitely smooth - and its covariance matrices are so ill-conditioned (eigenvalues
decaying faster than exponentially) that Cholesky needs a small "jitter" on the diagonal to run at all.

## Fast simulation by circulant embedding
"""),
    code(r"""rows = []
for n in (250, 500, 1000, 2000):
    tt = np.arange(n) * 0.01
    t0 = time.perf_counter(); gp.sample_cholesky(gp.matern32, tt, 1, R(2), ell=1.0); tc = time.perf_counter() - t0
    acov = gp.matern32(0, np.arange(n) * 0.01, ell=1.0)
    try:
        t0 = time.perf_counter(); gp.circulant_embedding(acov, 2, R(3)); te = time.perf_counter() - t0; ok = "yes"
    except ValueError:
        te, ok = np.nan, "needs padding"
    rows.append((n, tc * 1000, te * 1000, ok))
print(pd.DataFrame(rows, columns=["grid points", "Cholesky (ms)", "circulant embedding (ms)", "minimal embedding valid"]).to_string(index=False, float_format="%.1f"))
x = gp.circulant_embedding(gp.exponential(0, np.arange(2048) * 0.01, ell=0.5), 2000, R(4))["X"]
print("exponential kernel, sample covariance at lags 0, 25, 50:", np.round(np.cov(x[:, [0, 25, 50]].T)[0], 3), " theory", np.round(gp.exponential(0, np.array([0, 25, 50]) * 0.01, ell=0.5), 3))"""),
    md(r"""
On a regular grid the covariance matrix of a stationary process is Toeplitz; embedding it in a circulant matrix
twice as large diagonalises it by the FFT, so exact samples cost O(n log n) instead of Cholesky's O(n³). The
embedding must be non-negative definite - always true for the exponential kernel, true for Matérn 3/2 once the
window is long compared with the correlation length, and false for smooth kernels on short windows, where one
pads the grid.

## Long memory: fractional Brownian motion
"""),
    code(r"""fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for H in (0.25, 0.5, 0.75):
    B = gp.fbm_davies_harte(2048, H, 1.0, 1, R(int(100 * H)))[0]
    ax[0].plot(np.linspace(0, 1, 2049), B, lw=0.8, label=f"H = {H}")
ax[0].legend(); ax[0].set_title("fractional Brownian motion")
for H in (0.3, 0.5, 0.7, 0.9):
    est = gp.hurst_aggregated_variance(np.diff(gp.fbm_davies_harte(2**15, H, 1.0, 1, R(7))[0]))
    ax[1].loglog(est["block_sizes"], est["variances"], "o-", ms=3, label=f"H = {H}: estimate {est['H']:.3f}")
ax[1].legend(fontsize=8); ax[1].set_xlabel("block size m"); ax[1].set_title("Var(block mean) ~ m^(2H-2)"); plt.show()"""),
    md(r"""
Fractional Brownian motion has Var(B_t) = t^{2H}; its increments (fractional Gaussian noise) are negatively
correlated for H < ½ and positively correlated with a non-summable, power-law autocovariance for H > ½ - long
memory, as in river flows (Hurst's original data), network traffic and volatility. The Davies-Harte method is
circulant embedding of the increments, always valid for fGn. The variance of block means decays like
m^{2H−2} rather than m⁻¹, which is exactly what makes long-memory data hard: averages converge slowly.

## Conditioning on data: Gaussian-process regression
"""),
    code(r"""rng = R(8)
x_obs = np.sort(rng.random(12) * 10)
f_true = lambda x: np.sin(x) + 0.3 * np.cos(3 * x)
y_obs = f_true(x_obs) + 0.1 * rng.standard_normal(12)
xs = np.linspace(0, 10, 300)
best = max(((ell, gp.condition(gp.squared_exponential, x_obs, y_obs, xs, noise_var=0.01, ell=ell)) for ell in (0.3, 0.6, 1.0, 2.0, 4.0)), key=lambda p: p[1]["log_marginal_likelihood"])
for ell in (0.3, 0.6, 1.0, 2.0, 4.0):
    print(f"ell = {ell}: log marginal likelihood {gp.condition(gp.squared_exponential, x_obs, y_obs, xs, noise_var=0.01, ell=ell)['log_marginal_likelihood']:.2f}")
post = best[1]
plt.plot(xs, f_true(xs), "k--", label="truth"); plt.plot(x_obs, y_obs, "ko", label="data")
plt.plot(xs, post["mean"], label=f"posterior mean (ell = {best[0]})"); plt.fill_between(xs, post["mean"] - 2 * post["std"], post["mean"] + 2 * post["std"], alpha=0.25, label="+- 2 sd")
plt.legend(fontsize=8); plt.show()"""),
    md(r"""
Conditioning a Gaussian vector on some of its coordinates is linear algebra: the posterior mean is a weighted
sum of the observations, the posterior variance shrinks near the data and returns to the prior far from it.
The marginal likelihood of the data - available in closed form - chooses the kernel's length scale; this is
kriging in geostatistics and GP regression in machine learning.

## ARMA models: identification, estimation and checking
"""),
    code(r"""phi, theta = [0.6, -0.3], [0.4]
x = ts.simulate_arma(phi, theta, 3000, R(9))
print(f"causal: {ts.is_causal(phi)}, invertible: {ts.is_invertible(theta)}")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
lags = np.arange(21)
ax[0].stem(lags, ts.sample_acf(x, 20)); ax[0].plot(lags, ts.arma_acf(phi, theta, 20), "r_", ms=12); ax[0].axhspan(-ts.acf_bands(3000), ts.acf_bands(3000), alpha=0.2); ax[0].set_title("ACF (red: theory)")
ax[1].stem(lags, ts.sample_pacf(x, 20)); ax[1].axhspan(-ts.acf_bands(3000), ts.acf_bands(3000), alpha=0.2); ax[1].set_title("PACF")
f = np.linspace(0, 0.5, 400)
ax[2].semilogy(ts.periodogram(x)["f"], ts.periodogram(x)["psd"], alpha=0.3, label="periodogram"); w = ts.welch(x, 256)
ax[2].semilogy(w["f"], w["psd"], label=f"Welch ({w['segments']} segments)"); ax[2].semilogy(f, ts.arma_spectrum(phi, theta, f), "k--", label="ARMA spectrum"); ax[2].legend(fontsize=7); plt.show()
sel = ts.select_ar_order(x, 10)
fit = ts.ar_least_squares(x[10 - sel["p"]:], sel["p"])
print(f"AIC picks an AR({sel['p']}) approximation; residual Ljung-Box (20 lags): p = {ts.ljung_box(fit['residuals'], 20, sel['p'])['p_value']:.3f}")
ar2 = ts.ar_least_squares(x, 2)
print(f"an AR(2) is too short: residual Ljung-Box p = {ts.ljung_box(ar2['residuals'], 20, 2)['p_value']:.2e}")"""),
    md(r"""
The ACF of an ARMA(2, 1) decays as a damped oscillation; the PACF of a pure AR(p) would cut off after lag p,
but the MA part makes it tail off too - so a finite AR needs several lags to approximate it, and AIC picks the
order. The residual check is the essential last step: an AR(2) leaves autocorrelation that the Ljung-Box test
detects at once. The raw periodogram is unbiased but its variance does not fall with n; Welch's averaging of
segment periodograms makes it consistent at the price of frequency resolution.
"""),
]

NOTEBOOKS["18_markov_chain_monte_carlo"] = [
    md(r"""
# 18 · Markov chain Monte Carlo

When we cannot sample a distribution π directly - a Bayesian posterior, a spin system, a configuration of
interacting particles - we can often build a Markov chain whose stationary law is π and run it. Detailed
balance (notebook 06) tells us how: the Metropolis-Hastings acceptance rule. This notebook tunes random-walk
Metropolis, meets the Gibbs sampler and its weakness, uses gradients in Hamiltonian Monte Carlo, simulates the
Ising model across its phase transition, and diagnoses chains with autocorrelation times and R-hat.
"""),
    setup_cell(["from scipy import stats", "from engstoch import mcmc, output as oa"]),
    md("## Random-walk Metropolis and its tuning"),
    code(r"""logp = lambda x: -0.5 * np.sum(x**2)
rows = []
for d in (1, 10, 50):
    for s in (0.1, 2.38, 10.0):
        r = mcmc.metropolis(logp, np.zeros(d), 20_000, s / np.sqrt(d), R(d))
        rows.append((d, f"{s}/sqrt(d)", r["acceptance"], oa.iat(r["chain"][2000:, 0])["tau"] if r["acceptance"] > 0 else np.inf))
print(pd.DataFrame(rows, columns=["dimension", "proposal sd", "acceptance", "IAT of x_1"]).to_string(index=False, float_format="%.3f"))
ad = mcmc.metropolis(logp, np.zeros(50), 40_000, 1.0, R(5), adapt_until=10_000)
print(f"\nadaptive scaling in d = 50: acceptance {ad['acceptance']:.3f}, final scale {ad['final_scale'][0]:.3f} (2.38/sqrt(50) = {2.38 / np.sqrt(50):.3f})")"""),
    md(r"""
Propose x′ = x + sZ and accept with probability min(1, π(x′)/π(x)): detailed balance holds for any s, but the
efficiency does not. Tiny steps are always accepted and go nowhere; huge steps are always rejected. For
Gaussian-like targets the optimal scale is 2.38/√d with acceptance 0.234 (Roberts, Gelman and Gilks), and the
integrated autocorrelation time still grows linearly with the dimension. Adapting the scale during burn-in
finds it automatically.

## A Bayesian posterior
"""),
    code(r"""rng = R(6)
x = rng.random(40) * 10
y = (rng.random(40) < 1 / (1 + np.exp(-(-3.0 + 0.8 * x)))).astype(float)
def log_post(b):
    eta = b[0] + b[1] * x
    return float(np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * np.sum(b**2) / 100)
chains = [mcmc.metropolis(log_post, start, 20_000, [0.6, 0.12], R(10 + k))["chain"] for k, start in enumerate([[-5, 2], [0, 0], [3, -1], [-1, 1.5]])]
post = np.vstack([c[5000:] for c in chains])
print("posterior means", post.mean(axis=0).round(3), " sds", post.std(axis=0).round(3), " correlation", np.corrcoef(post.T)[0, 1].round(3))
for j, name in enumerate(["intercept", "slope"]):
    rh = oa.gelman_rubin([c[10_000:, j] for c in chains])["R_hat"]
    print(f"{name}: R-hat {rh:.4f}, ESS {sum(oa.ess(c[5000:, j]) for c in chains):.0f} of {len(post)} draws, 95 % credible interval {np.quantile(post[:, j], [0.025, 0.975]).round(3)}")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for c in chains: ax[0].plot(c[:3000, 1], lw=0.6)
ax[0].set_title("four chains from dispersed starts (slope)"); ax[1].scatter(post[::20, 0], post[::20, 1], s=2); ax[1].set_xlabel("intercept"); ax[1].set_ylabel("slope"); plt.show()"""),
    md(r"""
For a logistic regression the posterior is known only up to its normalising constant - which is all Metropolis
needs. Several chains from dispersed starting points let R-hat compare between- and within-chain variance;
values near 1 and an adequate effective sample size are the minimum evidence of convergence. The strong
posterior correlation between intercept and slope is what slows the chain down.

## Gibbs sampling and its weakness
"""),
    code(r"""rows = []
for rho in (0.0, 0.5, 0.9, 0.99):
    g = mcmc.gibbs_bivariate_normal(rho, 50_000, R(7))
    rows.append((rho, oa.iat(g[:, 0])["tau"], (1 + rho**2) / (1 - rho**2), np.corrcoef(g.T)[0, 1]))
print(pd.DataFrame(rows, columns=["rho", "IAT of x (measured)", "(1 + rho^2)/(1 - rho^2)", "sample corr"]).to_string(index=False, float_format="%.3f"))
g = mcmc.gibbs_bivariate_normal(0.99, 60, R(8), x0=(-3.0, -3.0))
plt.plot(g[:, 0], g[:, 1], "o-", ms=3, lw=0.8); plt.title("Gibbs zig-zag along a narrow ridge (rho = 0.99)"); plt.gca().set_aspect("equal"); plt.show()"""),
    md(r"""
Gibbs sampling updates one coordinate at a time from its full conditional - no tuning, every move accepted.
But it can only move parallel to the axes, so along a narrow diagonal ridge it crawls: the chain of x is an
AR(1) with coefficient ρ², and the autocorrelation time explodes as ρ → 1. Reparametrising (centring, blocking
correlated coordinates) is the cure.

## Hamiltonian Monte Carlo
"""),
    code(r"""scales = np.array([1.0, 10.0])
lp = lambda z: -0.5 * np.sum((z / scales) ** 2)
glp = lambda z: -z / scales**2
for step, L in ((0.3, 10), (0.9, 20), (2.1, 20)):
    h = mcmc.hmc(lp, glp, np.zeros(2), 4000, step, L, R(9))
    tau = oa.iat(h["chain"][500:, 1])["tau"] if h["acceptance"] > 0.01 else np.inf
    print(f"HMC step {step}, {L} leapfrog steps: acceptance {h['acceptance']:.3f}, IAT of the wide coordinate {tau:.1f}, sd {h['chain'][500:, 1].std():.2f} (10)")
rw = mcmc.metropolis(lp, np.zeros(2), 40_000, 2.0, R(10))
print(f"random-walk Metropolis, same target: acceptance {rw['acceptance']:.3f}, IAT {oa.iat(rw['chain'][2000:, 1])['tau']:.1f} per (cheaper) step")
errs = [abs(mcmc.leapfrog_energy_error(lambda z: -z, lambda z: -0.5 * float(np.sum(z**2)), [1.0], [0.5], h_, int(round(2 / h_)))) for h_ in (0.2, 0.1, 0.05)]
print("leapfrog energy error for step 0.2, 0.1, 0.05:", np.round(errs, 6), "-> second order")"""),
    md(r"""
HMC treats −log π as a potential, draws a random momentum, and follows Hamilton's equations with the leapfrog
integrator for many steps before a single accept/reject; the integrator is symplectic and nearly conserves the
energy, so long, distant moves are accepted. An integrated autocorrelation time below 1 means
successive draws are *negatively* correlated - better than independent for estimating the mean. The step size
must suit the *narrowest* direction: above twice the smallest scale (here 2) the leapfrog is unstable, the
energy error explodes and nothing is accepted - the price of not preconditioning.

## The Ising model and a phase transition
"""),
    code(r"""betas = np.linspace(0.25, 0.65, 12)
mags, ens = [], []
for b in betas:
    r = mcmc.ising(24, b, 1200, R(int(1000 * b)), "heatbath")
    mags.append(np.abs(r["magnetisation"][200:]).mean()); ens.append(r["energy"][200:].mean())
bb = np.linspace(0.25, 0.65, 200)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(betas, mags, "o", label="heat bath, 24 x 24"); ax[0].plot(bb, [mcmc.onsager_magnetisation(b) for b in bb], "k", label="Onsager-Yang (infinite lattice)"); ax[0].axvline(mcmc.BETA_C, ls=":"); ax[0].legend(fontsize=8); ax[0].set_xlabel("beta"); ax[0].set_title("|magnetisation|")
ax[1].plot(betas, ens, "o"); ax[1].plot(bb, [mcmc.onsager_energy(b) for b in bb], "k"); ax[1].set_xlabel("beta"); ax[1].set_title("energy per site"); plt.show()
for b in (0.35, mcmc.BETA_C, 0.55):
    r = mcmc.ising(32, b, 2000, R(11), "metropolis", start="hot")
    print(f"beta = {b:.4f}: IAT of the magnetisation {oa.iat(r['magnetisation'][300:])['tau']:.0f} sweeps")"""),
    md(r"""
Below the critical inverse temperature β_c = ½ln(1 + √2) spins are disordered; above it they align. Local
Markov chains reproduce Onsager's exact energy and the Onsager-Yang magnetisation away from β_c (the finite
lattice rounds the transition), but near β_c the autocorrelation time explodes - critical slowing down - which
cluster algorithms (Swendsen-Wang, Wolff) were invented to defeat.
"""),
]

NOTEBOOKS["19_project_staffing_a_call_centre"] = [
    md(r"""
# 19 · Project: staffing a call centre from data

A service line must answer 80 % of calls within 20 seconds, every half hour of the day, with as few agents as
possible. The data are 20 weekdays of logged calls (`call_centre.csv`): arrival times and handle times, with an
outage and some dropped connections. The project combines the course: estimate a non-homogeneous arrival
rate (notebook 09), model the service times (01, 12), staff each half hour by Erlang C (12), test the
schedule in a discrete-event simulation with impatient callers and a rate that changes through the day (13),
analyse the output honestly (14), test the stationary model and its textbook correction, and report.
"""),
    setup_cell(["from scipy import stats", "from engstoch import poisson as pp, queues as qu, des, montecarlo as mc, models, datasets"]),
    md("## 1. The data and their faults"),
    code(r"""cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
print(f"{len(cc)} calls over {cc.day.nunique()} days; zero handle times: {(cc.handle_min == 0).sum()}")
cnt = cc.groupby("day").size()
hh = cc.assign(slot=(cc.arrival_h * 2).astype(int) / 2).groupby(["day", "slot"]).size().unstack(fill_value=0)
print("days with empty half hours during opening:", {d: [float(s) for s in hh.columns[(hh.loc[d] == 0).to_numpy()]] for d in hh.index if (hh.loc[d] == 0).any()})
calls = cc[cc.handle_min > 0].copy()
outage = (calls.day == 8) & (calls.arrival_h >= 13.0) & (calls.arrival_h < 14.5)
print("weekday means:", cnt.groupby(cc.groupby("day").weekday.first()).mean().round(0).to_dict())"""),
    md(r"""
Two faults need decisions before any modelling. Day 8 has three empty half hours in the early afternoon - a
logging outage, not an absence of demand - so those slots are excluded from day 8 (the day's other slots are
kept). Zero handle times are dropped connections: they are not service and are removed from the handle-time
data (they are 0.4 % of calls, so whether they still count as arrivals hardly matters). Mondays are busier and
are planned separately; the rest of this project staffs a Tuesday-Friday day.

## 2. The arrival rate
"""),
    code(r"""edges = np.arange(8.0, 20.01, 0.5)
mid = edges[:-1] + 0.25
week = calls[(calls.weekday != "Mon")]
counts = []
for d, g in week.groupby("day"):
    c = pp.counts_in_windows(g.arrival_h.to_numpy(), edges).astype(float)
    if d == 8:
        c[(edges[:-1] >= 13.0) & (edges[:-1] < 14.5)] = np.nan
    counts.append(c)
counts = np.array(counts)
n_days = np.sum(~np.isnan(counts), axis=0)
lam_hat = np.nansum(counts, axis=0) / n_days / 0.5
se = np.sqrt(np.nansum(counts, axis=0)) / n_days / 0.5
disp = [pp.dispersion_test(counts[~np.isnan(counts[:, j]), j])["p_value"] for j in range(len(mid))]
print(f"per-slot dispersion tests across the {counts.shape[0]} days: smallest p = {min(disp):.3f}; fraction below 0.05 = {np.mean(np.array(disp) < 0.05):.2f}")
plt.errorbar(mid, lam_hat, 2 * se, fmt="o", ms=3, label="estimate +- 2 s.e. (Tue-Fri)"); plt.plot(np.linspace(8, 20, 300), models.call_centre_rate(np.linspace(8, 20, 300)), "k--", lw=1, label="generating profile (unknown in practice)")
plt.xlabel("hour"); plt.ylabel("calls per hour"); plt.legend(); plt.show()"""),
    md(r"""
Averaging the counts of each half hour over the 16 Tuesday-Friday days gives the rate profile with Poisson
standard errors of 5-8 %; the counts of a slot vary between days no more than Poisson variation allows (the
dispersion tests are unremarkable), so a non-homogeneous Poisson process with this profile is a reasonable
model. The peak at 10:30 is three and a half times the evening rate.

## 3. Service times
"""),
    code(r"""h = calls.handle_min.to_numpy()
ln_mu, ln_sd = np.log(h).mean(), np.log(h).std(ddof=1)
print(f"handle times: mean {h.mean():.3f} min, sd {h.std():.3f}, cv {h.std() / h.mean():.3f}")
print(f"lognormal fit: KS p = {stats.kstest(np.log(h), 'norm', args=(ln_mu, ln_sd)).pvalue:.3f};  exponential fit: KS p = {stats.kstest(h, 'expon', args=(0, h.mean())).pvalue:.1e}")
by_hour = calls.assign(hour=calls.arrival_h.astype(int)).groupby("hour").handle_min.mean()
print("mean handle time by hour of arrival:", by_hour.round(2).to_dict())
mean_h = h.mean() / 60                                  # hours"""),
    md(r"""
Handle times are lognormal with mean 4 minutes and cv 0.8, not exponential, and do not drift through the day.
Erlang C assumes exponential service; with cv < 1 it should be slightly pessimistic (notebook 12) - but the
bigger question is whether a *stationary* model can describe a half hour in which the rate changes.

## 4. A first schedule: Erlang C slot by slot
"""),
    code(r"""target, answer = 0.8, 20 / 3600
c_sipp = np.array([qu.staffing(l, 1 / mean_h, target, answer)["c"] for l in lam_hat])
print("agents per half hour (stationary independent period-by-period, 'SIPP'):", c_sipp.tolist())
print(f"agent-hours per day: {0.5 * c_sipp.sum():.1f}")
def agents_fn(schedule):
    return lambda t: int(schedule[min(max(int((t - 8.0) / 0.5), 0), len(schedule) - 1)])
rate_fn = lambda t: float(np.interp(t, mid, lam_hat, left=0, right=0)) if 8.0 <= t < 20.0 else 0.0
rate_max = 1.05 * lam_hat.max()"""),
    md(r"""
Each half hour is treated as a stationary M/M/c queue with its own arrival rate and staffed with the smallest c
that meets the target - the usual practice. The rate function used in the simulation interpolates the
half-hourly estimates linearly, which is closer to how demand really moves than a step function.

## 5. Testing the schedule by simulation
"""),
    code(r"""def run_day(schedule, seed, patience=True):
    r = des.call_centre(rate_fn, rate_max, lambda g: float(np.exp(ln_mu + ln_sd * g.standard_normal())) / 60, agents_fn(schedule), 21.0, R(seed),
                        patience_sampler=(lambda g: g.exponential(3.0 / 60)) if patience else None, answer_within=answer)
    slot = np.clip(((r["arrival"] - 8.0) / 0.5).astype(int), 0, len(mid) - 1)
    ok = (~r["abandoned"]) & (np.nan_to_num(r["wait"], nan=1.0) <= answer)
    sl = np.array([ok[slot == j].mean() if np.any(slot == j) else np.nan for j in range(len(mid))])
    ab = np.array([r["abandoned"][slot == j].mean() if np.any(slot == j) else np.nan for j in range(len(mid))])
    return sl, ab, ok.mean(), r["abandoned"].mean()
def evaluate(schedule, reps=40, seed0=0):
    out = [run_day(schedule, seed0 + k) for k in range(reps)]
    sl = np.array([o[0] for o in out]); ab = np.array([o[1] for o in out])
    day = mc.mean_ci([o[2] for o in out], method="t")
    return {"slot_sl": np.nanmean(sl, axis=0), "slot_se": np.nanstd(sl, axis=0, ddof=1) / np.sqrt(reps), "slot_ab": np.nanmean(ab, axis=0), "day_sl": day, "day_ab": float(np.mean([o[3] for o in out]))}
ev_sipp = evaluate(c_sipp)
print(f"SIPP schedule, 40 simulated days: daily service level {ev_sipp['day_sl']['estimate']:.3f} +- {ev_sipp['day_sl']['std_error']:.3f}, abandonment {ev_sipp['day_ab']:.3f}")
miss = mid[ev_sipp["slot_sl"] + 2 * ev_sipp["slot_se"] < target]
print("half hours significantly below 80 %:", miss.tolist())
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.errorbar(mid, ev_sipp["slot_sl"], 2 * ev_sipp["slot_se"], fmt="o-", ms=3, label="SIPP"); ax.axhline(target, color="k", ls="--"); ax.set_ylabel("answered within 20 s")
ax2 = ax.twinx(); ax2.plot(mid, lam_hat, color="0.7"); ax2.set_ylabel("calls / hour", color="0.5"); ax.legend(); plt.show()"""),
    md(r"""
Forty simulated days with lognormal handle times, 3-minute mean patience and a continuously varying rate give
the service level of every half hour with an interval. The literature warns that SIPP fails on a ramp: the rate
rises *within* the half hour and work arriving late in one slot spills into the next. Here no slot misses -
Erlang C's pessimism (exponential service where the real cv is 0.8, and no abandonment where 3 % of callers
leave) more than pays for the lag. The schedule meets the target with room to spare in most slots, which
suggests it is too expensive rather than too thin.

## 6. Accounting for the lag: the offered-load (infinite-server) approximation
"""),
    code(r"""tgrid = np.linspace(8.0, 20.0, 2401)
lam_t = np.array([rate_fn(t) for t in tgrid])
dt = tgrid[1] - tgrid[0]
s_grid = np.arange(0, 2.0, dt)
surv = stats.lognorm.sf(s_grid * 60, ln_sd, scale=np.exp(ln_mu))      # P(S > s), s in hours
load = np.convolve(lam_t, surv)[: len(tgrid)] * dt                     # m(t) = int lambda(t - s) P(S > s) ds  (M_t/G/inf mean)
c_lag = []
for j, t0 in enumerate(edges[:-1]):
    w = (tgrid >= t0) & (tgrid < t0 + 0.5)
    a_eff = np.max(load[w])                                           # staff for the busiest moment of the slot
    c_lag.append(qu.staffing(a_eff / mean_h, 1 / mean_h, target, answer)["c"])
c_lag = np.array(c_lag)
print("lag-aware schedule:", c_lag.tolist(), f"\nagent-hours {0.5 * c_lag.sum():.1f} (SIPP {0.5 * c_sipp.sum():.1f})")
ev_lag = evaluate(c_lag, seed0=1000)
print(f"lag-aware schedule: daily service level {ev_lag['day_sl']['estimate']:.3f} +- {ev_lag['day_sl']['std_error']:.3f}; half hours significantly below 80 %:", mid[ev_lag["slot_sl"] + 2 * ev_lag["slot_se"] < target].tolist())"""),
    md(r"""
The number of calls *in progress* in an infinite-server system with time-varying arrivals is
m(t) = ∫ λ(t − s) P(S > s) ds - the arrival rate smoothed and delayed by the service-time distribution
(Eick, Massey and Whitt). Staffing each slot for its peak offered load m(t) builds in both the within-slot rise
and the spill-over. It is the safe textbook remedy - and here it buys a service level of 95 % for eight extra
agent-hours that the target does not require. Both stationary schedules are starting points; the simulation
has to decide how many agents are really needed.

## 7. Trimming: the cheapest schedule that meets the target everywhere
"""),
    code(r"""sched = c_lag.copy()
ev = ev_lag
for it in range(3):
    slack = ev["slot_sl"] - 2 * ev["slot_se"] - target
    trial = sched.copy()
    trial[(slack > 0.06) & (sched > 1)] -= 1                           # remove an agent where the margin is large
    ev_t = evaluate(trial, seed0=2000 + 100 * it)
    if np.all(ev_t["slot_sl"] + 2 * ev_t["slot_se"] >= target):
        sched, ev = trial, ev_t
print("trimmed schedule:", sched.tolist(), f"\nagent-hours {0.5 * sched.sum():.1f}")
final = evaluate(sched, reps=80, seed0=9000)                           # fresh seeds for the final assessment
print(f"final check on 80 fresh days: service level {final['day_sl']['estimate']:.3f} +- {final['day_sl']['std_error']:.3f}, worst half hour {np.min(final['slot_sl']):.3f}, abandonment {final['day_ab']:.3f}")
fig, ax = plt.subplots(figsize=(9, 3.6))
for name, e in (("SIPP", ev_sipp), ("lag-aware", ev_lag), ("final (fresh seeds)", final)):
    ax.plot(mid, e["slot_sl"], "o-", ms=3, label=name)
ax.axhline(target, color="k", ls="--"); ax.legend(); ax.set_xlabel("hour"); ax.set_ylabel("answered within 20 s"); plt.show()"""),
    md(r"""
Removing agents only where the simulated margin is large, and accepting a change only if every half hour still
meets the target, gives a schedule cheaper than both stationary ones - a few agent-hours below SIPP - with its
worst half hour right at the target. Because the trimming used the
simulation to *choose*, the final schedule is re-evaluated on fresh random numbers - the honest estimate of
its performance (notebook 13's winner's curse).

## 8. The report
"""),
    code(r"""report = f'''# Staffing a service line: from call logs to a half-hourly schedule

Data: 20 weekdays, {len(cc)} calls. Removed: {int((cc.handle_min == 0).sum())} zero-length (dropped) calls from the handle-time data;
      day 8 13:00-14:30 excluded from the rate estimate (logging outage). Mondays (+{100 * (cnt[cc.groupby('day').weekday.first() == 'Mon'].mean() / cnt[cc.groupby('day').weekday.first() != 'Mon'].drop(8).mean() - 1):.0f} %) need their own schedule.
Arrivals: non-homogeneous Poisson, half-hourly rates {lam_hat.min():.0f}-{lam_hat.max():.0f} calls/h (s.e. 5-8 %); counts consistent with Poisson variation.
Service: lognormal, mean {h.mean():.2f} min, cv {h.std() / h.mean():.2f}; no trend through the day. Patience assumed exponential, mean 3 min (not in the data).
Target: 80 % answered within 20 s in every half hour.
SIPP (Erlang C per slot): {0.5 * c_sipp.sum():.1f} agent-hours; day {ev_sipp['day_sl']['estimate']:.3f}, {len(miss)} half hours significantly below target (Erlang C's pessimism covers the ramp).
Lag-aware (peak offered load of the M_t/G/inf model per slot): {0.5 * c_lag.sum():.1f} agent-hours; day {ev_lag['day_sl']['estimate']:.3f} - safe but more than needed.
Recommended (lag-aware, trimmed by simulation): {0.5 * sched.sum():.1f} agent-hours; on 80 fresh simulated days the daily service level is
      {final['day_sl']['estimate']:.3f} +- {final['day_sl']['std_error']:.3f}, worst half hour {np.min(final['slot_sl']):.3f}, abandonment {100 * final['day_ab']:.1f} %.
Schedule (agents per half hour from 08:00): {sched.tolist()}
Caveats: the patience distribution is assumed; shift constraints (agents work blocks, not half hours) are not modelled; the rate estimate has
      5-8 % error per slot, so the schedule should be re-checked against a rate 10 % higher in the peak (a one-line change above).
'''
print(report)"""),
    md(r"""
The report follows the course's method: clean the data on documented grounds, model what the data support
(Poisson arrivals with a daily profile, lognormal service) and state what they do not (patience), use the
closed-form theory where it applies and simulation where it does not, put intervals on every simulated
number, test the standard practice and its textbook correction instead of assuming either, and assess the final
choice on fresh random numbers. Its central finding - that both stationary rules over-staff this line and a
simulation-trimmed schedule saves agent-hours - came from the simulation, not from a rule.
"""),
]

# =====================================================================================================
# Teaching layer
def _x(objectives, prereq, time, inside, exercises=(), implement=""):
    return dict(objectives=objectives, prereq=prereq, time=time, inside=inside, exercises=list(exercises), implement=implement)


EXTRAS["00_randomness_and_generators"] = _x(
    ["explain how linear congruential generators work and when they reach full period", "recognise lattice structure and the RANDU defect",
     "apply and interpret statistical tests of uniformity and independence", "seed and spawn independent streams reproducibly"],
    "basic probability (uniform distribution, independence), Python and NumPy", "60 min",
    [md("## Inside the algorithm: PCG32 in four lines"), code(r"""state, inc, M64 = 0, (54 << 1) | 1, (1 << 64) - 1
def step():
    global state
    old = state; state = (old * 6364136223846793005 + inc) & M64
    xs, rot = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF, old >> 59
    return ((xs >> rot) | (xs << ((-rot) & 31))) & 0xFFFFFFFF
step(); state = (state + 42) & M64; step()
print([hex(step()) for _ in range(3)], "vs library", [hex(v) for v in (lambda g: [g.next32() for _ in range(3)])(rg.PCG32(42, 54))])""")],
    ["Find all multipliers a in 1..63 for which x -> a x + 1 mod 64 has full period, and check the Hull-Dobell theorem against `LCG.period` for every a.",
     "Generate 10^6 triples from Park-Miller and from RANDU and count how many fall in the slab 0.45 < 9u_1 - 6u_2 + u_3 - round(9u_1 - 6u_2 + u_3) < 0.55. What fraction does a perfect generator give, and what do the two generators give?"],
    "**Implement it yourself:** write Knuth's gap test for the interval [0, 0.1) from scratch (gap lengths 0..9 and >= 10, expected counts from the geometric law, the chi-square p-value from `special.chi2_sf`) and check it against `rngtests.gap_test` on 10^5 PCG32 numbers.")

EXTRAS["01_generating_random_variates"] = _x(
    ["generate variates by inversion for continuous and discrete laws", "design an acceptance-rejection sampler and predict its efficiency",
     "use the specialised normal, gamma and alias methods", "generate dependent vectors with Cholesky factors and copulas"],
    "notebook 00; distribution and density functions", "75 min",
    [md("## Inside the algorithm: Vose's alias table by hand"), code(r"""p = np.array([0.5, 0.3, 0.15, 0.05]); n = len(p)
scaled = p * n; small = [i for i in range(n) if scaled[i] < 1]; large = [i for i in range(n) if scaled[i] >= 1]
prob, alias = np.ones(n), np.arange(n)
while small and large:
    s, g = small.pop(), large.pop()
    prob[s], alias[s] = scaled[s], g
    scaled[g] -= 1 - scaled[s]
    (small if scaled[g] < 1 else large).append(g)
lib_prob, lib_alias = rg.alias_table(p)
print("by hand:", prob.round(3), alias, "| library:", lib_prob.round(3), lib_alias)
print("pmf recovered:", [round(float((prob[j] + sum(1 - prob[i] for i in range(n) if alias[i] == j and i != j)) / n), 6) for j in range(n)])""")],
    ["Generate standard Cauchy variates by inversion and by the ratio of two independent normals; show with a KS test that both are Cauchy, and show that the running mean of 10^6 of them does not converge.",
     "Sample the half-normal density sqrt(2/pi) exp(-x^2/2) on x > 0 by rejection from an Exp(lambda) proposal. Find the lambda that minimises M and check the observed acceptance rate against 1/M."],
    "**Implement it yourself:** write the Box-Muller transform and the polar method from scratch, time both for 10^6 normals, and test both with `rngtests.ks_test` against `special.norm_cdf`.")

EXTRAS["02_monte_carlo_estimation"] = _x(
    ["write a Monte Carlo estimate with its standard error and confidence interval", "check the coverage of an interval procedure by simulation",
     "choose the sample size for a target precision", "explain why rare events and heavy tails break crude Monte Carlo"],
    "notebooks 00-01; the central limit theorem", "60 min",
    [md("## Inside the algorithm: Welford's one-pass variance"), code(r"""x = h(R(11).random(100_000)) + 1e9          # a large offset ruins the naive formula
n, mean, m2 = 0, 0.0, 0.0
for v in x[:20_000]:
    n += 1; d = v - mean; mean += d / n; m2 += d * (v - mean)
naive = (np.sum(x[:20_000] ** 2) - 20_000 * np.mean(x[:20_000]) ** 2) / 19_999
print(f"Welford variance {m2 / (n - 1):.6f}, two-pass {np.var(x[:20_000], ddof=1):.6f}, naive sum-of-squares {naive:.1f}")""")],
    ["Estimate pi by throwing points in the unit square, with a 99 % interval, and find the number of points needed for six correct digits. How long would that take at 10^8 points per second?",
     "For the lognormal(0, 1.5) mean, compare the coverage at n = 1000 of the t interval and of a percentile bootstrap interval (2000 resamples) over 500 replications. Does the bootstrap fix the problem?"],
    "**Implement it yourself:** write the Wilson score interval for a proportion and use it to show by simulation that, for p = 0.001 and n = 2000, the Wald interval's coverage is far below 95 % while Wilson's is close.")

EXTRAS["03_variance_reduction"] = _x(
    ["apply antithetic variates and predict when they help or hurt", "build control-variate estimators with estimated optimal coefficients",
     "design an importance-sampling proposal for a rare event", "use stratification and common random numbers"],
    "notebook 02", "90 min",
    [md("## Inside the algorithm: the optimal control-variate coefficient"), code(r"""Y, C = payoff, ST
beta = np.cov(Y, C)[0, 1] / np.var(C, ddof=1)
adj = Y - beta * (C - S0 * np.exp(r_ * T))
print(f"beta by hand {beta:.4f} (library {cv['beta']:.4f}); estimate {adj.mean():.4f}; variance ratio {np.var(Y) / np.var(adj):.2f} = 1/(1 - rho^2) = {1 / (1 - np.corrcoef(Y, C)[0, 1] ** 2):.2f}")""")],
    ["Price an Asian (arithmetic-average) call on GBM with 12 monitoring dates, using the geometric-average Asian call (closed form: a lognormal) as a control variate. Report the variance reduction.",
     "Estimate P(S_100 > 30) for a sum of 100 Exp(1) variables minus 100 (i.e. P(sum > 130)) by exponential tilting of each summand. Find the best tilt and compare with the gamma tail from SciPy."],
    "**Implement it yourself:** write stratified sampling with Neyman allocation for E[h(U)] from scratch (pilot run, allocation, estimate and standard error), and reproduce the library's result for h(u) = 10u on u > 0.9.")

EXTRAS["04_quasi_monte_carlo"] = _x(
    ["construct van der Corput, Halton and lattice point sets", "measure discrepancy and relate it to integration error",
     "compare QMC and MC convergence on smooth integrands", "obtain error estimates by randomised QMC"],
    "notebooks 02-03", "60 min",
    [md("## Inside the algorithm: Warnock's formula"), code(r"""x = qmc.sobol(64, 2)
n, d = x.shape
t1 = 3.0**-d; t2 = 2.0 ** (1 - d) / n * np.sum(np.prod(1 - x**2, axis=1))
t3 = np.sum(np.prod(1 - np.maximum(x[:, None, :], x[None, :, :]), axis=2)) / n**2
print(f"L2-star discrepancy by hand {np.sqrt(t1 - t2 + t3):.6f}; library {qmc.l2_star_discrepancy(x):.6f}")""")],
    ["Price a European call by QMC: map Sobol points through the normal quantile and compare the error with MC for n = 2^8..2^16. Then do the same for a digital option (payoff 1 if S_T > K). Is the gain smaller, as one might expect for a discontinuous payoff?",
     "For the integrand prod_j (1 + (x_j - 1/2)/j^2) in d = 20, compare MC and randomly shifted Sobol (use d <= 10 Sobol plus 10 Halton coordinates if you prefer). Explain the result with the idea of effective dimension."],
    "**Implement it yourself:** write the radical inverse function in any base with integer arithmetic and reproduce `qmc.halton(100, 5)` exactly.")

EXTRAS["05_discrete_time_markov_chains"] = _x(
    ["build transition matrices and compute n-step probabilities", "classify states into communicating classes, recurrent and transient, with periods",
     "simulate chains and relate time averages to the stationary law", "estimate a chain from data and test its assumptions"],
    "notebook 00; matrices and linear systems", "75 min",
    [md("## Inside the algorithm: the period by breadth-first search"), code(r"""P_ = models.ehrenfest(6)["chain"].P
level, queue, g = {0: 0}, [0], 0
while queue:
    i = queue.pop(0)
    for j in np.flatnonzero(P_[i] > 0):
        if j not in level: level[j] = level[i] + 1; queue.append(j)
        else: g = math.gcd(g, level[i] + 1 - level[j])
print("period of the Ehrenfest chain by BFS levels:", g)""")],
    ["A machine is good, worn or broken; write a 3-state chain with a repair policy of your choice, find its stationary law and the long-run fraction of broken days, and check by simulating 10^5 days.",
     "Fit a separate two-state chain to each season of `weather.csv`, then simulate ten years from the seasonal model and from the homogeneous model. Compare the distribution of the longest wet spell per year with the data."],
    "**Implement it yourself:** write the MLE of P with standard errors from a path of state indices, and check it against `markov.estimate` on the weather data.")

EXTRAS["06_long_run_behaviour_and_mixing"] = _x(
    ["compute stationary laws by solving, iterating and simulating", "compute mean return and first-passage times",
     "measure mixing by total variation and the spectral gap", "check reversibility and compute PageRank"],
    "notebook 05", "75 min",
    [md("## Inside the algorithm: mean first-passage times by a linear solve"), code(r"""Pm = inv["chain"].P; j = 0
others = [i for i in range(7) if i != j]
m = np.linalg.solve(np.eye(6) - Pm[np.ix_(others, others)], np.ones(6))      # m_i = 1 + sum_{k != j} P_ik m_k
print("by first-step analysis:", m.round(3), "\nfundamental matrix:  ", np.delete(M[:, 0], 0).round(3))""")],
    ["For the lazy random walk on a cycle of n vertices compute the mixing time t_mix(1/4) for n = 8..128 and show that it grows like n^2. Compare with the relaxation time 1/gap.",
     "Find the stationary law of the (s, S) inventory chain for every s in 0..4 and S in s+1..10 with holding cost 1 per unit per week, lost sale cost 20 and ordering cost 15. Which policy minimises the long-run weekly cost?"],
    "**Implement it yourself:** write power iteration for PageRank on a sparse link list (dict of out-links), with dangling pages and damping, and reproduce `markov.pagerank` on the web graph.")

EXTRAS["07_absorption_hitting_and_random_walks"] = _x(
    ["solve absorption problems with the fundamental matrix and first-step analysis", "apply the reflection principle and the ballot theorem",
     "explain the arcsine law", "state Polya's theorem and recognise slow convergence in simulations"],
    "notebooks 05-06", "75 min",
    [md("## Inside the algorithm: ruin probabilities by first-step analysis"), code(r"""N, p = 10, 0.47
A = np.zeros((N + 1, N + 1)); b = np.zeros(N + 1)
A[0, 0] = A[N, N] = 1; b[0] = 1                         # u_0 = 1 (ruined), u_N = 0
for i in range(1, N):
    A[i, i], A[i, i + 1], A[i, i - 1] = 1, -p, -(1 - p)
u = np.linalg.solve(A, b)
print("u_i = p u_{i+1} + q u_{i-1}:", u[1:N].round(4), "\nclosed form:            ", np.array([wk.ruin_probability(i, N, p) for i in range(1, N)]).round(4))""")],
    ["In snakes and ladders, which single snake would you remove to shorten the expected game the most? Compute the expected game length for each removal.",
     "Simulate 10^4 symmetric walks of 10^4 steps and plot the distribution of the time of the last visit to 0 divided by n. Compare with the arcsine law."],
    "**Implement it yourself:** count the n-step paths that stay strictly positive after step 0 and end at k by brute-force enumeration for n = 12, and check the ballot-theorem formula (k/n) C(n, (n+k)/2).")

EXTRAS["08_branching_processes"] = _x(
    ["compute means and variances of generation sizes", "find extinction probabilities as fixed points of the pgf",
     "relate extinction to criticality and to offspring variability", "simulate branching processes efficiently"],
    "notebook 05; probability generating functions", "60 min",
    [md("## Inside the algorithm: Newton's method for q = G(q)"), code(r"""G = lambda s: np.exp(1.5 * (s - 1)); dG = lambda s: 1.5 * np.exp(1.5 * (s - 1))
q = 0.0
for k in range(6):
    q = q - (G(q) - q) / (dG(q) - 1); print(k, f"{q:.15f}")
print("library (fixed-point iteration):", wk.extinction_probability(wk.pgf_poisson(1.5)))""")],
    ["A new virus has offspring distribution negative binomial with mean R0 = 2.5 and dispersion k (variance R0 + R0^2/k). Compute the probability that a single introduction dies out for k = 0.1, 0.5, 1, 10 and infinity (Poisson).",
     "For Poisson(1.0) offspring (critical), estimate P(Z_n > 0) for n = 10..1000 by iterating the pgf and show that it behaves like 2/(sigma^2 n) (Kolmogorov's estimate)."],
    "**Implement it yourself:** simulate the total progeny of a subcritical Poisson(0.8) process 10^4 times and compare its distribution with the Borel law P(T = n) = e^{-mn} (mn)^{n-1} / n!.")

EXTRAS["09_the_poisson_process"] = _x(
    ["construct homogeneous and non-homogeneous Poisson processes", "estimate a time-varying rate and test the Poisson assumptions on data",
     "use compound, split and superposed processes", "simulate spatial Poisson processes"],
    "notebooks 01-02; the exponential and Poisson distributions", "75 min",
    [md("## Inside the algorithm: Lewis-Shedler thinning in five lines"), code(r"""g = R(21); t, kept = 0.0, []
while True:
    t += g.exponential(1 / 160.0)
    if t > 24: break
    if g.random() < models.call_centre_rate(t) / 160.0: kept.append(t)
print(f"{len(kept)} arrivals; expected {pp.cumulative_intensity(models.call_centre_rate, 24.0, 20001):.0f}")""")],
    ["Using the call-centre data, test whether the Monday profile is a scaled copy of the Tuesday-Friday profile: estimate half-hourly rates for both and test the ratio for constancy (a chi-square test on the Monday counts given the scaled profile).",
     "Simulate a Cox process whose daily rate is the call-centre profile times a Gamma(shape 20, mean 1) day factor, and show that it reproduces the overdispersion of daily totals that a Poisson process cannot."],
    "**Implement it yourself:** simulate a non-homogeneous Poisson process by inverting the cumulative intensity of lambda(t) = 2 + sin(t) on [0, 50] with a root finder, and check the counts in [0, 10] and [10, 20] against their Poisson laws.")

EXTRAS["10_renewal_processes"] = _x(
    ["compute and interpret the renewal function", "explain and quantify the inspection paradox",
     "price maintenance policies with the renewal-reward theorem", "compute availability of alternating renewal systems"],
    "notebook 09", "60 min",
    [md("## Inside the algorithm: one step of the renewal-equation scheme"), code(r"""F = lambda x: 1 - np.exp(-2 * x)
hh = 0.01; k = 3; tk = np.arange(k + 1) * hh; Fk = F(tk); dF = np.diff(Fk)
m_prev = 2.0 * tk[:k]                                    # exact m on the earlier grid points
rest = Fk[k] + 0.5 * np.dot(dF[:k], np.r_[0.0, m_prev[:0:-1]] + m_prev[::-1])
print(f"m(t_3) by the scheme {rest / (1 - 0.5 * dF[0]):.8f}, exact {2 * tk[k]:.8f}")""")],
    ["Buses arrive as a renewal process with gamma inter-arrival times of mean 10 min. For shape 0.5, 1, 4 and infinity (deterministic), compute and simulate a passenger's mean wait. What does a schedule with cv 1.5 cost passengers?",
     "A component has a lognormal lifetime (median 2000 h, sigma 0.6). Planned replacement costs 1, failure 8. Find the optimal replacement age and the saving over run-to-failure."],
    "**Implement it yourself:** write the renewal-equation solver from scratch for a given cdf on a uniform grid and check it against the Erlang-2 closed form.")

EXTRAS["11_continuous_time_markov_chains"] = _x(
    ["build generators and compute transition functions by exponentials and uniformisation", "estimate a generator from an event log, handling censoring",
     "analyse birth-death chains and absorption times", "simulate reaction networks by Gillespie's algorithm and tau-leaping"],
    "notebooks 05-06 and 09", "90 min",
    [md("## Inside the algorithm: Gillespie's direct method"), code(r"""x, t, g = np.array([497, 3, 0]), 0.0, R(30)
while True:
    a = np.array([0.3 * x[0] * x[1] / 500, 0.1 * x[1]]); a0 = a.sum()
    if a0 == 0: break
    t += g.exponential(1 / a0)
    j = 0 if g.random() * a0 < a[0] else 1
    x = x + np.array([[-1, 1, 0], [0, -1, 1]])[j]
print(f"epidemic over at t = {t:.1f} days; final state S, I, R = {x.tolist()}")""")],
    ["For the machine model, compute the probability that a machine that is up now is down at some point in the next 24 hours (make 'down' absorbing and use P(24)).",
     "Simulate the Lotka-Volterra reaction network from `models.lotka_volterra()` many times and estimate the probability that the predators go extinct before time 50. How does it depend on the initial numbers?"],
    "**Implement it yourself:** write uniformisation from scratch with the truncation chosen from the Poisson tail, and reproduce `CTMC.transition_matrix` for the machine at t = 1, 10 and 100 h.")

EXTRAS["12_queueing_theory"] = _x(
    ["compute M/M/1, M/M/c, M/M/c/K and M/G/1 performance measures", "use Little's law as a modelling and checking tool",
     "staff a multi-server system for a service-level target", "judge when exponential assumptions are adequate"],
    "notebooks 09-11", "75 min",
    [md("## Inside the algorithm: Erlang B by recursion and why not by factorials"), code(r"""a, c = 380.0, 400
b = 1.0
for k in range(1, c + 1):
    b = a * b / (k + a * b)
try:
    direct = a**c / math.factorial(c) / sum(a**k / math.factorial(k) for k in range(c + 1))
except OverflowError as e:
    direct = f"OverflowError ({e})"
print(f"recursion {b:.6e}; library {qu.erlang_b(c, a):.6e}; direct formula: {direct}")""")],
    ["A hospital ward has 30 beds, admissions are Poisson at 5 per day, stays average 5 days. Compute the probability that an admission is turned away (Erlang B) and the number of beds needed for 1 % blocking. Check one answer by simulating M/G/c/c with lognormal stays (insensitivity).",
     "Two call centres each receive 100 calls/h with 4-min calls and must answer 80 % in 20 s. Compare the agents needed separately and when pooled, and simulate the pooled centre with the measured handle times."],
    "**Implement it yourself:** write the Lindley recursion and an estimate of P(W > t) for M/M/1, and compare with the exact rho e^{-(mu - lambda) t}.")

EXTRAS["13_discrete_event_simulation"] = _x(
    ["build an event-scheduling simulation with an event list and statistics", "model abandonment, finite sources and inventory policies",
     "validate a simulation against analytical results", "optimise by simulation without fooling yourself"],
    "notebooks 09-12", "90 min",
    [md("## Inside the algorithm: an M/M/1 queue in twenty lines"), code(r"""sim = des.Simulator(); g = R(40); state = {"n": 0}; waits = []; queue = []
q_tw = des.TimeWeighted(sim)
def arrive():
    if state["n"] == 0: waits.append(0.0); sim.after(g.exponential(1.0), depart)
    else: queue.append(sim.now)
    state["n"] += 1; q_tw.update(state["n"]); sim.after(g.exponential(1 / 0.8), arrive)
def depart():
    state["n"] -= 1; q_tw.update(state["n"])
    if queue: waits.append(sim.now - queue.pop(0)); sim.after(g.exponential(1.0), depart)
sim.schedule(0.0, arrive); sim.run(until=50_000)
print(f"mean wait {np.mean(waits):.2f} (4), time-average number in system {q_tw.mean():.2f} (4)")""")],
    ["Add a callback option to the call-centre model: callers who abandon call back after an exponential time (mean 30 min) with probability 0.5. How much does it raise the load and the abandonment rate at 4 agents?",
     "Compare continuous review with periodic review (every 2 time units) in the (s, S) inventory model: optimise each and report the cost difference with a common-random-numbers interval."],
    "**Implement it yourself:** add a priority class to the M/M/1 example above (two customer types, non-preemptive priority) and check the mean waits against Cobham's formula.")

EXTRAS["14_output_analysis"] = _x(
    ["measure autocorrelation with the integrated autocorrelation time", "build valid intervals by batch means, replications and regeneration",
     "detect and remove the initial transient", "plan run lengths for a target precision"],
    "notebooks 02 and 13", "60 min",
    [md("## Inside the algorithm: batch means by hand"), code(r"""x = w[10_000:]; b = len(x) // 20
means = x[: 20 * b].reshape(20, b).mean(axis=1)
se = means.std(ddof=1) / np.sqrt(20)
print(f"batch means by hand: {means.mean():.4f} +- {se:.4f}; library: {oa.batch_means(x, 20)['std_error']:.4f}")""")],
    ["For M/M/1 at rho = 0.5, 0.8, 0.9 and 0.95 estimate the integrated autocorrelation time of the waits and compare its growth with (1 - rho)^-2, the heavy-traffic prediction. How many customers are needed for a 1 % relative half-width at rho = 0.95?",
     "Run the inventory simulation of notebook 13 with daily cost outputs and compare batch means, replications and the regenerative method (regeneration: orders placed at a given inventory position) on the same budget."],
    "**Implement it yourself:** write Sokal's adaptive-window estimate of the integrated autocorrelation time from scratch (autocovariance by FFT, the window rule M >= 5 tau) and check it on an AR(1) with phi = 0.95.")

EXTRAS["15_brownian_motion"] = _x(
    ["construct Brownian paths by increments, midpoint refinement and scaled walks", "explain quadratic variation and its consequences",
     "use the reflection principle for maxima, first passages and barriers", "fit geometric Brownian motion to price data and test it"],
    "notebooks 07 and 09; the normal distribution", "75 min",
    [md("## Inside the algorithm: the Brownian-bridge crossing probability"), code(r"""a, x0, x1, dt_ = 1.0, 0.8, 0.9, 0.01
p_formula = np.exp(-2 * (a - x0) * (a - x1) / dt_)
fine = R(50)
B = bm.bridge(dt_, 400, 40_000, fine, a=x0, b=x1)
print(f"P(bridge from 0.8 to 0.9 over dt = 0.01 crosses 1): formula {p_formula:.4f}, simulated with 400 sub-steps {np.mean(B.max(axis=1) >= a):.4f}")
print("(the sub-stepped simulation is itself discretely monitored and misses crossings between its sub-steps - the bias the formula removes)")""")],
    ["Estimate the distribution of the time at which Brownian motion on [0, 1] attains its maximum, and compare with the arcsine law.",
     "Price an up-and-out call (barrier 130, strike 100) with daily monitoring by plain Monte Carlo, with the bridge correction, and with the continuity correction of Broadie, Glasserman and Kou (shift the barrier by exp(0.5826 sigma sqrt(dt)))."],
    "**Implement it yourself:** write Levy's midpoint construction from scratch for 2^10 steps and verify the covariance min(s, t) at three pairs of times.")

EXTRAS["16_stochastic_differential_equations"] = _x(
    ["solve SDEs by Euler-Maruyama, Milstein and Heun", "measure strong and weak orders of convergence correctly",
     "distinguish Ito and Stratonovich interpretations", "use multilevel Monte Carlo for expectations"],
    "notebook 15; ordinary differential equations", "90 min",
    [md("## Inside the algorithm: one Milstein step"), code(r"""x0, dt_, dW = 1.0, 0.01, 0.07
em = x0 + mu * x0 * dt_ + sig * x0 * dW
mil = em + 0.5 * (sig * x0) * sig * (dW**2 - dt_)
ex = x0 * np.exp((mu - sig**2 / 2) * dt_ + sig * dW)
print(f"exact {ex:.8f}, Euler {em:.8f} (error {abs(em - ex):.1e}), Milstein {mil:.8f} (error {abs(mil - ex):.1e})")""")],
    ["For the Langevin equation dX = -X dt + sqrt(2) dW, compare the stationary variance of Euler-Maruyama with step h against the exact value 1. Derive the Euler value 1/(1 - h/2) and check it.",
     "Simulate the CIR process with 2 kappa theta < sigma^2 by plain Euler (with max(x, 0) inside the square root only) and by full truncation; compare their means at T = 1 with the exact value for h = 0.1, 0.01."],
    "**Implement it yourself:** write the MLMC level estimator for the GBM call (coupled coarse and fine Euler paths) and verify that Var(P_l - P_{l-1}) halves from one level to the next.")

EXTRAS["17_gaussian_processes_and_time_series"] = _x(
    ["relate a covariance function to the smoothness of paths", "simulate stationary processes by Cholesky and by circulant embedding",
     "condition a Gaussian process on data", "identify, fit and check ARMA models"],
    "notebooks 02 and 15; linear algebra", "90 min",
    [md("## Inside the algorithm: the Durbin-Levinson recursion"), code(r"""gam = ts.arma_acov([0.6, -0.3], [], 3)
phi1 = gam[1] / gam[0]; v1 = gam[0] * (1 - phi1**2)
phi22 = (gam[2] - phi1 * gam[1]) / v1
phi21 = phi1 - phi22 * phi1
print(f"order 2 by hand: ({phi21:.6f}, {phi22:.6f}); library: {ts.durbin_levinson(gam)['phi'][2, 1:3].round(6)}")""")],
    ["Sample a Matern 5/2 process on 2^14 points of [0, 100] by circulant embedding (pad if needed) and estimate its variogram; compare with the theoretical one.",
     "Fit AR models to the log-returns of `prices.csv` and to their squares. Are the returns uncorrelated (Ljung-Box)? Are the squared returns? What does that say about the regime change?"],
    "**Implement it yourself:** write circulant embedding for a stationary covariance from scratch (FFT of the first row, check of the eigenvalues, complex white noise) and check the sample covariance of 10^4 paths.")

EXTRAS["18_markov_chain_monte_carlo"] = _x(
    ["construct Metropolis-Hastings samplers and tune their proposals", "use Gibbs sampling and recognise when it mixes slowly",
     "apply Hamiltonian Monte Carlo with the leapfrog integrator", "diagnose convergence with R-hat, autocorrelation and effective sample size"],
    "notebooks 06 and 14", "90 min",
    [md("## Inside the algorithm: one Metropolis step"), code(r"""g = R(60); x = np.array([0.5]); lp = logp(x)
y = x + 1.0 * g.standard_normal(1); ly = logp(y)
alpha = min(1.0, np.exp(ly - lp)); accept = g.random() < alpha
print(f"proposal {y[0]:.3f}: acceptance probability {alpha:.3f} -> {'accepted' if accept else 'rejected (stay at x)'}")""")],
    ["Sample the posterior of the logistic regression in this notebook with HMC (write the gradient) and compare effective samples per second with random-walk Metropolis.",
     "Sample a mixture of two well-separated normals (means -4 and 4) with random-walk Metropolis from several starts. Show that R-hat detects the failure, then fix it with parallel tempering between two temperatures."],
    "**Implement it yourself:** write the checkerboard Metropolis sweep for the Ising model from scratch and reproduce the energy per site at beta = 0.3 against `mcmc.onsager_energy`.")

EXTRAS["19_project_staffing_a_call_centre"] = _x(
    ["estimate a time-varying arrival rate from logs and decide what to do with faulty data", "choose between closed-form models and simulation for a decision",
     "test a standard method and its textbook correction against simulation", "optimise by simulation and assess the result on fresh random numbers"],
    "the whole course", "4 h",
    [md("## Inside the analysis: the morning ramp in numbers"), code(r"""j = int(np.argmax(np.diff(lam_hat)))
print(f"steepest rise between {mid[j]:.2f} and {mid[j + 1]:.2f} h: {lam_hat[j]:.0f} -> {lam_hat[j + 1]:.0f} calls/h")
for jj in (j, j + 1):
    w_ = (tgrid >= edges[jj]) & (tgrid < edges[jj] + 0.5)
    print(f"slot {edges[jj]:.1f}: mean arrival rate x mean handle = {lam_hat[jj] * mean_h:.1f} erlangs; peak offered load m(t) = {load[w_].max():.1f}; SIPP {c_sipp[jj]}, lag-aware {c_lag[jj]}")""")],
    ["Build the Monday schedule from the Monday data alone (four days) and from the Tuesday-Friday profile scaled by the estimated Monday factor. Which is more reliable, and how do they compare in simulation?",
     "Patience is not in the data. Repeat the final assessment with mean patience 1, 3 and 10 minutes and with deterministic patience. How sensitive is the recommendation?"],
    "**Implement it yourself:** add shift constraints - agents work 4-hour blocks starting on the hour - and find the cheapest set of shifts whose half-hourly coverage is at least the recommended schedule (a small integer program; brute force or a greedy heuristic), then assess it by simulation.")


def apply_extras(name, cells):
    ex = EXTRAS[name]
    header = md(
        "**What you will learn**\n"
        + "\n".join(f"- {o}" for o in ex["objectives"])
        + f"\n\n**Before you start:** {ex['prereq']}  ·  **Time:** about {ex['time']}"
    )
    cells = [cells[0], header] + list(cells[1:])
    items = ex["exercises"] + [ex["implement"]]
    exercises = md("## Exercises\n" + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    return cells + list(ex["inside"]) + [exercises]


def write_all(run: bool = True, only=()):
    import nbclient

    for name in sorted(NOTEBOOKS):
        if only and not name.startswith(tuple(only)):
            continue
        cells = apply_extras(name, NOTEBOOKS[name])
        nb = new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        if run:
            nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name, flush=True)


if __name__ == "__main__":
    write_all(run="--no-run" not in sys.argv, only=[a for a in sys.argv[1:] if not a.startswith("--")])
