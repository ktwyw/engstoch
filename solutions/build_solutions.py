"""Build the worked solutions to every exercise of the course, as executed notebooks.

The exercise texts are taken from ../notebooks/build_notebooks.py, so questions and solutions cannot
drift apart; the build fails if a notebook's number of solutions differs from its number of exercises.

    python solutions/build_solutions.py            # build and execute all solution notebooks
    python solutions/build_solutions.py 05 12      # only those whose names start with 05 or 12
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("course", HERE.parent / "notebooks" / "build_notebooks.py")
course = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(course)
md, code = course.md, course.code
SOLUTIONS: dict[str, dict] = {}


def solutions(name, setup="", imports=()):
    """Register the solutions of one course notebook: a function returning one list of cells per exercise."""

    def deco(fn):
        SOLUTIONS[name] = {"setup": setup, "imports": list(imports), "answers": fn()}
        return fn

    return deco


def exercises(name) -> list[str]:
    cells = course.apply_extras(name, list(course.NOTEBOOKS[name]))
    text = [c.source for c in cells if c.cell_type == "markdown" and "## Exercises" in c.source][0]
    items = re.split(r"\n(?=\d+\. )", text.split("## Exercises")[1].strip())
    return [re.sub(r"^\d+\.\s*", "", it).strip() for it in items]


def notebook_cells(name) -> list:
    title = re.sub(r"^#\s*", "", course.NOTEBOOKS[name][0].source.strip().splitlines()[0])
    entry = SOLUTIONS[name]
    ex = exercises(name)
    if len(ex) != len(entry["answers"]):
        raise ValueError(f"{name}: {len(ex)} exercises but {len(entry['answers'])} solutions")
    cells = [
        md(f"""
# Solutions · {title}

Worked solutions to the exercises of [notebook {name[:2]}](../notebooks/{name}.ipynb). Try each exercise
yourself before reading its solution - the learning happens in the attempt. Solutions are one way to
answer each question; other correct approaches exist.
"""),
        course.setup_cell(entry["imports"]),
    ] + ([code(entry["setup"])] if entry["setup"] else [])
    for i, (text, answer) in enumerate(zip(ex, entry["answers"]), 1):
        cells.append(md(f"## Exercise {i}\n\n{text}"))
        cells.extend(answer)
    return cells


def write_all(only=()):
    import nbclient

    for name in SOLUTIONS:
        if only and not name.startswith(tuple(only)):
            continue
        nb = new_notebook(
            cells=notebook_cells(name),
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        path = HERE / f"{name[:2]}_solutions.ipynb"
        nbformat.write(nb, path)
        print("wrote", path.name, flush=True)


@solutions("00_randomness_and_generators", imports=["from engstoch import rng as rg, rngtests as rt, special as sf"])
def _():
    return [
        [code(r"""full = [a for a in range(1, 64) if rg.LCG(a, 1, 64).period() == 64]
hd = [a for a in range(1, 64) if rg.full_period_conditions(a, 1, 64)]
print("full period (by running the generator):", full)
print("Hull-Dobell prediction agrees:", full == hd, "-> a = 1 mod 4:", all(a % 4 == 1 for a in full), f"({len(full)} multipliers)")"""),
         md(r"""
With m = 64 = 2⁶ and c = 1 (odd, so coprime to m), the theorem requires a − 1 divisible by 2 (the only prime
factor) and by 4 (because 4 divides m): a ≡ 1 (mod 4). Exactly those 16 multipliers give period 64 - though
a = 1, the identity plus a shift, shows that full period alone says nothing about randomness.
""")],
        [code(r"""rows = []
for name, gen in (("Park-Miller", rg.LCG.park_miller(7)), ("RANDU", rg.LCG.randu(7))):
    u = gen.random(3 * 10**6).reshape(-1, 3)
    s = 9 * u[:, 0] - 6 * u[:, 1] + u[:, 2]
    frac = s - np.round(s)
    rows.append((name, float(np.mean((frac > 0.45) & (frac < 0.55)) + np.mean((frac < -0.45) & (frac > -0.55)))))
print(pd.DataFrame(rows, columns=["generator", "fraction in the slab |frac| in (0.45, 0.55)"]).to_string(index=False))
v = R(1).random((10**6, 3)); s = 9 * v[:, 0] - 6 * v[:, 1] + v[:, 2]; f = s - np.round(s)
print("a perfect generator (NumPy PCG64):", np.mean(np.abs(f) > 0.45).round(4), "(0.1: the fractional part of a sum of continuous variables is ~uniform)")"""),
         md(r"""
For a good generator the fractional part of 9u₁ − 6u₂ + u₃ is essentially uniform, so a slab of width 0.1 (the
two halves near ±½) holds 10 % of the triples - Park-Miller gives that. For RANDU the combination is *exactly*
an integer (up to rounding), the fractional part is 0, and the slab is empty: a test aimed at the right
direction exposes the defect with a handful of numbers.
""")],
        [code(r"""def gap_test_mine(u, a=0.0, b=0.1, t=10):
    hits = np.flatnonzero((u >= a) & (u < b))
    gaps = np.diff(hits) - 1
    p = b - a
    counts = np.bincount(np.minimum(gaps, t), minlength=t + 1)
    probs = np.r_[p * (1 - p) ** np.arange(t), (1 - p) ** t]
    expected = probs * len(gaps)
    stat = np.sum((counts - expected) ** 2 / expected)
    return stat, sf.chi2_sf(stat, t)
u = rg.PCG32(3, 9).random(100_000)
mine = gap_test_mine(u)
lib = rt.gap_test(u, 0.0, 0.1, 10)
print(f"mine: chi2 = {mine[0]:.4f}, p = {mine[1]:.4f};  library: chi2 = {lib['statistic']:.4f}, p = {lib['p_value']:.4f}")"""),
         md(r"""
The gaps between visits to an interval of probability p are geometric: P(gap = r) = p(1 − p)ʳ. Pooling the tail
at t keeps every expected count large enough for the chi-square approximation (t + 1 cells, t degrees of
freedom). The hand-written test reproduces the library's statistic exactly.
""")],
    ]


@solutions("01_generating_random_variates", imports=["import time", "from scipy import stats", "from engstoch import rng as rg, rngtests as rt, special as sf"])
def _():
    return [
        [code(r"""g = R(1)
c_inv = np.tan(np.pi * (g.random(100_000) - 0.5))
c_rat = g.standard_normal(100_000) / g.standard_normal(100_000)
print(f"inversion: KS p = {stats.kstest(c_inv, 'cauchy').pvalue:.3f};  ratio of normals: KS p = {stats.kstest(c_rat, 'cauchy').pvalue:.3f}")
x = np.tan(np.pi * (R(2).random(10**6) - 0.5))
n = np.arange(1, len(x) + 1)
plt.semilogx(n, np.cumsum(x) / n); plt.xlabel("n"); plt.ylabel("running mean"); plt.title("Cauchy: no law of large numbers"); plt.show()
print("running mean at n = 1e4, 1e5, 1e6:", [round(float(np.mean(x[:k])), 3) for k in (10**4, 10**5, 10**6)])"""),
         md(r"""
Both constructions give the standard Cauchy law (F⁻¹(u) = tan(π(u − ½)); the ratio of two independent normals
is Cauchy because the angle of a 2-D normal vector is uniform). The Cauchy has no mean: the running mean keeps
jumping whenever a huge value arrives, and in fact the mean of n Cauchy variables is again standard Cauchy -
averaging does not help at all.
""")],
        [code(r"""lams = np.linspace(0.3, 3, 271)
M = np.sqrt(2 / np.pi) / lams * np.exp(lams**2 / 2)        # max of f/g, attained at x = lambda
lam_best = lams[np.argmin(M)]
print(f"best lambda = {lam_best:.3f} (theory 1), M = {M.min():.4f} (sqrt(2e/pi) = {np.sqrt(2 * np.e / np.pi):.4f})")
f = lambda x: np.sqrt(2 / np.pi) * np.exp(-x**2 / 2)
r = rg.rejection(f, lambda m: R(3).exponential(1 / lam_best, m), lambda x: lam_best * np.exp(-lam_best * x), M.min() * 1.0001, R(4), 50_000)
print(f"observed acceptance {r['acceptance']:.4f} vs 1/M = {1 / M.min():.4f}; KS p vs half-normal = {stats.kstest(r['x'], 'halfnorm').pvalue:.3f}")"""),
         md(r"""
The ratio f/g = √(2/π) e^{−x²/2 + λx}/λ is maximal at x = λ, so M(λ) = √(2/π) e^{λ²/2}/λ, minimised at λ = 1 with
M = √(2e/π) ≈ 1.315: three proposals in four are accepted. Attaching a random sign gives a normal - one of
the classical exact normal generators.
""")],
        [code(r"""def box_muller_mine(g, n):
    u1, u2 = 1 - g.random(n // 2), g.random(n // 2)
    r = np.sqrt(-2 * np.log(u1))
    return np.r_[r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)]
def polar_mine(g, n):
    out = []
    while len(out) < n:
        v = 2 * g.random((n, 2)) - 1
        s = np.sum(v**2, axis=1); ok = (s > 0) & (s < 1)
        out.extend((v[ok] * np.sqrt(-2 * np.log(s[ok]) / s[ok])[:, None]).ravel())
    return np.array(out[:n])
for name, fn in (("Box-Muller", box_muller_mine), ("polar", polar_mine)):
    t0 = time.perf_counter(); z = fn(R(5), 10**6); t1 = time.perf_counter() - t0
    print(f"{name:<10}: {t1 * 1000:.0f} ms for 1e6 normals, KS p = {rt.ks_test(z, sf.norm_cdf)['p_value']:.3f}, mean {z.mean():+.4f}, var {z.var():.4f}")"""),
         md(r"""
In vectorised NumPy Box-Muller is several times faster: the polar method saves the trigonometric functions but
throws away 21 % of its uniforms and pays for boolean indexing and a Python-level loop. In compiled code the polar method used to win;
modern generators (NumPy's ziggurat) beat both by avoiding logarithms almost always.
""")],
    ]


@solutions("02_monte_carlo_estimation", imports=["from scipy import stats", "from engstoch import montecarlo as mc, output as oa"])
def _():
    return [
        [code(r"""n = 10**6
u = R(1).random((n, 2))
hit = (np.sum(u**2, axis=1) <= 1).astype(float) * 4
r = mc.mean_ci(hit, level=0.99)
print(f"pi ~ {r['estimate']:.5f}, 99 % CI ({r['ci'][0]:.5f}, {r['ci'][1]:.5f}); sigma = {r['std_dev']:.4f}")
n6 = mc.sample_size(r["std_dev"], 0.5e-5, 0.99)
print(f"six correct digits (half-width 5e-6 at 99 %): n = {n6:.2e} points, {n6 / 1e8 / 3600:.1f} hours at 1e8 points per second")"""),
         md(r"""
Each point gives 4·1{inside}, with standard deviation 4√(p(1 − p)) ≈ 1.64. Six digits need a half-width of
5·10⁻⁶, i.e. n ≈ (2.58 × 1.64 / 5·10⁻⁶)² ≈ 7·10¹¹ points - about two hours at a hundred million per second. Monte
Carlo is the wrong tool for computing π; it is the right one when there is no better method.
""")],
        [code(r"""truth = np.exp(1.125)
rows = []
cov_t, cov_b = [], []
for k in range(500):
    x = R(100 + k).lognormal(0, 1.5, 1000)
    cov_t.append(mc.mean_ci(x, method="t")["ci"])
    boot = R(9000 + k).choice(x, (2000, len(x))).mean(axis=1)
    cov_b.append(tuple(np.quantile(boot, [0.025, 0.975])))
print("t interval:          coverage", oa.coverage([a for a, b in cov_t], [b for a, b in cov_t], truth))
print("percentile bootstrap: coverage", oa.coverage([a for a, b in cov_b], [b for a, b in cov_b], truth))"""),
         md(r"""
The bootstrap does not fix it: both intervals cover the true mean about 92-93 % of the time instead of 95 %.
The bootstrap resamples the observed data, so it cannot know about the large values that a sample of 1000
rarely contains - the source of the problem. For heavy tails one needs more data, a parametric model for the
tail, or a transformation (estimate the lognormal parameters and transform back).
""")],
        [code(r"""p, n = 0.001, 2000
z = 1.959963984540054
cw, cwi = [], []
for k in range(20_000):
    s = R(k).binomial(n, p)
    ph = s / n
    half = z * np.sqrt(ph * (1 - ph) / n)
    cw.append(ph - half <= p <= ph + half)
    centre = (ph + z**2 / (2 * n)) / (1 + z**2 / n)
    hw = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    cwi.append(centre - hw <= p <= centre + hw)
print(f"coverage at p = 0.001, n = 2000: Wald {np.mean(cw):.3f}, Wilson {np.mean(cwi):.3f}")
print("library Wilson for 2 successes:", mc.proportion_ci(2, 2000)["ci"])"""),
         md(r"""
With np = 2 the count is 0 in 13.5 % of runs, and the Wald interval is then the single point 0 - it can never
cover. Its coverage is about 86 %; Wilson's interval, which inverts the score test rather than plugging in p̂,
stays near 95 %. For rare-event estimation by crude Monte Carlo, Wald intervals are unusable.
""")],
    ]


@solutions("03_variance_reduction", imports=["from scipy import stats", "from engstoch import montecarlo as mc, brownian as bm, special as sf"])
def _():
    return [
        [code(r"""S0, K, r, sig, T, m = 100.0, 100.0, 0.05, 0.2, 1.0, 12
n = 100_000
S = bm.gbm(S0, r, sig, T, m, n, R(1))[:, 1:]
arith = np.exp(-r * T) * np.maximum(S.mean(axis=1) - K, 0)
geo = np.exp(-r * T) * np.maximum(np.exp(np.log(S).mean(axis=1)) - K, 0)
t = np.arange(1, m + 1) * T / m
mu_g = np.log(S0) + (r - sig**2 / 2) * t.mean()
var_g = sig**2 / m**2 * np.sum(np.minimum.outer(t, t))
d1 = (mu_g - np.log(K) + var_g) / np.sqrt(var_g)
geo_exact = np.exp(-r * T) * (np.exp(mu_g + var_g / 2) * sf.norm_cdf(d1) - K * sf.norm_cdf(d1 - np.sqrt(var_g)))
plain = mc.mean_ci(arith)
cv = mc.control_variate(arith, geo, geo_exact)
print(f"geometric Asian (closed form) {geo_exact:.4f}, simulated {geo.mean():.4f}")
print(f"arithmetic Asian: plain {plain['estimate']:.4f} +- {plain['std_error']:.4f}; with the control {cv['estimate']:.4f} +- {cv['std_error']:.5f}; variance reduction {cv['variance_reduction']:.0f}x")"""),
         md(r"""
The log of the geometric average of GBM prices is normal (a linear combination of Brownian values) with mean
log S₀ + (r − σ²/2)t̄ and variance σ²/m² Σᵢⱼ min(tᵢ, tⱼ), so its call has a Black-Scholes-type formula. The two
averages are almost perfectly correlated, and the control variate cuts the variance by several hundred - the
classical Kemna-Vorst trick.
""")],
        [code(r"""n, a = 100, 130.0
exact = stats.gamma.sf(a, n)
rows = []
for theta in (0.0, 0.1, 0.2, 0.23, 0.3, 0.4):
    g = R(int(100 * theta) + 1)
    X = g.exponential(1 / (1 - theta), (50_000, n))          # tilted Exp(1): rate 1 - theta
    S = X.sum(axis=1)
    lr = np.exp(-theta * S) * (1 - theta) ** (-n)              # f/g = prod e^{-x} / ((1-theta) e^{-(1-theta) x})
    est = mc.mean_ci((S > a) * lr)
    rows.append((theta, est["estimate"], mc.relative_error(est) if est["estimate"] > 0 else np.inf))
print(f"P(sum of 100 Exp(1) > 130) = {exact:.4e}")
print(pd.DataFrame(rows, columns=["tilt theta", "estimate", "relative error"]).to_string(index=False, float_format="%.3g"))
print("theory: the tilted mean n/(1 - theta) equals a at theta = 1 - n/a =", round(1 - n / a, 4))"""),
         md(r"""
Tilting each Exp(1) summand to Exp(1 − θ) multiplies the density by e^{θx}(1 − θ); the likelihood ratio of the
whole sample depends only on the sum. The best tilt makes the rare event typical - the tilted mean of the sum,
n/(1 − θ), equals the threshold 130 at θ = 0.23 - and gives about 1 % relative error with 5·10⁴ samples, where
crude Monte Carlo would need around 10⁸.
""")],
        [code(r"""h = lambda u: np.where(u > 0.9, 10 * u, 0.1 * u)
def stratified_neyman(h, n, k, g, pilot=200):
    edges = np.linspace(0, 1, k + 1); p = np.diff(edges)
    sd = np.array([np.std(h(edges[j] + p[j] * g.random(pilot)), ddof=1) for j in range(k)])
    nj = np.maximum(2, np.round(n * p * sd / np.sum(p * sd)).astype(int))
    m, v = np.zeros(k), np.zeros(k)
    for j in range(k):
        y = h(edges[j] + p[j] * g.random(nj[j])); m[j], v[j] = y.mean(), y.var(ddof=1)
    return np.sum(p * m), np.sqrt(np.sum(p**2 * v / nj)), nj
est, se, nj = stratified_neyman(h, 10_000, 10, R(5))
lib = mc.stratified(h, 10_000, 10, R(5), "neyman")
print(f"mine: {est:.5f} +- {se:.5f}, allocation {nj.tolist()}\nlibrary: {lib['estimate']:.5f} +- {lib['std_error']:.5f}, allocation {lib['allocation'].tolist()}\nexact {0.1 * 0.81 / 2 + 10 * 0.19 / 2:.5f}")"""),
         md(r"""
The pilot estimates each stratum's standard deviation; Neyman allocation puts most samples in the last
stratum, where the integrand jumps. With the same random stream the hand-written version reproduces the
library's allocation and estimate exactly.
""")],
    ]


@solutions("04_quasi_monte_carlo", imports=["from engstoch import qmc, montecarlo as mc, brownian as bm, special as sf"])
def _():
    return [
        [code(r"""S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
bs = bm.black_scholes_call(S0, K, r, sig, T)
d2 = (np.log(S0 / K) + (r - sig**2 / 2) * T) / (sig * np.sqrt(T))
digital = np.exp(-r * T) * sf.norm_cdf(d2)
def price(u, kind):
    z = sf.norm_ppf(np.clip(u, 1e-12, 1 - 1e-12))
    ST = S0 * np.exp((r - sig**2 / 2) * T + sig * np.sqrt(T) * z)
    return np.exp(-r * T) * (np.maximum(ST - K, 0) if kind == "call" else (ST > K).astype(float))
rows = []
for m in range(8, 17, 2):
    n = 2**m
    for kind, truth in (("call", bs), ("digital", digital)):
        e_mc = np.mean([abs(price(R(m * 10 + k).random(n), kind).mean() - truth) for k in range(10)])
        e_q = np.mean([abs(price(np.mod(qmc.van_der_corput(n, 2) + R(m * 100 + k).random(), 1), kind).mean() - truth) for k in range(10)])
        rows.append((n, kind, e_mc, e_q, e_mc / e_q))
print(pd.DataFrame(rows, columns=["n", "payoff", "MC error", "shifted-QMC error", "gain"]).to_string(index=False, float_format="%.2e"))"""),
         md(r"""
In one dimension the shifted van der Corput sequence is the 1-D Sobol sequence, and QMC's gain grows with n for
both payoffs. Surprisingly the digital option gains *more*: in one dimension a single jump has bounded
variation, so the Koksma-Hlawka bound still gives an O(log n / n) error, while the call's payoff, mapped through
the normal quantile, has a long unbounded tail near u = 1. The textbook trouble with discontinuities appears in
several dimensions, where the jump runs diagonally across the cube (a digital option on a basket or an Asian
average) and the integrand has infinite Hardy-Krause variation; conditioning out the jump restores QMC's rate.
""")],
        [code(r"""d = 20
f = lambda x: np.prod(1 + (x - 0.5) / np.arange(1, d + 1) ** 2, axis=1)
def pts(n):
    return np.hstack([qmc.sobol(n, 10), qmc.halton(n, 10, start=1)])
rows = []
for m in (8, 10, 12, 14):
    n = 2**m
    e_mc = np.mean([abs(f(R(m * 10 + k).random((n, d))).mean() - 1) for k in range(10)])
    rq = qmc.randomised_qmc(f, pts(n), 10, R(m))
    rows.append((n, e_mc, np.mean(np.abs(rq["replicates"] - 1))))
print(pd.DataFrame(rows, columns=["n", "MC error", "randomised QMC error"]).to_string(index=False, float_format="%.2e"))
var_j = (1 / 12) / np.arange(1, d + 1) ** 4
print("share of the variance carried by the first 1, 2, 5 coordinates (first-order terms):", (np.cumsum(var_j) / var_j.sum())[[0, 1, 4]].round(4))"""),
         md(r"""
The integral is exactly 1 and the nominal dimension is 20, yet QMC wins by orders of magnitude: the j-th
coordinate's influence decays like 1/j², so 92 % of the variance comes from the first coordinate alone. The
integrand has low *effective dimension*, and the first coordinates of Sobol points are the best distributed.
This is why QMC works for many high-dimensional problems in finance - and why ordering the inputs by importance
(for example by the Brownian-bridge construction) matters.
""")],
        [code(r"""PR = [2, 3, 5, 7, 11]
def radical_inverse_int(i, b):
    num, den = 0, 1
    while i > 0:
        i, d = divmod(i, b)
        num, den = num * b + d, den * b
    return num / den
H = np.array([[radical_inverse_int(i, b) for b in PR] for i in range(100)])
print("max difference from qmc.halton:", np.max(np.abs(H - qmc.halton(100, 5))))
print("first rows:\n", H[:4].round(4))"""),
         md(r"""
Accumulating the mirrored digits as an integer numerator over bᵏ and dividing once at the end avoids rounding
in the accumulation; the result agrees with the library to the last bit or two.
""")],
    ]


@solutions("05_discrete_time_markov_chains", imports=["from engstoch import markov as mk, models, datasets"])
def _():
    return [
        [code(r"""P = np.array([[0.90, 0.08, 0.02],      # good: wears or breaks
              [0.00, 0.85, 0.15],      # worn: preventive repair only when broken in this policy
              [0.70, 0.00, 0.30]])     # broken: repaired to good with prob. 0.7 per day
c = mk.MarkovChain(P, ["good", "worn", "broken"])
pi = c.stationary()
path = c.simulate(100_000, "good", R(1))
print("stationary law:", pi.round(4).tolist(), "\nsimulated fractions:", (np.bincount(path, minlength=3) / len(path)).round(4).tolist())
P2 = P.copy(); P2[1] = [0.5, 0.45, 0.05]                    # alternative: repair worn machines (half of them per day)
print("with preventive repair of worn machines: broken", mk.MarkovChain(P2).stationary()[2].round(4), "vs", pi[2].round(4))"""),
         md(r"""
Under this policy the machine is broken about 8.5 % of days; the simulation agrees. Changing one row - repairing
worn machines before they fail - cuts broken days by more than half. The stationary law turns a maintenance
policy into a number that can be costed.
""")],
        [code(r"""wd = pd.read_csv(datasets.path("weather.csv"), comment="#")
x = wd.wet.to_numpy(); doy = wd.day_of_year.to_numpy()
q = (doy - 1) // 91 % 4
P_s = []
for s in range(4):
    idx = np.flatnonzero(q[:-1] == s)
    c = np.zeros((2, 2)); np.add.at(c, (x[idx], x[idx + 1]), 1)
    P_s.append(c / c.sum(axis=1, keepdims=True))
P_h = mk.estimate(x, 2)["P"]
def simulate(Pfun, g, years=10):
    y = np.zeros(365 * years, dtype=int)
    for k in range(1, len(y)):
        y[k] = g.random() < Pfun(k)[y[k - 1], 1]
    return y
def longest(y):
    return [max((len(r) for r in np.split(yy, np.flatnonzero(np.diff(yy)) + 1) if r[0] == 1), default=0) for yy in y.reshape(-1, 365)]
obs = longest(x)
sea = sum((longest(simulate(lambda k: P_s[(k % 365) // 91 % 4], R(10 + i))) for i in range(10)), [])
hom = sum((longest(simulate(lambda k: P_h, R(30 + i))) for i in range(10)), [])
print(f"longest wet spell per year: data mean {np.mean(obs):.2f}, seasonal model {np.mean(sea):.2f}, homogeneous model {np.mean(hom):.2f}")
print(f"  upper quartile: data {np.quantile(obs, 0.75):.1f}, seasonal {np.quantile(sea, 0.75):.1f}, homogeneous {np.quantile(hom, 0.75):.1f}")"""),
         md(r"""
Neither chain reproduces the annual maximum: both give about 6.7 days against 7.8 in the data. The quarterly
chain is no better than the homogeneous one because averaging over a quarter still hides the peak of the wet
season - the generating probabilities vary smoothly through the year - and annual maxima are driven by exactly
that peak. (With ten years the data's mean has a standard error of about 0.6 days, so the gap is real.) The
next step is a chain whose transition probabilities vary smoothly with the day of the year, fitted by logistic
regression of tomorrow's state on today's state and a seasonal cosine.""")],
        [code(r"""def mle(path, m):
    counts = np.zeros((m, m)); np.add.at(counts, (path[:-1], path[1:]), 1)
    n_i = counts.sum(axis=1, keepdims=True)
    P = counts / n_i
    return P, np.sqrt(P * (1 - P) / n_i)
x = pd.read_csv(datasets.path("weather.csv"), comment="#").wet.to_numpy()
P, se = mle(x, 2)
lib = mk.estimate(x, 2)
print("mine:\n", P.round(5), "\n", se.round(5), "\nlibrary agrees:", np.allclose(P, lib["P"]) and np.allclose(se, lib["std_error"]))"""),
         md(r"""
Given the number of departures n_i from state i, the next states are multinomial, so the MLE is the observed
fraction and its standard error the binomial one. The counts matrix is all the data the first-order chain uses.
""")],
    ]


@solutions("06_long_run_behaviour_and_mixing", imports=["from engstoch import markov as mk, models"])
def _():
    return [
        [code(r"""rows = []
for n in (8, 16, 32, 64, 128):
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = 0.5; P[i, (i + 1) % n] += 0.25; P[i, (i - 1) % n] += 0.25
    c = mk.MarkovChain(P)
    rows.append((n, c.mixing_time(0.25), c.spectral_gap()["relaxation_time"]))
r = np.array(rows, float)
print(pd.DataFrame(rows, columns=["n", "t_mix(1/4)", "relaxation time"]).to_string(index=False, float_format="%.1f"))
print("t_mix / n^2:", (r[:, 1] / r[:, 0] ** 2).round(4).tolist(), " relaxation / n^2:", (r[:, 2] / r[:, 0] ** 2).round(4).tolist())"""),
         md(r"""
Both grow like n²: a random walk needs about n² steps to diffuse a distance n (spectral gap 1 − cos(2π/n) ≈
2π²/n² for the lazy walk, halved). Unlike the Ehrenfest urn there is no log factor - the cycle has no cutoff.
Diffusive mixing is the reason local MCMC moves are slow on long, thin targets.
""")],
        [code(r"""def weekly_cost(s, S, h=1.0, lost=20.0, order=15.0, mean=2.0):
    m = models.inventory_chain(s, S, mean); c = m["chain"]; pmf = m["demand_pmf"]
    pi = c.stationary(); k = np.arange(len(pmf))
    cost = 0.0
    for x in range(S + 1):
        start = S if x <= s else x
        cost += pi[x] * ((order if x <= s else 0.0) + lost * np.sum(pmf * np.maximum(k - start, 0)) + h * x)
    return cost
grid = [(s, S, weekly_cost(s, S)) for s in range(0, 5) for S in range(s + 1, 11)]
best = min(grid, key=lambda t: t[2])
print(f"best policy s = {best[0]}, S = {best[1]}: {best[2]:.3f} per week")
print("a few others:", [(s, S, round(float(c), 3)) for s, S, c in grid if (s, S) in ((2, 6), (0, 4), (3, 8), (4, 10))])"""),
         md(r"""
The weekly cost is the stationary expectation of ordering cost (when stock ends at or below s), expected lost
sales given the starting stock, and holding cost on the end-of-week stock. A full grid of policies is cheap
because each needs one linear solve. The best policy reorders at s = 2 but orders up to S = 10, the largest
value on the grid: with an ordering cost of 15 against a holding cost of 1 per unit-week, large infrequent
orders pay, and the grid should be extended before trusting the optimum.""")],
        [code(r"""links = {0: [1, 2], 1: [2, 3], 2: [0, 3, 4], 3: [4, 5], 4: [2, 5], 5: []}
n, d = 6, 0.85
r = np.full(n, 1 / n)
for it in range(200):
    new = np.full(n, (1 - d) / n)
    dangling = sum(r[i] for i in links if not links[i])
    new += d * dangling / n
    for i, outs in links.items():
        for j in outs:
            new[j] += d * r[i] / len(outs)
    if np.max(np.abs(new - r)) < 1e-14:
        break
    r = new
print(f"converged in {it} iterations:", r.round(6).tolist())
print("library:                   ", mk.pagerank(models.web_graph()["adjacency"]).round(6).tolist())"""),
         md(r"""
Only the links are stored; each iteration distributes every page's rank over its out-links, spreads the
dangling pages' rank uniformly, and adds the teleportation share. The error contracts by at least the factor d
per iteration, so 0.85ᵏ < 10⁻¹⁴ needs about 200 iterations at worst - and much fewer in practice.
""")],
    ]


@solutions("07_absorption_hitting_and_random_walks", imports=["from engstoch import markov as mk, walks as wk, models"])
def _():
    return [
        [code(r"""base = models.snakes_and_ladders()
E0 = base["chain"].absorption()["expected_steps"][0]
snakes = {k: v for k, v in base["jumps"].items() if v < k}
rows = []
for head in snakes:
    j = dict(base["jumps"]); del j[head]
    rows.append((f"{head} -> {snakes[head]}", models.snakes_and_ladders(jumps=j)["chain"].absorption()["expected_steps"][0]))
rows.sort(key=lambda t: t[1])
print(f"with all snakes: {E0:.3f} throws")
print(pd.DataFrame(rows, columns=["snake removed", "expected throws"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Each candidate board is a new absorbing chain, solved in microseconds. The most harmful snake is the one at 35,
one square before the finish: players near the end land on it often (a throw past 36 wastes the turn, so they
hover there), and it sends them back thirteen squares. Removing it shortens the expected game by more than four
throws; removing the equally long snake at 17 saves barely one.""")],
        [code(r"""n = 10_000
w = wk.simple_walk(n, R(1), n_paths=10_000)
last_zero = np.array([np.flatnonzero(row == 0)[-1] for row in w]) / n
x = np.linspace(0, 1, 101)
plt.hist(last_zero, bins=50, density=True, alpha=0.6, label="last visit to 0 / n"); plt.plot(x[1:-1], 1 / (np.pi * np.sqrt(x[1:-1] * (1 - x[1:-1]))), "k", label="arcsine density"); plt.legend(); plt.show()
print("P(last zero before n/10):", np.mean(last_zero < 0.1).round(4), " arcsine:", round(float(wk.arcsine_cdf(0.1)), 4))
print("P(last zero after 9n/10):", np.mean(last_zero > 0.9).round(4), " arcsine:", round(1 - float(wk.arcsine_cdf(0.9)), 4))"""),
         md(r"""
Lévy's second arcsine law: the last return to zero is as likely to fall in the first tenth of the game as in
the last tenth, and either is far more likely than the middle tenth. A walk that has been ahead for most of a
long game is the rule, not a sign of skill.
""")],
        [code(r"""import itertools
n = 12
for k in (2, 4, 6):
    count = 0
    for steps in itertools.product((1, -1), repeat=n):
        s = np.cumsum(steps)
        if s[-1] == k and np.all(s > 0):
            count += 1
    formula = k / n * wk.paths_count(n, k)
    print(f"n = {n}, end at {k}: brute force {count}, ballot formula (k/n) C(n, (n+k)/2) = {formula:.0f}")"""),
         md(r"""
Of the C(n, (n+k)/2) paths ending at k, a fraction k/n stays strictly positive - the ballot theorem with
a − b = k votes of margin among a + b = n. Brute force over the 4096 paths confirms it.
""")],
    ]


@solutions("08_branching_processes", imports=["from scipy import optimize", "from engstoch import walks as wk"])
def _():
    return [
        [code(r"""R0 = 2.5
rows = []
for k in (0.1, 0.5, 1.0, 10.0, np.inf):
    if np.isinf(k):
        G = lambda s: np.exp(R0 * (s - 1))
    else:
        G = lambda s, k=k: (1 + R0 / k * (1 - s)) ** (-k)
    q = optimize.brentq(lambda s: G(s) - s, 0, 1 - 1e-12) if G(0.0) < 1 else 1.0
    rows.append((k, q, 1 - q))
print(pd.DataFrame(rows, columns=["dispersion k", "P(extinct)", "P(major outbreak)"]).to_string(index=False, float_format="%.4f"))"""),
         md(r"""
The negative binomial pgf is (1 + (R₀/k)(1 − s))^{−k}. With strong overdispersion (k = 0.1, as for SARS) most
infected people infect nobody and a few infect many: a single introduction dies out 86 % of the
time even with R₀ = 2.5, whereas Poisson offspring gives only 11 %. Superspreading makes outbreaks rarer but
explosive when they happen.
""")],
        [code(r"""G = wk.pgf_poisson(1.0)
qn = wk.extinction_by_generation(G, 1000)
rows = [(n, 1 - qn[n], 2 / n, (1 - qn[n]) * n / 2) for n in (10, 30, 100, 300, 1000)]
print(pd.DataFrame(rows, columns=["n", "P(Z_n > 0)", "2/(sigma^2 n)", "ratio"]).to_string(index=False, float_format="%.5f"))"""),
         md(r"""
At criticality the survival probability decays only like 2/(σ²n) (Kolmogorov 1938; σ² = 1 for Poisson(1)),
with a ratio that tends to 1 slowly - the correction is of order log n / n. Critical populations die out
surely but slowly, and those still alive at generation n have size of order n.
""")],
        [code(r"""m = 0.8
tot = []
for i in range(10_000):
    z = wk.simulate_galton_watson(lambda k, r: r.poisson(m, k), 200, R(i))
    tot.append(z.sum())
tot = np.array(tot)
ns = np.arange(1, 16)
borel = np.exp(-m * ns) * (m * ns) ** (ns - 1) / np.array([math.factorial(int(k)) for k in ns])
emp = np.array([np.mean(tot == k) for k in ns])
print(pd.DataFrame({"n": ns, "simulated": emp.round(4), "Borel": borel.round(4)}).T.to_string(header=False))
print(f"mean total progeny {tot.mean():.3f} vs 1/(1 - m) = {wk.total_progeny_mean(wk.pgf_poisson(m)):.3f}")"""),
         md(r"""
The total progeny of a Poisson(m) branching process is Borel-distributed - a consequence of the hitting-time
theorem for the random walk that explores the family tree. Its mean 1/(1 − m) = 5 hides a long tail: one
family in a hundred exceeds 40.
""")],
    ]


@solutions("09_the_poisson_process", imports=["from scipy import stats, optimize", "from engstoch import poisson as pp, models, datasets, special as sf"])
def _():
    return [
        [code(r"""cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
edges = np.arange(8, 20.01, 1.0)
mon = np.array([pp.counts_in_windows(g.arrival_h, edges) for d, g in cc[cc.weekday == "Mon"].groupby("day")]).sum(axis=0)
rest = cc[(cc.weekday != "Mon") & (cc.day != 8)]
base = np.array([pp.counts_in_windows(g.arrival_h, edges) for d, g in rest.groupby("day")]).sum(axis=0)
factor = mon.sum() / 4 / (base.sum() / rest.day.nunique())
expected = base / rest.day.nunique() * 4 * factor
chi = sf.chi2_sf(np.sum((mon - expected) ** 2 / expected), len(mon) - 2)
print(f"Monday factor {factor:.3f}; chi-square test of a proportional profile: p = {chi:.3f}")
print("hourly ratio Monday / other days:", (mon / 4 / (base / rest.day.nunique())).round(3).tolist())"""),
         md(r"""
Given the Monday totals, a proportional profile predicts the Monday hourly counts; the chi-square test rejects
it (p = 0.004), but the rejection rests almost entirely on one hour - 13:00-14:00, with a ratio of 0.94 against
about 1.15 everywhere else. Since these data are synthetic we know the true profile is proportional, so this is
a chance fluctuation of the kind a test produces when many cells are compared. In real work the next step would
be to look at that hour on each of the four Mondays before believing in a Monday lunchtime dip; for staffing,
one factor of about 1.15 on the weekday profile is the sensible model.""")],
        [code(r"""lam_int = pp.cumulative_intensity(models.call_centre_rate, 24.0, 20001)
pois = [len(pp.thinning(models.call_centre_rate, 160.0, 24.0, R(k))["times"]) for k in range(400)]
cox = []
for k in range(400):
    g = R(1000 + k)
    f = g.gamma(20, 1 / 20)
    cox.append(len(pp.thinning(lambda t: f * models.call_centre_rate(t), 160.0 * f, 24.0, g)["times"]))
print(f"Poisson days: mean {np.mean(pois):.0f}, variance/mean {np.var(pois) / np.mean(pois):.2f}")
print(f"Cox days (Gamma day factor, cv {1 / np.sqrt(20):.2f}): mean {np.mean(cox):.0f}, variance/mean {np.var(cox) / np.mean(cox):.2f}; theory 1 + mean/shape = {1 + lam_int / 20:.1f}")"""),
         md(r"""
A random day factor F with mean 1 and variance 1/20 makes the daily count mixed Poisson: Var N = E N + Var(F)(E N)²,
so the variance-to-mean ratio is 1 + Λ/20 ≈ 56 for about 1100 calls a day. Even modest day-to-day variation of
the rate (cv 22 %) swamps Poisson noise at this volume - why staffing must plan for forecast error, not just
for randomness given the forecast.
""")],
        [code(r"""lam = lambda t: 2 + np.sin(t)
Lam = lambda t: 2 * t + 1 - np.cos(t)
e = pp.homogeneous(1.0, Lam(50.0), R(7))
times_one = np.array([optimize.brentq(lambda t: Lam(t) - ei, 0, 50) for ei in e])
counts = []
for k in range(3000):
    e = pp.homogeneous(1.0, Lam(20.0), R(100 + k))
    t = np.array([optimize.brentq(lambda s: Lam(s) - ei, 0, 20) for ei in e])
    counts.append((np.sum(t < 10), np.sum((t >= 10) & (t < 20))))
counts = np.array(counts)
for j, (a, b) in enumerate(((0, 10), (10, 20))):
    m = Lam(b) - Lam(a)
    print(f"N[{a}, {b}): mean {counts[:, j].mean():.3f}, variance {counts[:, j].var():.3f}, Lambda difference {m:.3f}")
print("correlation of the two counts (independent increments):", np.corrcoef(counts.T)[0, 1].round(3))"""),
         md(r"""
Inverting Λ(t) = 2t + 1 − cos t at each unit-rate arrival gives the non-homogeneous process; the counts in the
two windows are Poisson with means Λ(10) − Λ(0) and Λ(20) − Λ(10) and uncorrelated. A root finder per arrival is
slow but exact; tabulating Λ and interpolating, as notebook 09 did, is the fast version.
""")],
    ]


@solutions("10_renewal_processes", imports=["from scipy import stats", "from engstoch import renewal as rn"])
def _():
    return [
        [code(r"""rows = []
for shape in (0.5, 1.0, 4.0, 100.0):
    if np.isinf(shape):
        sampler, m2 = (lambda n, r: np.full(n, 10.0)), 100.0
    else:
        sampler, m2 = (lambda n, r, k=shape: r.gamma(k, 10.0 / k, n)), 100.0 * (1 + 1 / shape)
    ins = rn.inspect(sampler, 200.0, 3000, R(int(10 * min(shape, 9))))
    rows.append((shape, 1 / np.sqrt(shape) if np.isfinite(shape) else 0.0, ins["residual"].mean(), rn.mean_residual_life(10.0, m2)))
print(pd.DataFrame(rows, columns=["gamma shape", "cv", "simulated mean wait (min)", "E[X^2]/(2E[X])"]).to_string(index=False, float_format="%.2f"))
print("cv 1.5:", rn.mean_residual_life(10.0, 100 * (1 + 1.5**2)), "min vs 5 min for a perfectly regular service")"""),
         md(r"""
The mean wait is E[X²]/(2E[X]) = (μ/2)(1 + cv²): half a headway for regular buses, a full headway for
Poisson-like ones, and 16 minutes with cv 1.5 although buses still come every 10 minutes on average. Bunching
costs passengers more than a lower frequency would - the argument for headway-based (rather than
timetable-based) bus control.
""")],
        [code(r"""sigma, median = 0.6, 2000.0
surv = lambda x: stats.lognorm.sf(np.asarray(x), sigma, scale=median)
T = np.linspace(200, 6000, 400)
o = rn.optimal_age_replacement(surv, 1.0, 8.0, T)
mean_life = stats.lognorm.mean(sigma, scale=median)
print(f"optimal replacement age {o['T']:.0f} h, cost rate {o['cost_rate'] * 1000:.4f} per 1000 h; run to failure {8 / mean_life * 1000:.4f} per 1000 h; saving {100 * (1 - o['cost_rate'] * mean_life / 8):.0f} %")
haz = stats.lognorm.pdf(T, sigma, scale=median) / surv(T)
print(f"the lognormal hazard rises then falls: maximum at {T[np.argmax(haz)]:.0f} h")"""),
         md(r"""
Replacing at about 780 h saves 45 % of the cost of running to failure. The lognormal hazard is not
monotone - it rises to a maximum and then decreases - so very old survivors are actually *better* than middle-
aged ones, and the cost curve flattens beyond the optimum; for a Weibull with shape > 1 the hazard keeps rising.
""")],
        [code(r"""def renewal_eq(cdf, t_end, n):
    h = t_end / n; t = np.arange(n + 1) * h; F = cdf(t); dF = np.diff(F); m = np.zeros(n + 1)
    for k in range(1, n + 1):
        known = sum(dF[j - 1] * 0.5 * ((m[k - j + 1] if j > 1 else 0.0) + m[k - j]) for j in range(1, k + 1))
        m[k] = (F[k] + known) / (1 - 0.5 * dF[0])
    return t, m
F = lambda x: 1 - np.exp(-2 * x) * (1 + 2 * x)
t, m = renewal_eq(F, 5.0, 500)
print("max error vs the Erlang-2 closed form:", np.max(np.abs(m - rn.renewal_function_erlang2(t, 2.0))))
print("max difference from the library:", np.max(np.abs(m - rn.renewal_function(F, 5.0, 500)[1])))"""),
         md(r"""
Each m_k depends on all earlier values, so the scheme is a triangular recursion (O(n²) work, which a convolution
by FFT could reduce). The trapezoidal treatment of each cell makes the m_k term appear on both sides; solving
for it gives the division by 1 − dF₁/2.
""")],
    ]


@solutions("11_continuous_time_markov_chains", imports=["from engstoch import ctmc, models"])
def _():
    return [
        [code(r"""Q = models.machine()["Q"].copy()
Qa = Q.copy(); Qa[2] = 0.0                                     # make 'down' absorbing
P24 = ctmc.CTMC(Qa).transition_matrix(24.0)
print(f"P(machine up now is down at some time in the next 24 h) = {P24[0, 2]:.4f}")
print(f"P(down exactly at t = 24 h) = {ctmc.CTMC(Q).transition_matrix(24.0)[0, 2]:.4f} (smaller: it may have been repaired)")
hits = []
for k in range(4000):
    p = models.machine()["chain"].simulate(24.0, "up", R(k)); hits.append(np.any(p["states"] == 2))
print(f"simulation: {np.mean(hits):.4f}")"""),
         md(r"""
"Ever down within 24 hours" is a first-passage question; making the down state absorbing turns it into "down
at 24 hours" for the modified chain. It is about nine times the probability of being down at the 24-hour mark:
failures are frequent but short, and most are repaired before the mark.
""")],
        [code(r"""lv = models.lotka_volterra()
rows = []
for x0 in ((50, 100), (100, 50), (20, 20), (200, 20)):
    ext = []
    for k in range(100):
        r = ctmc.gillespie(np.array(x0), lv["stoich"], lv["propensity"], 50.0, R(k), max_events=100_000)
        ext.append(r["x"][-1, 1] == 0)
    rows.append((x0, np.mean(ext)))
print(pd.DataFrame(rows, columns=["(prey, predators) at t = 0", "P(predators extinct by t = 50)"]).to_string(index=False))"""),
         md(r"""
The deterministic Lotka-Volterra equations cycle for ever; the stochastic system does not - each cycle passes
through a trough of predators where a run of deaths ends the population. Starting with few predators or far
from equilibrium (big oscillations, deep troughs) makes extinction much more likely. Demographic noise is
decisive in small populations, which is why conservation biology uses stochastic models.
""")],
        [code(r"""def uniformise(Q, t, tol=1e-12):
    L = np.max(-np.diag(Q)); Rm = np.eye(len(Q)) + Q / L
    w = np.exp(-L * t); total = w; P = w * np.eye(len(Q)); Rk = np.eye(len(Q)); k = 0
    while 1 - total > tol:
        k += 1; w *= L * t / k; Rk = Rk @ Rm; P += w * Rk; total += w
    return P, k
c = models.machine()["chain"]
for t in (1.0, 10.0, 100.0):
    P, k = uniformise(c.Q, t)
    print(f"t = {t:5.0f} h: {k + 1} terms, max difference from exp(Qt) {np.max(np.abs(P - c.transition_matrix(t))):.1e}")"""),
         md(r"""
The number of terms grows like Λt + a few √(Λt): at t = 100 h with Λ = 1.2 per hour about 170 terms. For stiff
chains (a fast rate next to a slow one) and long times uniformisation becomes expensive, while scaling and
squaring does not care; uniformisation's advantages are positivity and an a-priori error bound.
""")],
    ]


@solutions("12_queueing_theory", imports=["from engstoch import queues as qu, models"])
def _():
    return [
        [code(r"""a = 5 * 5.0
print(f"30 beds, 25 erlangs: P(turned away) = {qu.erlang_b(30, a):.4f}")
c = next(c for c in range(1, 100) if qu.erlang_b(c, a) <= 0.01)
print(f"beds for at most 1 % blocking: {c} (Erlang B {qu.erlang_b(c, a):.4f})")
n = 200_000
arr = np.cumsum(R(1).exponential(1 / 5.0, n)); stay = R(2).lognormal(np.log(5.0) - 0.5 * 0.8**2, 0.8, n)
import heapq
beds = []; blocked = 0
for t, s in zip(arr, stay):
    while beds and beds[0] <= t: heapq.heappop(beds)
    if len(beds) < 30: heapq.heappush(beds, t + s)
    else: blocked += 1
print(f"simulated M/G/30/30 with lognormal stays (mean 5 days): blocking {blocked / n:.4f} - insensitive to the stay distribution")"""),
         md(r"""
Erlang's loss formula depends on the service-time distribution only through its mean (Sevastyanov's
insensitivity theorem), so it applies to lognormal hospital stays: about 5 % of admissions are turned away with
30 beds, and 1 % blocking needs 36. The simulation keeps a heap of discharge times and confirms the formula.
""")],
        [code(r"""lam, mean_h = 100 / 60, 4.0
sep = qu.staffing(lam, 1 / mean_h, 0.8, 20 / 60)
pool = qu.staffing(2 * lam, 1 / mean_h, 0.8, 20 / 60)
print(f"separately: 2 x {sep['c']} = {2 * sep['c']} agents; pooled: {pool['c']} agents ({2 * sep['c'] - pool['c']} fewer)")
from engstoch import datasets
cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#"); h = cc.handle_min[cc.handle_min > 0].to_numpy()
arr = np.cumsum(R(3).exponential(1 / (2 * lam), 300_000))
w = qu.simulate_ggc(arr, R(4).choice(h, 300_000), pool["c"])["wait"][20_000:]
print(f"pooled centre simulated with the measured handle times: P(wait <= 20 s) = {np.mean(w <= 20 / 60):.3f} (Erlang C predicts {pool['service_level']:.3f})")"""),
         md(r"""
Pooling saves agents because the variability of the combined load grows only like the square root of the
load. With the real (less variable than exponential) handle times the pooled centre does better than Erlang C
predicts - the exponential model is conservative here.
""")],
        [code(r"""lam, mu, n = 0.8, 1.0, 500_000
a = R(5).exponential(1 / lam, n); s = R(6).exponential(1 / mu, n)
w = np.zeros(n)
for k in range(1, n):
    w[k] = max(0.0, w[k - 1] + s[k - 1] - a[k])
w = w[20_000:]
for t in (0.0, 1.0, 5.0, 10.0):
    print(f"P(W > {t:4}) simulated {np.mean(w > t):.4f}, exact rho e^(-(mu - lambda) t) = {lam / mu * np.exp(-(mu - lam) * t):.4f}")"""),
         md(r"""
In M/M/1 the waiting time is 0 with probability 1 − ρ and otherwise exponential with rate μ − λ. The
recursion reproduces the tail, though slowly converging estimates at large t remind us that the waits are
strongly autocorrelated (notebook 14).
""")],
    ]


@solutions("13_discrete_event_simulation", imports=["from collections import deque", "from engstoch import des, montecarlo as mc"])
def _():
    return [
        [code(r"""def centre_with_callbacks(c, retry_prob, seed, T=20_000.0, lam=4.0, mu=1.0, theta=0.5, retry_mean=0.5):
    g = R(seed); sim = des.Simulator(); queue = deque(); busy = [0]; stats_ = {"fresh": 0, "retries": 0, "abandon": 0, "served": 0}
    def arrive(kind):
        stats_[kind] += 1
        call = {"start": None}
        if busy[0] < c and not queue: start(call)
        else:
            queue.append(call); sim.after(g.exponential(1 / theta), abandon, call)
    def fresh():
        arrive("fresh"); sim.after(g.exponential(1 / lam), fresh)
    def start(call):
        call["start"] = sim.now; busy[0] += 1; sim.after(g.exponential(1 / mu), done)
    def done():
        busy[0] -= 1; stats_["served"] += 1
        if queue: start(queue.popleft())
    def abandon(call):
        if call["start"] is None and call in queue:
            queue.remove(call); stats_["abandon"] += 1
            if g.random() < retry_prob: sim.after(g.exponential(retry_mean), arrive, "retries")
    sim.schedule(0.0, fresh); sim.run(until=T)
    return stats_
for p in (0.0, 0.5):
    s = centre_with_callbacks(4, p, 1)
    tot = s["fresh"] + s["retries"]
    print(f"retry probability {p}: offered calls per fresh call {tot / s['fresh']:.3f}, abandonment per attempt {s['abandon'] / tot:.3f}, fresh callers never served {1 - s['served'] / s['fresh']:.3f}")"""),
         md(r"""
Callbacks add load exactly when the system is congested, which lengthens queues and causes more abandonment
per attempt; but because half the abandoners try again, the fraction of *customers* never served falls. Which
number a manager watches (abandoned attempts or lost customers) changes the conclusion - and logs that do not
link retries to callers overstate demand.
""")],
        [code(r"""def cost(s, S, seed, review=None):
    return des.inventory_sS(s, S, 2.0, lambda g: 1 + g.poisson(1.0), lambda g: g.uniform(1.0, 3.0), 2000.0, R(seed), review=review)["cost_rate"]
grid = [(s, S) for s in range(4, 22, 2) for S in range(s + 10, s + 45, 5)]
best_c = min(grid, key=lambda k: np.mean([cost(*k, seed) for seed in range(4)]))
best_p = min(grid, key=lambda k: np.mean([cost(*k, seed, 2.0) for seed in range(4)]))
d = [cost(*best_p, 500 + k, 2.0) - cost(*best_c, 500 + k) for k in range(20)]
r = mc.mean_ci(d, method="t")
print(f"continuous review best {best_c}, periodic review (every 2) best {best_p}")
print(f"extra cost of periodic review: {r['estimate']:.2f} per unit time, 95 % CI ({r['ci'][0]:.2f}, {r['ci'][1]:.2f})")"""),
         md(r"""
Periodic review must protect against demand over the lead time *plus* the review period, so its best reorder
point is higher and its cost a little larger; the common-random-numbers interval shows the difference is real.
In practice the saving in monitoring effort may be worth it.
""")],
        [code(r"""def priority_mm1(lam1, lam2, mu, T, seed):
    g = R(seed); sim = des.Simulator(); queues = [deque(), deque()]; busy = [False]; waits = [[], []]
    def arrive(k, lam):
        if not busy[0]: start(k, sim.now)
        else: queues[k].append(sim.now)
        sim.after(g.exponential(1 / lam), arrive, k, lam)
    def start(k, t_arr):
        busy[0] = True; waits[k].append(sim.now - t_arr); sim.after(g.exponential(1 / mu), done)
    def done():
        busy[0] = False
        for k in (0, 1):
            if queues[k]: start(k, queues[k].popleft()); break
    sim.schedule(0.0, arrive, 0, lam1); sim.schedule(0.0, arrive, 1, lam2); sim.run(until=T)
    return [np.mean(w[1000:]) for w in waits]
lam1, lam2, mu = 0.3, 0.5, 1.0
w = priority_mm1(lam1, lam2, mu, 200_000.0, 1)
W0 = (lam1 + lam2) / mu**2                                   # sum lambda_i E[S_i^2]/2 with E[S^2] = 2/mu^2
s1, s = lam1 / mu, (lam1 + lam2) / mu
print(f"high priority: simulated {w[0]:.3f}, Cobham {W0 / (1 - s1):.3f}")
print(f"low priority:  simulated {w[1]:.3f}, Cobham {W0 / ((1 - s1) * (1 - s)):.3f}")"""),
         md(r"""
Cobham's formula W_k = W₀ / ((1 − σ_{k−1})(1 − σ_k)) with W₀ = Σλᵢ E[Sᵢ²]/2 gives the mean wait of each class
under non-preemptive priority: about 1.1 for the high class and 5.7 for the low one, against 4.0 for everybody
under FCFS at the same total load 0.8. The load-weighted waits are conserved - 0.3 × 1.14 + 0.5 × 5.71 =
0.8 × 4.0 (Kleinrock's conservation law): priority moves waiting between classes, it does not remove it.
""")],
    ]


@solutions("14_output_analysis", imports=["from engstoch import output as oa, queues as qu, montecarlo as mc, des"])
def _():
    return [
        [code(r"""rows = []
for rho in (0.5, 0.8, 0.9, 0.95):
    n = 400_000
    w = qu.lindley(R(int(100 * rho)).exponential(1 / rho, n), R(int(100 * rho) + 1).exponential(1.0, n))[n // 10:]
    tau = oa.iat(w)["tau"]
    rows.append((rho, tau, tau * (1 - rho) ** 2))
print(pd.DataFrame(rows, columns=["rho", "tau (measured)", "tau (1 - rho)^2"]).to_string(index=False, float_format="%.3f"))
rho = 0.95; Wq = rho / (1 - rho); var_w = rho * (2 - rho) / (1 - rho) ** 2
tau95 = rows[-1][1]
n_needed = (1.96 * np.sqrt(var_w * tau95) / (0.01 * Wq)) ** 2
print(f"rho = 0.95: customers for a 1 % relative half-width ~ {n_needed:.2e}")"""),
         md(r"""
τ(1 − ρ)² stays between about 2.5 and 3, confirming the heavy-traffic scaling τ ∝ (1 − ρ)⁻²: the waits' memory
(in customers) grows like the relaxation time of the queue. Since Var(W) itself grows like (1 − ρ)⁻², the
asymptotic variance τ Var(W) of the mean wait grows like (1 − ρ)⁻⁴. At ρ = 0.95 a 1 % half-width needs tens of
millions of customers - the reason heavy-traffic simulation studies use regenerative or variance-reduction
methods, or report coarser precision.
""")],
        [code(r"""reps = oa.replications([des.inventory_sS(10, 35, 2.0, lambda r: 1 + r.poisson(1.0), lambda r: r.uniform(1, 3), 500.0, R(k))["cost_rate"] for k in range(40)])
long = des.inventory_sS(10, 35, 2.0, lambda r: 1 + r.poisson(1.0), lambda r: r.uniform(1, 3), 500.0 * 40, R(1000))["cost_rate"]
print(f"40 replications of 500 time units: {reps['estimate']:.3f} +- {reps['std_error']:.3f}")
print(f"one run of 20000 time units: {long:.3f} (a single number: an interval needs batch means of its output series)")"""),
         md(r"""
With the library's inventory model, which reports the long-run rate of one run, replications are the natural
interval method; batch means need the cost recorded per period (a few lines added to the model), and the
regenerative method needs a regeneration state (an order placed when the position is exactly s, for unit
demand). For this lightly correlated system all three agree; the replication interval also absorbs the small
start-up bias of each run.
""")],
        [code(r"""def iat_sokal(x, c=5.0):
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, m); acov = np.fft.irfft(f * np.conj(f), m)[:n] / n
    rho = acov / acov[0]
    for M in range(1, n):
        tau = 1 + 2 * np.sum(rho[1:M + 1])
        if M >= c * tau: return tau, M
phi = 0.95
e = R(7).standard_normal(500_000); x = np.zeros_like(e)
for k in range(1, len(e)): x[k] = phi * x[k - 1] + e[k]
tau, M = iat_sokal(x)
print(f"mine: tau = {tau:.2f} (window {M}); library {oa.iat(x)['tau']:.2f}; theory (1 + phi)/(1 - phi) = {(1 + phi) / (1 - phi):.2f}")"""),
         md(r"""
The FFT gives all autocovariances in O(n log n); the window stops the sum where the remaining autocorrelations
are mostly noise. With φ = 0.95 the estimate is within a few percent of 39, the exact value.
""")],
    ]


@solutions("15_brownian_motion", imports=["from engstoch import brownian as bm, walks as wk"])
def _():
    return [
        [code(r"""W = bm.paths(1.0, 4000, 20_000, R(1))
argmax = np.argmax(W, axis=1) / 4000
for x in (0.05, 0.25, 0.5):
    print(f"P(argmax <= {x}) simulated {np.mean(argmax <= x):.4f}, arcsine {float(wk.arcsine_cdf(x)):.4f}")
plt.hist(argmax, bins=50, density=True); xx = np.linspace(0.005, 0.995, 200); plt.plot(xx, 1 / (np.pi * np.sqrt(xx * (1 - xx))), "k"); plt.title("time of the maximum on [0, 1]"); plt.show()"""),
         md(r"""
Lévy's arcsine law again: the maximum of a Brownian path on [0, 1] is most likely near the ends of the
interval. For an investor this says the best time to have sold was probably very early or very late - a fact
that makes "if only I had sold at the top" regret almost guaranteed.
""")],
        [code(r"""S0, K, B, r, sig, T, n = 100.0, 100.0, 130.0, 0.05, 0.2, 1.0, 252
paths = bm.gbm(S0, r, sig, T, n, 100_000, R(2)); dt = T / n
pay = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0)
plain = np.mean(pay * np.all(paths < B, axis=1))
lw = np.log(paths); pcross = np.exp(-2 * (np.log(B) - lw[:, :-1]) * (np.log(B) - lw[:, 1:]) / (sig**2 * dt))
alive = np.all(paths < B, axis=1) & np.all(R(3).random(pcross.shape) >= np.where((paths[:, :-1] < B) & (paths[:, 1:] < B), pcross, 0), axis=1)
bridge = np.mean(pay * alive)
B_shift = B * np.exp(-0.5826 * sig * np.sqrt(dt))
bgk = np.mean(pay * np.all(paths < B_shift, axis=1))
print(f"daily monitoring (plain MC) {plain:.4f}; continuous monitoring by the bridge correction {bridge:.4f}")
print(f"BGK: the continuous price from discrete simulation with the barrier shifted down to {B_shift:.2f}: {bgk:.4f}")"""),
         md(r"""
The plain daily-monitoring price, the bridge-corrected (continuous-monitoring) price and the BGK estimate show
the direction of the effect: continuous monitoring knocks out more paths, so the continuous price is lower.
Broadie, Glasserman and Kou's correction converts between the two by shifting the barrier by
exp(±0.5826σ√Δt) (here downwards to price continuous monitoring with daily simulation); it agrees with the
bridge correction to about a cent.
""")],
        [code(r"""levels = 10; n = 2**levels; g = R(4); n_paths = 20_000
W = np.zeros((n_paths, n + 1)); W[:, -1] = g.standard_normal(n_paths)
step = n
while step > 1:
    half = step // 2
    left, right = W[:, 0:n - half:step], W[:, step::step]
    W[:, half::step] = 0.5 * (left + right) + np.sqrt(step / n / 4) * g.standard_normal(left.shape)
    step = half
for s, t in ((0.25, 0.5), (0.5, 0.75), (0.1, 0.9)):
    i, j = int(s * n), int(t * n)
    print(f"Cov(W_{s}, W_{t}) = {np.cov(W[:, i], W[:, j])[0, 1]:.4f} (min = {min(s, t)})")"""),
         md(r"""
Starting from W₁ ~ N(0, 1), each pass fills the midpoints of the current intervals with the bridge mean plus
N(0, Δ/4) noise, where Δ is the current interval length. Ten passes give 1024 steps; the covariance min(s, t)
is reproduced at every pair.
""")],
    ]


@solutions("16_stochastic_differential_equations", imports=["from engstoch import sde, brownian as bm"])
def _():
    return [
        [code(r"""rows = []
for h in (0.5, 0.2, 0.1, 0.02):
    X = sde.euler_maruyama(lambda t, x: -x, lambda t, x: np.sqrt(2.0) + 0 * x, 0.0, 200.0, int(200 / h), 400, R(int(100 * h)))["X"]
    rows.append((h, X[:, int(20 / h):].var(), 1 / (1 - h / 2)))
print(pd.DataFrame(rows, columns=["h", "simulated stationary variance", "1/(1 - h/2)"]).to_string(index=False, float_format="%.4f"))"""),
         md(r"""
Euler gives the AR(1) X_{k+1} = (1 − h)X_k + √(2h) Z with stationary variance 2h / (1 − (1 − h)²) =
1/(1 − h/2): the error in the *invariant measure* is first order in h, which matters for long-time simulation
(molecular dynamics, MCMC by Langevin dynamics, where a Metropolis correction removes it).
""")],
        [code(r"""x0, kappa, theta, sigma, T = 0.04, 0.5, 0.04, 0.4, 1.0          # 2 kappa theta = 0.04 < sigma^2 = 0.16
exact = float(sde.cir_mean(x0, kappa, theta, T))
for h in (0.1, 0.01):
    n = int(T / h); g = R(int(1 / h))
    x = np.full(200_000, x0)
    for k in range(n):
        x = x + kappa * (theta - x) * h + sigma * np.sqrt(np.maximum(x, 0) * h) * g.standard_normal(len(x))
    ft = sde.cir_full_truncation(x0, kappa, theta, sigma, T, n, 200_000, R(int(1 / h) + 7))[:, -1]
    print(f"h = {h}: absorption-style Euler mean {x.mean():.5f} (negative values {np.mean(x < 0):.3f}), full truncation {ft.mean():.5f}, exact {exact:.5f}")"""),
         md(r"""
When the Feller condition fails, zero is reachable and Euler steps overshoot below it. With max(x, 0) only
inside the square root, a quarter of the paths are negative at h = 0.1 - impossible values for a rate or a
variance. Its *mean* happens to be close here (negative excursions and the missing noise near zero partly
cancel), while full truncation is biased upwards at h = 0.1; both biases vanish as h shrinks. The case for full
truncation is not the mean but positivity and robustness: in the Heston model the CIR value is a variance whose
square root drives the price, and a negative value breaks the simulation.
""")],
        [code(r"""def level_pair(level, n, g, S0=100.0, r=0.05, sig=0.2, T=1.0):
    nf = 2**level; hf = T / nf
    dW = np.sqrt(hf) * g.standard_normal((n, nf))
    Sf = np.full(n, S0)
    for k in range(nf): Sf = Sf * (1 + r * hf + sig * dW[:, k])
    if level == 0: return np.exp(-r * T) * np.maximum(Sf - 100, 0)
    Sc = np.full(n, S0); dWc = dW[:, 0::2] + dW[:, 1::2]
    for k in range(nf // 2): Sc = Sc * (1 + r * 2 * hf + sig * dWc[:, k])
    return np.exp(-r * T) * (np.maximum(Sf - 100, 0) - np.maximum(Sc - 100, 0))
v = [np.var(level_pair(l, 100_000, R(l))) for l in range(1, 9)]
print("Var(P_l - P_l-1), l = 1..8:", np.round(v, 5).tolist())
print("ratios V_l / V_l+1:", np.round(np.array(v[:-1]) / np.array(v[1:]), 2).tolist())"""),
         md(r"""
Driving the coarse path with the sums of pairs of fine increments couples the two; their payoffs differ by
O(√h) (the strong error), so the variance of the difference is O(h) and halves from level to level. That decay
is what lets MLMC spend most of its samples on the cheap levels.
""")],
    ]


@solutions("17_gaussian_processes_and_time_series", imports=["from engstoch import gaussian as gp, timeseries as ts, datasets"])
def _():
    return [
        [code(r"""n, L_dom, ell = 2**14, 100.0, 2.0
h = L_dom / n
for pad in (0, n // 2, n):
    try:
        acov = gp.matern52(0, np.arange(n + pad) * h, ell=ell)
        X = gp.circulant_embedding(acov, 20, R(1), pad=pad)["X"]
        print(f"padding {pad}: embedding valid; {X.shape[0]} paths of {X.shape[1]} points"); break
    except ValueError as e:
        print(f"padding {pad}: {e}")
vg = gp.empirical_variogram(X, 400)
lags = np.arange(401) * h
plt.plot(lags, vg, label="empirical (20 paths)"); plt.plot(lags, 1 - gp.matern52(0, lags, ell=ell), "k--", label="1 - C(h)"); plt.legend(); plt.xlabel("lag"); plt.show()"""),
         md(r"""
On [0, 100] with ℓ = 2 the window is fifty correlation lengths long, and the minimal embedding is already
non-negative definite (the covariance has decayed to ~10⁻²⁰ at half the window); on a shorter window padding
would be needed. The variogram of twenty paths follows 1 − C(h) and levels off at the variance.
""")],
        [code(r"""pr = pd.read_csv(datasets.path("prices.csv"), comment="#")
lr = np.diff(np.log(pr.close.to_numpy()))
print(f"log-returns: Ljung-Box (20 lags) p = {ts.ljung_box(lr, 20)['p_value']:.3f}; AIC order {ts.select_ar_order(lr, 10)['p']}")
sq = (lr - lr.mean()) ** 2
print(f"squared returns: Ljung-Box p = {ts.ljung_box(sq, 20)['p_value']:.2e}")
print(f"squared returns within each regime: p = {ts.ljung_box(sq[:1000], 20)['p_value']:.3f} and {ts.ljung_box(sq[1000:], 20)['p_value']:.3f}")"""),
         md(r"""
The returns themselves are uncorrelated (as GBM says), but their squares are not - over the whole record.
Within each regime they are, so the "volatility clustering" here is entirely the regime change: a slowly
varying volatility makes squared returns autocorrelated. In real markets volatility varies continuously and
GARCH or stochastic-volatility models describe it.
""")],
        [code(r"""def circulant(acov, n_paths, g):
    c = np.asarray(acov, float); row = np.r_[c, c[-2:0:-1]]; m = len(row)
    lam = np.fft.fft(row).real
    assert lam.min() > -1e-10 * lam.max(), "embedding not non-negative definite"
    Z = g.standard_normal((n_paths, m)) + 1j * g.standard_normal((n_paths, m))
    Y = np.fft.fft(np.sqrt(np.maximum(lam, 0) / m) * Z, axis=1)
    return np.vstack([Y.real, Y.imag])[:, : len(c)]
t = np.arange(200) * 0.05
X = circulant(gp.exponential(0, t, ell=0.5), 5000, R(2))
print("sample covariance at lags 0, 5, 20:", np.round(np.cov(X[:, [0, 5, 20]].T)[0], 3), " theory", np.round(gp.exponential(0, t[[0, 5, 20]], ell=0.5), 3))"""),
         md(r"""
A circulant matrix is diagonalised by the Fourier matrix, its eigenvalues are the FFT of its first row, and
F diag(√λ/√m) Z with complex white noise Z has covariance equal to the circulant - whose top-left n × n block is
the wanted Toeplitz covariance. Real and imaginary parts are two independent samples.
""")],
    ]


@solutions("18_markov_chain_monte_carlo", imports=["import time", "from engstoch import mcmc, output as oa"])
def _():
    return [
        [code(r"""rng = R(6)
x = rng.random(40) * 10
y = (rng.random(40) < 1 / (1 + np.exp(-(-3.0 + 0.8 * x)))).astype(float)
def lp(b):
    eta = b[0] + b[1] * x
    return float(np.sum(y * eta - np.logaddexp(0, eta)) - 0.5 * np.sum(b**2) / 100)
def glp(b):
    p = 1 / (1 + np.exp(-(b[0] + b[1] * x)))
    return np.array([np.sum(y - p), np.sum((y - p) * x)]) - b / 100
rows = []
for name, run in (("random-walk Metropolis", lambda: mcmc.metropolis(lp, [0, 0], 20_000, [0.6, 0.12], R(1))["chain"]),
                  ("HMC (step 0.05, 20 leapfrogs)", lambda: mcmc.hmc(lp, glp, np.zeros(2), 2000, 0.05, 20, R(2))["chain"])):
    t0 = time.perf_counter(); ch = run(); dt = time.perf_counter() - t0
    e = min(oa.ess(ch[len(ch) // 5:, j]) for j in range(2))
    rows.append((name, len(ch), e, e / dt))
print(pd.DataFrame(rows, columns=["sampler", "iterations", "min ESS", "ESS per second"]).to_string(index=False, float_format="%.1f"))"""),
         md(r"""
HMC produces nearly independent draws per iteration but each costs twenty gradient evaluations; random-walk
Metropolis is cheap per step but strongly autocorrelated along the posterior's correlated ridge. Per second of
computing they are within a small factor here; HMC's advantage grows with dimension, where random walks slow
down linearly and HMC only like d^{1/4}.
""")],
        [code(r"""def logp(z):
    return float(np.logaddexp(-0.5 * (z[0] - 4) ** 2, -0.5 * (z[0] + 4) ** 2))
chains = [mcmc.metropolis(logp, [s], 20_000, 1.0, R(10 + k))["chain"][:, 0] for k, s in enumerate((-4, -4, 4, 4))]
print("single-temperature chains: means", [round(float(c.mean()), 2) for c in chains], " R-hat", round(oa.gelman_rubin([c[10_000:] for c in chains])["R_hat"], 2))
def tempering(n, g, betas=(1.0, 0.1)):
    x = np.array([4.0, 4.0]); out = np.empty(n)
    for k in range(n):
        for j, b in enumerate(betas):
            y = x[j] + 1.5 * g.standard_normal()
            if np.log(g.random()) < b * (logp([y]) - logp([x[j]])): x[j] = y
        if np.log(g.random()) < (betas[0] - betas[1]) * (logp([x[1]]) - logp([x[0]])): x = x[::-1].copy()
        out[k] = x[0]
    return out
pt = [tempering(20_000, R(20 + k)) for k in range(4)]
print("parallel tempering: means", [round(float(c.mean()), 2) for c in pt], " R-hat", round(oa.gelman_rubin([c[10_000:] for c in pt])["R_hat"], 3), " P(z > 0)", round(float(np.mean(np.concatenate(pt) > 0)), 3))"""),
         md(r"""
Started in different modes, plain Metropolis chains rarely cross the valley between them: their means range
from −1 to 1.4 although the truth is 0, and R-hat (1.16, far above 1.01) flags the failure. A second chain at inverse temperature 0.1 sees a
flattened target and crosses freely; swapping states between the two (accepted with the ratio of the tempered
densities) carries the crossings to the cold chain, which now spends half its time in each mode.
""")],
        [code(r"""L, beta, sweeps = 32, 0.3, 2000
g = R(30); s = g.choice(np.array([-1, 1]), (L, L))
ii, jj = np.indices((L, L)); masks = [(ii + jj) % 2 == 0, (ii + jj) % 2 == 1]
E = []
for k in range(sweeps):
    for mask in masks:
        nb = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2 * s * nb
        flip = mask & (g.random((L, L)) < np.exp(-beta * np.maximum(dE, 0)))
        s = np.where(flip, -s, s)
    E.append(-np.sum(s * (np.roll(s, 1, 0) + np.roll(s, 1, 1))) / L**2)
E = np.array(E[300:])
print(f"energy per site {E.mean():.4f} +- {E.std() * np.sqrt(oa.iat(E)['tau'] / len(E)):.4f}; Onsager {mcmc.onsager_energy(beta):.4f}")"""),
         md(r"""
Sites of one checkerboard colour have neighbours only of the other colour, so all of them can be updated at
once without violating detailed balance; a sweep is two vectorised half-sweeps. At β = 0.3 (high temperature)
the 32 × 32 lattice already reproduces the infinite-lattice energy within its error bar.
""")],
    ]


PROJECT_SETUP = r'''from engstoch import poisson as pp, queues as qu, des, montecarlo as mc, models, datasets
cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
calls = cc[cc.handle_min > 0]
edges = np.arange(8.0, 20.01, 0.5); mid = edges[:-1] + 0.25
def slot_rates(df, exclude_outage=True):
    counts = []
    for d, g in df.groupby("day"):
        c = pp.counts_in_windows(g.arrival_h.to_numpy(), edges).astype(float)
        if exclude_outage and d == 8:
            c[(edges[:-1] >= 13.0) & (edges[:-1] < 14.5)] = np.nan
        counts.append(c)
    counts = np.array(counts)
    return np.nansum(counts, axis=0) / np.sum(~np.isnan(counts), axis=0) / 0.5
lam_week = slot_rates(calls[calls.weekday != "Mon"])
h = calls.handle_min.to_numpy(); ln_mu, ln_sd = np.log(h).mean(), np.log(h).std(ddof=1); mean_h = h.mean() / 60
target, answer = 0.8, 20 / 3600
def agents_fn(schedule):
    return lambda t: int(schedule[min(max(int((t - 8.0) / 0.5), 0), len(schedule) - 1)])
def evaluate(schedule, lam, reps=30, seed0=0, patience=lambda g: g.exponential(3.0 / 60)):
    rate_fn = lambda t: float(np.interp(t, mid, lam)) if 8.0 <= t < 20.0 else 0.0
    out = []
    for k in range(reps):
        r = des.call_centre(rate_fn, 1.05 * lam.max(), lambda g: float(np.exp(ln_mu + ln_sd * g.standard_normal())) / 60, agents_fn(schedule), 21.0, R(seed0 + k), patience_sampler=patience, answer_within=answer)
        slot = np.clip(((r["arrival"] - 8.0) / 0.5).astype(int), 0, len(mid) - 1)
        ok = (~r["abandoned"]) & (np.nan_to_num(r["wait"], nan=1.0) <= answer)
        out.append(([ok[slot == j].mean() for j in range(len(mid))], ok.mean(), r["abandoned"].mean()))
    sl = np.array([o[0] for o in out])
    return {"slot_sl": sl.mean(axis=0), "slot_se": sl.std(axis=0, ddof=1) / np.sqrt(reps), "day_sl": float(np.mean([o[1] for o in out])), "day_ab": float(np.mean([o[2] for o in out]))}
def lag_schedule(lam):
    from scipy import stats
    tgrid = np.linspace(8.0, 20.0, 2401); dt = tgrid[1] - tgrid[0]
    lam_t = np.interp(tgrid, mid, lam)
    surv = stats.lognorm.sf(np.arange(0, 2.0, dt) * 60, ln_sd, scale=np.exp(ln_mu))
    load = np.convolve(lam_t, surv)[: len(tgrid)] * dt
    return np.array([qu.staffing(load[(tgrid >= t0) & (tgrid < t0 + 0.5)].max() / mean_h, 1 / mean_h, target, answer)["c"] for t0 in edges[:-1]])
'''


@solutions("19_project_staffing_a_call_centre", imports=["from scipy import stats"], setup=PROJECT_SETUP)
def _():
    return [
        [code(r"""lam_mon_direct = slot_rates(calls[calls.weekday == "Mon"])
week_days = calls[calls.weekday != "Mon"].groupby("day").size().drop(8)
factor = calls[calls.weekday == "Mon"].groupby("day").size().mean() / week_days.mean()
lam_mon_scaled = lam_week * factor
print(f"Monday factor {factor:.3f} (s.e. about {np.sqrt(1 / (4 * 1100) + 1 / (15 * 1080)):.3f})")
s_direct, s_scaled = lag_schedule(lam_mon_direct), lag_schedule(lam_mon_scaled)
print("direct  :", s_direct.tolist(), 0.5 * s_direct.sum(), "agent-hours")
print("scaled  :", s_scaled.tolist(), 0.5 * s_scaled.sum(), "agent-hours")
truth = models.call_centre_rate(mid) * models.CALL_CENTRE["monday_factor"]
for name, s in (("direct", s_direct), ("scaled", s_scaled)):
    ev = evaluate(s, truth, reps=30, seed0=500)
    print(f"{name}: tested against the true Monday profile - day {ev['day_sl']:.3f}, worst half hour {ev['slot_sl'].min():.3f}")"""),
         md(r"""
Four Mondays give half-hourly rates with standard errors near 15 %, so a schedule built on them inherits that
noise slot by slot; scaling the sixteen-day weekday profile by one well-estimated factor is far more precise.
Tested against the true Monday rate (known here because the data are synthetic) the scaled schedule is the
more reliable. Borrowing strength across days is the general lesson: estimate shape and level separately.
""")],
        [code(r"""sched = lag_schedule(lam_week)
truth = models.call_centre_rate(mid)
rows = []
for name, pat in (("exponential, mean 1 min", lambda g: g.exponential(1 / 60)), ("exponential, mean 3 min", lambda g: g.exponential(3 / 60)),
                  ("exponential, mean 10 min", lambda g: g.exponential(10 / 60)), ("deterministic 3 min", lambda g: 3 / 60)):
    ev = evaluate(sched, truth, reps=20, seed0=700, patience=pat)
    rows.append((name, ev["day_sl"], ev["slot_sl"].min(), ev["day_ab"]))
print(pd.DataFrame(rows, columns=["patience", "service level", "worst half hour", "abandonment"]).to_string(index=False, float_format="%.3f"))"""),
         md(r"""
Patience changes abandonment by a factor of several but the service level only modestly: impatient callers
leave, which shortens the queue for everyone else, so short patience *raises* the measured service level.
The recommendation (based on 20-second answering) is therefore robust, while any abandonment target would
need patience data - which the logs, recording only completed calls, do not contain.
""")],
        [code(r"""need = lag_schedule(lam_week)                        # required agents per half hour from 08:00
starts = np.arange(8.0, 16.01, 1.0)                    # 4-hour shifts starting on the hour, ending by 20:00
cover = np.array([[1 if s <= t0 < s + 4 else 0 for s in starts] for t0 in edges[:-1]])
shifts = np.zeros(len(starts), dtype=int)
while np.any(cover @ shifts < need):                   # greedy: add the shift covering the largest unmet need
    deficit = np.maximum(need - cover @ shifts, 0)
    shifts[np.argmax(cover.T @ deficit)] += 1
improved = True
while improved:                                        # drop redundant shifts
    improved = False
    for j in range(len(starts)):
        if shifts[j] > 0:
            shifts[j] -= 1
            if np.all(cover @ shifts >= need): improved = True
            else: shifts[j] += 1
print("shifts starting at", starts.tolist(), ":", shifts.tolist())
cov = cover @ shifts
print(f"agent-hours: shifts {4 * shifts.sum()}, required {0.5 * need.sum():.1f}; overstaffing in {np.sum(cov > need)} of {len(need)} half hours")
ev = evaluate(cov, models.call_centre_rate(mid), reps=40, seed0=900)
print(f"shift schedule on 40 fresh days: service level {ev['day_sl']:.3f}, worst half hour {ev['slot_sl'].min():.3f}")"""),
         md(r"""
Covering a half-hourly requirement with 4-hour shifts is a set-covering integer program; a greedy heuristic
(add the shift that covers the most unmet need, then remove redundant ones) gets close to optimal for small
instances like this. The shift structure costs extra agent-hours - the requirement curve has a sharp morning
peak that 4-hour blocks cannot follow - and the extra cover shows up as a higher service level. Split shifts or
part-timers are how real centres buy back the difference.
""")],
    ]


if __name__ == "__main__":
    write_all(only=[a for a in sys.argv[1:] if not a.startswith("--")])
