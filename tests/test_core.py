import math

import numpy as np
import pytest

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


def R(seed):
    return np.random.default_rng(seed)


def test_special_edges():
    assert sf.norm_ppf(0.0) == -math.inf and sf.norm_ppf(1.0) == math.inf
    with pytest.raises(ValueError):
        sf.norm_ppf(1.5)
    assert sf.gammainc(2.0, 0.0) == 0.0 and sf.gammaincc(2.0, 0.0) == 1.0
    assert sf.betainc(2.0, 3.0, 0.0) == 0.0 and sf.betainc(2.0, 3.0, 1.0) == 1.0
    assert sf.t_ppf(0.5, 4.0) == 0.0 and sf.t_ppf(0.975, 1e9) == pytest.approx(1.959963984540054)
    assert sf.poisson_cdf(-1, 2.0) == 0.0
    assert sf.kolmogorov_sf(0.0) == 1.0 and 0 < sf.kolmogorov_sf(0.1) <= 1
    assert sf.chi2_ppf(0.5, 2.0) == pytest.approx(2 * math.log(2), rel=1e-10)


def test_generators():
    g = rg.LCG(5, 3, 16)
    assert isinstance(g.random(), float) and len(g.random(4)) == 4
    assert rg.LCG(2, 0, 16, seed=1).period(limit=100) is None
    assert not rg.full_period_conditions(5, 0, 16) and not rg.full_period_conditions(5, 2, 16)
    assert rg.full_period_conditions(4, 1, 9)  # m = 9: a - 1 divisible by 3, m not divisible by 4
    assert 0 <= rg.XorShift64Star(0).random() < 1 and 0 <= rg.PCG32().random() < 1
    assert len(rg.PCG32().random(3)) == 3


def test_variates():
    g = R(0)
    with pytest.raises(ValueError):
        rg.rejection(lambda x: 2 * x, lambda m: g.random(m), lambda x: np.ones_like(x), 1.0, g, 10)
    assert rg.discrete_inverse([1, 1], g, 5).max() <= 1
    prob, alias = rg.alias_table([1.0])
    assert prob[0] == 1.0 and alias[0] == 0
    assert np.all(rg.poisson(0.0, g, 10) == 0)
    assert np.all(rg.binomial(5, 1.0, g, 10) == 5) and np.all(rg.binomial(5, 0.0, g, 10) == 0)
    assert rg.multivariate_normal([0, 0], np.eye(2), g.standard_normal, 3).shape == (3, 2)
    assert rg.inverse_transform(lambda u: u * 2, g, 4).max() < 2
    assert rg.weibull(2.0, 1.0, g, 5).min() > 0 and rg.exponential(1.0, g, 5).min() > 0


def test_rngtests():
    u = R(1).random(5000)
    assert set(rt.runs_up_test(u)) >= {"statistic", "counts", "p_value"}
    assert rt.autocorrelation_test(u, lag=2)["p_value"] > 0
    assert rt.gap_test(u)["df"] == 10
    s = rt.spectral_test_2d(3, 31)
    assert s["nu"] > 0 and 0 < s["merit"] <= 1.0001


def test_montecarlo():
    g = R(2)
    r = mc.mean_ci(g.random(100), method="t")
    assert r["ci"][0] < r["estimate"] < r["ci"][1]
    assert mc.relative_error({"estimate": 0.0, "std_error": 1.0}) == math.inf
    cv = mc.control_variate(g.random(100), g.random((100, 2)), [0.5, 0.5])
    assert np.shape(cv["beta"]) == (2,)
    a = mc.antithetic(lambda u: np.sum(u, axis=1), 100, g, dim=2)
    assert a["estimate"] == pytest.approx(1.0, abs=1e-12)
    isn = mc.importance_sampling(lambda x: x, lambda x: -0.5 * x**2, lambda n: g.normal(size=n), lambda x: -0.5 * x**2, 1000)
    assert isn["ess"] == pytest.approx(1000)
    assert mc.exponential_tilting_normal(1.0)(0.0) == pytest.approx(0.5)
    ny = mc.stratified(lambda u: np.zeros_like(u), 100, 4, g, "neyman")
    assert ny["estimate"] == 0.0


def test_qmc():
    with pytest.raises(ValueError):
        qmc.sobol(8, 11)
    assert qmc.sobol(4, 2, skip_zero=True).shape == (4, 2)
    h = qmc.halton(10, 3, rng=R(3), scramble="permute")
    assert h.shape == (10, 3) and h.min() >= 0 and h.max() < 1
    assert qmc.radical_inverse(0).tolist() == [0.0]


def test_markov_edges():
    with pytest.raises(ValueError):
        mk.MarkovChain([[0.5, 0.6], [0.5, 0.5]])
    with pytest.raises(ValueError):
        mk.MarkovChain([[1.0, 0.0]])
    c = mk.MarkovChain([[0, 1], [1, 0]], ["a", "b"])
    assert "MarkovChain" in repr(c) and c.labels([0, 1]) == ["a", "b"]
    assert c.distribution([1, 0], 3).tolist() == [0.0, 1.0]
    with pytest.raises(RuntimeError):
        c.mixing_time(0.1, k_max=50)
    assert mk.MarkovChain([[0.9, 0.1], [0.5, 0.5]]).stationary_power(max_iter=3)["iterations"] == 3
    k = mk.MarkovChain([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]]).expected_hitting_times([0])
    assert k[0] == 0 and math.isinf(k[1])
    single = mk.MarkovChain([[1.0]])
    assert single.classify()["classes"][0]["period"] == 1
    assert mk.estimate([0, 0, 1], 3)["P"][2].tolist() == pytest.approx([1 / 3] * 3)


def test_walks_and_branching():
    assert wk.paths_count(3, 0) == 0 and wk.paths_count(2, 4) == 0
    assert wk.lattice_walk(5, 2, R(4), 3).shape == (3, 6, 2)
    assert wk.total_progeny_mean(wk.pgf_poisson(1.5)) == math.inf
    assert wk.generation_moments(wk.pgf_poisson(1.0), 4)["var"] == pytest.approx(4.0)
    d = wk.pgf_discrete([0.2, 0.5, 0.3])
    assert d["mean"] == pytest.approx(1.1) and float(d["pmf"](5)) == 0.0
    assert wk.extinction_probability(wk.pgf_poisson(1.0), max_iter=10)["iterations"] == 10
    assert float(wk.pgf_geometric(0.5)["pmf"](0)) == 0.5 and float(wk.pgf_poisson(2.0)["pmf"](0)) == pytest.approx(math.exp(-2))
    z = wk.simulate_galton_watson(lambda k, r: np.full(k, 3), 12, R(5), cap=100)
    assert z[-1] == z[-2] > 100


def test_ctmc_edges():
    with pytest.raises(ValueError):
        ctmc.CTMC([[-1.0, 2.0], [1.0, -1.0]])
    c = ctmc.CTMC([[0.0, 0.0], [1.0, -1.0]])
    assert "CTMC" in repr(c) and c.jump_chain().P[0, 0] == 1.0
    assert c.simulate(10.0, 0, R(6))["states"].tolist() == [0]
    assert c.distribution([0, 1], 50.0)[0] == pytest.approx(1.0)
    assert ctmc.expm(np.zeros((2, 2))).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    t = ctmc.tau_leap([1], [[-5]], lambda x: np.array([10.0]), 1.0, 0.5, R(7))
    assert t["clipped_steps"] >= 1 and t["x"].min() >= 0
    big = ctmc.CTMC([[-2000.0, 2000.0], [1.0, -1.0]]).uniformisation(5.0)
    assert np.allclose(big["P"].sum(axis=1), 1.0)
    g = ctmc.gillespie([0], [[1]], lambda x: np.array([0.0]), 1.0, R(8))
    assert len(g["t"]) == 1


def test_poisson_renewal():
    with pytest.raises(ValueError):
        pp.thinning(lambda t: 5.0 + 0 * t, 1.0, 100.0, R(9))
    assert pp.rate_mle(0, 1.0)["ci"][0] == 0.0
    assert len(pp.superpose([1.0, 3.0], [2.0])) == 3
    assert pp.compound(1.0, 0.0, lambda n, r: r.random(n), R(10))["total"] == 0.0
    t, m = renewal.renewal_function(lambda x: 1 - np.exp(-x), 1.0, 10)
    assert m[0] == 0.0
    assert renewal.availability(9.0, 1.0) == 0.9
    assert renewal.weibull_moments(1.0, 2.0)["mean"] == pytest.approx(2.0)
    assert len(renewal.count_at(lambda n, r: r.exponential(1.0, n), 2.0, 5, R(11))) == 5
    assert renewal.equilibrium_cdf(0.0, lambda s: np.exp(-s), 1.0)[0] == 0.0


def test_queues_edges():
    with pytest.raises(ValueError):
        queues.mm1(1.0, 1.0)
    with pytest.raises(ValueError):
        queues.mmc(3.0, 1.0, 3)
    with pytest.raises(ValueError):
        queues.mg1(1.0, 1.0, 2.0)
    assert queues.erlang_c(2, 3.0) == 1.0 and queues.service_level(3.0, 1.0, 2, 1.0) == 0.0
    assert queues.little(lam=2.0, W=3.0) == 6.0 and queues.little(L=6.0, W=3.0) == 2.0 and queues.little(L=6.0, lam=2.0) == 3.0
    assert queues.mm_inf(2.0, 1.0)["L"] == 2.0
    assert queues.allen_cunneen(0.5, 1.0, 1, 1.0, 1.0) == pytest.approx(queues.mm1(0.5, 1.0)["Wq"])
    assert queues.time_average([0.0, 1.0], [2.0, 4.0], 2.0) == 3.0


def test_des_engine():
    sim = des.Simulator()
    sim.schedule(1.0, lambda: None)
    sim.run(until=0.5)
    assert sim.now == 0.5 and sim.n_events == 0
    with pytest.raises(ValueError):
        sim.schedule(0.1, lambda: None)
    t = des.Tally()
    assert math.isnan(t.mean()) and math.isnan(t.var())
    for v in (1.0, 2.0, 3.0):
        t.add(v)
    assert t.mean() == 2.0 and t.var() == 1.0
    tw = des.TimeWeighted(des.Simulator())
    assert tw.mean() == 0.0
    r = des.inventory_sS(2, 6, 1.0, lambda g: 1, lambda g: 0.5, 50.0, R(12), review=1.0)
    assert r["order_rate"] > 0
    cc = des.call_centre(lambda t: 2.0, 2.0, lambda g: 0.1, lambda t: 1 if t < 5 else 2, 10.0, R(13))
    assert 0 <= cc["service_level"] <= 1


def test_output_edges():
    x = R(14).normal(size=2000)
    assert oa.batch_means(x, 3)["lag1_corr"] != oa.batch_means(x, 3)["lag1_corr"]  # nan for few batches
    assert oa.coverage([0, 0], [1, 1], 0.5)["coverage"] == 1.0
    assert len(oa.welch_moving_average(np.ones((2, 10)), 2)) == 8
    assert oa.ess(x) > 1000 and oa.replications([1.0, 2.0, 3.0])["estimate"] == 2.0


def test_brownian_sde_gaussian():
    g = R(15)
    assert bm.donsker(10, 2, g).shape == (2, 11)
    assert bm.arcsine_positive_time(np.zeros((1, 5)))[0] == 0.0
    assert bm.first_passage_simulated(100.0, 1.0, 10, 5, g).tolist() == [np.inf] * 5
    em = sde.euler_maruyama(lambda t, x: -x, lambda t, x: np.zeros((len(x), 2, 2)), [1.0, 1.0], 1.0, 10, 3, g)
    assert em["X"].shape == (3, 11, 2)
    m = sde.milstein(lambda t, x: 0 * x, lambda t, x: 0 * x, lambda t, x: 0 * x, [1.0, 2.0], 1.0, 4, 2, g)
    assert m["X"].shape == (2, 5, 2)
    assert sde.heun_stratonovich(lambda t, x: 0 * x, lambda t, x: 0 * x, [1.0, 2.0], 1.0, 4, 2, g)["X"].shape == (2, 5, 2)
    assert gp.periodic(0.0, 1.0, period=1.0) == pytest.approx(1.0)
    assert gp.fbm(0.0, 1.0, 0.5) == 0.0
    s = gp.sample_cholesky(gp.exponential, [0.0, 1.0], 2, g, jitter=1e-9, mean=[1.0, 1.0])
    assert s["X"].shape == (2, 2)
    with pytest.raises(ValueError):
        gp.circulant_embedding(gp.squared_exponential(0, np.linspace(0, 1, 50), ell=1.0), 1, g)


def test_timeseries_mcmc_models():
    assert ts.is_causal([]) and ts.is_invertible([]) and not ts.is_invertible([2.0])
    assert ts.psi_weights([0.5], [], 3).tolist() == [1.0, 0.5, 0.25]
    assert ts.acf_bands(100) == pytest.approx(0.196, abs=1e-3)
    x = ts.simulate_arma([0.5], [], 300, R(16), mean=10.0)
    assert abs(x.mean() - 10.0) < 1.0
    assert len(ts.sample_pacf(x, 3)) == 4 and len(ts.periodogram(np.arange(7.0))["psd"]) == 4
    c = mcmc.gibbs([lambda x, r: 1.0], [0.0], 3, R(17))
    assert c.tolist() == [[1.0]] * 3
    sa = mcmc.simulated_annealing(lambda v: (v - 3) ** 2, 0, lambda v, r: v + r.choice([-1, 1]), 2000, 5.0, R(18))
    assert sa["x"] == 3
    assert mcmc.onsager_magnetisation(0.3) == 0.0
    assert mcmc.ising(4, 0.4, 2, R(19), h=0.1)["spins"].shape == (4, 4)
    assert models.mmck_chain(1.0, 1.0, 1, 3)["Q"].shape == (4, 4)
    assert models.sir()["R0"] == pytest.approx(3.0)
    assert models.lotka_volterra()["propensity"](np.array([1, 1])).shape == (3,)
    assert np.all(models.call_centre_rate(np.array([7.0, 21.0])) == 0)


def test_datasets():
    assert datasets.available() == sorted(datasets.INFO)
    assert "call" in datasets.describe("call_centre.csv")
    with pytest.raises(KeyError):
        datasets.path("missing.csv")
    assert datasets.path("weather.csv").is_file()
