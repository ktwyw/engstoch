"""engstoch: stochastic processes and simulation from scratch.

special      normal, gamma, beta, chi-square, t, Poisson and Kolmogorov distribution functions
rng          LCG/RANDU, xorshift64*, PCG32; inversion, rejection, Box-Muller, alias tables, gamma, copulas
rngtests     chi-square, KS, serial, runs, gap tests; the spectral test; RANDU's planes
montecarlo   Monte Carlo estimates and intervals; antithetic, control variates, importance and stratified sampling
qmc          van der Corput, Halton, Sobol, lattice rules, discrepancy, randomised QMC
markov       discrete-time Markov chains: classes, stationarity, absorption, passage times, mixing, estimation
walks        random walks, gambler's ruin, reflection, Polya recurrence, Galton-Watson branching
poisson      homogeneous, non-homogeneous, compound and spatial Poisson processes; rate inference
renewal      renewal function, renewal-reward, inspection paradox, age replacement, availability
ctmc         continuous-time chains: exp(Qt), uniformisation, birth-death, estimation, Gillespie, tau-leaping
queues       M/M/1, M/M/c (Erlang B/C), M/G/1, approximations, staffing, Jackson networks, Lindley, G/G/c
des          event-scheduling simulation engine; call centre, (s, S) inventory, machine repair
output       autocorrelation time, batch means, MSER, regenerative method, replications, R-hat
brownian     Brownian motion, bridge, Levy construction, GBM, OU, reflection, first passage, barriers
sde          Euler-Maruyama, Milstein, Heun, strong and weak order, CIR, multilevel Monte Carlo
gaussian     covariance kernels, Cholesky and circulant-embedding sampling, fBM, GP conditioning
timeseries   ARMA simulation, ACF/PACF, Durbin-Levinson, Yule-Walker, Ljung-Box, periodogram, Welch
mcmc         Metropolis, independence sampler, Gibbs, HMC, Ising model, simulated annealing
models       the course's example chains, queues, reaction networks and the call-centre profile
datasets     the course data files (call centre, weather, machine log, prices)
"""

from . import (
    brownian,
    ctmc,
    datasets,
    des,
    gaussian,
    markov,
    mcmc,
    models,
    montecarlo,
    output,
    poisson,
    qmc,
    queues,
    renewal,
    rng,
    rngtests,
    sde,
    special,
    timeseries,
    walks,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "brownian",
    "ctmc",
    "datasets",
    "des",
    "gaussian",
    "markov",
    "mcmc",
    "models",
    "montecarlo",
    "output",
    "poisson",
    "qmc",
    "queues",
    "renewal",
    "rng",
    "rngtests",
    "sde",
    "special",
    "timeseries",
    "walks",
]
