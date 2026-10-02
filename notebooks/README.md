# Course notebooks

Twenty executed notebooks on stochastic processes and simulation, built on one working method: model the
randomness, compute what the model implies exactly where possible, simulate where not, and treat every
simulation as a statistical experiment - with intervals, checks against theory, and decisions reassessed on
fresh random numbers. Every notebook states its learning objectives, prerequisites and study time, rebuilds a
key algorithm in plain Python ("Inside the algorithm", checked against the library), and ends with exercises
including an "Implement it yourself" task. Worked solutions are in [`../solutions`](../solutions).

| # | Notebook | Time | Before you start | Solutions |
|---|---|---|---|---|
| [00](00_randomness_and_generators.ipynb) | Randomness and pseudo-random generators | 60 min | basic probability (uniform distribution, independence), Python and NumPy | [solutions](../solutions/00_solutions.ipynb) |
| [01](01_generating_random_variates.ipynb) | Generating random variates | 75 min | notebook 00; distribution and density functions | [solutions](../solutions/01_solutions.ipynb) |
| [02](02_monte_carlo_estimation.ipynb) | Monte Carlo estimation and its error | 60 min | notebooks 00-01; the central limit theorem | [solutions](../solutions/02_solutions.ipynb) |
| [03](03_variance_reduction.ipynb) | Variance reduction | 90 min | notebook 02 | [solutions](../solutions/03_solutions.ipynb) |
| [04](04_quasi_monte_carlo.ipynb) | Quasi-Monte Carlo | 60 min | notebooks 02-03 | [solutions](../solutions/04_solutions.ipynb) |
| [05](05_discrete_time_markov_chains.ipynb) | Discrete-time Markov chains | 75 min | notebook 00; matrices and linear systems | [solutions](../solutions/05_solutions.ipynb) |
| [06](06_long_run_behaviour_and_mixing.ipynb) | Long-run behaviour and mixing | 75 min | notebook 05 | [solutions](../solutions/06_solutions.ipynb) |
| [07](07_absorption_hitting_and_random_walks.ipynb) | Absorption, hitting times and random walks | 75 min | notebooks 05-06 | [solutions](../solutions/07_solutions.ipynb) |
| [08](08_branching_processes.ipynb) | Branching processes | 60 min | notebook 05; probability generating functions | [solutions](../solutions/08_solutions.ipynb) |
| [09](09_the_poisson_process.ipynb) | The Poisson process | 75 min | notebooks 01-02; the exponential and Poisson distributions | [solutions](../solutions/09_solutions.ipynb) |
| [10](10_renewal_processes.ipynb) | Renewal processes | 60 min | notebook 09 | [solutions](../solutions/10_solutions.ipynb) |
| [11](11_continuous_time_markov_chains.ipynb) | Continuous-time Markov chains | 90 min | notebooks 05-06 and 09 | [solutions](../solutions/11_solutions.ipynb) |
| [12](12_queueing_theory.ipynb) | Queueing theory | 75 min | notebooks 09-11 | [solutions](../solutions/12_solutions.ipynb) |
| [13](13_discrete_event_simulation.ipynb) | Discrete-event simulation | 90 min | notebooks 09-12 | [solutions](../solutions/13_solutions.ipynb) |
| [14](14_output_analysis.ipynb) | Output analysis | 60 min | notebooks 02 and 13 | [solutions](../solutions/14_solutions.ipynb) |
| [15](15_brownian_motion.ipynb) | Brownian motion | 75 min | notebooks 07 and 09; the normal distribution | [solutions](../solutions/15_solutions.ipynb) |
| [16](16_stochastic_differential_equations.ipynb) | Stochastic differential equations | 90 min | notebook 15; ordinary differential equations | [solutions](../solutions/16_solutions.ipynb) |
| [17](17_gaussian_processes_and_time_series.ipynb) | Gaussian processes and stationary time series | 90 min | notebooks 02 and 15; linear algebra | [solutions](../solutions/17_solutions.ipynb) |
| [18](18_markov_chain_monte_carlo.ipynb) | Markov chain Monte Carlo | 90 min | notebooks 06 and 14 | [solutions](../solutions/18_solutions.ipynb) |
| [19](19_project_staffing_a_call_centre.ipynb) | Project: staffing a call centre from data | 4 h | the whole course | [solutions](../solutions/19_solutions.ipynb) |

**Suggested order:** simulation foundations (00-04: generators, variates, Monte Carlo, variance reduction, QMC),
Markov chains and their relatives (05-08), point processes (09-10), continuous time, queues and simulation
studies (11-14), continuous paths (15-17), MCMC (18) and the project (19), which draws on most of them.

The notebooks are generated from `build_notebooks.py` (`python notebooks/build_notebooks.py` rebuilds and executes
them; pass notebook numbers to rebuild only some). They open in Google Colab, where the first cell installs
`engstoch` from GitHub.
