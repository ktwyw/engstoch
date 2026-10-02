"""README images: the hero animation and the gallery, computed with engstoch. Run: python tools/make_images.py"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engstoch import brownian as bm  # noqa: E402
from engstoch import (  # noqa: E402
    ctmc,
    datasets,
    mcmc,  # noqa: E402
    models,
    qmc,
    sde,
)
from engstoch import poisson as pp  # noqa: E402
from engstoch import rng as rg  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def hero():
    """Brownian paths growing, with the reflection-principle law of the running maximum filling in."""
    rng = np.random.default_rng(7)
    n, n_paths = 400, 400
    t = np.linspace(0, 1, n + 1)
    W = bm.paths(1.0, n, n_paths, rng)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=80, gridspec_kw={"width_ratios": [2, 1]})
    lines = [ax.plot([], [], lw=0.7, color=f"C{k % 10}", alpha=0.8)[0] for k in range(12)]
    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_title("Brownian paths and the level a = 1")
    ax.set_xlabel("t")
    a_grid = np.linspace(0, 3, 200)
    ax2.plot(a_grid, bm.max_cdf(a_grid, 1.0), "k", lw=1.5, label="2(1 - Phi(a)): reflection")
    (emp,) = ax2.plot([], [], "C3", lw=1.5, label="paths simulated so far")
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("a")
    ax2.set_title("P(max W >= a)")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    frames = 60

    def update(f):
        k = int((f + 1) / frames * n)
        for j, ln in enumerate(lines):
            ln.set_data(t[: k + 1], W[j, : k + 1])
        m = int((f + 1) / frames * n_paths)
        mx = W[:m].max(axis=1)
        emp.set_data(a_grid, [np.mean(mx >= a) for a in a_grid])
        return lines + [emp]

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(OUT / "hero.gif", writer=PillowWriter(fps=12))
    plt.close(fig)


def randu():
    u = rg.LCG.randu(1).random(30_000).reshape(-1, 3)
    p = rg.LCG.park_miller(1).random(30_000).reshape(-1, 3)
    fig = plt.figure(figsize=(8, 3.6))
    for i, (name, x) in enumerate((("RANDU", u), ("Park-Miller", p))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(*x[:4000].T, s=0.5)
        ax.view_init(elev=10, azim=-58)
        ax.set_title(f"{name}: consecutive triples")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.tight_layout()
    fig.savefig(OUT / "randu.png")
    plt.close(fig)


def qmc_points():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
    for ax, (name, x) in zip(axes, (("random", np.random.default_rng(1).random((512, 2))), ("Sobol", qmc.sobol(512, 2)))):
        ax.scatter(*x.T, s=3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}: L2* = {qmc.l2_star_discrepancy(x):.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "qmc.png")
    plt.close(fig)


def sir():
    s = models.sir(N=500, beta=0.3, gamma=0.1, I0=3)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for k in range(30):
        r = ctmc.gillespie(s["x0"], s["stoich"], s["propensity"], 200.0, np.random.default_rng(k))
        ax.step(r["t"], r["x"][:, 1], where="post", lw=0.7, alpha=0.7)
    ax.set_xlabel("days")
    ax.set_ylabel("infected")
    ax.set_title("30 stochastic SIR epidemics (Gillespie)")
    fig.tight_layout()
    fig.savefig(OUT / "sir.png")
    plt.close(fig)


def strong_order():
    a, b = (lambda t, x: 0.1 * x), (lambda t, x: 0.5 * x)
    ex = lambda t, W: np.exp((0.1 - 0.125) * t + 0.5 * W)  # noqa: E731
    steps = [8, 16, 32, 64, 128, 256, 512]
    e = sde.strong_order(a, b, ex, 1.0, 1.0, steps, 3000, np.random.default_rng(2))
    m = sde.strong_order(a, b, ex, 1.0, 1.0, steps, 3000, np.random.default_rng(2), "milstein", lambda t, x: 0.5 + 0 * x)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.loglog(e["h"], e["error"], "o-", label=f"Euler-Maruyama, slope {e['order']:.2f}")
    ax.loglog(m["h"], m["error"], "s-", label=f"Milstein, slope {m['order']:.2f}")
    ax.set_xlabel("step h")
    ax.set_title("strong error on geometric Brownian motion")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sde_order.png")
    plt.close(fig)


def ising():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for ax, b in zip(axes, (0.35, mcmc.BETA_C, 0.5)):
        r = mcmc.ising(96, b, 400, np.random.default_rng(3), "heatbath", start="hot")
        ax.imshow(r["spins"], cmap="gray", interpolation="nearest")
        ax.set_title(f"beta = {b:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Ising model by heat-bath MCMC: below, at and above beta_c", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ising.png")
    plt.close(fig)


def call_centre():
    cc = pd.read_csv(datasets.path("call_centre.csv"), comment="#")
    edges = np.arange(8.0, 20.01, 0.5)
    days = [g.arrival_h.to_numpy() for d, g in cc[(cc.weekday != "Mon") & (cc.day != 8)].groupby("day")]
    ie = pp.intensity_estimate(days, edges)
    fig, ax = plt.subplots(figsize=(6, 3.3))
    ax.step(edges[:-1], ie["rate"], where="post", label="estimated from 15 days of logs")
    ax.fill_between(edges[:-1], ie["rate"] - 2 * ie["std_error"], ie["rate"] + 2 * ie["std_error"], step="post", alpha=0.25)
    tt = np.linspace(8, 20, 300)
    ax.plot(tt, models.call_centre_rate(tt), "k--", lw=1, label="generating profile")
    ax.set_xlabel("hour")
    ax.set_ylabel("calls per hour")
    ax.set_title("project: the arrival rate behind a staffing plan")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "call_centre.png")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    randu()
    qmc_points()
    sir()
    strong_order()
    ising()
    call_centre()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:<18} {f.stat().st_size / 1024:7.1f} kB")
