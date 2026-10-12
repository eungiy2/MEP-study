"""Experiment result visualization -- reproducing the style of paper Figs. 4/5/6.

Outputs are saved to `outputs/` (or the current directory if unspecified).
"""

from pathlib import Path

import matplotlib.pyplot as plt

from ..optimization.mep import MepResult


def _save(fig, name: str, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost_vs_stage(results: list[MepResult], outdir: str | Path = "outputs") -> None:
    """Fig. 6 -- optimal cost versus the number of allowed stages"""
    ms = [len(r.path) for r in results]
    # use the cost of the optimal path (argmax p)
    costs = [float(r.costs[int(r.probs.argmax())]) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ms, costs, "o-")
    ax.set_xlabel("number of stages M")
    ax.set_ylabel("optimal process cost")
    ax.grid(alpha=0.3)
    _save(fig, "fig6_cost_vs_stage", outdir)


def plot_cost_vs_alpha(results: list[MepResult], alphas: list[float], outdir: str | Path = "outputs") -> None:
    """Fig. 5 -- optimal path/cost versus the relative HA cost weight alpha"""
    costs = [float(r.costs[int(r.probs.argmax())]) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, costs, "s-")
    ax.set_xlabel(r"$\alpha$ (relative HA cost weight)")
    ax.set_ylabel("optimal process cost")
    ax.grid(alpha=0.3)
    _save(fig, "fig5_cost_vs_alpha", outdir)


def plot_temperature_profile(result: MepResult, outdir: str | Path = "outputs") -> None:
    """Fig. 4(e) -- temperature profile of the optimal sequence"""
    etas = result.etas
    t_ticks = [0.0]
    for e in etas:
        t_ticks.append(t_ticks[-1] + e.t)
    temps = [e.T for e in etas] + [etas[-1].T]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(t_ticks, temps, where="post")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("air temperature [°C]")
    ax.set_ylim(20, 80)
    ax.set_title(f"optimal path: {'->'.join(map(str, result.path))}")
    ax.grid(alpha=0.3)
    _save(fig, "fig4e_temperature_profile", outdir)


def plot_beta_history(result: MepResult, outdir: str | Path = "outputs") -> None:
    """Convergence curve -- free energy F* per iteration"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.history)
    ax.set_xlabel("homotopy iteration")
    ax.set_ylabel("free energy F*")
    ax.grid(alpha=0.3)
    _save(fig, "convergence_Fstar", outdir)