"""Reproduce paper Figs. 4/5/6.

Usage:
    uv run python experiments/run_paper_figures.py            # default (M sweep)
    uv run python experiments/run_paper_figures.py --alpha     # alpha sweep
    uv run python experiments/run_paper_figures.py --all       # both
    uv run python experiments/run_paper_figures.py --outdir outputs
"""

import argparse
import json
from pathlib import Path

from drying.experiment import (
    alpha_scenarios,
    plot_cost_vs_alpha,
    plot_cost_vs_stage,
    plot_temperature_profile,
    stage_count_scenarios,
)
from drying.optimization import MepConfig, MepResult, solve


def _run_scenarios(scenarios, outdir: Path) -> list[MepResult]:
    results: list[MepResult] = []
    for sc in scenarios:
        cfg = MepConfig(
            m=sc.m,
            x0_wet=sc.x0_wet,
            xd_wet=sc.xd_wet,
            alpha=sc.alpha,
            p_us=sc.p_us,
            t_ambient_c=sc.t_ambient_c,
        )
        res = solve(cfg)
        print(f"[{sc.name}] path={res.path} "
              f"cost={res.costs[int(res.probs.argmax())]:.4f} "
              f"max_p={res.probs.max():.4f} converged={res.converged}")
        results.append(res)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce ACC2023 drying paper figures")
    parser.add_argument("--all", action="store_true", help="run both M and alpha sweeps")
    parser.add_argument("--alpha", action="store_true", help="run alpha sweep (Fig. 5)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    outdir: Path = args.outdir
    run_m = args.all or not args.alpha
    run_a = args.all or args.alpha

    payload: dict = {}
    if run_m:
        print("== stage count sweep (Fig. 4/6) ==")
        results = _run_scenarios(stage_count_scenarios(), outdir)
        plot_cost_vs_stage(results, outdir)
        # like paper Fig. 4(a)-(d): temperature profile for every stage count
        for r in results:
            plot_temperature_profile(r, outdir)
        payload["stage_sweep"] = {
            str(len(r.path)): {
                "cost": float(r.costs[int(r.probs.argmax())]),
                "path": list(r.path),
                "etas": [{"t": e.t, "T": e.T} for e in r.etas],
                "converged": r.converged,
            }
            for r in results
        }

    if run_a:
        print("== alpha sweep (Fig. 5) ==")
        scenarios = alpha_scenarios()
        results = _run_scenarios(scenarios, outdir)
        alphas = [sc.alpha for sc in scenarios]
        plot_cost_vs_alpha(results, alphas, outdir)
        payload["alpha_sweep"] = {
            sc.name: {
                "cost": float(r.costs[int(r.probs.argmax())]),
                "path": list(r.path),
            }
            for sc, r in zip(scenarios, results)
        }

    (outdir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results/figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()