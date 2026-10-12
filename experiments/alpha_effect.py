"""Fig. 5 -- effect of the relative cost weight alpha of the HA process.

Usage:  uv run python experiments/alpha_effect.py [--outdir outputs]
"""

import argparse
import json
from pathlib import Path

from drying.experiment import alpha_scenarios, plot_cost_vs_alpha
from drying.optimization import MepConfig, solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    scenarios = alpha_scenarios()
    results = []
    for sc in scenarios:
        cfg = MepConfig(
            m=sc.m, x0_wet=sc.x0_wet, xd_wet=sc.xd_wet,
            alpha=sc.alpha, p_us=sc.p_us, t_ambient_c=sc.t_ambient_c,
        )
        res = solve(cfg)
        cost = float(res.costs[int(res.probs.argmax())])
        print(f"{sc.name}: cost={cost:.4f}, path={res.path}, max_p={res.probs.max():.3f}")
        results.append(res)

    plot_cost_vs_alpha(results, [sc.alpha for sc in scenarios], args.outdir)
    payload = {
        sc.name: {
            "alpha": sc.alpha,
            "cost": float(r.costs[int(r.probs.argmax())]),
            "path": list(r.path),
        }
        for sc, r in zip(scenarios, results)
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "alpha_effect.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()