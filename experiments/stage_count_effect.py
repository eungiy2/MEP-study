"""Fig. 4/6 -- effect of the number of allowed stages M.

Usage:  uv run python experiments/stage_count_effect.py [--outdir outputs]
"""

import argparse
import json
from pathlib import Path

from drying.experiment import plot_cost_vs_stage, stage_count_scenarios
from drying.optimization import MepConfig, solve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    results = []
    for sc in stage_count_scenarios():
        cfg = MepConfig(
            m=sc.m, x0_wet=sc.x0_wet, xd_wet=sc.xd_wet,
            alpha=sc.alpha, p_us=sc.p_us, t_ambient_c=sc.t_ambient_c,
        )
        res = solve(cfg)
        cost = float(res.costs[int(res.probs.argmax())])
        print(f"M={sc.m}: cost={cost:.4f}, path={res.path}, max_p={res.probs.max():.3f}")
        results.append(res)

    plot_cost_vs_stage(results, args.outdir)
    payload = {
        f"M={len(r.path)}": {
            "cost": float(r.costs[int(r.probs.argmax())]),
            "path": list(r.path),
            "converged": r.converged,
        }
        for r in results
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "stage_effect.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()