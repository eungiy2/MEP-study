"""Algorithm 1 -- MEP-based combinatorial optimization main loop.

Homotopy strategy (Section III, Algorithm 1):

- Start beta at beta_min and multiply it by zeta each iteration.
- Near beta ~ 0, the entropy term dominates the free energy F*, making it
  **convex** -> the global minimum is tracked reliably.
- As beta grows, the objective transitions to the original non-convex cost
  function, warm-starting each inner solve from the previous local solution.
- The loop stops when max_omega p(omega) >= 1 - eps: the distribution has
  converged to {0,1}, restoring the discrete path choice.

Each iteration alternate-minimizes (i) the parameter optimization (inner)
and (ii) the weight recomputation, so Fbar decreases monotonically, which
guarantees convergence.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..model.cost import ControlParams, path_cost
from ..model.paths import Path, enumerate_paths
from .gibbs import free_energy, gibbs_probs
from .solver import solve_inner


@dataclass(frozen=True)
class MepConfig:
    """Algorithm 1 execution settings.

    Parameters
    ----------
    m:              maximum number of allowed stages M
    x0_wet, xd_wet: initial / target wet-basis moisture content
    alpha:          relative cost weight of the HA process alpha (Eq. 16)
    p_us:           ultrasound transducer power consumption [W]
    t_ambient_c:    ambient air temperature [C]
    t_min:          minimum residence time t_0 per stage [min]  (Eq. 9 lower bound)
    t_low_deg_c, t_high_deg_c:  admissible air temperature range T in [30, 70] C (Eq. 9)
    beta_min:       starting beta of the homotopy
    beta_growth:    beta growth factor zeta (>= 1)
    eps:            convergence tolerance (max_p >= 1 - eps)
    max_iter:       maximum number of outer iterations
    init_t_min:     initial residence time [min]
    init_t_deg_c:   initial air temperature [C]
    solver:         inner constrained-optimization solver (scipy.optimize.minimize)
    """

    m: int
    x0_wet: float
    xd_wet: float
    alpha: float
    t_ambient_c: float = 25.0
    p_us: float = 15.0
    t_min: float = 2.0
    t_low_deg_c: float = 30.0
    t_high_deg_c: float = 70.0
    beta_min: float = 1e-4
    beta_growth: float = 1.5
    eps: float = 1e-3
    max_iter: int = 300
    init_t_min: float = 20.0
    init_t_deg_c: float = 50.0
    solver: str = "SLSQP"


@dataclass
class MepResult:
    """Algorithm 1 result."""

    path: Path
    etas: List[ControlParams]
    probs: np.ndarray
    costs: np.ndarray
    beta_final: float
    converged: bool
    history: List[float] = field(default_factory=list)   # per-iteration F* values


def _flatten(etas: List[ControlParams]) -> np.ndarray:
    """Flatten into a 1-D vector ordered as (t_1, T_1, ..., t_M, T_M)."""
    return np.array([v for e in etas for v in (e.t, e.T)], dtype=float)


def _unflatten(x: np.ndarray, m: int) -> List[ControlParams]:
    x = np.asarray(x, dtype=float)
    return [ControlParams(t=x[2 * k], T=x[2 * k + 1]) for k in range(m)]


def _evaluate(paths: List[Path], etas: List[ControlParams], cfg: MepConfig) -> np.ndarray:
    return np.array(
        [
            path_cost(
                p,
                etas,
                cfg.x0_wet,
                cfg.xd_wet,
                alpha=cfg.alpha,
                t_ambient_c=cfg.t_ambient_c,
                p_us=cfg.p_us,
            )
            for p in paths
        ]
    )


def solve(cfg: MepConfig) -> MepResult:
    """Run Algorithm 1 -- MEP combinatorial optimization.

    Returns
    -------
    MepResult
        Optimal path omega*, optimal parameters {eta*_k}, the final probability
        distribution, and whether the loop converged.
    """
    paths = list(enumerate_paths(cfg.m))
    etas = [ControlParams(t=cfg.init_t_min, T=cfg.init_t_deg_c) for _ in range(cfg.m)]
    x0 = _flatten(etas)
    bounds = [(cfg.t_min, None), (cfg.t_low_deg_c, cfg.t_high_deg_c)] * cfg.m

    beta = cfg.beta_min
    converged = False
    history: List[float] = []

    for _ in range(cfg.max_iter):
        costs = _evaluate(paths, etas, cfg)
        p = gibbs_probs(beta, costs)
        history.append(free_energy(beta, costs))

        if p.max() > 1.0 - cfg.eps:          # the path choice is effectively fixed
            converged = True
            break

        # [Step 1] Inner constrained optimization: minimize F*(beta; eta) (Eq. 27), warm-start
        def obj(x: np.ndarray) -> float:
            est = _unflatten(x, cfg.m)
            return free_energy(beta, _evaluate(paths, est, cfg))

        res = solve_inner(
            obj,
            x0,
            bounds=bounds,
            method=cfg.solver,
            maxiter=300,
            ftol=1e-10,
        )
        if not res.success:
            # TODO: log solver failures -- use the result and continue for now
            pass

        x0 = res.x
        etas = _unflatten(x0, cfg.m)

        # [Step 3] Increase beta by the growth factor zeta (Algorithm 1,
        #          "beta <- zeta * beta").  Multiplicative growth traces the
        #          homotopy from the convex (beta ~ 0) to the non-convex
        #          (large beta) objective; linear (+=) growth would skip the
        #          low-beta convex phase entirely.
        beta *= cfg.beta_growth

    # --- After the loop: pick the optimal path from the final distribution ---
    costs = _evaluate(paths, etas, cfg)
    p = gibbs_probs(beta, costs)
    if not converged and p.max() > 1.0 - cfg.eps:
        converged = True

    best_idx = int(np.argmax(p))
    return MepResult(
        path=paths[best_idx],
        etas=etas,
        probs=p,
        costs=costs,
        beta_final=beta,
        converged=converged,
        history=history,
    )