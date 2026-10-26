"""Inner constrained parameter optimization (Eq. 27) -- scipy.optimize wrapper.

The inner problem of Algorithm 1 minimizes the free energy

    F*(beta; eta) = -(1/beta) * log sum_omega exp(-beta * D(omega, eta))   Eq. (26)

w.r.t. the control parameters eta = (t_1, T_1, ..., t_M, T_M), subject to

    U = {(t, T) : t >= t_0,  T_low <= T <= T_high}                         Eq. (9)

As a study-mode note: the objective is smooth in eta (once the moisture
penalty term is accounted for), the variable scale of t (~minutes) and T
(~tens of degrees C) differs by ~1-2 orders of magnitude, and the homotopy
warm-starts each inner solve from the previous solution.  A bounded,
gradient-free, derivative-aware SLSQP run with moderate tolerances has
worked well here; the wrapper exists so the solver choice is configurable
without touching Algorithm 1.
"""

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize, OptimizeResult


def solve_inner(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    method: str = "SLSQP",
    maxiter: int = 300,
    ftol: float = 1e-10,
    verbose: bool = False,
) -> OptimizeResult:
    """Minimize ``objective`` over the box ``bounds``, warm-started at ``x0``.

    Parameters
    ----------
    objective : callable
        F*(beta; eta) as a function of the flattened control vector.
    x0 : np.ndarray
        Flattened (t_1, T_1, ..., t_M, T_M) -- previous homotopy solution.
    bounds : sequence of (lo, hi)
        Per-coordinate box constraints (Eq. 9).
    method, maxiter, ftol : solver settings (defaults tuned for this problem).
    verbose : bool
        If True, return the raw scipy result verbatim; otherwise still
        return it (callers inspect ``res.success`` / ``res.x``).

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``res.x`` is the optimal control vector; check ``res.success``.
    """
    res = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    if verbose or not res.success:
        return res
    return res