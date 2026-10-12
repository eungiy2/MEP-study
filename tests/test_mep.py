"""Integration tests for Algorithm 1 (MEP) -- structure / probability / convergence."""

import numpy as np

from drying.model.paths import enumerate_paths
from drying.optimization import MepConfig, solve
from drying.optimization.gibbs import gibbs_probs


def _cfg(m=2, **kw):
    defaults = dict(
        m=m,
        x0_wet=0.755,
        xd_wet=0.075,
        alpha=0.5,
        p_us=15.0,
        t_ambient_c=25.0,
        max_iter=120,
    )
    defaults.update(kw)
    return MepConfig(**defaults)


def test_solve_returns_valid_path():
    cfg = _cfg(m=2)
    res = solve(cfg)
    assert len(res.path) == 2
    assert all(g in (0, 1) for g in res.path)
    assert len(res.etas) == 2
    assert np.isclose(res.probs.sum(), 1.0)
    # the optimal path is argmax p
    assert res.path == list(enumerate_paths(2))[int(np.argmax(res.probs))]


def test_solve_m1():
    res = solve(_cfg(m=1))
    assert len(res.probs) == 2          # |Omega| = 2^1
    assert res.costs.shape == (2,)
    assert all(np.isfinite(res.costs))


def test_solve_converges_or_bounded():
    """Must converge within max_iter, or at least keep beta_final finite."""
    res = solve(_cfg(m=3, max_iter=200, beta_growth=2.0))
    assert np.isfinite(res.beta_final)
    assert 0.0 < res.probs.max() <= 1.0
    assert len(res.history) >= 1


def test_probs_progress_toward_binary():
    """As beta grows, max p must approach 1."""
    cfg = _cfg(m=2, max_iter=200, beta_growth=3.0)
    res = solve(cfg)
    intermediate = gibbs_probs(1e-4, res.costs)
    assert res.probs.max() >= intermediate.max() - 1e-9


def test_more_stages_raw_cost_higher_with_same_parameters():
    """Replicating the same stage eta increases the total cost with the stage count."""
    from drying.model.constants import XD_WET_TARGET
    from drying.model.cost import ControlParams, path_cost

    etas = [ControlParams(t=20.0, T=50.0)] * 4
    c4 = path_cost((1, 1, 1, 1), etas, 0.755, XD_WET_TARGET,
                   alpha=0.5, t_ambient_c=25.0, p_us=15.0)
    c3 = path_cost((1, 1, 1), etas[:3], 0.755, XD_WET_TARGET,
                   alpha=0.5, t_ambient_c=25.0, p_us=15.0)
    assert c4 > c3