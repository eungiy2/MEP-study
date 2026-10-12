"""Unit tests for cost -- energy & path cost, Eq. (14)-(18)."""

import pytest

from drying.model.constants import T0_DEG_C
from drying.model.cost import ControlParams, g0, g1, path_cost, penalty
from drying.model.paths import enumerate_paths

A = 0.5
P = 20.0
T0 = T0_DEG_C


def eta(t=20.0, T=50.0):
    return ControlParams(t=t, T=T)


def test_g0_positive_and_monotone():
    assert g0(eta(t=10), A, T0) > 0
    assert g0(eta(t=30), A, T0) > g0(eta(t=10), A, T0)            # higher t -> higher cost
    assert g0(eta(T=60), A, T0) > g0(eta(T=40), A, T0)            # higher T -> higher cost


def test_g1_is_g0_plus_ultrasound():
    e = eta(t=20, T=50)
    assert g1(e, A, T0, P) == pytest.approx(g0(e, A, T0) + P * e.t)


def test_penalty():
    assert penalty(x_wet=0.070, xd_wet=0.075) == 0.0
    assert penalty(x_wet=0.09, xd_wet=0.075) > 0.0


def test_path_cost_increases_with_us_usage():
    """For the same (t, T), a HA/US (gamma=1) path is costlier than HA (gamma=0)."""
    M = 2
    etas = [eta(t=20, T=50), eta(t=20, T=50)]
    paths = list(enumerate_paths(M))
    cost = lambda p: path_cost(p, etas, x0_wet=0.755, xd_wet=0.075,
                               alpha=A, t_ambient_c=T0, p_us=P)
    costs = {p: cost(p) for p in paths}
    assert costs[(0, 0)] < costs[(1, 1)]
    assert costs[(0, 1)] < costs[(1, 1)]
    assert all(c > 0 for c in costs.values())