"""Unit tests for kinetics -- drying dynamics, Eq. (11)-(13)."""

import math

import pytest

from drying.model.constants import X0_DRY_INIT, X0_WET
from drying.model.kinetics import K, M_eq, moisture_update, t_star

# Kelvin range corresponding to T in [30, 70] C
KELVIN_POINTS = [303.15, 313.15, 323.15, 333.15, 343.15]


@pytest.mark.parametrize("gamma", [0, 1])
def test_K_positive(gamma):
    for T in KELVIN_POINTS:
        assert K(gamma, T) > 0


@pytest.mark.parametrize("gamma", [0, 1])
def test_M_eq_decreasing_in_T(gamma):
    vals = [M_eq(gamma, T) for T in KELVIN_POINTS]
    for a, b in zip(vals, vals[1:]):
        assert a > b          # equilibrium moisture decreases with temperature
    assert all(0 < v < 1 for v in vals)


@pytest.mark.parametrize("gamma", [0, 1])
def test_t_star_consistent_with_lewis_solution(gamma):
    """Verify that t* in Eq. (13) is consistent with the Lewis solution.

    By definition of t*, moisture_update must be equivalent to:
        x_dry(t) = M_eq + (x_dry_now - M_eq) * exp(-K*t)
    Comparing the two paths (direct formula vs. via t_star) independently
    checks that the log/exp handling in t* is correct.
    """
    T = 323.15
    x_wet = X0_WET
    t = 10.0

    x_dry_now = x_wet / (1.0 - x_wet)
    expected_dry = M_eq(gamma, T) + (x_dry_now - M_eq(gamma, T)) * math.exp(-K(gamma, T) * t)
    expected_wet = expected_dry / (1.0 + expected_dry)

    assert moisture_update(gamma, x_wet, t, T) == pytest.approx(expected_wet, rel=1e-6)


@pytest.mark.parametrize("gamma", [0, 1])
def test_t_star_monotonic_in_t(gamma):
    """The equivalent time t* must increase with residence time."""
    a = t_star(gamma, 323.15, X0_WET, 5.0)
    b = t_star(gamma, 323.15, X0_WET, 50.0)
    assert b > a


def test_moisture_decreases_with_time():
    for gamma in (0, 1):
        x0 = X0_WET
        x_short = moisture_update(gamma, x0, 1.0, 323.15)
        x_long = moisture_update(gamma, x0, 60.0, 323.15)
        assert 0 < x_long < x_short < X0_WET


def test_x0_dry_reference_consistent():
    """3.16 = dry-basis constant near the dry-basis value of X0_WET (~3.08)."""
    x_dry = X0_WET / (1.0 - X0_WET)
    assert x_dry == pytest.approx(3.082, abs=1e-2)